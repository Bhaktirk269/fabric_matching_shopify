# app.py
import os, re, unicodedata, json
import io
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
from rapidfuzz import fuzz
# Supabase removed: materials are sourced from Google Sheets only
from dotenv import load_dotenv
import csv
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Load variables from a local .env file if present
load_dotenv()

SHOPIFY_ORDER_API = os.getenv("SHOPIFY_ORDER_API")

# Materials source: Google Sheets only
MATERIALS_SOURCE = "gsheet"
MATERIALS_GSHEET_URL = os.getenv("MATERIALS_GSHEET_URL")
MATERIALS_GSHEET_GID = os.getenv("MATERIALS_GSHEET_GID")

# Optional toggle for filtering only active materials (set MATERIALS_REQUIRE_ACTIVE=false to include all)
REQUIRE_ACTIVE = os.getenv("MATERIALS_REQUIRE_ACTIVE", "true").lower() == "true"

# Normalize upstream base: strip trailing /orderNumber or /orderNmuber if provided
if SHOPIFY_ORDER_API:
    SHOPIFY_ORDER_API = re.sub(r"/(orderNumber|orderNmuber)/?$", "", SHOPIFY_ORDER_API.rstrip("/"))

if not SHOPIFY_ORDER_API:
    raise RuntimeError("Missing env var: SHOPIFY_ORDER_API")
app = FastAPI(title="Fabric Matcher")

# ------------------------
# Utilities / Normalizers
# ------------------------

def norm(s: str) -> str:
    """
    Normalize: ASCII fold, lowercase, remove boilerplate tokens, strip filler words,
    unify separators, collapse spaces. (Colors are kept.)
    """
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    s = s.lower()

    # remove common boilerplate tokens
    s = re.sub(r"\bsr\.?\s*no\.?\b", " ", s)   # "Sr No", "Sr. No." -> remove label (number remains elsewhere)
    s = re.sub(r"^\s*a\s*-\s*", " ", s)        # leading "A -" marker at start of string

    # strip generic filler words
    s = re.sub(r"\b(fabric|fabrics|material|collection|cover|covers)\b", " ", s)

    # unify separators / punctuation
    s = s.replace("&", " and ")
    s = re.sub(r"[®™'\"()+/\\._-]+", " ", s)

    # keep alnum & space; collapse spaces
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    """Normalize and split into tokens; keep all words."""
    s = norm(s)
    toks = [t for t in s.split() if t]
    # dedupe preserving order
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def clean_list(values: List[str]) -> List[str]:
    dedup, seen = [], set()
    for v in values or []:
        vv = (v or "").strip()
        if vv and vv not in seen:
            dedup.append(vv); seen.add(vv)
    return dedup

# ------------------------
# Schema helpers
# ------------------------

def _coalesce_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    # Try exact keys
    for k in candidates:
        if k in d:
            return k
    # Try case-insensitive
    lower_map = {str(k).lower(): k for k in d.keys()}
    for k in candidates:
        lk = str(k).lower()
        if lk in lower_map:
            return lower_map[lk]
    # Try normalized (remove non-alnum)
    def nk(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())
    norm_map = {nk(k): k for k in d.keys()}
    for k in candidates:
        nn = nk(k)
        if nn in norm_map:
            return norm_map[nn]
    return None

def _standardize_material_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(raw)
    # id candidates
    id_key = _coalesce_key(raw, [
        "id", "ID", "uuid", "row_id", "rowid", "pk", "material_id", "materialId"
    ])
    out["id"] = raw.get(id_key) if id_key else None

    # material_name candidates
    mn_key = _coalesce_key(raw, [
        "material_name", "Material name", "materialName", "name", "Material", "material"
    ])
    if mn_key:
        out["material_name"] = raw.get(mn_key)

    # collection_name candidates
    cn_key = _coalesce_key(raw, [
        "collection_name", "Collection name", "collection", "Collection", "fabric_collection"
    ])
    if cn_key:
        out["collection_name"] = raw.get(cn_key)

    # website_name candidates
    wn_key = _coalesce_key(raw, [
        "website_name", "Website name", "Website Name", "website", "Website"
    ])
    if wn_key:
        out["website_name"] = raw.get(wn_key)

    # aliases candidates
    al_key = _coalesce_key(raw, [
        "aliases", "Aliases", "alias", "Alias", "aka", "AKAs"
    ])
    if al_key:
        out["aliases"] = raw.get(al_key)

    # active candidates
    ac_key = _coalesce_key(raw, [
        "active", "Active", "is_active", "enabled", "available"
    ])
    if ac_key is not None:
        val = raw.get(ac_key)
        if isinstance(val, bool):
            out["active"] = val
        elif isinstance(val, (int, float)):
            out["active"] = bool(val)
        elif isinstance(val, str):
            out["active"] = val.strip().lower() in {"true", "1", "yes", "y"}

    return out

# ------------------------
# Debug endpoints
# ------------------------

@app.get("/debug/order-raw/{order_number}")
async def debug_order_raw(order_number: str):
    async with httpx.AsyncClient(timeout=25) as client:
        # Try query param first, then path param fallback
        r = await client.get(SHOPIFY_ORDER_API, params={"orderNumber": str(order_number)})
        if r.status_code != 200:
            # Try known upstream typo key as a second attempt
            r = await client.get(SHOPIFY_ORDER_API, params={"orderNmuber": str(order_number)})
        if r.status_code != 200:
            base = (SHOPIFY_ORDER_API or "").rstrip("/")
            r = await client.get(f"{base}/{order_number}")
    # show a snippet of the upstream response for debugging
    return {
        "url": str(getattr(getattr(r, "request", None), "url", SHOPIFY_ORDER_API)),
        "status": r.status_code,
        "headers": dict(r.headers),
        "body_preview": r.text[:1000]
    }

@app.get("/debug/materials")
def debug_materials():
    mats = fetch_materials()
    sample = mats[:5]
    return {
        "count": len(mats),
        "sample": sample,
        "fields": ["id", "material_name", "collection_name", "aliases", "active"]
    }

# (Removed) /debug/supabase endpoint to avoid exposing Supabase configuration

@app.get("/debug/config")
def debug_config():
    # Return config without performing any external calls
    return {
        "require_active": REQUIRE_ACTIVE,
        "shopify_order_api": SHOPIFY_ORDER_API,
        "materials_source": MATERIALS_SOURCE,
        "gsheet_url": MATERIALS_GSHEET_URL,
        "gsheet_gid": MATERIALS_GSHEET_GID,
    }

# (Removed) /debug/rpc endpoint since Supabase RPC is not used in Google Sheets mode

# ------------------------
# Upstream + DB fetchers
# ------------------------

async def fetch_order(order_number: str) -> Dict[str, Any]:
    params = {"orderNumber": order_number}
    async with httpx.AsyncClient(timeout=25) as client:
        # Attempt query parameter style first
        r = await client.get(SHOPIFY_ORDER_API, params=params)
        # Try upstream typo param name if needed
        if r.status_code != 200:
            r = await client.get(SHOPIFY_ORDER_API, params={"orderNmuber": str(order_number)})
        # Fallback to path style if still non-200
        if r.status_code != 200:
            base = (SHOPIFY_ORDER_API or "").rstrip("/")
            r = await client.get(f"{base}/{order_number}")
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Order API error: {r.status_code}")
        try:
            data = r.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=502, detail="Order API returned non-JSON")
    # Support a few common shapes:
    if isinstance(data, dict) and "order" in data:
        return data["order"]
    # Upstream variant: { orderNumber, productProperties: [ { ... } ] }
    if isinstance(data, dict) and isinstance(data.get("productProperties"), list):
        return {"line_items": data["productProperties"]}
    return data  # assume top-level is the order

def _parse_aliases(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x) for x in arr]
            except Exception:
                pass
        return [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    return []

def fetch_materials_from_gsheet() -> List[Dict[str, Any]]:
    if not MATERIALS_GSHEET_URL:
        raise HTTPException(status_code=500, detail="MATERIALS_GSHEET_URL not set for Google Sheets source")
    try:
        csv_url = _gsheet_export_csv_url(sheet_url=MATERIALS_GSHEET_URL, gid=MATERIALS_GSHEET_GID)
        with httpx.Client(timeout=60) as client:
            r = client.get(csv_url, follow_redirects=True)
            if r.status_code != 200:
                snippet = (r.text or "")[:300]
                raise HTTPException(status_code=502, detail=f"Google Sheets fetch error: {r.status_code} {snippet}")
            text = r.text
        f = io.StringIO(text)
        reader = csv.DictReader(f)
        out: List[Dict[str, Any]] = []
        for row in reader:
            material_name = row.get("Material name") or row.get("material_name") or row.get("Material") or row.get("name")
            if not material_name:
                continue
            collection_name = row.get("Collection name") or row.get("collection_name") or row.get("collection")
            website_name = row.get("Website Name") or row.get("website_name") or row.get("Website name") or row.get("Website")
            aliases = _parse_aliases(row.get("Aliases") or row.get("aliases") or "")
            active_raw = row.get("Active") or row.get("active")
            active_val: Optional[bool] = None
            if active_raw is not None:
                ar = str(active_raw).strip().lower()
                active_val = ar in {"true", "1", "yes", "y"}
            else:
                active_val = True
            item = {
                "id": None,
                "material_name": material_name,
                "collection_name": collection_name,
                "website_name": website_name,
                "aliases": aliases,
                "active": active_val,
            }
            out.append(item)
        if REQUIRE_ACTIVE:
            out = [r for r in out if r.get("active") is True]
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GSheet fetch error: {e}")

def fetch_materials() -> List[Dict[str, Any]]:
    """Fetch materials from the configured source; parse aliases into a list; optionally require active=true."""
    return fetch_materials_from_gsheet()

# ------------------------
# Candidate extraction
# ------------------------

def extract_candidates_from_line_item(li: Dict[str, Any]) -> Dict[str, List[str]]:
    # properties
    props = li.get("properties") or li.get("line_item_properties") or []
    kv = {}
    if props and isinstance(props, list):
        for p in props:
            # accept both dict {"name","value"} and {"key","value"} shapes
            k = (p.get("name") or p.get("key") or "").strip()
            v = (p.get("value") or "").strip()
            if k:
                kv[k] = v
    else:
        # If it's a flat dict of properties (from upstream), pull string keys
        for k, v in li.items():
            if isinstance(k, str) and isinstance(v, str):
                kk = k.strip()
                if kk:
                    kv[kk] = v.strip()

    # keys likely to contain fabrics
    def split_candidates(text: str) -> List[str]:
        # split on commas/semicolons; keep clean pieces
        parts = [p.strip() for p in re.split(r"[;,]", text or "") if (p or "").strip()]
        return parts or ([text] if text else [])

    key_hits = []
    for k, v in kv.items():
        lk = k.lower()
        if any(t in lk for t in ["fabric", "material", "collection", "cover"]) or ("website" in lk and "name" in lk):
            key_hits.extend(split_candidates(v))

    variant_title = li.get("variant_title") or ""
    title = li.get("title") or ""  # some shops put fabric or collection here

    # pull explicit collection property, if present
    explicit_collection = (
        kv.get("Fabric Collection")
        or kv.get("Fabrics Collections")
        or kv.get("Collection")
        or ""
    )
    # pull explicit website name, if present
    explicit_website = (
        kv.get("Website Name")
        or kv.get("website name")
        or kv.get("Website")
        or kv.get("website")
        or kv.get("WebsiteName")
        or ""
    )

    # include product title as a weaker fallback candidate; split comma-separated lists
    fabric_option_candidates = clean_list(
        key_hits
        + split_candidates(variant_title)
        + split_candidates(title)
        + split_candidates(explicit_website)
    )
    fabric_collection_candidates = clean_list([explicit_collection])

    return {
        "fabric_option_candidates": fabric_option_candidates,
        "fabric_collection_candidates": fabric_collection_candidates
    }

# ------------------------
# Matching logic
# ------------------------

def match_one(cands: Dict[str, List[str]], materials: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not materials:
        return None

    # Precompute normalized (material_name only)
    mats = []
    for m in materials:
        base_norm = norm(m["material_name"])
        alias_norms = []
        try:
            for a in (m.get("aliases") or []):
                if isinstance(a, str) and a.strip():
                    alias_norms.append(norm(a))
        except Exception:
            pass
        name_norms = [n for n in [base_norm] + alias_norms if n]
        mats.append({
            **m,
            "norm": base_norm,
            "name_norms": name_norms,
            "collection_norm": norm(m.get("collection_name") or ""),
            "website_norm": norm(m.get("website_name") or "")
        })

    option_norms = [norm(x) for x in cands.get("fabric_option_candidates", []) if x]
    coll_norms = [norm(x) for x in cands.get("fabric_collection_candidates", []) if x]

    def coll_ok(m) -> bool:
        # Allow collection candidates to match collection_name OR material_name
        if not coll_norms:
            return False
        fields = [m.get("collection_norm") or "", m.get("norm") or ""]
        for field in fields:
            if not field:
                continue
            for c in coll_norms:
                if c == field or (c in field) or (field in c):
                    return True
        return False

    def simple(s: str) -> str:
        return re.sub(r"\s+", "", s or "")

    # -------- PASS W: WEBSITE NAME PRIORITY (exact then substring) --------
    for cand in option_norms:
        cand_simple = simple(cand)
        for m in mats:
            wn = m.get("website_norm") or ""
            if not wn:
                continue
            wn_simple = simple(wn)
            # exact
            if cand == wn:
                return {
                    "id": m.get("id"),
                    "name": m["material_name"],
                    "collection": m.get("collection_name"),
                    "score": 1.0,
                    "via": "website_exact",
                    "coll_match": True
                }
            # substring either direction (no-space too)
            if (wn in cand or cand in wn) or (wn_simple and (wn_simple in cand_simple or cand_simple in wn_simple)):
                return {
                    "id": m.get("id"),
                    "name": m["material_name"],
                    "collection": m.get("collection_name"),
                    "score": 0.99,
                    "via": "website_substring",
                    "coll_match": True
                }

    # -------- PASS 0: SUBSTRING (both directions, no-space form too) --------
    for cand in option_norms:
        cand_simple = simple(cand)
        for m in mats:
            for nrm in m.get("name_norms", [m.get("norm", "")]):
                m_simple = simple(nrm)
                if (nrm and (nrm in cand or cand in nrm)) \
                   or (m_simple and (m_simple in cand_simple or cand_simple in m_simple)):
                    return {
                        "id": m.get("id"),
                        "name": m["material_name"],
                        "collection": m.get("collection_name"),
                        "score": 0.99,
                        "via": "substring",
                        "coll_match": coll_ok(m)
                    }

    # -------- PASS 1: TOKEN-SUBSET (order-agnostic containment) --------
    def token_subset_ratio(q_tokens: List[str], t_tokens: List[str]) -> float:
        q, t = set(q_tokens), set(t_tokens)
        return len(q & t) / max(1, len(q))

    for cand in option_norms:
        q_tokens = cand.split()
        for m in mats:
            name_tokens_variants = [(nrm or "").split() for nrm in m.get("name_norms", [m.get("norm", "")])]
            for t_tokens in name_tokens_variants:
                cov = token_subset_ratio(q_tokens, t_tokens)
                rev = token_subset_ratio(t_tokens, q_tokens)
                if cov >= 0.90 or rev >= 0.90 or (len(q_tokens) >= 3 and cov >= 0.80):
                    return {
                        "id": m.get("id"),
                        "name": m["material_name"],
                        "collection": m.get("collection_name"),
                        "score": 0.96,
                        "via": "token_subset",
                        "coll_match": coll_ok(m)
                    }

    # -------- PASS 2: EXACT (normalized) --------
    for cand in option_norms:
        for m in mats:
            for nrm in m.get("name_norms", [m.get("norm", "")]):
                if cand == nrm:
                    return {
                        "id": m.get("id"),
                        "name": m["material_name"],
                        "collection": m.get("collection_name"),
                        "score": 1.0,
                        "via": "exact",
                        "coll_match": coll_ok(m)
                    }

    # -------- PASS 2.5: COLLECTION SUBSTRING FALLBACK (broadened) --------
    if coll_norms:
        for coll in coll_norms:
            coll_simple = simple(coll)
            for m in mats:
                for field in [m.get("collection_norm") or "", m.get("norm") or ""]:
                    if not field:
                        continue
                    field_simple = simple(field)
                    if (field and (coll in field or field in coll)) or (field_simple and (coll_simple in field_simple or field_simple in coll_simple)):
                        return {
                            "id": m.get("id"),
                            "name": m["material_name"],
                            "collection": m.get("collection_name"),
                            "score": 0.9,
                            "via": "collection_substring",
                            "coll_match": True
                        }

    # -------- PASS 3: FUZZY BACKUP --------
    best = None
    for cand in option_norms:
        for m in mats:
            for nrm in m.get("name_norms", [m.get("norm", "")]):
                r_base = fuzz.ratio(cand, nrm) / 100.0
                r_part = fuzz.partial_ratio(cand, nrm) / 100.0
                r_tok  = fuzz.token_set_ratio(cand, nrm) / 100.0
                score = max(r_base, r_part, r_tok)
                if nrm in cand or cand in nrm:
                    score = max(score, 0.95)
                co = coll_ok(m)
                if (not best) or (score > best["score"]) or (score == best["score"] and co and not best["coll_match"]):
                    best = {"mat": m, "score": score, "coll_match": co}

    if best and (best["score"] >= 0.90 or (best["score"] >= 0.80 and best["coll_match"])):
        m = best["mat"]
        return {
            "id": m.get("id"),
            "name": m["material_name"],
            "collection": m.get("collection_name"),
            "score": round(float(best["score"]), 3),
            "via": "fuzzy",
            "coll_match": best["coll_match"]
        }
    # No DB RPC fallback in Google Sheets-only mode
    return None

# ------------------------
# API Models & Routes
# ------------------------

class MatchResponse(BaseModel):
    order_number: str
    matches: List[Dict[str, Any]]

@app.get("/match/order/{order_number}", response_model=MatchResponse)
async def match_order(order_number: str, skip_empty: bool = False):
    order = await fetch_order(order_number)
    line_items = order.get("line_items") or order.get("items") or []
    if not isinstance(line_items, list):
        raise HTTPException(status_code=422, detail="Order payload missing line items")

    materials = fetch_materials()
    results = []
    for li in line_items:
        cands = extract_candidates_from_line_item(li)
        no_cands = not cands.get("fabric_option_candidates") and not cands.get("fabric_collection_candidates")
        if skip_empty and no_cands:
            continue
        match = match_one(cands, materials)
        if not match:
            unmatched = {"id": None, "score": 0.0, "via": "unmatched"}
            if no_cands:
                unmatched["reason"] = "no_candidates"
            match = unmatched
        results.append({
            "line_item_id": li.get("id"),
            "sku": li.get("sku"),
            "title": li.get("title"),
            "candidates": cands,
            "match": match
        })

    return MatchResponse(order_number=order_number, matches=results)

@app.get("/match/order/{order_number}/summary")
async def match_order_summary(order_number: str, skip_empty: bool = True):
    order = await fetch_order(order_number)
    line_items = order.get("line_items") or order.get("items") or []
    if not isinstance(line_items, list):
        raise HTTPException(status_code=422, detail="Order payload missing line items")

    materials = fetch_materials()
    lines = []
    for idx, li in enumerate(line_items, start=1):
        cands = extract_candidates_from_line_item(li)
        no_cands = not cands.get("fabric_option_candidates") and not cands.get("fabric_collection_candidates")
        if skip_empty and no_cands:
            continue
        m = match_one(cands, materials) or {"id": None, "score": 0.0, "via": "unmatched"}
        lines.append({
            "line_index": idx,
            "title": li.get("title"),
            "sku": li.get("sku"),
            "fabric_option_candidates": cands.get("fabric_option_candidates", []),
            "fabric_collection_candidates": cands.get("fabric_collection_candidates", []),
            "match": {
                # "id": m.get("id"),
                "name": m.get("name"),
                "collection": m.get("collection"),
                "score": m.get("score"),
                "via": m.get("via")
            }
        })
    return {"order_number": order_number, "lines": lines}

@app.get("/match/order/{order_number}/chosen")
async def match_order_chosen(order_number: str, skip_empty: bool = True):
    order = await fetch_order(order_number)
    line_items = order.get("line_items") or order.get("items") or []
    materials = fetch_materials()
    chosen = []
    for li in line_items:
        cands = extract_candidates_from_line_item(li)
        if skip_empty and not (cands.get("fabric_option_candidates") or cands.get("fabric_collection_candidates")):
            continue
        m = match_one(cands, materials)
        chosen.append({
            "title": li.get("title"),
            "sku": li.get("sku"),
            "chosen_fabric": m.get("name") if m else None,
            "via": m.get("via") if m else "unmatched",
            "score": m.get("score") if m else 0.0
        })
    return {"order_number": order_number, "chosen": chosen}

@app.get("/materials")
def list_materials(limit: int = 100, offset: int = 0, active: Optional[bool] = None):
    rows = fetch_materials()
    if active is not None:
        rows = [r for r in rows if r.get("active") is True] if active else [r for r in rows if not r.get("active")]
    total = len(rows)
    start = max(0, offset)
    end = start + max(0, limit)
    items = rows[start:end]
    return {"total": total, "limit": limit, "offset": offset, "items": items, "source": MATERIALS_SOURCE}

@app.get("/match/orders")
async def match_orders(orderNumbers: str, skip_empty: bool = False):
    nums = [n.strip() for n in (orderNumbers or "").split(",") if n.strip()]
    if not nums:
        raise HTTPException(status_code=400, detail="Provide comma-separated orderNumbers query param")
    materials = fetch_materials()
    out = {}
    for n in nums:
        order = await fetch_order(n)
        line_items = order.get("line_items") or order.get("items") or []
        if not isinstance(line_items, list):
            out[n] = {"error": "Order payload missing line items"}
            continue
        results = []
        for li in line_items:
            cands = extract_candidates_from_line_item(li)
            no_cands = not cands.get("fabric_option_candidates") and not cands.get("fabric_collection_candidates")
            if skip_empty and no_cands:
                continue
            match = match_one(cands, materials)
            if not match:
                unmatched = {"id": None, "score": 0.0, "via": "unmatched"}
                if no_cands:
                    unmatched["reason"] = "no_candidates"
                match = unmatched
            results.append({
                "line_item_id": li.get("id"),
                "sku": li.get("sku"),
                "title": li.get("title"),
                "candidates": cands,
                "match": match
            })
        out[n] = {"order_number": n, "matches": results}
    return out

@app.get("/")
def health():
    # show the routes so we can see if /match/order/{order_number} is registered
    return {"ok": True, "routes": [getattr(r, "path", str(r)) for r in app.routes]}

# ------------------------
# Fabric master endpoints
# ------------------------

FABRIC_MASTER_DIR = Path(__file__).parent / "fabric_master"
FABRIC_MASTER_DIR.mkdir(exist_ok=True)

# Frontend mount
FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)
if (FRONTEND_DIR / "index.html").exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
else:
    # Provide a hint file if missing
    @app.get("/ui")
    def ui_placeholder():
        return {"info": "Place index.html in frontend/ to enable the UI."}

@app.get("/fabric-master/files")
def fabric_master_list():
    try:
        files = []
        for p in sorted(FABRIC_MASTER_DIR.glob("*")):
            if p.is_file():
                files.append({
                    "name": p.name,
                    "size": p.stat().st_size
                })
        return {"directory": str(FABRIC_MASTER_DIR), "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fabric-master/upload")
async def fabric_master_upload(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        dest = FABRIC_MASTER_DIR / file.filename
        content = await file.read()
        dest.write_bytes(content)
        return {"saved": dest.name, "bytes": len(content)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _read_csv_preview(path: Path, limit: int = 10):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= max(0, limit):
                break
            rows.append(row)
        return {"headers": reader.fieldnames or [], "rows": rows}

@app.get("/fabric-master/preview")
def fabric_master_preview(filename: str, limit: int = 10):
    try:
        path = FABRIC_MASTER_DIR / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        if path.suffix.lower() not in {".csv"}:
            raise HTTPException(status_code=400, detail="Only .csv supported")
        return _read_csv_preview(path, limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fabric-master/import")
def fabric_master_import(filename: str, require_active: Optional[bool] = None):
    try:
        path = FABRIC_MASTER_DIR / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        if path.suffix.lower() not in {".csv"}:
            raise HTTPException(status_code=400, detail="Only .csv supported")
        preview = _read_csv_preview(path, limit=0)
        headers = preview.get("headers", [])
        if not headers:
            raise HTTPException(status_code=400, detail="CSV has no headers")
        inserted, failed = 0, 0
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Accept multiple common header variants
                    material_name = row.get("Material name") or row.get("material_name") or row.get("Material") or row.get("name")
                    collection_name = row.get("Collection name") or row.get("collection_name") or row.get("collection")
                    website_name = row.get("Website Name") or row.get("website_name") or row.get("Website name") or row.get("Website")
                    aliases_raw = row.get("Aliases") or row.get("aliases") or ""
                    active_raw = row.get("Active") or row.get("active")
                    active_val = None
                    if require_active is not None:
                        active_val = bool(require_active)
                    elif active_raw is not None:
                        ar = str(active_raw).strip().lower()
                        active_val = ar in {"true","1","yes","y"}
                    aliases = []
                    if isinstance(aliases_raw, str):
                        if aliases_raw.strip().startswith("["):
                            try:
                                arr = json.loads(aliases_raw)
                                if isinstance(arr, list):
                                    aliases = [str(x) for x in arr]
                                else:
                                    aliases = [a.strip() for a in re.split(r"[;,]", aliases_raw) if a.strip()]
                            except Exception:
                                aliases = [a.strip() for a in re.split(r"[;,]", aliases_raw) if a.strip()]
                        else:
                            aliases = [a.strip() for a in re.split(r"[;,]", aliases_raw) if a.strip()]
                    elif isinstance(aliases_raw, list):
                        aliases = [str(x) for x in aliases_raw]

                    if not material_name:
                        failed += 1
                        continue

                    payload = {
                        "Material name": material_name,
                        "Collection name": collection_name,
                        "Aliases": aliases,
                    }
                    if website_name:
                        # Store as "Website Name" to align with the CSV and table schema
                        payload["Website Name"] = website_name
                    # In Google Sheets-only mode, we do not insert into any database.
                    # Count as preview-only import.
                    inserted += 1
                except Exception:
                    failed += 1
                    continue
        return {"inserted": inserted, "failed": failed}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------
# Google Sheets CSV import (no local files required)
# ------------------------

def _gsheet_export_csv_url(sheet_url: Optional[str] = None, spreadsheet_id: Optional[str] = None, gid: Optional[str] = None) -> str:
    if sheet_url:
        p = urlparse(sheet_url)
        m = re.search(r"/d/([^/]+)/", p.path or "")
        if m:
            spreadsheet_id = spreadsheet_id or m.group(1)
        q = parse_qs(p.query or "")
        if (not gid) and ("gid" in q and q["gid"]):
            gid = q["gid"][0]
        # Some links carry gid in the fragment (#gid=...)
        if not gid and (p.fragment or ""): 
            fq = parse_qs(p.fragment)
            if "gid" in fq and fq["gid"]:
                gid = fq["gid"][0]
    if not spreadsheet_id:
        raise HTTPException(status_code=400, detail="Provide a Google Sheets sharing URL or spreadsheet_id")
    base = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"
    if gid:
        base += f"&gid={gid}"
    return base

def _read_csv_text_preview(csv_text: str, limit: int = 10):
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    rows = []
    for i, row in enumerate(reader):
        if i >= max(0, limit):
            break
        rows.append(row)
    return {"headers": reader.fieldnames or [], "rows": rows}

@app.get("/fabric-master/gsheet/preview")
def gsheet_preview(url: Optional[str] = None, spreadsheet_id: Optional[str] = None, gid: Optional[str] = None, limit: int = 10):
    try:
        csv_url = _gsheet_export_csv_url(sheet_url=url, spreadsheet_id=spreadsheet_id, gid=gid)
        with httpx.Client(timeout=30) as client:
            r = client.get(csv_url, follow_redirects=True)
            if r.status_code != 200:
                snippet = (r.text or "")[:300]
                raise HTTPException(status_code=502, detail=f"Google Sheets fetch error: {r.status_code} {snippet}")
            text = r.text
        return _read_csv_text_preview(text, limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fabric-master/gsheet/import")
def gsheet_import(url: Optional[str] = None, spreadsheet_id: Optional[str] = None, gid: Optional[str] = None, require_active: Optional[bool] = None):
    try:
        csv_url = _gsheet_export_csv_url(sheet_url=url, spreadsheet_id=spreadsheet_id, gid=gid)
        with httpx.Client(timeout=60) as client:
            r = client.get(csv_url, follow_redirects=True)
            if r.status_code != 200:
                snippet = (r.text or "")[:300]
                raise HTTPException(status_code=502, detail=f"Google Sheets fetch error: {r.status_code} {snippet}")
            text = r.text
        f = io.StringIO(text)
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise HTTPException(status_code=400, detail="CSV has no headers")
        inserted, failed = 0, 0
        for row in reader:
            try:
                material_name = row.get("Material name") or row.get("material_name") or row.get("Material") or row.get("name")
                collection_name = row.get("Collection name") or row.get("collection_name") or row.get("collection")
                website_name = row.get("Website Name") or row.get("website_name") or row.get("Website name") or row.get("Website")
                aliases_raw = row.get("Aliases") or row.get("aliases") or ""
                active_raw = row.get("Active") or row.get("active")
                active_val = None
                if require_active is not None:
                    active_val = bool(require_active)
                elif active_raw is not None:
                    ar = str(active_raw).strip().lower()
                    active_val = ar in {"true","1","yes","y"}
                aliases = []
                if isinstance(aliases_raw, str):
                    if aliases_raw.strip().startswith("["):
                        try:
                            arr = json.loads(aliases_raw)
                            if isinstance(arr, list):
                                aliases = [str(x) for x in arr]
                            else:
                                aliases = [a.strip() for a in re.split(r"[;,]", aliases_raw) if a.strip()]
                        except Exception:
                            aliases = [a.strip() for a in re.split(r"[;,]", aliases_raw) if a.strip()]
                    else:
                        aliases = [a.strip() for a in re.split(r"[;,]", aliases_raw) if a.strip()]
                elif isinstance(aliases_raw, list):
                    aliases = [str(x) for x in aliases_raw]

                if not material_name:
                    failed += 1
                    continue

                payload = {
                    "Material name": material_name,
                    "Collection name": collection_name,
                    "Aliases": aliases,
                }
                if website_name:
                    payload["Website Name"] = website_name
                if active_val is not None:
                    payload["Active"] = active_val
                # No DB insert in Google Sheets-only mode.
                inserted += 1
            except Exception:
                failed += 1
                continue
        return {"inserted": inserted, "failed": failed}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
