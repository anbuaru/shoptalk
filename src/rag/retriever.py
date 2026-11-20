"""Stage 2 of the ShopTalk RAG pipeline: retrieval.

  1. Embed the semantic query with the (fine-tuned) embedding model, using the same
     query prefix the index was built for (E5: "query: ").
  2. Search Chroma with hard metadata filters where the data supports them (price).
     Price must be stored as a NUMBER in Chroma metadata, otherwise $gte/$lte never match.
  3. Apply soft attribute matching (color, gender, brand, size) in Python, because ABO
     attribute values are free-form ("Navy Blue", "Mens") and Chroma only does exact matches.
  4. Optionally rerank the candidates with a cross-encoder.
  5. Product type ("shirt" -> SHIRT) is searched as a Chroma filter first; other types are used
     only when none of the requested type passes the filters. Order by (type match, #attributes matched, relevance), return top-k.
  6. If even the best match is below `min_similarity` (and no item of the requested product type was
     found), return nothing, so the generator says "no match" instead of recommending something unrelated.

ABO price coverage is sparse in the raw data. `missing_price_policy` controls what happens when a
price filter leaves too few results:
  * "strict"   - only priced products that satisfy the filter
  * "fallback" - top up with products that have no price, flagged price_unknown=True (default)
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

from .query_parser import COLOR_SYNONYMS, GENDER_PATTERNS, ParsedQuery

KNOWN_DOC_PREFIXES = ("passage: ", "query: ")


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def strip_doc_prefix(doc: str) -> str:
    """Documents indexed with an E5 'passage: ' prefix: drop it before showing text to the reranker/LLM."""
    for p in KNOWN_DOC_PREFIXES:
        if doc.startswith(p):
            return doc[len(p):]
    return doc


def load_sentence_model(path: str, device: str = "cuda"):
    """Load a SentenceTransformer from a full fine-tune folder, a LoRA adapter folder, or an HF id."""
    import json
    import os

    from sentence_transformers import SentenceTransformer

    adapter_cfg = os.path.join(str(path), "adapter_config.json")
    if os.path.exists(adapter_cfg) and not os.path.exists(os.path.join(str(path), "modules.json")):
        from peft import PeftModel
        base = json.load(open(adapter_cfg))["base_model_name_or_path"]
        model = SentenceTransformer(base, device=device)
        model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, str(path)).merge_and_unload()
        return model
    return SentenceTransformer(str(path), device=device)


class Embedder:
    """Thin wrapper so the query prefix (E5: 'query: ', BGE: 'Represent this sentence...')
    matches what was used in Phase 5. Use the same model and prefixes as the index."""

    def __init__(self, model, query_prefix: str = "", doc_prefix: str = ""):
        self.model = model  # a sentence_transformers.SentenceTransformer
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_query(self, text: str) -> list:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0].tolist()

    def embed_docs(self, texts: list, batch_size: int = 64) -> list:
        return self.model.encode([self.doc_prefix + t for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=False).tolist()


class CrossEncoderReranker:
    def __init__(self, model, max_chars: int = 1000):
        self.model = model  # a sentence_transformers.CrossEncoder
        self.max_chars = max_chars

    def scores(self, query: str, docs: list) -> list:
        if not docs:
            return []
        return [float(s) for s in self.model.predict([(query, d[: self.max_chars]) for d in docs])]


def _size_regex(size: str) -> str:
    s = re.escape(size.lower())
    if size[0].isdigit():
        # "size 10", "10 M US", "10 D(M)", "10 wide" - but not any stray "10" in the text
        return rf"\bsize\s*:?\s*{s}\b|\b{s}\s*(?:us|uk|eu|m|w|d|b|wide|medium|narrow|\(m\)|d\(m\))\b"
    if len(size) <= 1:
        # single letters (S/M/L) only count when written as a size
        return rf"\bsize\s*:?\s*{s}\b|\({s}\)"
    return rf"\bsize\s*:?\s*{s}\b|\b{s}\b"


class ChromaRetriever:
    def __init__(
        self,
        collection,
        embedder: Embedder,
        reranker: Optional[CrossEncoderReranker] = None,
        candidate_k: int = 60,
        price_key: str = "price",
        color_key: str = "color",
        brand_key: str = "brand",
        title_key: str = "title",
        type_key: Optional[str] = None,
        missing_price_policy: str = "fallback",
        fallback_margin: float = 0.15,
        min_similarity: Optional[float] = None,
    ):
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.price_key, self.color_key = price_key, color_key
        self.brand_key, self.title_key = brand_key, title_key
        self.type_key = type_key
        assert missing_price_policy in ("strict", "fallback")
        self.missing_price_policy = missing_price_policy
        self.fallback_margin = fallback_margin
        self.min_similarity = min_similarity
        self.last_best_similarity: Optional[float] = None

    # ---------- Chroma search ----------
    def build_where(self, pq: ParsedQuery, with_type: bool = False) -> Optional[dict]:
        conds = []
        if with_type and self.type_key and pq.product_types:
            types = list(pq.product_types)
            conds.append({self.type_key: types[0]} if len(types) == 1 else {self.type_key: {"$in": types}})
        if pq.min_price is not None:
            conds.append({self.price_key: {"$gte": float(pq.min_price)}})
        if pq.max_price is not None:
            conds.append({self.price_key: {"$lte": float(pq.max_price)}})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def _search(self, qvec: list, where: Optional[dict], n: int) -> list:
        n = max(1, min(n, self.collection.count()))
        res = self.collection.query(query_embeddings=[qvec], n_results=n, where=where,
                                    include=["documents", "metadatas", "distances"])
        out = []
        for pid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            out.append({"id": pid, "document": strip_doc_prefix(doc or ""), "metadata": meta or {},
                        "distance": float(dist), "similarity": 1.0 - float(dist)})
        return out

    # ---------- soft attribute matching ----------
    def _attr_matches(self, p: dict, pq: ParsedQuery) -> dict:
        meta, doc = p["metadata"], p["document"].lower()
        title = str(meta.get(self.title_key, "")).lower()
        matched = {}
        if pq.colors:
            pcolor = str(meta.get(self.color_key, "")).lower()
            wanted = set()
            for c in pq.colors:
                wanted |= COLOR_SYNONYMS.get(c, {c})
            if pcolor:
                hay = pcolor
            else:  # no color field: look in the text, minus the brand ("RED WAGON" is not red)
                brand = str(meta.get(self.brand_key, "")).lower()
                hay = (title + " " + doc[:300]).replace(brand, " ") if brand else title + " " + doc[:300]
            matched["color"] = any(re.search(rf"\b{re.escape(w)}\b", hay) for w in wanted)
        if pq.gender:
            hay = title + " " + doc[:500]
            is_w = bool(re.search(GENDER_PATTERNS["women"], hay))
            is_m = bool(re.search(GENDER_PATTERNS["men"], hay))
            is_u = bool(re.search(GENDER_PATTERNS["unisex"], hay))
            is_k = bool(re.search(GENDER_PATTERNS["kids"], hay))
            if pq.gender == "men":
                matched["gender"] = (is_m or is_u) and not is_k
            elif pq.gender == "women":
                matched["gender"] = (is_w or is_u) and not is_k
            elif pq.gender == "kids":
                matched["gender"] = is_k
            else:
                matched["gender"] = is_u
        if pq.brand:
            matched["brand"] = pq.brand.lower() in str(meta.get(self.brand_key, "")).strip().lower()
        if pq.product_types and self.type_key:
            matched["type"] = meta.get(self.type_key) in set(pq.product_types)
        elif getattr(pq, "type_unavailable", False):
            matched["type"] = False   # the catalog doesn't sell the requested kind of product
        if pq.size:
            matched["size"] = bool(re.search(_size_regex(pq.size), title + " " + doc))
        return matched

    # ---------- main entry ----------
    def retrieve(self, pq: ParsedQuery, top_k: int = 10, exclude_ids: Optional[list] = None,
                 followup: bool = False) -> list:
        """followup=True relaxes the min_similarity floor by `fallback_margin`: the topic was already accepted
        in the previous turn, and excluding the products already shown lowers the best score."""
        exclude = set(exclude_ids or [])
        qvec = self.embedder.embed_query(pq.semantic_query)
        where = self.build_where(pq)
        use_type = bool(self.type_key and pq.product_types)
        n = self.candidate_k + len(exclude)
        if use_type:
            # 1) only the requested product type; 2) other types only if NONE of that type passes the filters
            #    (2 real shirts beat 2 shirts padded with 3 beanies; the answer then says what is missing)
            cands = [c for c in self._search(qvec, self.build_where(pq, with_type=True), n) if c["id"] not in exclude]
            if not cands:
                cands = [c for c in self._search(qvec, where, n) if c["id"] not in exclude]
                for c in cands:
                    c["type_fallback"] = True
        else:
            cands = [c for c in self._search(qvec, where, n) if c["id"] not in exclude]
            if getattr(pq, "type_unavailable", False):
                for c in cands:
                    c["type_fallback"] = True   # only alternatives that match the user's other wishes survive
        for c in cands:
            c["price_unknown"] = _is_missing(c["metadata"].get(self.price_key))

        # Price filter left too few results -> top up with unpriced products
        if where is not None and self.missing_price_policy == "fallback" and len(cands) < top_k:
            seen = {c["id"] for c in cands} | exclude
            extra = [c for c in self._search(qvec, None, self.candidate_k * 2)
                     if c["id"] not in seen and _is_missing(c["metadata"].get(self.price_key))]
            for c in extra:
                c["price_unknown"] = True
                c["fallback"] = True
            cands += extra

        self.last_best_similarity = max((c["similarity"] for c in cands), default=None)
        if not cands:
            return []
        # nothing in the catalog is close enough -> let the generator say "no match"
        # Items of the requested product type exist -> the request is in-catalog; the floor is meant to catch
        # requests the catalog doesn't sell ("car tires"), so it is skipped then.
        type_hit = use_type and not any(c.get("type_fallback") for c in cands)
        floor = None if self.min_similarity is None or type_hit else \
            self.min_similarity - (self.fallback_margin if followup else 0.0)
        if floor is not None and self.last_best_similarity < floor:
            return []

        if self.reranker is not None:
            for c, s in zip(cands, self.reranker.scores(pq.semantic_query, [c["document"] for c in cands])):
                c["rerank_score"] = s
        for c in cands:
            c["matched"] = self._attr_matches(c, pq)
            c["n_matched"] = sum(c["matched"].values())
            c["relevance"] = c.get("rerank_score", c["similarity"])
        # unpriced top-ups must satisfy every requested attribute and be close to the best match
        best = self.last_best_similarity
        cands = [c for c in cands if not c.get("fallback") or
                 (c["n_matched"] == len(c["matched"]) and c["similarity"] >= best - self.fallback_margin)]
        # other-type alternatives only make sense when they share the user's other wishes
        # (no shirts under $50 -> red men's items are fair alternatives; "shirt under $15" -> no mugs)
        cands = [c for c in cands if not c.get("type_fallback") or
                 (len(c["matched"]) > 1 and all(v for k, v in c["matched"].items() if k != "type"))]
        if not cands:
            return []

        # the requested product type first (a red beanie is not a red shirt), then priced-and-in-range,
        # then the number of other attributes matched, then relevance
        has_price = pq.min_price is not None or pq.max_price is not None
        cands.sort(key=lambda c: (c["matched"].get("type", True),
                                  (not c["price_unknown"]) if has_price else True,
                                  c["n_matched"], c["relevance"]), reverse=True)
        return cands[:top_k]


def clean_metadata(row: dict) -> dict:
    """Chroma metadata must be str/int/float/bool and cannot be None/NaN.
    Numbers stay numbers (needed for $gte/$lte price filters)."""
    out = {}
    for k, v in row.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        elif getattr(v, "ndim", 0) > 0:  # numpy array (e.g. a list column read from Parquet)
            v = ", ".join(str(x) for x in v.tolist())
        if _is_missing(v):
            continue
        if hasattr(v, "item"):  # numpy scalar
            v = v.item()
            if _is_missing(v):
                continue
        if isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def catalog_metadata(df, id_col: str, meta_cols: list, price_col: Optional[str] = "price",
                     image_urls: Optional[dict] = None, image_id_col: Optional[str] = None) -> dict:
    """{item_id: metadata dict} with numeric price and (optionally) an image_url field."""
    import pandas as pd

    df = df.drop_duplicates(subset=[id_col]).copy()
    if price_col and price_col in df.columns:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    cols = [c for c in meta_cols if c in df.columns]
    out = {}
    for rid, rec in zip(df[id_col].astype(str), df[cols].to_dict("records")):
        meta = clean_metadata(rec)
        if image_urls and image_id_col and meta.get(image_id_col) in image_urls:
            meta["image_url"] = image_urls[meta[image_id_col]]
        out[rid] = meta
    return out


def build_chroma_index(df, embedder: Embedder, client, collection_name: str, id_col: str, text_col: str,
                       meta_cols: list, price_col: Optional[str] = "price", batch_size: int = 512,
                       image_urls: Optional[dict] = None, image_id_col: Optional[str] = None):
    """(Re)build a cosine Chroma collection from a DataFrame. Documents are stored WITHOUT the
    doc prefix; the prefix is only added when embedding."""
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    metas = catalog_metadata(df, id_col, meta_cols, price_col, image_urls, image_id_col)
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        ids = chunk[id_col].astype(str).tolist()
        texts = chunk[text_col].fillna("").astype(str).tolist()
        col.add(ids=ids, documents=texts, embeddings=embedder.embed_docs(texts),
                metadatas=[metas[i] for i in ids])
    return col


def patch_collection_metadata(collection, metas: dict, batch_size: int = 2000) -> int:
    """Overwrite metadata of an existing collection in place (no re-embedding). Returns #updated."""
    ids = [i for i in collection.get(include=[])["ids"] if i in metas]
    for s in range(0, len(ids), batch_size):
        b = ids[s:s + batch_size]
        collection.update(ids=b, metadatas=[metas[i] or {"_": ""} for i in b])
    return len(ids)
