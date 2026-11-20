"""Build a ShopTalkPipeline from rag_config.json, exactly as Phase 6 evaluated it.

Phase 7 (FastAPI) only needs:

    from src.rag.factory import build_pipeline
    pipeline = build_pipeline("src/rag/rag_config.json")   # once, at startup
    result = pipeline.run(query, history)
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .generator import HFGenerator
from .pipeline import ShopTalkPipeline
from .query_parser import LLMQueryParser, ProductTypeLexicon, RuleBasedParser
from .retriever import ChromaRetriever, CrossEncoderReranker, Embedder, load_sentence_model


def _resolve(path: str, base_dir: str) -> str:
    """Config paths are stored relative to the repo root so the API can run outside Colab."""
    if not path or os.path.isabs(path) or not os.path.exists(os.path.join(base_dir, path)):
        return path
    return os.path.join(base_dir, path)


def build_pipeline(config_path: str, load_llm: bool = True, device: Optional[str] = None,
                   repo_dir: Optional[str] = None, local_chroma_dir: Optional[str] = None) -> ShopTalkPipeline:
    """local_chroma_dir: copy the Chroma folder there first and query the copy. Use it whenever the
    index lives on a network/Drive mount: metadata filters read SQLite, which is slow and spiky there."""
    import chromadb
    import torch
    from sentence_transformers import CrossEncoder

    cfg = json.load(open(config_path))
    base = repo_dir or cfg.get("repo_dir_hint") or os.getcwd()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    keys = cfg["keys"]

    embedder = Embedder(load_sentence_model(_resolve(cfg["embed_model"], base), device=device),
                        query_prefix=cfg.get("query_prefix", ""), doc_prefix=cfg.get("doc_prefix", ""))
    chroma_path = _resolve(cfg["chroma_dir"], base)
    if local_chroma_dir:
        import shutil
        if not os.path.exists(local_chroma_dir):
            shutil.copytree(chroma_path, local_chroma_dir)
        chroma_path = local_chroma_dir
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(cfg["collection"])
    expected = cfg.get("evaluated", {}).get("index_size")
    if expected and collection.count() < 0.95 * expected:
        raise RuntimeError(f"Chroma collection has {collection.count()} vectors, expected ~{expected}: "
                           f"wrong chroma_dir or a damaged index.")

    r = cfg["retrieval"]
    reranker = CrossEncoderReranker(CrossEncoder(r["reranker"], device=device)) if r.get("reranker") else None
    retriever = ChromaRetriever(collection, embedder, reranker=reranker, candidate_k=r["candidate_k"],
                                price_key=keys["price"], color_key=keys["color"], brand_key=keys["brand"],
                                title_key=keys["title"], type_key=keys.get("type") if r.get("type_filter", True) else None,
                                missing_price_policy=r["missing_price_policy"],
                                min_similarity=r.get("min_similarity"))

    brands = []
    if cfg.get("brands_file"):
        bf = _resolve(cfg["brands_file"], base)
        if os.path.exists(bf):
            brands = json.load(open(bf))
    lexicon = None   # product types + "not sold here" are part of the type filter: off together
    if cfg.get("type_lexicon_file") and r.get("type_filter", True):
        lf = _resolve(cfg["type_lexicon_file"], base)
        if os.path.exists(lf):
            lexicon = ProductTypeLexicon.load(lf)
    parser = RuleBasedParser(known_brands=brands, type_lexicon=lexicon)

    generator = None
    if load_llm:
        l = cfg["llm"]
        generator = HFGenerator(l["model_id"], load_in_4bit=l.get("load_in_4bit", False),
                                max_new_tokens=l.get("max_new_tokens", 100),
                                context_max_chars=l.get("context_max_chars", 350),
                                context_keys=dict(title_key=keys["title"], brand_key=keys["brand"],
                                                  price_key=keys["price"], color_key=keys["color"],
                                                  type_key=keys.get("type")))
        if cfg.get("parser") == "llm":
            parser = LLMQueryParser(generator.as_text_fn(), fallback=parser)

    return ShopTalkPipeline(parser, retriever, generator, retrieve_k=r["retrieve_k"], context_k=r["context_k"],
                            link_template=cfg.get("link_template"), title_key=keys["title"],
                            brand_key=keys["brand"], price_key=keys["price"], color_key=keys["color"],
                            image_key=keys.get("image"), type_key=keys.get("type"))
