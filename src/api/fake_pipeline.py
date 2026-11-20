"""Mock vector search and generation pipeline for low-compute environments.

Provides standard data validation patterns, parsing verification, and streaming tests
without using any GPU memory, LLM loading, or ChromaDB index files.
"""
from __future__ import annotations

import re
import time

from src.rag.generator import resolve_citations
from src.rag.pipeline import ShopTalkPipeline
from src.rag.query_parser import RuleBasedParser

# Seed catalog containing static product cards for mock matching
PRODUCTS = [
    ("FAKE000001", "Men's Red Running Sneakers", "Symbol", 31.99, "Red", "SHOES"),
    ("FAKE000002", "Men's Red Canvas Sneakers", "Symbol", 25.99, "Red", "SHOES"),
    ("FAKE000003", "Men's Black Leather Sneakers", "find.", 45.00, "Black", "SHOES"),
    ("FAKE000004", "Men's Grey Knit Sneakers", "Symbol", 19.99, "Grey", "SHOES"),
    ("FAKE000005", "Women's Waterproof Hiking Boot", "206 Collective", 89.00, "Brown", "BOOT"),
    ("FAKE000006", "Men's Black Leather Bifold Wallet", "Goodthreads", 24.00, "Black", "WALLET"),
    ("FAKE000007", "Happy Belly Dark Roast Coffee Pods", "Happy Belly", 18.50, None, "COFFEE"),
    ("FAKE000008", "Cozy Small Loveseat Sofa", "Rivet", 299.00, "Grey", "SOFA"),
]


class FakeRetriever:
    """Simulated retriever filtering catalog records based on rule parameters."""
    size = len(PRODUCTS)

    def __init__(self):
        self.last_best_similarity = None

    def retrieve(self, pq, top_k=10, exclude_ids=None, followup=False):
        """
        Performs basic keyword matching and price constraints over static seed list.

        Returns:
            List[dict]: Formatted dictionary mapping representing mock retrieved results.
        """
        words = set(re.findall(r"[a-z]+", pq.semantic_query.lower()))
        out = []
        for pid, title, brand, price, color, ptype in PRODUCTS:
            if pid in (exclude_ids or []):
                continue
            if pq.max_price is not None and price > pq.max_price:
                continue
            if pq.min_price is not None and price < pq.min_price:
                continue
            twords = set(re.findall(r"[a-z]+", title.lower()))
            sim = len(words & twords) / max(1, len(words))
            if sim == 0:
                continue
            matched = {}
            if pq.colors:
                matched["color"] = bool(color) and color.lower() in pq.colors
            if pq.gender:
                matched["gender"] = pq.gender in ("men", "women") and title.lower().startswith(pq.gender)
            out.append({"id": pid, "document": title, "similarity": sim, "relevance": sim, "matched": matched,
                        "n_matched": sum(matched.values()), "price_unknown": False,
                        "metadata": {"title": title, "brand": brand, "price": price, "color": color, "ptype": ptype}})
        out.sort(key=lambda c: (c["n_matched"], c["relevance"]), reverse=True)
        self.last_best_similarity = out[0]["similarity"] if out else None
        return out[:top_k]


class FakeGenerator:
    """Mock text generator generating static response templates with catalog citation references."""
    model_id = "fake-llm"

    def _text(self, products, unmet):
        intro = "None of these is exactly what you asked for; closest options: " if unmet else "Here are good matches: "
        return intro + " ".join(f"[{i}] {p['metadata']['title']}." for i, p in enumerate(products[:2], 1))

    def generate(self, query, products, parsed=None, unmet=None):
        """Generates static recommendation text."""
        raw = self._text(products, unmet)
        return resolve_citations(raw, products), {"seconds": 0.0, "output_tokens": len(raw.split()), "raw": raw,
                                                  "hit_token_cap": False}

    def stream(self, query, products, parsed=None, unmet=None):
        """Streams tokens simulating latency patterns."""
        for word in resolve_citations(self._text(products, unmet), products).split(" "):
            time.sleep(0.001)
            yield word + " "


def build_fake_pipeline() -> ShopTalkPipeline:
    """
    Constructs a valid mock pipeline instance conforming to standard class interfaces.

    Returns:
        ShopTalkPipeline: Built fake pipeline context structure.
    """
    return ShopTalkPipeline(RuleBasedParser(), FakeRetriever(), FakeGenerator(), retrieve_k=10, context_k=3,
                            type_key="ptype", image_key=None)
