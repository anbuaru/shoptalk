"""ShopTalk RAG pipeline: parse -> retrieve -> generate, each stage swappable.

`ShopTalkPipeline.run(query, history)` returns a JSON-serialisable dict shaped like the
Phase 7 /recommend response: {answer, products[{id,title,price,link,...}], latency_ms, ...}.
"""
from __future__ import annotations

import time
from dataclasses import fields, replace
from typing import Optional

from .generator import NO_MATCH_ANSWER, check_grounding, enforce_ids, remove_unshown_ids
from .query_parser import ParsedQuery

_PQ_FIELDS = {f.name for f in fields(ParsedQuery)}


def unmet_constraints(context: list) -> list:
    """Requested attributes that NONE of the products shown satisfies (e.g. ["type", "color"])."""
    if not context:
        return []
    keys = {k for p in context for k in p.get("matched", {})}
    return sorted(k for k in keys if not any(p.get("matched", {}).get(k) for p in context))


class ShopTalkPipeline:
    def __init__(self, parser, retriever, generator=None, retrieve_k: int = 20, context_k: int = 5,
                 link_template: str = "https://www.amazon.com/dp/{id}",
                 title_key="title", brand_key="brand", price_key="price", color_key="color",
                 image_key: Optional[str] = "image_url", type_key: Optional[str] = None):
        self.parser, self.retriever, self.generator = parser, retriever, generator
        self.retrieve_k, self.context_k = retrieve_k, context_k
        self.link_template = link_template
        self.keys = dict(title=title_key, brand=brand_key, price=price_key, color=color_key,
                         image=image_key, type=type_key)

    # ---------- follow-ups ("something similar but cheaper") ----------
    def resolve_followup(self, pq: ParsedQuery, history: Optional[list]) -> tuple:
        """Merge a follow-up with the previous turn. Returns (parsed_query, ids_to_exclude)."""
        if not pq.is_followup or not history:
            return pq, []
        prev = history[-1]
        p = prev["parsed"]
        prev_pq = p if isinstance(p, ParsedQuery) else ParsedQuery(**{k: v for k, v in p.items() if k in _PQ_FIELDS})
        prev_products = prev.get("products", [])
        merged = replace(
            prev_pq, raw=pq.raw,
            colors=pq.colors or prev_pq.colors, gender=pq.gender or prev_pq.gender,
            brand=pq.brand or prev_pq.brand, size=pq.size or prev_pq.size,
            product_types=pq.product_types or prev_pq.product_types, type_term=pq.type_term or prev_pq.type_term,
            type_unavailable=pq.type_unavailable or (prev_pq.type_unavailable and not pq.product_types),
            min_price=pq.min_price if pq.min_price is not None else prev_pq.min_price,
            max_price=pq.max_price if pq.max_price is not None else prev_pq.max_price,
            is_followup=True, wants_cheaper=pq.wants_cheaper, wants_pricier=pq.wants_pricier,
        )
        # "cheaper" / "pricier" is relative to the top recommendation of the previous turn
        # (relative to the cheapest product shown would often leave nothing to show)
        ref = next((p["price"] for p in prev_products if p.get("price") is not None), None)
        if pq.wants_cheaper and ref is not None:
            merged.max_price = round(ref - 0.01, 2)
            if merged.min_price is not None and merged.min_price > merged.max_price:
                merged.min_price = None
        if pq.wants_pricier and ref is not None:
            merged.min_price = round(ref + 0.01, 2)
            if merged.max_price is not None and merged.max_price < merged.min_price:
                merged.max_price = None
        return merged, [p["id"] for p in prev_products]

    # ---------- formatting ----------
    def _product_card(self, p: dict) -> dict:
        m, k = p["metadata"], self.keys
        try:
            price = float(m.get(k["price"])) if m.get(k["price"]) not in (None, "") else None
        except (TypeError, ValueError):
            price = None
        return {
            "id": p["id"],
            "title": m.get(k["title"], ""),
            "brand": m.get(k["brand"]),
            "price": price,
            "color": m.get(k["color"]),
            "product_type": m.get(k["type"]) if k["type"] else None,
            "image_url": m.get(k["image"]) if k["image"] else None,
            "link": self.link_template.format(id=p["id"]) if self.link_template else None,
            "score": round(float(p.get("relevance", p.get("similarity", 0.0))), 4),
            "matched": p.get("matched", {}),
            "price_unknown": p.get("price_unknown", price is None),
        }

    # ---------- main entry ----------
    def run(self, query: str, history: Optional[list] = None, generate: bool = True) -> dict:
        t_start = time.perf_counter()
        pq = self.parser.parse(query)
        pq, exclude = self.resolve_followup(pq, history)
        t_parse = time.perf_counter()

        is_followup_turn = bool(pq.is_followup and history)
        retrieved = self.retriever.retrieve(pq, top_k=self.retrieve_k, exclude_ids=exclude,
                                            followup=is_followup_turn)
        context = retrieved[: self.context_k]
        t_retr = time.perf_counter()

        unmet = unmet_constraints(context)
        gen_info = {}
        if not context:
            answer = NO_MATCH_ANSWER
        elif generate and self.generator is not None:
            # follow-ups are generated against the merged request, so the LLM knows what "similar" means
            gen_query = query if not is_followup_turn else f"{query} (previous request: {pq.semantic_query})"
            answer, gen_info = self.generator.generate(gen_query, context, parsed=pq, unmet=unmet)
        else:
            answer = ""
        grounding = check_grounding(answer, context) if answer and context else \
            {"cited_ids": [], "hallucinated_ids": [], "lists_ids": False, "grounded_ids": True}
        if answer and context:
            # grounding (above) is measured on the RAW model output; then unshown IDs are removed and,
            # if no valid ID is left, the shown IDs are appended
            answer, removed = remove_unshown_ids(answer, context)
            grounding["removed_ids"] = removed
            answer = enforce_ids(answer, context, check_grounding(answer, context))
        t_gen = time.perf_counter()

        return {
            "query": query,
            "parsed": pq.to_dict(),
            "answer": answer,
            "products": [self._product_card(p) for p in context],
            "retrieved_ids": [p["id"] for p in retrieved],
            "no_match": not context,
            "unmet_constraints": unmet,
            "best_similarity": getattr(self.retriever, "last_best_similarity", None),
            "grounding": grounding,
            "llm": getattr(self.generator, "model_id", None) if generate else None,
            "generation": {k: v for k, v in gen_info.items() if k != "raw"},
            "latency_ms": {
                "parse": round((t_parse - t_start) * 1000, 1),
                "retrieve": round((t_retr - t_parse) * 1000, 1),
                "generate": round((t_gen - t_retr) * 1000, 1),
                "total": round((t_gen - t_start) * 1000, 1),
            },
        }

    @staticmethod
    def history_entry(result: dict) -> dict:
        """What to keep from a turn so the next query can be a follow-up."""
        return {"parsed": result["parsed"], "products": result["products"]}
