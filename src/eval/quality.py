"""Automatic answer-quality proxies (complement - not replace - the manual ratings)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CITE_RE = re.compile(r"\[ID:\s*([^\]]+)\]", re.I)
PRICE_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
CLEAN_END = tuple('.!?)"\'*]`')


def cited_ids(answer: str) -> list[str]:
    out = []
    for m in CITE_RE.finditer(answer or ""):
        out += [x.strip() for x in re.split(r"[,;\s]+", m.group(1)) if x.strip()]
    return out


def _price(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else None


def answer_proxies(answer: str, products: list[dict], no_match: bool = False) -> dict:
    answer = answer or ""
    returned = [str(p.get("id")) for p in products or []]
    cited = cited_ids(answer)
    cited_set, ret_set = set(cited), set(returned)
    valid = cited_set & ret_set
    prices = [float(x.replace(",", "")) for x in PRICE_RE.findall(answer)]
    prod_prices = [p for p in (_price(pr.get("price")) for pr in products or []) if p is not None]
    grounded_prices = [x for x in prices if any(abs(x - pp) < 0.011 for pp in prod_prices)]
    stripped = answer.rstrip()
    return {
        "answer_words": len(answer.split()),
        "answer_chars": len(answer),
        "empty_answer": not stripped,
        "n_returned": len(returned),
        "n_cited": len(cited_set),
        "cites_any": bool(cited_set),
        "citation_valid_rate": (len(valid) / len(cited_set)) if cited_set else np.nan,
        "hallucinated_ids": sorted(cited_set - ret_set),
        "coverage": (len(valid) / len(ret_set)) if ret_set else np.nan,
        "n_prices": len(prices),
        "price_grounded_rate": (len(grounded_prices) / len(prices)) if prices else np.nan,
        "ends_cleanly": bool(stripped) and stripped.endswith(CLEAN_END),
        "no_match": bool(no_match),
    }


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    has_products = df["n_returned"] > 0
    return {
        "n": int(len(df)),
        "mean_words": float(df["answer_words"].mean()),
        "p95_words": float(df["answer_words"].quantile(0.95)),
        "empty_rate": float(df["empty_answer"].mean()),
        "cites_any_rate": float(df.loc[has_products, "cites_any"].mean()) if has_products.any() else np.nan,
        "citation_valid_rate": float(np.nanmean(df["citation_valid_rate"])) if df["citation_valid_rate"].notna().any() else np.nan,
        "hallucinated_id_rate": float((df["hallucinated_ids"].map(len) > 0).mean()),
        "coverage": float(np.nanmean(df["coverage"])) if df["coverage"].notna().any() else np.nan,
        "price_grounded_rate": float(np.nanmean(df["price_grounded_rate"])) if df["price_grounded_rate"].notna().any() else np.nan,
        "truncation_rate": float((~df["ends_cleanly"] & ~df["empty_answer"]).mean()),
        "no_match_rate": float(df["no_match"].mean()),
    }
