"""Phase 9 retrieval metrics + Phase 6 eval-set loading.

The loader is tolerant of the common shapes a Phase 6 eval file can take
(.jsonl / .json / .csv; id lists, graded dicts, or attribute constraints) so the
same query set and labels can be reused without rewriting them.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

QUERY_KEYS = ("query", "question", "text", "q", "user_query")
QID_KEYS = ("qid", "query_id", "id", "case_id")
GRADED_KEYS = ("relevance", "graded", "qrels", "labels", "judgments", "grades")
LIST_KEYS = ("relevant_ids", "relevant_product_ids", "relevant", "gold_ids",
             "positives", "expected_ids", "gold", "product_ids")
ATTR_KEYS = ("expected", "constraints", "expected_attributes", "expected_filters", "filters")
NO_MATCH_KEYS = ("expect_no_match", "expected_no_match", "no_match", "should_be_no_match")


# ----------------------------------------------------------------------------- loading
def _parse_listish(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if s[0] in "[{":
            try:
                return json.loads(s)
            except Exception:
                pass
        return [x.strip() for x in re.split(r"[;|,]", s) if x.strip()]
    return v


def load_records(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        out = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("queries", "items", "data", "eval", "examples", "cases", "test"):
                if isinstance(data.get(k), list):
                    return data[k]
            # dict keyed by qid
            if all(isinstance(v, dict) for v in data.values()):
                return [{"qid": k, **v} for k, v in data.items()]
        raise ValueError(f"Unrecognised JSON layout in {p}")
    if p.suffix in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
        recs = df.to_dict(orient="records")
        for r in recs:
            for k in list(r):
                if k in GRADED_KEYS + LIST_KEYS + ATTR_KEYS:
                    r[k] = _parse_listish(r[k])
        return recs
    raise ValueError(f"Unsupported eval file type: {p.suffix}")


def _qrels_from(rec: dict) -> dict[str, float]:
    for k in GRADED_KEYS:
        v = _parse_listish(rec.get(k))
        if isinstance(v, dict) and v:
            return {str(pid): float(g) for pid, g in v.items()}
        if isinstance(v, list) and v and isinstance(v[0], dict):
            out = {}
            for d in v:
                pid = d.get("product_id", d.get("id", d.get("pid")))
                g = d.get("grade", d.get("relevance", d.get("label", 1)))
                if pid is not None:
                    out[str(pid)] = float(g)
            if out:
                return out
    for k in LIST_KEYS:
        v = _parse_listish(rec.get(k))
        if isinstance(v, (list, tuple)) and v:
            return {str(x): 1.0 for x in v}
        if isinstance(v, str) and v:
            return {v: 1.0}
    return {}


def normalize_eval_set(records: Iterable[dict]) -> list[dict]:
    """-> [{qid, query, qrels{pid: grade}, expected{attr: val}, expect_no_match, history, raw}]"""
    out = []
    for i, rec in enumerate(records):
        query = next((rec[k] for k in QUERY_KEYS if isinstance(rec.get(k), str) and rec[k].strip()), None)
        if query is None:
            continue
        qid = next((str(rec[k]) for k in QID_KEYS if rec.get(k) not in (None, "")), f"q{i:03d}")
        expected = {}
        for k in ATTR_KEYS:
            v = _parse_listish(rec.get(k))
            if isinstance(v, dict) and v:
                expected = v
                break
        enm = next((rec[k] for k in NO_MATCH_KEYS if k in rec), None)
        out.append({
            "qid": qid,
            "query": query.strip(),
            "qrels": _qrels_from(rec),
            "expected": expected,
            "expect_no_match": bool(enm) if enm is not None and not (isinstance(enm, float) and math.isnan(enm)) else None,
            "history": rec.get("history") if isinstance(rec.get("history"), list) else None,
            "raw": rec,
        })
    # make qids unique
    seen: dict[str, int] = {}
    for r in out:
        if r["qid"] in seen:
            seen[r["qid"]] += 1
            r["qid"] = f"{r['qid']}_{seen[r['qid']]}"
        else:
            seen[r["qid"]] = 0
    return out


# ----------------------------------------------------------------------------- metrics
def _is_rel(pid: str, qrels: dict, thr: float) -> bool:
    return qrels.get(str(pid), 0.0) >= thr


def precision_at_k(ranked, qrels, k, thr=1.0):
    """Standard P@k: relevant-in-top-k / k (missing slots count as non-relevant)."""
    return sum(_is_rel(p, qrels, thr) for p in ranked[:k]) / k


def judged_precision_at_k(ranked, qrels, k, thr=1.0):
    """Precision over judged items only (unlabelled products ignored) - shows label holes."""
    judged = [p for p in ranked[:k] if str(p) in qrels]
    return float("nan") if not judged else sum(_is_rel(p, qrels, thr) for p in judged) / len(judged)


def recall_at_k(ranked, qrels, k, thr=1.0):
    n_rel = sum(1 for g in qrels.values() if g >= thr)
    return float("nan") if n_rel == 0 else sum(_is_rel(p, qrels, thr) for p in ranked[:k]) / n_rel


def hit_at_k(ranked, qrels, k, thr=1.0):
    return float(any(_is_rel(p, qrels, thr) for p in ranked[:k]))


def mrr_at_k(ranked, qrels, k, thr=1.0):
    for i, p in enumerate(ranked[:k], 1):
        if _is_rel(p, qrels, thr):
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, qrels, k):
    dcg = sum((2 ** qrels.get(str(p), 0.0) - 1) / math.log2(i + 1) for i, p in enumerate(ranked[:k], 1))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return float("nan") if idcg == 0 else dcg / idcg


def _num(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"\d+(?:[.,]\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else None


def attr_match(meta: dict, expected: dict) -> bool:
    """Does a product's metadata satisfy the expected attribute constraints?"""
    if not expected:
        return False
    meta_l = {str(k).lower(): v for k, v in (meta or {}).items()}
    for key, want in expected.items():
        k = str(key).lower()
        if want in (None, "", []):
            continue
        if k in ("max_price", "price_max", "price_lte", "budget"):
            p = _num(meta_l.get("price"))
            if p is None or p > float(want) + 1e-6:
                return False
            continue
        if k in ("min_price", "price_min", "price_gte"):
            p = _num(meta_l.get("price"))
            if p is None or p < float(want) - 1e-6:
                return False
            continue
        have = meta_l.get(k)
        if have is None:
            return False
        wants = want if isinstance(want, (list, tuple, set)) else [want]
        have_s = str(have).lower()
        if not any(str(w).lower() in have_s for w in wants):
            return False
    return True


def score_query(ranked: list, qrels: dict, ks=(1, 3, 5, 10), thr=1.0,
                metas: list[dict] | None = None, expected: dict | None = None) -> dict:
    ranked = [str(p) for p in ranked]
    row: dict[str, float] = {"n_returned": len(ranked), "has_labels": float(bool(qrels))}
    if qrels:
        for k in ks:
            row[f"precision@{k}"] = precision_at_k(ranked, qrels, k, thr)
            row[f"judged_precision@{k}"] = judged_precision_at_k(ranked, qrels, k, thr)
            row[f"recall@{k}"] = recall_at_k(ranked, qrels, k, thr)
            row[f"hit@{k}"] = hit_at_k(ranked, qrels, k, thr)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, qrels, k)
        row["mrr"] = mrr_at_k(ranked, qrels, max(ks), thr)
    if expected and metas is not None:
        for k in ks:
            top = metas[:k]
            row[f"attr_precision@{k}"] = sum(attr_match(m, expected) for m in top) / k
    return row


def score_runs(runs: dict[str, list], evalset: list[dict], ks=(1, 3, 5, 10), thr=1.0,
               metas_by_qid: dict[str, list[dict]] | None = None) -> pd.DataFrame:
    rows = []
    for e in evalset:
        if e["qid"] not in runs:
            continue
        metas = (metas_by_qid or {}).get(e["qid"])
        r = score_query(runs[e["qid"]], e["qrels"], ks, thr, metas, e.get("expected"))
        rows.append({"qid": e["qid"], "query": e["query"], **r})
    return pd.DataFrame(rows)


def mean_metrics(df: pd.DataFrame) -> dict[str, float]:
    cols = [c for c in df.columns if "@" in c or c == "mrr"]
    return {c: float(np.nanmean(df[c].astype(float))) for c in cols if df[c].notna().any()}


def paired_bootstrap(a, b, n=2000, seed=0) -> dict:
    """Mean(b - a) with a 95% CI from a paired bootstrap over queries."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return {"delta": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": int(len(d))}


# ----------------------------------------------------------------------------- Phase 6 baseline helpers
def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = d
    return out


_ALIASES = [
    (r"judged[_ ]?prec(?:ision)?", "judged_precision"),
    (r"attr(?:ibute)?[_ ]?prec(?:ision)?", "attr_precision"),
    (r"precision|prec|p", "precision"),
    (r"recall|rec|r", "recall"),
    (r"ndcg", "ndcg"),
    (r"hit[_ ]?rate|hits?", "hit"),
]
_NOT_METRIC = re.compile(r"latency|_ms\b|^ms$|time|sec|tokens?|size|count|^n_|^num|top_?k|temperature|lora_|rank$|alpha|dropout|epoch|lr|batch")


def _canon_component(comp: str) -> str | None:
    c = comp.lower().replace("-", "_").strip()
    if _NOT_METRIC.search(c):
        return None
    if re.search(r"(?<![a-z])mrr(?![a-z])", c):
        return "mrr"
    for pat, canon in _ALIASES:
        # metric name not glued to other letters, then @k / _at_k / _k / k (e.g. P@5, precision_at_5_ft, ft_p5)
        m = re.search(rf"(?<![a-z])(?:{pat})\s*(?:@|_at_|_at|at_|_| )?(\d{{1,2}})(?!\d)", c)
        if m and 1 <= int(m.group(1)) <= 50:
            return f"{canon}@{int(m.group(1))}"
    return None


def canon_metric(name: str) -> str | None:
    """Canonical metric for a flattened config path, looking at every component from the leaf up:
    'retrieval.finetuned.P@5', 'precision_at_5_ft', 'p@5.lora' -> 'precision@5'; None if not a metric."""
    for comp in reversed(name.split(".")):
        m = _canon_component(comp)
        if m:
            return m
    return None


def _as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(%?)\s*", v)
        if m:
            return float(m.group(1)) / (100 if m.group(2) else 1)
    return None


def phase6_metrics(evaluated_block: Any) -> pd.DataFrame:
    """All numeric leaves of rag_config['evaluated'] (numbers or numeric strings), with a canonical
    metric name where one is recognised (metric=None rows are printed so nothing is silently dropped)."""
    rows = []
    for path, v in flatten(evaluated_block).items():
        if isinstance(v, list) and v and all(_as_number(x) is not None for x in v):
            continue  # per-query arrays are not summary metrics
        x = _as_number(v)
        if x is None:
            continue
        rows.append({"path": path, "value": x, "metric": canon_metric(path)})
    return pd.DataFrame(rows, columns=["path", "value", "metric"])
