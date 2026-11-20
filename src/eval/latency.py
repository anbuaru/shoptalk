"""Latency harness + percentile tables for Phase 9 (reused in Phase 10 for speech-to-text)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd


def percentiles(values, ps=(50, 95, 99)) -> dict:
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    if v.size == 0:
        return {"n": 0, **{f"p{p}": np.nan for p in ps}, "mean": np.nan, "max": np.nan}
    return {"n": int(v.size), "mean": float(v.mean()), **{f"p{p}": float(np.percentile(v, p)) for p in ps},
            "max": float(v.max())}


def latency_table(df: pd.DataFrame, cols: dict[str, str], ps=(50, 95, 99)) -> pd.DataFrame:
    """cols = {label: column_name}"""
    rows = []
    for label, col in cols.items():
        if col in df.columns:
            rows.append({"stage": label, **percentiles(pd.to_numeric(df[col], errors="coerce"), ps)})
    out = pd.DataFrame(rows)
    return out.round(1)


def run_requests(fn: Callable[[str, int], dict], queries: list[str], n: int, warmup: int = 3,
                 concurrency: int = 1, desc: str = "requests") -> list[dict]:
    """Call fn(query, i) n times (cycling through queries) after `warmup` unrecorded calls."""
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731
    for i in range(warmup):
        try:
            fn(queries[i % len(queries)], -1 - i)
        except Exception:
            pass
    jobs = [(i, queries[i % len(queries)]) for i in range(n)]
    results: list[dict] = []

    def _one(i, q):
        t0 = time.time()
        try:
            r = fn(q, i)
            return {"i": i, "query": q, "ok": True, "started": t0, **r}
        except Exception as e:
            return {"i": i, "query": q, "ok": False, "error": repr(e)[:300], "started": t0}

    if concurrency <= 1:
        for i, q in tqdm(jobs, desc=desc):
            results.append(_one(i, q))
    else:
        with ThreadPoolExecutor(concurrency) as ex:
            futs = [ex.submit(_one, i, q) for i, q in jobs]
            for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
                results.append(f.result())
    return sorted(results, key=lambda r: r["i"])
