"""Turn production logs (Phase 7 JSONL) into Phase 9 evaluation cases."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_TEST_PREFIXES = ("phase7-", "phase8-", "phase9-", "test-")
DOWN = {"down", "thumbs_down", "thumbsdown", "-1", "bad", "negative", "dislike", "👎", "no", "false"}
UP = {"up", "thumbs_up", "thumbsup", "1", "+1", "good", "positive", "like", "👍", "yes", "true"}


def read_jsonl(path: str | Path) -> tuple[list[dict], int]:
    p = Path(path)
    if not p.exists():
        return [], 0
    out, bad = [], 0
    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                out.append(rec)
            else:
                bad += 1
        except Exception:
            bad += 1
    return out, bad


def is_test_session(sid, prefixes=DEFAULT_TEST_PREFIXES) -> bool:
    return isinstance(sid, str) and sid.lower().startswith(tuple(p.lower() for p in prefixes))


def classify_feedback(rec: dict) -> str | None:
    kind = str(rec.get("kind") or rec.get("type") or rec.get("event") or "").lower()
    rating = next((rec[k] for k in ("rating", "vote", "thumbs", "value", "score") if k in rec), None)
    if "click" in kind or (rating is None and "product_id" in rec):
        return "click"
    if isinstance(rating, bool):
        return "up" if rating else "down"
    if isinstance(rating, (int, float)):
        return "up" if rating > 0 else "down" if rating < 0 else None
    if rating is not None:
        r = str(rating).strip().lower()
        if r in DOWN:
            return "down"
        if r in UP:
            return "up"
    return None


def truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    try:
        if v != v:  # NaN
            return False
    except Exception:
        pass
    return bool(v)


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", str(q or "").strip().lower())


def _latency_total(v):
    if isinstance(v, dict):
        return v.get("total")
    return v if isinstance(v, (int, float)) else None


def mine(requests_path, feedback_path, prefixes=DEFAULT_TEST_PREFIXES) -> dict:
    reqs, bad_r = read_jsonl(requests_path)
    fbs, bad_f = read_jsonl(feedback_path)
    rq = pd.DataFrame(reqs)
    if rq.empty:
        rq = pd.DataFrame(columns=["request_id", "query", "session_id"])
    for c in ("request_id", "query", "session_id", "no_match", "product_ids", "answer", "latency"):
        if c not in rq.columns:
            rq[c] = None
    rq["is_test"] = rq["session_id"].map(lambda s: is_test_session(s, prefixes)).astype(bool)
    real = rq[~rq["is_test"]].copy()
    sess_by_req = dict(zip(rq["request_id"], rq["session_id"]))

    fb = pd.DataFrame(fbs)
    if fb.empty:
        fb = pd.DataFrame(columns=["request_id"])
    fb["fb_type"] = [classify_feedback(r) for r in fbs] if fbs else []
    if "session_id" not in fb.columns:
        fb["session_id"] = None
    fb["session_id"] = [s if isinstance(s, str) else sess_by_req.get(r)
                        for s, r in zip(fb["session_id"], fb["request_id"])]
    fb["is_test"] = fb["session_id"].map(lambda s: is_test_session(s, prefixes)).astype(bool)
    fb_real = fb[~fb["is_test"]].copy()

    comment_col = next((c for c in ("comment", "text", "note") if c in fb_real.columns), None)
    downs = fb_real[fb_real["fb_type"] == "down"]
    down_comments = downs.groupby("request_id")[comment_col].apply(
        lambda s: " | ".join(str(x) for x in s if isinstance(x, str) and x.strip())) if comment_col and not downs.empty else pd.Series(dtype=str)

    cases = []
    down_ids = set(downs["request_id"])
    for _, r in real.iterrows():
        reasons = []
        if r["request_id"] in down_ids:
            reasons.append("thumbs_down")
        if truthy(r.get("no_match")):
            reasons.append("no_match")
        if not reasons or not isinstance(r["query"], str) or not r["query"].strip():
            continue
        cases.append({
            "request_id": r["request_id"], "session_id": r["session_id"], "query": r["query"].strip(),
            "reason": "+".join(reasons), "comment": down_comments.get(r["request_id"], ""),
            "logged_product_ids": r["product_ids"] if isinstance(r["product_ids"], list) else [],
            "logged_no_match": truthy(r.get("no_match")), "logged_answer": r["answer"],
            "logged_latency_total": _latency_total(r["latency"]), "ts": r.get("ts", r.get("timestamp")),
        })
    cases_df = pd.DataFrame(cases)
    n_before_dedup = len(cases_df)
    if not cases_df.empty:
        cases_df["qnorm"] = cases_df["query"].map(_norm_q)
        agg_reason = cases_df.groupby("qnorm")["reason"].apply(lambda s: "+".join(sorted(set("+".join(s).split("+")))))
        cases_df = cases_df.drop_duplicates("qnorm", keep="last").copy()
        cases_df["reason"] = cases_df["qnorm"].map(agg_reason)
        cases_df = cases_df.drop(columns="qnorm").reset_index(drop=True)
        cases_df.insert(0, "case_id", [f"log{i:03d}" for i in range(len(cases_df))])

    clicks = fb_real[fb_real["fb_type"] == "click"]
    n_real = len(real)
    stats = {
        "requests_total": int(len(rq)), "requests_test_filtered": int(rq["is_test"].sum()),
        "requests_real": int(n_real), "bad_lines": int(bad_r + bad_f),
        "feedback_total": int(len(fb)), "feedback_test_filtered": int(fb["is_test"].sum()) if len(fb) else 0,
        "thumbs_up": int((fb_real["fb_type"] == "up").sum()), "thumbs_down": int((fb_real["fb_type"] == "down").sum()),
        "clicks": int(len(clicks)),
        "ctr": float(clicks["request_id"].nunique() / n_real) if n_real else float("nan"),
        "mean_click_rank": float(pd.to_numeric(clicks.get("rank"), errors="coerce").mean()) if len(clicks) else float("nan"),
        "no_match_rate": float(real["no_match"].map(truthy).mean()) if n_real else float("nan"),
        "cases_before_dedup": int(n_before_dedup), "cases": int(len(cases_df)),
    }
    return {"cases": cases_df, "stats": stats, "requests_real": real, "feedback_real": fb_real}
