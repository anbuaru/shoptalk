"""Failure taxonomy + markdown write-up for Phase 9."""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

CATEGORIES = {
    "retrieval_miss": {
        "desc": "Labelled query where no relevant product reached the top-k.",
        "fix": "Mine these queries as hard negatives/positives for the next E5 LoRA round; add a sparse "
               "(BM25) leg for brand / model-number / size tokens and fuse with the dense scores (RRF).",
        "measure": "precision@5 / hit@5 on the Phase 6 set + these cases",
        "severity": 3},
    "low_attr_precision": {
        "desc": "Fewer than half of the top-5 products satisfy the query's stated attributes.",
        "fix": "Tighten the query parser (colour/type/brand synonyms, price phrases) and apply parsed filters "
               "before ranking; fall back progressively (drop colour, then brand) instead of dropping all.",
        "measure": "attr_precision@5",
        "severity": 2},
    "false_no_match": {
        "desc": "System said 'no match' although the query is answerable from the catalog.",
        "fix": "Relax filter cascade before declaring no-match; log which constraint emptied the result set; "
               "normalise colour/product_type vocab to the ABO values.",
        "measure": "no-match accuracy on queries labelled expect_no_match=False",
        "severity": 3},
    "missed_no_match": {
        "desc": "Query labelled as unanswerable, but the system recommended products anyway.",
        "fix": "Add a relevance-score floor + constraint check before generation; answer honestly when nothing fits.",
        "measure": "no-match accuracy on queries labelled expect_no_match=True",
        "severity": 2},
    "unmet_constraints": {
        "desc": "Products returned while some parsed constraints were not met.",
        "fix": "State the relaxed constraint explicitly in the answer and offer the nearest alternatives.",
        "measure": "share of answers that mention every unmet constraint",
        "severity": 1},
    "hallucinated_citation": {
        "desc": "Answer cites an [ID: ...] that is not among the returned products.",
        "fix": "Post-validate citations against the retrieved IDs (strip/repair) and keep the IDs in the prompt "
               "as a closed list; penalise in the prompt with an explicit instruction.",
        "measure": "hallucinated_id_rate (target 0)",
        "severity": 3},
    "uncited_answer": {
        "desc": "Products were returned but the answer cites none of them.",
        "fix": "Few-shot example with citations in the system prompt; validator that appends IDs if missing.",
        "measure": "cites_any_rate",
        "severity": 1},
    "truncated_answer": {
        "desc": "Answer stops mid-sentence (hit max_new_tokens).",
        "fix": "Ask for a fixed, short format (e.g. 3 bullets, 1 line each); trim at the last full sentence; "
               "set max_new_tokens from the Phase 9 sweep.",
        "measure": "truncation_rate",
        "severity": 2},
    "slow": {
        "desc": "Client round-trip above the latency target.",
        "fix": "Lower max_new_tokens, shorten the prompt (fewer/shorter product cards), 4-bit/AWQ weights or vLLM; "
               "keep streaming so first token stays < 1 s.",
        "measure": "P95 total / full-answer latency",
        "severity": 2},
    "low_relevance_rating": {"desc": "Manual relevance rating ≤ 2.", "fix": "See retrieval fixes; inspect query type.",
                             "measure": "mean relevance rating", "severity": 3},
    "low_groundedness_rating": {"desc": "Manual groundedness rating ≤ 2.",
                                "fix": "Restrict the answer to fields present in the product cards; lower temperature.",
                                "measure": "mean groundedness rating", "severity": 3},
    "low_fluency_rating": {"desc": "Manual fluency rating ≤ 2.", "fix": "Prompt for structure; check repetition penalty.",
                           "measure": "mean fluency rating", "severity": 1},
    "user_thumbs_down": {"desc": "Real user gave a thumbs-down (from feedback.jsonl).",
                         "fix": "Triage the comment; route into the categories above.", "measure": "thumbs-down rate",
                         "severity": 2},
}


def _f(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except Exception:
        return None


def classify(row: dict, k: int = 5, slow_ms: float = 5000) -> list[str]:
    tags = []
    hit = _f(row.get(f"hit@{k}"))
    if row.get("has_labels") and hit == 0.0:
        tags.append("retrieval_miss")
    ap = _f(row.get(f"attr_precision@{k}"))
    if ap is not None and ap < 0.5 and not row.get("no_match"):
        tags.append("low_attr_precision")
    enm = row.get("expect_no_match")
    if enm is False and row.get("no_match"):
        tags.append("false_no_match")
    if enm is True and not row.get("no_match"):
        tags.append("missed_no_match")
    if row.get("unmet_constraints") and (row.get("n_returned") or 0) > 0:
        tags.append("unmet_constraints")
    if row.get("hallucinated_ids"):
        tags.append("hallucinated_citation")
    if (row.get("n_returned") or 0) > 0 and not row.get("cites_any") and not row.get("no_match"):
        tags.append("uncited_answer")
    if not row.get("ends_cleanly", True) and not row.get("empty_answer"):
        tags.append("truncated_answer")
    if (_f(row.get("client_ms")) or 0) > slow_ms:
        tags.append("slow")
    for dim in ("relevance", "groundedness", "fluency"):
        v = _f(row.get(dim))
        if v is not None and v <= 2:
            tags.append(f"low_{dim}_rating")
    if "thumbs_down" in str(row.get("reason") or ""):
        tags.append("user_thumbs_down")
    return tags


def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    df = df.copy()
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"
            cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_report(cases: pd.DataFrame, headline: dict, sections: dict[str, pd.DataFrame],
                 n_examples: int = 3, title: str = "ShopTalk Phase 9 — Failure cases & improvement plan") -> str:
    """cases needs columns: case_id, source, query, tags(list), answer, top_titles."""
    md = [f"# {title}", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_\n", "## 1. Headline results\n"]
    md.append(_md_table(pd.DataFrame([{"metric": k, "value": v} for k, v in headline.items()])))
    for name, df in sections.items():
        md += [f"\n### {name}\n", _md_table(df)]

    exploded = cases.explode("tags").dropna(subset=["tags"])
    counts = exploded.groupby("tags")["case_id"].nunique().rename("cases").reset_index()
    counts["share"] = counts["cases"] / max(len(cases), 1)
    counts["severity"] = counts["tags"].map(lambda t: CATEGORIES.get(t, {}).get("severity", 1))
    counts["priority_score"] = counts["cases"] * counts["severity"]
    counts = counts.sort_values("priority_score", ascending=False)

    md += ["\n## 2. Failure taxonomy\n", f"{len(cases)} evaluated cases; "
           f"{cases['tags'].map(bool).sum()} have at least one failure tag.\n",
           _md_table(counts.rename(columns={"tags": "category"}))]

    md.append("\n## 3. Examples per category\n")
    for cat in counts["tags"]:
        info = CATEGORIES.get(cat, {})
        md.append(f"\n### {cat}\n_{info.get('desc', '')}_\n")
        ex = cases[cases["tags"].map(lambda t: cat in t)].head(n_examples)
        for _, r in ex.iterrows():
            ans = str(r.get("answer") or "").replace("\n", " ")
            md.append(f"- **{r['case_id']}** ({r.get('source', '')}) — “{r['query']}”  \n"
                      f"  top products: {r.get('top_titles', '')}  \n"
                      f"  answer: {ans[:300]}{'…' if len(ans) > 300 else ''}  \n"
                      f"  tags: {', '.join(r['tags'])}")
    md.append("\n## 4. Improvement plan (ranked by cases × severity)\n")
    plan = [{"#": i + 1, "category": c, "cases": int(n), "proposed fix": CATEGORIES.get(c, {}).get("fix", ""),
             "how we measure": CATEGORIES.get(c, {}).get("measure", "")}
            for i, (c, n) in enumerate(zip(counts["tags"], counts["cases"]))]
    md.append(_md_table(pd.DataFrame(plan)))
    md.append("\n> Review and edit this plan: the fixes are starting points generated from the failure "
              "categories; confirm each against the examples above before scheduling the work.\n")
    return "\n".join(md)
