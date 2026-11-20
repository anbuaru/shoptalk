"""Runtime pieces of the API that are independent of FastAPI: request logging, rolling latency
metrics, and the streaming version of ShopTalkPipeline.run().

Logs are JSON Lines, one record per line, so they can be read with pandas.read_json(lines=True).
They are the raw material for the feedback loop and for the Phase 9 evaluation:
  requests.jsonl  - every /recommend call: query, parsed filters, product IDs, answer, latency
  feedback.jsonl  - explicit feedback (thumbs up/down) and implicit feedback (product clicks)
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Iterator, Optional

import numpy as np

from src.rag.generator import NO_MATCH_ANSWER, check_grounding, enforce_ids, remove_unshown_ids
from src.rag.pipeline import ShopTalkPipeline, unmet_constraints


def utc_now() -> str:
    """
    Generates an ISO-8601 formatted UTC timestamp.

    Returns:
        str: Timestamp string with millisecond precision.
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JsonlLogger:
    """
    Append-only, thread-safe JSON Lines telemetry writer.

    Provides safe logging concurrency across background streaming worker threads.
    """

    def __init__(self, log_dir: str):
        """
        Initializes the logger workspace.

        Args:
            log_dir (str): Base destination directory for telemetry records.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, name: str, record: dict) -> None:
        """
        Appends a dictionary entry into a JSONLines tracking log file.

        Args:
            name (str): Log prefix (e.g. 'requests' or 'feedback').
            record (dict): Context data dictionary payload to archive.
        """
        line = json.dumps({"ts": utc_now(), **record}, default=str, ensure_ascii=False)
        with self._lock, open(os.path.join(self.log_dir, f" {name}.jsonl".strip()), "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Metrics:
    """
    Maintains in-memory counts and a rolling queue of query latencies.

    Handles metric calculations without writing database transactions.
    """

    STAGES = ("asr", "parse", "retrieve", "generate", "total", "tts")

    def __init__(self, window: int = 1000):
        """
        Initializes rolling metrics tracking.

        Args:
            window (int): The max size window of recent requests used for percentile calculations.
        """
        self._lock = threading.Lock()
        self.latencies = deque(maxlen=window)
        self.counts = {"requests": 0, "stream_requests": 0, "errors": 0, "no_match": 0,
                       "feedback_up": 0, "feedback_down": 0, "clicks": 0,
                       "transcribe": 0, "audio_requests": 0, "tts": 0}
        self.started = time.time()

    def inc(self, key: str, n: int = 1) -> None:
        """
        Safely increments an event counter.

        Args:
            key (str): Name of the counter.
            n (int): Value increment size.
        """
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + n

    def observe(self, latency_ms: dict) -> None:
        """
        Adds a new latency measurement to the rolling queue.

        Args:
            latency_ms (dict): Phase duration map in milliseconds.
        """
        with self._lock:
            self.latencies.append(latency_ms)

    def snapshot(self) -> dict:
        """
        Generates a metrics snapshot including P50, P95, and P99 percentiles.

        Returns:
            dict: Detailed running execution snapshot.
        """
        with self._lock:
            lat = list(self.latencies)
            counts = dict(self.counts)
        out = {"uptime_s": round(time.time() - self.started, 1), "counts": counts, "window": len(lat), "latency_ms": {}}
        for stage in self.STAGES:
            vals = [x[stage] for x in lat if stage in x]
            if vals:
                out["latency_ms"][stage] = {p: round(float(np.percentile(vals, q)), 1)
                                            for p, q in (("p50", 50), ("p95", 95), ("p99", 99))}
        return out


def log_record(request_id: str, session_id: Optional[str], endpoint: str, result: dict) -> dict:
    """
    Filters and formats query output into a standard logging payload structure.

    Args:
        request_id (str): Current query hash id.
        session_id (str): Current tracking session label.
        endpoint (str): Target web interface endpoint (e.g., '/recommend').
        result (dict): Complete result payload produced by pipeline.

    Returns:
        dict: Compact formatted record for disk storage.
    """
    p = result.get("parsed", {})
    return {
        "request_id": request_id, "session_id": session_id, "endpoint": endpoint,
        "query": result.get("query"),
        "parsed": {k: p.get(k) for k in ("semantic_query", "min_price", "max_price", "colors", "gender",
                                          "brand", "size", "product_types", "type_term", "type_unavailable",
                                          "is_followup") if p.get(k) not in (None, [], "", False)},
        "product_ids": [x["id"] for x in result.get("products", [])],
        "no_match": result.get("no_match"),
        "unmet_constraints": result.get("unmet_constraints", []),
        "answer": (result.get("answer") or "")[:1500],
        "hallucinated_ids_raw": result.get("grounding", {}).get("hallucinated_ids", []),
        "llm": result.get("llm"),
        "latency_ms": result.get("latency_ms"),
    }


def sse(event: str, data: dict) -> str:
    """
    Formats a data payload block into a standard Server-Sent Event text string.

    Args:
        event (str): Target action tag (e.g. 'token', 'meta').
        data (dict): Data payload to transmit.

    Returns:
        str: Standard formatted SSE transfer string.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


def run_stream(pipeline: ShopTalkPipeline, query: str, history: Optional[list] = None) -> Iterator[tuple]:
    """
    Streaming pipeline worker execution function.

    Yields intermediate and final pipeline outcomes as SSE compatible tuples.

    Args:
        pipeline (ShopTalkPipeline): Active model and DB execution structure.
        query (str): Plain-text request string.
        history (list, optional): Historical conversational turn states.

    Yields:
        tuple: Step label with correlated details dictionary.
    """
    t_start = time.perf_counter()
    pq = pipeline.parser.parse(query)
    pq, exclude = pipeline.resolve_followup(pq, history)
    is_followup_turn = bool(pq.is_followup and history)
    t_parse = time.perf_counter()

    retrieved = pipeline.retriever.retrieve(pq, top_k=pipeline.retrieve_k, exclude_ids=exclude,
                                            followup=is_followup_turn)
    context = retrieved[: pipeline.context_k]
    unmet = unmet_constraints(context)
    cards = [pipeline._product_card(p) for p in context]
    t_retr = time.perf_counter()
    yield "meta", {"parsed": pq.to_dict(), "products": cards, "no_match": not context,
                   "unmet_constraints": unmet,
                   "latency_ms": {"parse": round((t_parse - t_start) * 1000, 1),
                                  "retrieve": round((t_retr - t_parse) * 1000, 1)}}

    gen = pipeline.generator
    ttft = None
    if not context:
        answer = NO_MATCH_ANSWER
        yield "token", {"text": answer}
    else:
        gen_query = query if not is_followup_turn else f"{query} (previous request: {pq.semantic_query})"
        if hasattr(gen, "stream"):
            chunks = []
            for chunk in gen.stream(gen_query, context, parsed=pq, unmet=unmet):
                if ttft is None:
                    ttft = time.perf_counter() - t_retr
                chunks.append(chunk)
                yield "token", {"text": chunk}
            answer = "".join(chunks).strip()
        else:
            answer, _ = gen.generate(gen_query, context, parsed=pq, unmet=unmet)
            yield "token", {"text": answer}

    grounding = check_grounding(answer, context) if context else \
        {"cited_ids": [], "hallucinated_ids": [], "lists_ids": False, "grounded_ids": True}
    if context:
        answer, removed = remove_unshown_ids(answer, context)
        grounding["removed_ids"] = removed
        answer = enforce_ids(answer, context, check_grounding(answer, context))
    t_gen = time.perf_counter()

    yield "done", {
        "query": query, "parsed": pq.to_dict(), "answer": answer, "products": cards,
        "retrieved_ids": [p["id"] for p in retrieved], "no_match": not context,
        "unmet_constraints": unmet, "grounding": grounding,
        "best_similarity": getattr(pipeline.retriever, "last_best_similarity", None),
        "llm": getattr(gen, "model_id", None), "generation": {},
        "latency_ms": {"parse": round((t_parse - t_start) * 1000, 1),
                       "retrieve": round((t_retr - t_parse) * 1000, 1),
                       "generate": round((t_gen - t_retr) * 1000, 1),
                       "ttft": round(ttft * 1000, 1) if ttft is not None else None,
                       "total": round((t_gen - t_start) * 1000, 1)},
    }
