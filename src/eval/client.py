"""HTTP client for the Phase 7 ShopTalk API (JSON + SSE streaming), used by Phase 9."""
from __future__ import annotations

import json
import time
from typing import Iterable, Iterator

import requests


def parse_sse(lines: Iterable[str | bytes | None]) -> Iterator[tuple[str, str]]:
    """Minimal spec-compliant SSE parser -> (event, data) tuples."""
    event, data = "message", []
    for raw in lines:
        if raw is None:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r")
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


def _token_text(data: str) -> str:
    try:
        obj = json.loads(data)
    except Exception:
        return data
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ("token", "text", "delta", "content"):
            if isinstance(obj.get(k), str):
                return obj[k]
    return ""


class ShopTalkClient:
    def __init__(self, base_url: str, timeout: float = 180, query_field: str = "query",
                 extra_body: dict | None = None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.query_field = query_field
        self.extra_body = dict(extra_body or {})
        self.s = requests.Session()

    # ------------------------------------------------------------------ basics
    def health(self) -> dict:
        r = self.s.get(f"{self.base}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def metrics(self) -> str:
        r = self.s.get(f"{self.base}/metrics", timeout=10)
        r.raise_for_status()
        return r.text

    def wait_healthy(self, timeout_s: float = 900, poll_s: float = 5, proc=None, verbose=True) -> dict:
        t0, last = time.time(), None
        while time.time() - t0 < timeout_s:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"API process exited with code {proc.returncode}")
            try:
                last = self.health()
                if str(last.get("status", "")).lower() == "ok":
                    if verbose:
                        print(f"✅ /health ok after {time.time() - t0:.0f}s: {last}")
                    return last
            except Exception as e:  # not up yet
                last = {"error": repr(e)[:120]}
            time.sleep(poll_s)
        raise TimeoutError(f"/health not ok after {timeout_s}s; last={last}")

    def body(self, query: str, session_id: str, history: list | None = None, **kw) -> dict:
        b = {**self.extra_body, self.query_field: query, "session_id": session_id}
        if history:
            b["history"] = history
        b.update(kw)
        return b

    # ------------------------------------------------------------------ endpoints
    def recommend(self, query: str, session_id: str, history: list | None = None, **kw) -> dict:
        t0 = time.perf_counter()
        r = self.s.post(f"{self.base}/recommend", json=self.body(query, session_id, history, **kw),
                        timeout=self.timeout)
        client_ms = (time.perf_counter() - t0) * 1000
        try:
            out = r.json()
        except Exception:
            out = {"error": r.text[:500]}
        if not isinstance(out, dict):
            out = {"payload": out}
        out["_client_ms"] = client_ms
        out["_http_status"] = r.status_code
        return out

    def recommend_stream(self, query: str, session_id: str, history: list | None = None, **kw) -> dict:
        t0 = time.perf_counter()
        meta, done, err = None, None, None
        ttft = meta_ms = None
        tokens: list[str] = []
        with self.s.post(f"{self.base}/recommend/stream", json=self.body(query, session_id, history, **kw),
                         stream=True, timeout=self.timeout,
                         headers={"Accept": "text/event-stream"}) as r:
            status = r.status_code
            for ev, data in parse_sse(r.iter_lines(decode_unicode=True)):
                now = (time.perf_counter() - t0) * 1000
                if ev == "meta":
                    meta_ms = now
                    try:
                        meta = json.loads(data)
                    except Exception:
                        meta = {"raw": data}
                elif ev == "token":
                    if ttft is None:
                        ttft = now
                    tokens.append(_token_text(data))
                elif ev == "done":
                    try:
                        done = json.loads(data)
                    except Exception:
                        done = {"raw": data}
                    break
                elif ev == "error":
                    err = data
                    break
        total = (time.perf_counter() - t0) * 1000
        streamed = "".join(tokens)
        answer = (done or {}).get("answer") if isinstance(done, dict) else None
        return {
            "http_status": status, "ttft_ms": ttft, "meta_ms": meta_ms, "total_ms": total,
            "answer": answer if isinstance(answer, str) and answer else streamed,
            "n_token_events": len(tokens), "meta": meta, "done": done, "error": err,
        }
