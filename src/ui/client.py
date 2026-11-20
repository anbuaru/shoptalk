"""HTTP client for the ShopTalk inference API (Phase 7), used by the Phase 8 UI.

Keeps the UI free of HTTP details: the Gradio handlers call these methods and get plain dicts back.
Only `requests` is needed, so the same client works in Colab, in tests and later on AWS (Phase 11).

Endpoints used:
  GET  /health                          -> status banner
  POST /recommend/stream                -> SSE events: meta (products), token (answer text), done, error
  POST /recommend                       -> non-streaming fallback
  POST /feedback                        -> thumbs up / down
  GET  /click/{request_id}/{product_id} -> logs the click, then 307-redirects to the product page
  POST /transcribe                      -> speech-to-text for the microphone input (Phase 10)
  POST /speak                           -> optional text-to-speech of the answer (Phase 10)
"""
from __future__ import annotations

import json
import mimetypes
import os
from typing import Iterator, Optional

import requests


class ApiError(RuntimeError):
    """Raised when the API is unreachable or answers with an error the user should see."""


class ShopTalkClient:
    """
    Thin wrapper around the Phase 7 endpoints.

    Args:
        base_url (str): API address, e.g. "http://127.0.0.1:8000" or a public tunnel URL.
        timeout (float): Seconds to wait for a full response (streaming reads use it per chunk).
    """

    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # ---------------- helpers ----------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _raise_for(r: requests.Response) -> None:
        """Turns HTTP errors into ApiError with the API's own message (e.g. 503 while loading)."""
        if r.status_code < 400:
            return
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(f"API error {r.status_code}: {detail}")

    # ---------------- endpoints ----------------
    def health(self) -> dict:
        """Returns /health, or {'status': 'unreachable', 'error': ...} if the API cannot be reached."""
        try:
            r = self.session.get(self._url("/health"), timeout=10)
            self._raise_for(r)
            return r.json()
        except (requests.RequestException, ApiError) as e:
            return {"status": "unreachable", "error": str(e)}

    def recommend(self, query: str, history: Optional[list] = None, session_id: Optional[str] = None) -> dict:
        """Non-streaming call to POST /recommend."""
        try:
            r = self.session.post(self._url("/recommend"), timeout=self.timeout,
                                  json={"query": query, "history": history or [], "session_id": session_id})
        except requests.RequestException as e:
            raise ApiError(f"API unreachable: {e}") from e
        self._raise_for(r)
        return r.json()

    def stream(self, query: str, history: Optional[list] = None,
               session_id: Optional[str] = None) -> Iterator[tuple]:
        """
        Calls POST /recommend/stream and yields (event, data) as each Server-Sent Event arrives.

        Events: ("meta", {request_id, products, ...}), ("token", {text}), ("done", {...full response}),
        ("error", {request_id, detail}).
        """
        try:
            r = self.session.post(self._url("/recommend/stream"), stream=True, timeout=(10, self.timeout),
                                  json={"query": query, "history": history or [], "session_id": session_id})
        except requests.RequestException as e:
            raise ApiError(f"API unreachable: {e}") from e
        with r:
            self._raise_for(r)
            yield from parse_sse(r.iter_content(chunk_size=None))

    def feedback(self, request_id: str, rating: str, comment: Optional[str] = None,
                 session_id: Optional[str] = None) -> dict:
        """POST /feedback with rating 'up' or 'down'."""
        try:
            r = self.session.post(self._url("/feedback"), timeout=10, json={
                "request_id": request_id, "rating": rating, "comment": comment or None, "session_id": session_id})
        except requests.RequestException as e:
            raise ApiError(f"API unreachable: {e}") from e
        self._raise_for(r)
        return r.json()

    def transcribe(self, audio_path: str, session_id: Optional[str] = None) -> dict:
        """POST /transcribe with a recorded file. Returns {request_id, text, model, audio_s, latency_ms}."""
        mime = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
        try:
            with open(audio_path, "rb") as f:
                r = self.session.post(self._url("/transcribe"), timeout=60,
                                      files={"audio": (os.path.basename(audio_path), f, mime)},
                                      data={"session_id": session_id} if session_id else {})
        except requests.RequestException as e:
            raise ApiError(f"API unreachable: {e}") from e
        self._raise_for(r)
        return r.json()

    def speak(self, text: str) -> bytes:
        """POST /speak. Returns WAV bytes; raises ApiError (e.g. 404 when TTS is disabled on the server)."""
        try:
            r = self.session.post(self._url("/speak"), json={"text": text[:2000]}, timeout=60)
        except requests.RequestException as e:
            raise ApiError(f"API unreachable: {e}") from e
        self._raise_for(r)
        return r.content

    def click(self, request_id: str, product_id: str, rank: Optional[int] = None,
              session_id: Optional[str] = None) -> Optional[str]:
        """
        Logs a click via GET /click/... without following the redirect.

        Returns:
            str | None: The product page URL from the redirect, or None if the API did not redirect.
        """
        params = {k: v for k, v in (("rank", rank), ("session_id", session_id)) if v is not None}
        try:
            r = self.session.get(self._url(f"/click/{request_id}/{product_id}"), params=params,
                                 allow_redirects=False, timeout=10)
        except requests.RequestException:
            return None
        return r.headers.get("location") if r.status_code in (301, 302, 303, 307, 308) else None


def parse_sse(chunks) -> Iterator[tuple]:
    """
    Parses a Server-Sent Events byte stream into (event, data) tuples.

    Works on raw chunks (not lines) so tokens are shown as soon as they arrive; requests' iter_lines
    would buffer them. Events are separated by a blank line; `data:` is JSON.
    """
    buf = ""
    for chunk in chunks:
        if not chunk:
            continue
        buf += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        buf = buf.replace("\r\n", "\n")
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            event, data_lines = "message", []
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    yield event, json.loads(raw)
                except ValueError:
                    yield event, {"text": raw}
