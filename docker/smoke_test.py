"""ShopTalk deployment smoke test (Phase 11). Standard library only, so it runs anywhere:

    docker exec shoptalk python /app/docker/smoke_test.py                # inside the container
    python docker/smoke_test.py --url http://localhost:8000              # from the host
    python docker/smoke_test.py --latency 30                             # + P50/P95 over 30 requests
    python docker/smoke_test.py --audio my_query.wav                     # use a real recording for /transcribe

Checks: /health (waits while models load), /recommend, a follow-up with history, /transcribe and
/recommend/audio, /speak (if enabled), /feedback, /click, /metrics. Exit code 0 = all passed.
Sessions are tagged "phase11-smoke" so they can be filtered out of log mining.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave

SESSION = "phase11-smoke"
RESULTS = []


def _req(url, method="GET", data=None, headers=None, timeout=180):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body, status, ctype = r.read(), r.status, r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        body, status, ctype = e.read(), e.code, e.headers.get("Content-Type", "")
    ms = (time.perf_counter() - t0) * 1000
    payload = json.loads(body) if ctype.startswith("application/json") and body else body
    return status, payload, ms


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None   # report the 307 itself instead of following it to the shop


def status_no_redirect(url, timeout=30):
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(url, timeout=timeout) as r:
            return r.status, r.headers.get("Location")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Location")


def post_json(url, obj, timeout=180):
    return _req(url, "POST", json.dumps(obj).encode(), {"Content-Type": "application/json"}, timeout)


def post_audio(url, wav_bytes, fields):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="q.wav"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n".encode() + wav_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return _req(url, "POST", b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def tone_wav(seconds=1.5, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * i / sr)))
                               for i in range(int(seconds * sr))))
    return buf.getvalue()


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'✅' if ok else '❌'} {name:22s} {detail}")
    return ok


def pct(values, p):
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--wait", type=int, default=1200, help="seconds to wait for /health = ok")
    ap.add_argument("--latency", type=int, default=0, help="extra sequential /recommend calls for P50/P95")
    ap.add_argument("--audio", help="a real recording (wav/m4a/mp3) for the voice checks")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    # 1) health — models load in the background after the container starts
    t0, health = time.time(), {}
    while time.time() - t0 < a.wait:
        try:
            _, health, _ = _req(base + "/health", timeout=10)
            if isinstance(health, dict) and health.get("status") in ("ok", "error"):
                break
        except Exception:
            pass
        print(f"   … waiting for models ({time.time() - t0:.0f}s) {health.get('status') if isinstance(health, dict) else ''}")
        time.sleep(15)
    if not check("health", isinstance(health, dict) and health.get("status") == "ok",
                 json.dumps({k: health.get(k) for k in ("llm", "asr", "tts", "index_size", "error")} if isinstance(health, dict) else health)):
        sys.exit(1)

    # 2) text recommendation + follow-up
    s, r, ms = post_json(base + "/recommend", {"query": "white sneakers", "session_id": SESSION})
    ok = s == 200 and isinstance(r, dict) and r.get("products")
    check("recommend", ok, f"{ms:,.0f} ms, {len(r.get('products', [])) if isinstance(r, dict) else 0} products")
    if ok:
        s2, r2, ms2 = post_json(base + "/recommend", {"query": "only under $50", "session_id": SESSION,
                                                       "history": [r["history_entry"]]})
        check("follow-up", s2 == 200, f"{ms2:,.0f} ms, parsed={r2.get('parsed', {}) if isinstance(r2, dict) else r2}")
        s3, _, _ = post_json(base + "/feedback", {"request_id": r["request_id"], "rating": "up", "session_id": SESSION,
                                                  "comment": "phase11 smoke test"})
        check("feedback", s3 == 200)
        pid = r["products"][0]["id"]
        s4, loc = status_no_redirect(f"{base}/click/{r['request_id']}/{pid}?rank=1&session_id={SESSION}")
        check("click", s4 in (302, 303, 307), f"http {s4} → {loc}")

    # 3) voice (Phase 10)
    wav = open(a.audio, "rb").read() if a.audio else tone_wav()
    s, t, ms = post_audio(base + "/transcribe", wav, {"session_id": SESSION})
    check("transcribe", s == 200, f"{ms:,.0f} ms, heard: {t.get('text')!r}" if isinstance(t, dict) else f"http {s}")
    s, v, ms = post_audio(base + "/recommend/audio", wav, {"session_id": SESSION})
    # a tone has no words: 422 "No speech" is the correct answer for it
    check("recommend/audio", s == 200 or (s == 422 and not a.audio), f"http {s}, {ms:,.0f} ms")

    # 4) read-aloud (only if the server enabled it)
    if health.get("tts"):
        s, w, ms = post_json(base + "/speak", {"text": "Here are two white sneakers under $50."})
        check("speak", s == 200 and isinstance(w, (bytes, bytearray)) and w[:4] == b"RIFF", f"{ms:,.0f} ms")

    s, m, _ = _req(base + "/metrics", timeout=10)
    check("metrics", s == 200, json.dumps(m.get("counts")) if isinstance(m, dict) else "")

    # 5) optional latency sample
    if a.latency:
        qs = ["white sneakers", "a gray sofa", "black leather wallet", "gold hoop earrings for women", "black tote bag"]
        times = []
        for i in range(a.latency):
            s, r, ms = post_json(base + "/recommend", {"query": qs[i % len(qs)], "session_id": SESSION})
            if s == 200:
                times.append(ms)
        if times:
            print(f"   latency over {len(times)} requests: P50 {pct(times, 50):,.0f} ms · P95 {pct(times, 95):,.0f} ms"
                  f" · mean {statistics.mean(times):,.0f} ms")
        check("latency sample", len(times) == a.latency, f"{len(times)}/{a.latency} ok")

    failed = [n for n, ok in RESULTS if not ok]
    print("\nALL CHECKS PASSED" if not failed else f"\nFAILED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
