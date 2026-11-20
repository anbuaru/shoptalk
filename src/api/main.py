"""FastAPI routing engine for the ShopTalk Inference API (Phase 7, voice added in Phase 10).

This application exposes endpoints for grounded catalog query recommendations,
Server-Sent Events streaming, explicit and implicit feedback logging, and system telemetry.

Phase 10 (voice): speech-to-text runs in front of the unchanged text pipeline.
  POST /transcribe        audio file -> transcript (the UI shows it so the user can fix mistakes)
  POST /recommend/audio   audio file -> transcript -> same pipeline as /recommend
  POST /speak             text -> WAV (optional text-to-speech, only when SHOPTALK_TTS=1)
Environment: SHOPTALK_ASR=0 disables speech-to-text; SHOPTALK_ASR_MODEL overrides rag_config["asr"]["model_id"].

To start manually:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse

from src.asr.audio import AudioError, decode_audio
from src.asr.text_norm import transcript_to_query

from .schemas import (AudioRecommendResponse, FeedbackRequest, HealthResponse, HistoryTurn, RecommendRequest,
                      RecommendResponse, SpeakRequest, TranscribeResponse)
from .service import JsonlLogger, Metrics, log_record, run_stream, sse

API_VERSION = "0.10.0"
MAX_HISTORY = 3
MAX_AUDIO_BYTES = 10 * 1024 * 1024   # 10 MB upload limit
MAX_AUDIO_S = 30.0                   # Whisper's window; spoken shopping queries are a few seconds
_PID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


def _default_pipeline_factory() -> Callable[[], object]:
    """
    Factory function resolving which pipeline (real GPU vs mock fake) to construct.

    Returns:
        Callable: Constructor lambda that builds the active execution pipeline.
    """
    def build():
        if os.environ.get("SHOPTALK_FAKE") == "1":
            from .fake_pipeline import build_fake_pipeline
            return build_fake_pipeline()
        from src.rag.factory import build_pipeline
        repo = os.environ.get("SHOPTALK_REPO_DIR", os.getcwd())
        config = os.environ.get("SHOPTALK_CONFIG", os.path.join(repo, "src/rag/rag_config.json"))
        local = os.environ.get("SHOPTALK_LOCAL_CHROMA", "/tmp/shoptalk_chroma")

        # Build and return the real GPU-based RAG pipeline
        pipeline = build_pipeline(config, load_llm=True, repo_dir=repo, local_chroma_dir=local)
        import json
        pipeline.embed_model_name = json.load(open(config)).get("embed_model")
        return pipeline
    return build


def create_app(pipeline_factory: Optional[Callable[[], object]] = None, log_dir: Optional[str] = None,
               load_in_background: bool = True) -> FastAPI:
    """
    Instantiates and configures the FastAPI web application registry.

    Args:
        pipeline_factory (Callable, optional): Custom builder callback.
        log_dir (str, optional): Target directory for JSONLines logs.
        load_in_background (bool): If True, model loading runs asynchronously.

    Returns:
        FastAPI: Registered web application instance.
    """
    factory = pipeline_factory or _default_pipeline_factory()
    repo = os.environ.get("SHOPTALK_REPO_DIR", os.getcwd())
    state = {"pipeline": None, "status": "loading", "error": None, "started": time.time(),
             "asr": None, "asr_status": "loading", "asr_error": None,
             "tts": None, "tts_status": "disabled", "tts_error": None}
    gpu_lock = threading.Lock() # One GPU context: lock restricts model inference to 1 concurrency
    logger = JsonlLogger(log_dir or os.environ.get("SHOPTALK_LOG_DIR", os.path.join(repo, "logs")))
    metrics = Metrics()

    def load_voice():
        """Phase 10: speech-to-text (loaded once, on the GPU) and optional text-to-speech (CPU).
        A failure here never takes the text API down: /health reports it in asr_status / tts_status."""
        if os.environ.get("SHOPTALK_ASR", "1") == "0":
            state["asr_status"] = "disabled"
        else:
            t0 = time.time()
            try:
                from src.asr.whisper_asr import build_asr
                cfg_path = os.environ.get("SHOPTALK_CONFIG", os.path.join(repo, "src/rag/rag_config.json"))
                cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
                state["asr"] = build_asr(cfg)
                state["asr_status"] = "ok"
                logger.write("events", {"event": "asr_loaded", "model": state["asr"].model_id,
                                        "load_s": round(time.time() - t0, 1)})
            except Exception as e:
                state["asr_status"], state["asr_error"] = "error", f"{type(e).__name__}: {e}"
                logger.write("events", {"event": "asr_failed", "error": state["asr_error"],
                                        "trace": traceback.format_exc()[-2000:]})
        from src.tts.mms_tts import build_tts, tts_enabled
        if tts_enabled():
            state["tts_status"] = "loading"
            try:
                state["tts"] = build_tts()
                state["tts_status"] = "ok"
                logger.write("events", {"event": "tts_loaded", "model": state["tts"].model_id})
            except Exception as e:
                state["tts_status"], state["tts_error"] = "error", f"{type(e).__name__}: {e}"
                logger.write("events", {"event": "tts_failed", "error": state["tts_error"],
                                        "trace": traceback.format_exc()[-2000:]})

    def load():
        """Worker function executed at startup to initialize the pipeline components."""
        t0 = time.time()
        try:
            pipeline = factory()
            load_voice()          # before status=ok, so /health "ok" means every model is loaded
            state["pipeline"] = pipeline
            state["status"] = "ok"
            logger.write("events", {"event": "startup", "load_s": round(time.time() - t0, 1),
                                    "llm": getattr(state["pipeline"].generator, "model_id", None),
                                    "asr": getattr(state["asr"], "model_id", None)})
        except Exception as e:
            state["status"], state["error"] = "error", f"{type(e).__name__}: {e}"
            logger.write("events", {"event": "startup_failed", "error": state["error"],
                                    "trace": traceback.format_exc()[-2000:]})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Asynchronous lifecycle hook controlling startup loads and shutdowns."""
        if load_in_background:
            threading.Thread(target=load, daemon=True).start()
        else:
            load()
        yield

    app = FastAPI(title="ShopTalk API", version=API_VERSION, lifespan=lifespan,
                  description="Grounded product recommendations from the ABO catalog (RAG).")
    origins = [o.strip() for o in os.environ.get("SHOPTALK_CORS_ORIGINS", "*").split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    app.state.shoptalk = state
    app.state.metrics = metrics
    app.state.logger = logger

    def ready_pipeline():
        """Asserts model components are fully ready; raises 503 HTTP errors if still loading."""
        if state["status"] == "loading":
            raise HTTPException(503, "Models are still loading; retry in a few seconds (see GET /health).")
        if state["status"] == "error":
            raise HTTPException(503, f"Pipeline failed to load: {state['error']}")
        return state["pipeline"]

    def history_dicts(req: RecommendRequest) -> list:
        """Cuts history stack list down to the maximum supported conversational turns."""
        return [h.model_dump() for h in req.history[-MAX_HISTORY:]]

    def ready_asr():
        """Speech-to-text model, or 503 while loading / when it failed or is disabled."""
        ready_pipeline()
        if state["asr_status"] != "ok":
            detail = {"disabled": "Speech-to-text is disabled on this server (SHOPTALK_ASR=0).",
                      "error": f"Speech-to-text failed to load: {state['asr_error']}"}.get(
                state["asr_status"], "Speech-to-text is still loading; retry in a few seconds.")
            raise HTTPException(503, detail)
        return state["asr"]

    async def read_audio(audio: UploadFile) -> bytes:
        data = await audio.read()
        if not data:
            raise HTTPException(400, "Empty audio file.")
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, f"Audio file is larger than {MAX_AUDIO_BYTES // (1024 * 1024)} MB.")
        return data

    def decode_or_400(data: bytes):
        """Audio -> 16 kHz mono (CPU, outside the GPU lock). Returns (waveform, seconds, decode_ms)."""
        t0 = time.perf_counter()
        try:
            wav, seconds = decode_audio(data, max_seconds=MAX_AUDIO_S)
        except AudioError as e:
            raise HTTPException(400, f"Could not use this audio: {e}")
        return wav, seconds, round((time.perf_counter() - t0) * 1000, 1)

    # ---------------- endpoints ----------------
    @app.get("/health", response_model=HealthResponse)
    def health():
        """Reports application loading status, model IDs, active collection sizes, and uptimes."""
        p = state["pipeline"]
        index_size, embed = None, None
        if p is not None:
            try:
                index_size = p.retriever.collection.count()
            except Exception:
                index_size = getattr(p.retriever, "size", None)
            embed = getattr(p, "embed_model_name", None)
        return HealthResponse(status=state["status"], llm=getattr(getattr(p, "generator", None), "model_id", None),
                              embed_model=embed, index_size=index_size,
                              uptime_s=round(time.time() - state["started"], 1), error=state["error"],
                              asr=getattr(state["asr"], "model_id", None),
                              asr_status=state["asr_status"] if state["asr_status"] != "error"
                              else f"error: {state['asr_error']}",
                              tts=getattr(state["tts"], "model_id", None),
                              tts_status=state["tts_status"] if state["tts_status"] != "error"
                              else f"error: {state['tts_error']}")

    @app.post("/recommend", response_model=RecommendResponse)
    async def recommend(req: RecommendRequest):
        """Synchronous recommendation endpoint matching, generating, and logging query solutions."""
        pipeline = ready_pipeline()
        rid = uuid.uuid4().hex[:16]
        t_arrive = time.perf_counter()

        def work():
            with gpu_lock:
                t_lock = time.perf_counter()
                result = pipeline.run(req.query, history=history_dicts(req))
            result["latency_ms"]["queue"] = round((t_lock - t_arrive) * 1000, 1)
            return result

        try:
            result = await run_in_threadpool(work)
        except Exception as e:
            metrics.inc("errors")
            logger.write("errors", {"request_id": rid, "query": req.query, "error": f"{type(e).__name__}: {e}",
                                    "trace": traceback.format_exc()[-2000:]})
            raise HTTPException(500, f"Internal error (request_id={rid})")

        metrics.inc("requests")
        metrics.inc("no_match", int(bool(result["no_match"])))
        metrics.observe(result["latency_ms"])
        logger.write("requests", log_record(rid, req.session_id, "/recommend", result))
        return RecommendResponse(
            request_id=rid, query=req.query, answer=result["answer"], products=result["products"],
            no_match=result["no_match"], unmet_constraints=result.get("unmet_constraints", []),
            parsed=result["parsed"], grounding=result["grounding"], llm=result.get("llm"),
            latency_ms=result["latency_ms"],
            history_entry={"parsed": result["parsed"], "products": result["products"]})

    @app.post("/recommend/stream")
    def recommend_stream(req: RecommendRequest):
        """Server-Sent Events streaming endpoint delivering metadata and tokens step-by-step."""
        pipeline = ready_pipeline()
        rid = uuid.uuid4().hex[:16]
        t_arrive = time.perf_counter()

        def events():
            gpu_lock.acquire()
            try:
                queue_ms = round((time.perf_counter() - t_arrive) * 1000, 1)
                for kind, data in run_stream(pipeline, req.query, history_dicts(req)):
                    if kind == "meta":
                        yield sse("meta", {"request_id": rid, **data})
                    elif kind == "token":
                        yield sse("token", data)
                    else:
                        data["latency_ms"]["queue"] = queue_ms
                        metrics.inc("stream_requests")
                        metrics.inc("no_match", int(bool(data["no_match"])))
                        metrics.observe({k: v for k, v in data["latency_ms"].items() if v is not None})
                        logger.write("requests", log_record(rid, req.session_id, "/recommend/stream", data))
                        yield sse("done", {"request_id": rid, **{k: data[k] for k in (
                            "query", "answer", "products", "no_match", "unmet_constraints", "parsed",
                            "grounding", "llm", "latency_ms")},
                            "history_entry": {"parsed": data["parsed"], "products": data["products"]}})
            except Exception as e:
                metrics.inc("errors")
                logger.write("errors", {"request_id": rid, "query": req.query, "error": f"{type(e).__name__}: {e}",
                                        "trace": traceback.format_exc()[-2000:]})
                yield sse("error", {"request_id": rid, "detail": "Internal error"})
            finally:
                gpu_lock.release()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------------- Phase 10: voice ----------------
    @app.post("/transcribe", response_model=TranscribeResponse)
    async def transcribe(audio: UploadFile = File(..., description="Recording: wav, webm, ogg, mp3, m4a"),
                         session_id: Optional[str] = Form(None)):
        """Speech -> query text. The UI shows the transcript so the user can correct it before searching."""
        asr = ready_asr()
        data = await read_audio(audio)
        rid = uuid.uuid4().hex[:16]
        wav, seconds, decode_ms = decode_or_400(data)
        t_arrive = time.perf_counter()        # queue = waiting for the GPU lock, after decoding

        def work():
            with gpu_lock:
                t_lock = time.perf_counter()
                out = asr.transcribe(wav)
            out["queue_ms"] = round((t_lock - t_arrive) * 1000, 1)
            return out

        try:
            out = await run_in_threadpool(work)
        except Exception as e:
            metrics.inc("errors")
            logger.write("errors", {"request_id": rid, "endpoint": "/transcribe", "error": f"{type(e).__name__}: {e}",
                                    "trace": traceback.format_exc()[-2000:]})
            raise HTTPException(500, f"Internal error (request_id={rid})")
        latency = {"decode": decode_ms, "queue": out["queue_ms"], "asr": out["ms"]}
        query = transcript_to_query(out["text"])
        metrics.inc("transcribe")
        metrics.observe({"asr": out["ms"]})
        logger.write("asr", {"request_id": rid, "session_id": session_id, "endpoint": "/transcribe",
                             "text": query, "raw_text": out["text"], "model": out["model"],
                             "audio_s": round(seconds, 2), "bytes": len(data), "content_type": audio.content_type,
                             "latency_ms": latency})
        return TranscribeResponse(request_id=rid, text=query, raw_text=out["text"], model=out["model"],
                                  audio_s=round(seconds, 2), latency_ms=latency)

    @app.post("/recommend/audio", response_model=AudioRecommendResponse)
    async def recommend_audio(audio: UploadFile = File(...), session_id: Optional[str] = Form(None),
                              history: Optional[str] = Form(None, description="JSON list of earlier history_entry")):
        """One call for voice clients: speech-to-text, then exactly the /recommend pipeline on the transcript."""
        asr = ready_asr()
        pipeline = ready_pipeline()
        data = await read_audio(audio)
        try:
            hist = [HistoryTurn(**h).model_dump() for h in (json.loads(history) if history else [])][-MAX_HISTORY:]
        except (ValueError, TypeError) as e:
            raise HTTPException(422, f"history must be a JSON list of history entries: {e}")
        rid = uuid.uuid4().hex[:16]
        wav, seconds, decode_ms = decode_or_400(data)
        t_arrive = time.perf_counter()        # queue = waiting for the GPU lock, after decoding

        def work():
            with gpu_lock:
                t_lock = time.perf_counter()
                heard = asr.transcribe(wav)
                text = transcript_to_query(heard["text"]).strip()
                result = pipeline.run(text, history=hist) if text else None
            return heard, result, round((t_lock - t_arrive) * 1000, 1)

        try:
            heard, result, queue_ms = await run_in_threadpool(work)
        except Exception as e:
            metrics.inc("errors")
            logger.write("errors", {"request_id": rid, "endpoint": "/recommend/audio",
                                    "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
            raise HTTPException(500, f"Internal error (request_id={rid})")
        if result is None:
            logger.write("asr", {"request_id": rid, "session_id": session_id, "endpoint": "/recommend/audio",
                                 "text": "", "model": heard["model"], "audio_s": round(seconds, 2)})
            raise HTTPException(422, "No speech was recognised in the recording.")
        lat = result["latency_ms"]
        lat.update(queue=queue_ms, decode=decode_ms, asr=heard["ms"],
                   total_with_asr=round(lat["total"] + heard["ms"] + decode_ms, 1))
        metrics.inc("requests")
        metrics.inc("audio_requests")
        metrics.inc("no_match", int(bool(result["no_match"])))
        metrics.observe(lat)
        rec = log_record(rid, session_id, "/recommend/audio", result)
        rec.update(input="voice", transcript=result["query"], raw_transcript=heard["text"], asr_model=heard["model"],
                   audio_s=round(seconds, 2))
        logger.write("requests", rec)
        return AudioRecommendResponse(
            request_id=rid, query=result["query"], answer=result["answer"], products=result["products"],
            no_match=result["no_match"], unmet_constraints=result.get("unmet_constraints", []),
            parsed=result["parsed"], grounding=result["grounding"], llm=result.get("llm"), latency_ms=lat,
            history_entry={"parsed": result["parsed"], "products": result["products"]},
            transcript=result["query"], raw_transcript=heard["text"], asr_model=heard["model"],
            audio_s=round(seconds, 2))

    @app.post("/speak")
    def speak(req: SpeakRequest):
        """Optional text-to-speech (SHOPTALK_TTS=1): returns audio/wav of the answer, IDs removed."""
        if state["tts_status"] == "disabled":
            raise HTTPException(404, "Text-to-speech is disabled on this server (set SHOPTALK_TTS=1).")
        if state["tts_status"] != "ok":
            raise HTTPException(503, f"Text-to-speech is not ready: {state['tts_status']} {state['tts_error'] or ''}")
        try:
            out = state["tts"].synthesize(req.text)
        except ValueError as e:
            raise HTTPException(400, str(e))
        metrics.inc("tts")
        metrics.observe({"tts": out["ms"]})
        return Response(content=out["wav"], media_type="audio/wav",
                        headers={"X-TTS-ms": str(out["ms"]), "X-Audio-Seconds": str(out["audio_s"])})

    @app.post("/feedback")
    def feedback(fb: FeedbackRequest):
        """Logs explicit user thumbs-up/down ratings and comments into telemetry."""
        metrics.inc("feedback_up" if fb.rating == "up" else "feedback_down")
        logger.write("feedback", {"type": "explicit", **fb.model_dump()})
        return {"status": "recorded"}

    @app.get("/click/{request_id}/{product_id}")
    def click(request_id: str, product_id: str, rank: Optional[int] = None, session_id: Optional[str] = None):
        """Registers implicit product links click feedback, redirecting standard requests to the store link."""
        if not _PID_RE.match(product_id) or not _PID_RE.match(request_id):
            raise HTTPException(400, "Invalid id")
        metrics.inc("clicks")
        logger.write("feedback", {"type": "click", "request_id": request_id, "product_id": product_id,
                                  "rank": rank, "session_id": session_id})
        template = getattr(state["pipeline"], "link_template", None) or "https://www.amazon.com/dp/{id}"
        return RedirectResponse(template.format(id=product_id), status_code=307)

    @app.get("/metrics")
    def get_metrics():
        """Returns in-memory requests counters and rolling percentiles timing statistics."""
        return metrics.snapshot()

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        """Catches and drops unhandled system failures into clean 500 responses."""
        return JSONResponse(status_code=500, content={"detail": "Internal error"})

    return app


app = create_app()
