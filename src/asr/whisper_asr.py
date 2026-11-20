"""Speech-to-text for ShopTalk (Phase 10): open-source Whisper models via Hugging Face transformers.

Loaded once at API startup and kept on the GPU next to the LLM. Uses the model + processor directly
(not `pipeline()`), which works the same for whisper-tiny/base/small(.en), distil-whisper and
whisper-large-v3-turbo.

    asr = WhisperASR("openai/whisper-base.en")
    out = asr.transcribe(waveform_16k)   # {"text", "ms", "audio_s", "model"}
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from .audio import SAMPLE_RATE

DEFAULT_MODEL = "openai/whisper-base.en"


class WhisperASR:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: Optional[str] = None, fp16: bool = True,
                 language: str = "en", max_new_tokens: int = 96):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and self.device.startswith("cuda")) else torch.float32
        self.language = language
        self.max_new_tokens = max_new_tokens
        # English-only checkpoints (*.en) reject the language/task arguments
        self.multilingual = not model_id.rstrip("/").endswith(".en")
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id)
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=self.dtype, low_cpu_mem_usage=True)
        except TypeError:   # older transformers: torch_dtype
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.dtype, low_cpu_mem_usage=True)
        self.model = model.to(self.device).eval()
        self.load_s = time.perf_counter() - t0
        self._torch = torch

    def _sync(self):
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        if sr != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr}")
        torch = self._torch
        self._sync()
        t0 = time.perf_counter()
        feats = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = feats.input_features.to(self.device, self.dtype)
        kwargs = {"max_new_tokens": self.max_new_tokens}
        if self.multilingual:
            kwargs.update(language=self.language, task="transcribe")
        with torch.inference_mode():
            ids = self.model.generate(inputs, **kwargs)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        self._sync()
        return {"text": text, "ms": round((time.perf_counter() - t0) * 1000, 1),
                "audio_s": round(len(audio) / SAMPLE_RATE, 2), "model": self.model_id}

    def gpu_mb(self) -> Optional[float]:
        if not self.device.startswith("cuda"):
            return None
        return round(sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6, 1)

    def unload(self):
        del self.model
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()


class FakeASR:
    """No-GPU stand-in (SHOPTALK_FAKE=1): returns SHOPTALK_FAKE_TRANSCRIPT, or a fixed sentence."""
    model_id = "fake-asr"
    load_s = 0.0

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        t0 = time.perf_counter()
        text = os.environ.get("SHOPTALK_FAKE_TRANSCRIPT", "Show me a red shirt for men under $50")
        return {"text": text, "ms": round((time.perf_counter() - t0) * 1000, 1),
                "audio_s": round(len(audio) / sr, 2), "model": self.model_id}


def build_asr(config: Optional[dict] = None):
    """ASR from env / rag_config['asr'].  Env wins: SHOPTALK_ASR_MODEL, SHOPTALK_ASR_DEVICE."""
    if os.environ.get("SHOPTALK_FAKE") == "1":
        return FakeASR()
    cfg = (config or {}).get("asr", {}) or {}
    return WhisperASR(os.environ.get("SHOPTALK_ASR_MODEL") or cfg.get("model_id") or DEFAULT_MODEL,
                      device=os.environ.get("SHOPTALK_ASR_DEVICE") or cfg.get("device"),
                      max_new_tokens=int(cfg.get("max_new_tokens", 96)))
