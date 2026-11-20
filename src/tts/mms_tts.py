"""Optional text-to-speech for the ShopTalk UI (Phase 10): reads the answer aloud.

Default voice: facebook/mms-tts-eng (Meta MMS, a small VITS model, ~36M parameters) through Hugging Face
transformers, so it adds no new packages to the pinned API environment. It runs on CPU by default, so it
never waits for the GPU lock held by the LLM. Licence: check the model card before any commercial use
(MMS checkpoints are published for non-commercial use); `SHOPTALK_TTS_MODEL` swaps in another VITS
checkpoint whose tokenizer does not need a phonemizer.

The MMS English tokenizer only knows letters, so text goes through `speakable()` first: product IDs are
removed and numbers/prices are spelled out ("$11.99" -> "eleven dollars ninety nine").
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from src.asr.audio import to_wav_bytes
from src.asr.text_norm import speakable

DEFAULT_TTS_MODEL = "facebook/mms-tts-eng"
MAX_CHARS = 600


class MMSTTS:
    def __init__(self, model_id: str = DEFAULT_TTS_MODEL, device: Optional[str] = None, seed: int = 0):
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.model_id = model_id
        self.device = device or "cpu"
        self.seed = seed
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).to(self.device).eval()
        self.sample_rate = int(self.model.config.sampling_rate)
        self.load_s = time.perf_counter() - t0
        self._torch = torch

    def synthesize(self, text: str) -> dict:
        """-> {"wav": bytes, "ms": float, "audio_s": float, "text": spoken text}"""
        torch = self._torch
        spoken = speakable(text)[:MAX_CHARS]
        if not spoken:
            raise ValueError("nothing to say")
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)          # VITS samples durations: fixed seed = same voice every time
        inputs = self.tokenizer(spoken, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            wave = self.model(**inputs).waveform[0].float().cpu().numpy()
        ms = (time.perf_counter() - t0) * 1000
        return {"wav": to_wav_bytes(wave, self.sample_rate), "ms": round(ms, 1),
                "audio_s": round(len(wave) / self.sample_rate, 2), "text": spoken, "sample_rate": self.sample_rate}


class FakeTTS:
    """No-model stand-in (SHOPTALK_FAKE=1): a short 440 Hz beep per sentence."""
    model_id = "fake-tts"
    sample_rate = 16000
    load_s = 0.0

    def synthesize(self, text: str) -> dict:
        spoken = speakable(text)
        if not spoken:
            raise ValueError("nothing to say")
        t = np.arange(int(0.3 * self.sample_rate)) / self.sample_rate
        wave = 0.2 * np.sin(2 * np.pi * 440 * t)
        return {"wav": to_wav_bytes(wave, self.sample_rate), "ms": 0.1, "audio_s": 0.3, "text": spoken,
                "sample_rate": self.sample_rate}


def tts_enabled() -> bool:
    return os.environ.get("SHOPTALK_TTS", "0") == "1"


def build_tts():
    if os.environ.get("SHOPTALK_FAKE") == "1":
        return FakeTTS()
    return MMSTTS(os.environ.get("SHOPTALK_TTS_MODEL", DEFAULT_TTS_MODEL),
                  device=os.environ.get("SHOPTALK_TTS_DEVICE", "cpu"))
