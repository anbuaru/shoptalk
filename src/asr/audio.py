"""Audio decoding for speech-to-text (Phase 10).

Every input (browser recording, uploaded file, WAV from the notebook) is turned into 16 kHz mono float32,
which is what Whisper expects. ffmpeg handles any container/codec (webm, ogg, mp3, m4a, wav); a pure-stdlib
WAV reader is the fallback when ffmpeg is not installed (e.g. a slim Docker image in Phase 11).
"""
from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Union

import numpy as np

SAMPLE_RATE = 16000
AudioInput = Union[bytes, str, Path]


class AudioError(ValueError):
    """The audio could not be decoded, or is empty / too long."""


def _read_bytes(src: AudioInput) -> bytes:
    if isinstance(src, (str, Path)):
        return Path(src).read_bytes()
    return bytes(src)


def _ffmpeg_decode(data: bytes, sr: int) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, input=data, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not decode the audio: {proc.stderr.decode(errors='ignore')[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _wav_decode(data: bytes, sr: int) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(data)) as w:
            n_ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError) as e:
        raise AudioError(f"not a PCM WAV file and ffmpeg is not installed: {e}") from e
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise AudioError(f"unsupported WAV sample width: {width} bytes")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    if rate != sr and len(x):
        t_new = np.arange(int(round(len(x) * sr / rate))) / sr
        x = np.interp(t_new, np.arange(len(x)) / rate, x).astype(np.float32)
    return x


def decode_audio(src: AudioInput, sr: int = SAMPLE_RATE, max_seconds: float = 30.0) -> tuple:
    """Returns (float32 mono waveform at `sr`, duration in seconds). Raises AudioError."""
    data = _read_bytes(src)
    if not data:
        raise AudioError("empty audio")
    x = _ffmpeg_decode(data, sr) if shutil.which("ffmpeg") else _wav_decode(data, sr)
    seconds = len(x) / sr
    if seconds < 0.1:
        raise AudioError("audio is shorter than 0.1 s")
    if max_seconds and seconds > max_seconds:
        raise AudioError(f"audio is {seconds:.1f} s; the limit is {max_seconds:.0f} s")
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 1.0:
        x = x / peak
    return x.astype(np.float32), seconds


def to_wav_bytes(x: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """float waveform -> 16-bit PCM WAV bytes (stdlib only)."""
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


def write_wav(path: Union[str, Path], x: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_wav_bytes(x, sr))
    return path
