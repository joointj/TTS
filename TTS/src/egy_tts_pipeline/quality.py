from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .config import QualityConfig


def _load_wav(audio_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count)

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=dtype_map[sample_width]).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_width == 1:
        audio = (audio - 128.0) / 128.0
    else:
        audio = audio / float(2 ** (8 * sample_width - 1))
    return audio, sample_rate


def compute_audio_quality(audio_path: Path, text: str, config: QualityConfig) -> dict[str, object]:
    audio, sample_rate = _load_wav(audio_path)
    duration_sec = float(len(audio) / sample_rate) if sample_rate else 0.0
    char_count = max(len(text), 1)
    chars_per_sec = char_count / max(duration_sec, 1e-6)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    silence_mask = np.abs(audio) <= config.silence_threshold
    silence_ratio = float(np.mean(silence_mask)) if len(audio) else 1.0

    flags: list[str] = []
    if duration_sec < config.min_duration_sec:
        flags.append("too_short_audio")
    if duration_sec > config.max_duration_sec:
        flags.append("too_long_audio")
    if chars_per_sec < config.min_chars_per_sec:
        flags.append("slow_speech_or_long_pause")
    if chars_per_sec > config.max_chars_per_sec:
        flags.append("fast_speech_or_alignment_risk")
    if silence_ratio > config.silence_ratio_warn:
        flags.append("high_silence_ratio")
    if rms < config.low_rms_warn:
        flags.append("low_energy")
    if peak >= config.clipping_peak_warn:
        flags.append("clipping_risk")

    score = max(0.0, 1.0 - 0.12 * len(flags))
    return {
        "duration_sec": round(duration_sec, 4),
        "sample_rate": sample_rate,
        "chars_per_sec": round(chars_per_sec, 4),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "silence_ratio": round(silence_ratio, 6),
        "quality_score": round(score, 4),
        "quality_flags": flags,
    }
