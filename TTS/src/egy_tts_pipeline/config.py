from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    name: str = "egy_ar_synth_v1"
    output_root: str = "outputs"


class GenerationConfig(BaseModel):
    target_count: int = 120
    seed: int = 13
    min_words: int = 3
    max_words: int = 18
    allow_arabizi: bool = False
    allow_latin_tokens: bool = False
    domain_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "daily_life": 1.0,
            "transport": 1.0,
            "shopping": 1.0,
            "customer_support": 1.0,
            "family_social": 1.0,
            "work_admin": 1.0,
            "home_services": 1.0,
            "payments": 1.0,
        }
    )


class NormalizationConfig(BaseModel):
    remove_diacritics: bool = True
    normalize_alef: bool = True
    normalize_alef_maqsura: bool = True
    strip_tatweel: bool = True
    normalize_whitespace: bool = True
    normalize_punctuation_spacing: bool = True


class SpeakerProfile(BaseModel):
    id: str
    reference_wavs: list[str]
    weight: float = 1.0
    notes: str | None = None


class TTSConfig(BaseModel):
    engine: Literal["coqui_xtts"] = "coqui_xtts"
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    language: str = "ar"
    device: str = "auto"
    split_sentences: bool = False
    batch_size: int = 8
    max_attempts: int = 2
    overwrite_audio: bool = False
    pause_between_batches_ms: int = 0
    speaker_selection: Literal["round_robin", "weighted_random"] = "round_robin"
    speaker_profiles: list[SpeakerProfile] = Field(default_factory=list)


class ReviewConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7861
    page_size: int = 12


class ExportConfig(BaseModel):
    dataset_name: str = "egy_ar_stt_ready_v1"
    format: Literal["nemo_jsonl"] = "nemo_jsonl"
    copy_audio: bool = False
    relative_audio_paths: bool = True
    include_raw_text: bool = True
    include_quality_metadata: bool = True
    re_normalize_approved_text: bool = True


class QualityConfig(BaseModel):
    min_duration_sec: float = 0.7
    max_duration_sec: float = 18.0
    min_chars_per_sec: float = 3.0
    max_chars_per_sec: float = 22.0
    silence_threshold: float = 0.015
    silence_ratio_warn: float = 0.35
    low_rms_warn: float = 0.01
    clipping_peak_warn: float = 0.995


class PipelineConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)


def load_config(config_path: str | Path) -> PipelineConfig:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(data)
