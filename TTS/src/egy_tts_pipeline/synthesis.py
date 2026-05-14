from __future__ import annotations

import importlib
import time
from pathlib import Path

import torch

from .config import TTSConfig
from .db import PipelineDB
from .quality import compute_audio_quality
from .utils import PipelineContext, coerce_path_strings


class SpeakerPicker:
    def __init__(self, config: TTSConfig, workspace_root: Path, seed: int):
        self.config = config
        self.workspace_root = workspace_root
        self.counter = 0
        self.rng = __import__("random").Random(seed)
        self._validate_profiles()

    def _validate_profiles(self) -> None:
        if not self.config.speaker_profiles:
            raise RuntimeError("At least one speaker profile is required for synthesis.")
        missing_paths: list[str] = []
        for profile in self.config.speaker_profiles:
            for wav_path in coerce_path_strings(profile.reference_wavs, self.workspace_root):
                if not wav_path.exists():
                    missing_paths.append(f"{profile.id}: {wav_path}")
        if missing_paths:
            joined = "\n".join(missing_paths[:10])
            raise RuntimeError(f"Missing reference WAVs for synthesis:\n{joined}")

    def pick(self) -> tuple[str, list[Path]]:
        profiles = self.config.speaker_profiles
        if self.config.speaker_selection == "weighted_random":
            weights = [profile.weight for profile in profiles]
            profile = self.rng.choices(profiles, weights=weights, k=1)[0]
        else:
            profile = profiles[self.counter % len(profiles)]
            self.counter += 1
        return profile.id, coerce_path_strings(profile.reference_wavs, self.workspace_root)


class CoquiXTTSBackend:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.api = self._load_api()
        self.cached_speakers: set[str] = set()

    @staticmethod
    def _resolve_device(configured_device: str) -> str:
        if configured_device != "auto":
            return configured_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_api(self):
        try:
            tts_module = importlib.import_module("TTS.api")
            api_class = getattr(tts_module, "TTS")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Coqui TTS is not installed. Install `coqui-tts` after torch/torchaudio."
            ) from exc

        use_gpu = self.device.startswith("cuda")
        try:
            api = api_class(model_name=self.config.model_name)
            if hasattr(api, "to"):
                api = api.to(self.device)
        except TypeError:
            api = api_class(model_name=self.config.model_name, gpu=use_gpu)
        return api

    def synthesize_to_file(
        self,
        text: str,
        audio_path: Path,
        speaker_id: str,
        speaker_wavs: list[Path],
    ) -> None:
        kwargs = {
            "text": text,
            "file_path": str(audio_path),
            "language": self.config.language,
            "speaker": speaker_id,
            "split_sentences": self.config.split_sentences,
        }
        if speaker_id not in self.cached_speakers:
            kwargs["speaker_wav"] = [str(path) for path in speaker_wavs]

        self.api.tts_to_file(**kwargs)
        self.cached_speakers.add(speaker_id)


def run_synthesis(
    context: PipelineContext,
    db: PipelineDB,
    limit: int | None = None,
) -> dict[str, int]:
    picker = SpeakerPicker(
        config=context.config.tts,
        workspace_root=context.paths.workspace_root,
        seed=context.config.generation.seed,
    )
    backend = CoquiXTTSBackend(context.config.tts)

    processed = 0
    succeeded = 0
    failed = 0
    batch_size = context.config.tts.batch_size
    max_attempts = context.config.tts.max_attempts

    while True:
        remaining = batch_size if limit is None else min(batch_size, max(limit - processed, 0))
        if remaining <= 0:
            break
        queue = db.get_synthesis_queue(
            limit=remaining,
            max_attempts=max_attempts,
            overwrite_audio=context.config.tts.overwrite_audio,
        )
        if not queue:
            break

        for sample in queue:
            speaker_id, speaker_wavs = picker.pick()
            audio_path = context.paths.audio_dir / f"{sample['id']}.wav"
            db.mark_synthesis_running(sample["id"], speaker_id)
            try:
                context.logger.info("Synthesizing %s with speaker %s", sample["id"], speaker_id)
                backend.synthesize_to_file(
                    text=sample["tts_text"],
                    audio_path=audio_path,
                    speaker_id=speaker_id,
                    speaker_wavs=speaker_wavs,
                )
                metrics = compute_audio_quality(
                    audio_path=audio_path,
                    text=sample["normalized_text"],
                    config=context.config.quality,
                )
                db.mark_synthesis_success(
                    sample_id=sample["id"],
                    speaker_id=speaker_id,
                    audio_path=str(audio_path.relative_to(context.paths.run_dir)),
                    metrics=metrics,
                    model_name=context.config.tts.model_name,
                    language=context.config.tts.language,
                )
                succeeded += 1
            except Exception as exc:  # pragma: no cover - exercised in integration use
                context.logger.exception("Synthesis failed for %s", sample["id"])
                db.mark_synthesis_failure(sample["id"], speaker_id, str(exc))
                failed += 1
            processed += 1
            if limit is not None and processed >= limit:
                break
        if context.config.tts.pause_between_batches_ms > 0:
            time.sleep(context.config.tts.pause_between_batches_ms / 1000)
        if limit is not None and processed >= limit:
            break

    return {"processed": processed, "succeeded": succeeded, "failed": failed}
