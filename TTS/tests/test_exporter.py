import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from egy_tts_pipeline.db import PipelineDB
from egy_tts_pipeline.exporter import export_dataset
from egy_tts_pipeline.utils import build_context, close_logger


def write_sine_wav(path: Path, sample_rate: int = 24000, duration_sec: float = 0.5) -> None:
    samples = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 220 * samples) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())


class ExporterTests(unittest.TestCase):
    def test_export_only_includes_approved_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "configs").mkdir()
            (root / "src" / "egy_tts_pipeline" / "templates").mkdir(parents=True)
            (root / "src" / "egy_tts_pipeline" / "templates" / "review.html").write_text("ok", encoding="utf-8")
            config_path = root / "configs" / "pipeline.yaml"
            config_path.write_text(
                """
run:
  name: "test_run"
  output_root: "outputs"
tts:
  speaker_profiles:
    - id: "speaker_1"
      reference_wavs: ["data/reference_speakers/ref.wav"]
""",
                encoding="utf-8",
            )
            context = build_context(config_path=config_path, workspace_root=root)
            db = PipelineDB(context.paths.db_path)
            db.initialize()
            try:
                db.insert_generated_samples(
                    [
                        {
                            "id": "test-000001",
                            "prompt_text": "أنا عايز كشري",
                            "tts_text": "أنا عايز كشري",
                            "normalized_text": "انا عايز كشري",
                            "approved_text": None,
                            "domain": "shopping",
                            "intent": "order_food",
                            "template_id": "shopping_01",
                            "speaker_id": None,
                            "text_features": {"word_count": 3},
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                        },
                        {
                            "id": "test-000002",
                            "prompt_text": "معلش استنى شوية",
                            "tts_text": "معلش استنى شوية",
                            "normalized_text": "معلش استني شويه",
                            "approved_text": None,
                            "domain": "daily_life",
                            "intent": "coordination",
                            "template_id": "daily_04",
                            "speaker_id": None,
                            "text_features": {"word_count": 3},
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                        },
                    ]
                )
                for sample_id in ("test-000001", "test-000002"):
                    audio_path = context.paths.audio_dir / f"{sample_id}.wav"
                    write_sine_wav(audio_path)
                    db.mark_synthesis_success(
                        sample_id=sample_id,
                        speaker_id="speaker_1",
                        audio_path=str(audio_path.relative_to(context.paths.run_dir)),
                        metrics={
                            "duration_sec": 0.5,
                            "sample_rate": 24000,
                            "quality_score": 1.0,
                            "quality_flags": [],
                        },
                        model_name="xtts",
                        language="ar",
                    )
                db.update_review("test-000001", "approved", "good", "أنا عايز كشري")
                db.update_review("test-000002", "rejected", "bad", "معلش استنى شوية")

                export_root = export_dataset(context=context, db=db)
                manifest_lines = (export_root / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(manifest_lines), 1)
                self.assertIn("أنا عايز كشري", manifest_lines[0])
            finally:
                db.close()
                close_logger(context.logger)


if __name__ == "__main__":
    unittest.main()
