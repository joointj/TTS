from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

from .db import PipelineDB
from .text_normalization import normalize_egyptian_text
from .utils import PipelineContext, json_dumps


def export_dataset(
    context: PipelineContext,
    db: PipelineDB,
    include_pending_review: bool = False,
) -> Path:
    rows = db.fetch_export_rows(include_pending_review=include_pending_review)
    export_root = context.paths.exports_dir / context.config.export.dataset_name
    export_root.mkdir(parents=True, exist_ok=True)
    audio_export_dir = export_root / "audio"
    if context.config.export.copy_audio:
        audio_export_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = export_root / "manifest.jsonl"
    metadata_path = export_root / "metadata.csv"
    summary_path = export_root / "summary.json"

    manifest_rows: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []

    for row in rows:
        source_audio_path = context.paths.run_dir / row["audio_path"]
        if context.config.export.copy_audio:
            target_audio_path = audio_export_dir / f"{row['id']}.wav"
            shutil.copy2(source_audio_path, target_audio_path)
        else:
            target_audio_path = source_audio_path

        approved_text = row["approved_text"] or row["normalized_text"]
        if context.config.export.re_normalize_approved_text:
            approved_text = normalize_egyptian_text(approved_text, context.config.normalization)

        if context.config.export.relative_audio_paths:
            audio_filepath = os.path.relpath(target_audio_path, export_root)
        else:
            audio_filepath = str(target_audio_path.resolve())

        manifest_row = {
            "audio_filepath": audio_filepath,
            "text": approved_text,
            "duration": row["duration_sec"],
            "speaker_id": row["speaker_id"],
            "domain": row["domain"],
            "dialect": "arz",
            "synthetic": True,
            "tts_model": row["tts_model"],
            "quality_flags": row["quality_flags"],
        }
        if context.config.export.include_raw_text:
            manifest_row["text_raw"] = row["prompt_text"]
        if context.config.export.include_quality_metadata:
            manifest_row["quality_score"] = row["quality_score"]

        manifest_rows.append(manifest_row)
        csv_rows.append(
            {
                "id": row["id"],
                "audio_filepath": audio_filepath,
                "text": approved_text,
                "text_raw": row["prompt_text"],
                "review_status": row["review_status"],
                "speaker_id": row["speaker_id"],
                "duration_sec": row["duration_sec"],
                "quality_score": row["quality_score"],
                "quality_flags": "|".join(row["quality_flags"]),
                "domain": row["domain"],
            }
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json_dumps(row))
            handle.write("\n")

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()) if csv_rows else [
            "id",
            "audio_filepath",
            "text",
            "text_raw",
            "review_status",
            "speaker_id",
            "duration_sec",
            "quality_score",
            "quality_flags",
            "domain",
        ])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    summary = {
        "dataset_name": context.config.export.dataset_name,
        "sample_count": len(manifest_rows),
        "format": context.config.export.format,
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "copied_audio": context.config.export.copy_audio,
        "include_pending_review": include_pending_review,
    }
    summary_path.write_text(json_dumps(summary), encoding="utf-8")
    return export_root
