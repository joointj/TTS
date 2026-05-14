from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .config import PipelineConfig, load_config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-") or "sample"


def json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row))
            handle.write("\n")


@dataclass(frozen=True)
class PipelinePaths:
    workspace_root: Path
    config_path: Path
    run_dir: Path
    db_path: Path
    logs_dir: Path
    manifests_dir: Path
    audio_dir: Path
    exports_dir: Path
    template_dir: Path


@dataclass
class PipelineContext:
    config: PipelineConfig
    paths: PipelinePaths
    logger: logging.Logger


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("egy_tts_pipeline")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def build_context(config_path: str | Path, workspace_root: str | Path | None = None) -> PipelineContext:
    resolved_config_path = Path(config_path).resolve()
    root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
    config = load_config(resolved_config_path)
    output_root = resolve_path(root, config.run.output_root)
    run_dir = output_root / config.run.name
    paths = PipelinePaths(
        workspace_root=root,
        config_path=resolved_config_path,
        run_dir=run_dir,
        db_path=run_dir / "pipeline.db",
        logs_dir=run_dir / "logs",
        manifests_dir=run_dir / "manifests",
        audio_dir=run_dir / "audio",
        exports_dir=run_dir / "exports",
        template_dir=(root / "src" / "egy_tts_pipeline" / "templates").resolve(),
    )
    for directory in (
        paths.run_dir,
        paths.logs_dir,
        paths.manifests_dir,
        paths.audio_dir,
        paths.exports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(paths.logs_dir / "pipeline.log")
    return PipelineContext(config=config, paths=paths, logger=logger)


def coerce_path_strings(paths: Sequence[str], root: Path) -> list[Path]:
    return [resolve_path(root, item) for item in paths]
