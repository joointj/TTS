from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .db import PipelineDB
from .exporter import export_dataset
from .prompt_generation import PromptGenerator
from .review_app import create_review_app
from .synthesis import run_synthesis
from .utils import build_context, slugify, write_jsonl


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic Egyptian Arabic speech dataset pipeline.")
    parser.add_argument("--config", default="configs/pipeline.example.yaml", help="Path to YAML config file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-prompts", help="Generate Egyptian Arabic prompt corpus.")
    generate_parser.add_argument("--count", type=int, default=None, help="Target total prompt count for this run.")

    synthesize_parser = subparsers.add_parser("synthesize", help="Run Coqui XTTS synthesis.")
    synthesize_parser.add_argument("--limit", type=int, default=None, help="Optional max samples to synthesize this run.")

    review_parser = subparsers.add_parser("serve-review", help="Launch review UI.")
    review_parser.add_argument("--host", default=None)
    review_parser.add_argument("--port", type=int, default=None)

    export_parser = subparsers.add_parser("export", help="Export approved dataset manifest.")
    export_parser.add_argument("--include-pending-review", action="store_true")

    subparsers.add_parser("status", help="Print pipeline status summary.")

    run_all_parser = subparsers.add_parser("run-all", help="Generate prompts then synthesize all pending samples.")
    run_all_parser.add_argument("--count", type=int, default=None)
    run_all_parser.add_argument("--limit", type=int, default=None)
    run_all_parser.add_argument("--export-reviewed", action="store_true")

    return parser


def _open_db(context) -> PipelineDB:
    db = PipelineDB(context.paths.db_path)
    db.initialize()
    return db


def _sample_ids_for_count(db: PipelineDB, run_name: str, count: int) -> list[str]:
    sequence_values = db.reserve_counter_values("sample", count)
    run_prefix = slugify(run_name)
    return [f"{run_prefix}-{value:06d}" for value in sequence_values]


def command_generate_prompts(args: argparse.Namespace) -> None:
    context = build_context(args.config)
    db = _open_db(context)
    try:
        desired_total = args.count or context.config.generation.target_count
        existing_total = db.count_samples()
        to_create = max(desired_total - existing_total, 0)
        if to_create == 0:
            context.logger.info("Prompt generation skipped: already have %s samples.", existing_total)
            return
        sample_ids = _sample_ids_for_count(db, context.config.run.name, to_create)
        generator = PromptGenerator(context.config.generation, context.config.normalization)
        rows = generator.generate(sample_ids=sample_ids, existing_texts=db.fetch_existing_normalized_texts())
        inserted = db.insert_generated_samples(rows)
        write_jsonl(context.paths.manifests_dir / "generated_prompts.jsonl", db.fetch_all_samples())
        context.logger.info("Generated %s new prompts. Total samples: %s", inserted, db.count_samples())
    finally:
        db.close()


def command_synthesize(args: argparse.Namespace) -> None:
    context = build_context(args.config)
    db = _open_db(context)
    try:
        result = run_synthesis(context=context, db=db, limit=args.limit)
        write_jsonl(context.paths.manifests_dir / "synthesis_snapshot.jsonl", db.fetch_all_samples())
        context.logger.info(
            "Synthesis processed=%s succeeded=%s failed=%s",
            result["processed"],
            result["succeeded"],
            result["failed"],
        )
    finally:
        db.close()


def command_status(args: argparse.Namespace) -> None:
    context = build_context(args.config)
    db = _open_db(context)
    try:
        for key, value in db.get_status_counts().items():
            print(f"{key}: {value}")
    finally:
        db.close()


def command_review(args: argparse.Namespace) -> None:
    context = build_context(args.config)
    app = create_review_app(context)
    host = args.host or context.config.review.host
    port = args.port or context.config.review.port
    uvicorn.run(app, host=host, port=port)


def command_export(args: argparse.Namespace) -> None:
    context = build_context(args.config)
    db = _open_db(context)
    try:
        export_root = export_dataset(
            context=context,
            db=db,
            include_pending_review=args.include_pending_review,
        )
        write_jsonl(context.paths.manifests_dir / "export_snapshot.jsonl", db.fetch_export_rows(args.include_pending_review))
        context.logger.info("Exported dataset to %s", export_root)
    finally:
        db.close()


def command_run_all(args: argparse.Namespace) -> None:
    generate_args = argparse.Namespace(config=args.config, count=args.count)
    command_generate_prompts(generate_args)

    synth_args = argparse.Namespace(config=args.config, limit=args.limit)
    command_synthesize(synth_args)

    if args.export_reviewed:
        export_args = argparse.Namespace(config=args.config, include_pending_review=False)
        command_export(export_args)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    command_map = {
        "generate-prompts": command_generate_prompts,
        "synthesize": command_synthesize,
        "serve-review": command_review,
        "export": command_export,
        "status": command_status,
        "run-all": command_run_all,
    }
    command_map[args.command](args)
