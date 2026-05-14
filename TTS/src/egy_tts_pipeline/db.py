from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .utils import json_dumps, utc_now_iso


class PipelineDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection = sqlite3.connect(str(db_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS counters (
                name TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS samples (
                id TEXT PRIMARY KEY,
                prompt_text TEXT NOT NULL,
                tts_text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                approved_text TEXT,
                domain TEXT NOT NULL,
                intent TEXT,
                template_id TEXT,
                speaker_id TEXT,
                text_features_json TEXT NOT NULL,
                synthesis_status TEXT NOT NULL DEFAULT 'pending',
                review_status TEXT NOT NULL DEFAULT 'pending',
                audio_path TEXT,
                duration_sec REAL,
                sample_rate INTEGER,
                quality_score REAL,
                quality_flags_json TEXT NOT NULL DEFAULT '[]',
                tts_model TEXT,
                tts_language TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                review_notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                reviewed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_samples_synthesis_status ON samples (synthesis_status);
            CREATE INDEX IF NOT EXISTS idx_samples_review_status ON samples (review_status);
            CREATE INDEX IF NOT EXISTS idx_samples_domain ON samples (domain);
            """
        )
        self.connection.commit()
        self.connection.execute(
            """
            UPDATE samples
            SET synthesis_status = 'pending', updated_at = ?
            WHERE synthesis_status = 'running'
            """,
            (utc_now_iso(),),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def reserve_counter_values(self, name: str, count: int) -> list[int]:
        current = self.connection.execute(
            "SELECT value FROM counters WHERE name = ?",
            (name,),
        ).fetchone()
        start = int(current["value"]) if current else 0
        end = start + count
        self.connection.execute(
            """
            INSERT INTO counters (name, value) VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET value = excluded.value
            """,
            (name, end),
        )
        self.connection.commit()
        return list(range(start + 1, end + 1))

    def insert_generated_samples(self, rows: list[dict[str, Any]]) -> int:
        inserted = 0
        for row in rows:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO samples (
                    id, prompt_text, tts_text, normalized_text, approved_text, domain,
                    intent, template_id, speaker_id, text_features_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["prompt_text"],
                    row["tts_text"],
                    row["normalized_text"],
                    row["approved_text"],
                    row["domain"],
                    row["intent"],
                    row["template_id"],
                    row["speaker_id"],
                    json_dumps(row["text_features"]),
                    row["created_at"],
                    row["updated_at"],
                ),
            )
            inserted += cursor.rowcount
        self.connection.commit()
        return inserted

    def fetch_all_samples(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM samples ORDER BY id").fetchall()
        return [self._row_to_dict(row) for row in rows]

    def fetch_existing_normalized_texts(self) -> set[str]:
        rows = self.connection.execute("SELECT normalized_text FROM samples").fetchall()
        return {row["normalized_text"] for row in rows}

    def count_samples(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM samples").fetchone()
        return int(row["count"])

    def count_ready_for_review(self) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM samples
            WHERE synthesis_status = 'complete'
            """
        ).fetchone()
        return int(row["count"])

    def get_status_counts(self) -> dict[str, int]:
        counts = {
            "total": self.count_samples(),
            "ready_for_review": self.count_ready_for_review(),
        }
        for key in ("pending", "running", "complete", "failed"):
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM samples WHERE synthesis_status = ?",
                (key,),
            ).fetchone()
            counts[f"synthesis_{key}"] = int(row["count"])
        for key in ("pending", "approved", "rejected", "needs_fix"):
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM samples WHERE review_status = ?",
                (key,),
            ).fetchone()
            counts[f"review_{key}"] = int(row["count"])
        return counts

    def get_synthesis_queue(self, limit: int, max_attempts: int, overwrite_audio: bool) -> list[dict[str, Any]]:
        if overwrite_audio:
            query = """
                SELECT *
                FROM samples
                WHERE attempt_count < ?
                ORDER BY id
                LIMIT ?
            """
            params = (max_attempts, limit)
        else:
            query = """
                SELECT *
                FROM samples
                WHERE synthesis_status IN ('pending', 'failed')
                  AND attempt_count < ?
                ORDER BY id
                LIMIT ?
            """
            params = (max_attempts, limit)
        rows = self.connection.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def mark_synthesis_running(self, sample_id: str, speaker_id: str) -> None:
        self.connection.execute(
            """
            UPDATE samples
            SET synthesis_status = 'running',
                speaker_id = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (speaker_id, utc_now_iso(), sample_id),
        )
        self.connection.commit()

    def mark_synthesis_success(
        self,
        sample_id: str,
        speaker_id: str,
        audio_path: str,
        metrics: dict[str, Any],
        model_name: str,
        language: str,
    ) -> None:
        self.connection.execute(
            """
            UPDATE samples
            SET synthesis_status = 'complete',
                speaker_id = ?,
                audio_path = ?,
                duration_sec = ?,
                sample_rate = ?,
                quality_score = ?,
                quality_flags_json = ?,
                tts_model = ?,
                tts_language = ?,
                error_message = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (
                speaker_id,
                audio_path,
                metrics["duration_sec"],
                metrics["sample_rate"],
                metrics["quality_score"],
                json_dumps(metrics["quality_flags"]),
                model_name,
                language,
                utc_now_iso(),
                sample_id,
            ),
        )
        self.connection.commit()

    def mark_synthesis_failure(self, sample_id: str, speaker_id: str, error_message: str) -> None:
        self.connection.execute(
            """
            UPDATE samples
            SET synthesis_status = 'failed',
                speaker_id = ?,
                attempt_count = attempt_count + 1,
                error_message = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (speaker_id, error_message[:2000], utc_now_iso(), sample_id),
        )
        self.connection.commit()

    def list_samples(
        self,
        limit: int,
        offset: int,
        review_status: str | None = None,
        only_flagged: bool = False,
    ) -> list[dict[str, Any]]:
        filters = ["synthesis_status = 'complete'"]
        params: list[Any] = []
        if review_status:
            filters.append("review_status = ?")
            params.append(review_status)
        if only_flagged:
            filters.append("quality_flags_json != '[]'")
        where_clause = " AND ".join(filters)
        query = f"""
            SELECT *
            FROM samples
            WHERE {where_clause}
            ORDER BY id
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        rows = self.connection.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_review(
        self,
        sample_id: str,
        review_status: str,
        review_notes: str | None,
        approved_text: str | None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE samples
            SET review_status = ?,
                review_notes = ?,
                approved_text = ?,
                reviewed_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                review_status,
                review_notes,
                approved_text,
                utc_now_iso(),
                utc_now_iso(),
                sample_id,
            ),
        )
        self.connection.commit()

    def fetch_export_rows(self, include_pending_review: bool = False) -> list[dict[str, Any]]:
        if include_pending_review:
            query = """
                SELECT *
                FROM samples
                WHERE synthesis_status = 'complete'
                  AND review_status IN ('pending', 'approved')
                ORDER BY id
            """
            rows = self.connection.execute(query).fetchall()
        else:
            query = """
                SELECT *
                FROM samples
                WHERE synthesis_status = 'complete'
                  AND review_status = 'approved'
                ORDER BY id
            """
            rows = self.connection.execute(query).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["text_features"] = json.loads(data.pop("text_features_json"))
        data["quality_flags"] = json.loads(data.pop("quality_flags_json"))
        return data
