from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from .db import PipelineDB
from .text_normalization import normalize_egyptian_text
from .utils import PipelineContext


class ReviewUpdate(BaseModel):
    review_status: str
    review_notes: str | None = None
    approved_text: str | None = None


def create_review_app(context: PipelineContext) -> FastAPI:
    templates = Jinja2Templates(directory=str(context.paths.template_dir))
    app = FastAPI(title="Egyptian Arabic TTS Review")
    app.mount("/run-assets", StaticFiles(directory=str(context.paths.run_dir)), name="run-assets")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "review.html",
            {
                "request": request,
                "page_size": context.config.review.page_size,
                "run_name": context.config.run.name,
            },
        )

    @app.get("/api/stats")
    async def stats() -> dict[str, int]:
        db = PipelineDB(context.paths.db_path)
        try:
            return db.get_status_counts()
        finally:
            db.close()

    @app.get("/api/samples")
    async def samples(
        limit: int = 12,
        offset: int = 0,
        review_status: str | None = "pending",
        only_flagged: bool = False,
    ) -> dict[str, object]:
        db = PipelineDB(context.paths.db_path)
        try:
            rows = db.list_samples(limit=limit, offset=offset, review_status=review_status, only_flagged=only_flagged)
        finally:
            db.close()

        for row in rows:
            row["audio_url"] = f"/run-assets/{row['audio_path'].replace('\\', '/')}"
        return {"items": rows, "count": len(rows)}

    @app.post("/api/review/{sample_id}")
    async def update_review(sample_id: str, payload: ReviewUpdate) -> dict[str, object]:
        if payload.review_status not in {"pending", "approved", "rejected", "needs_fix"}:
            raise HTTPException(status_code=400, detail="Unsupported review status.")
        approved_text = payload.approved_text
        if approved_text:
            approved_text = normalize_egyptian_text(approved_text, context.config.normalization)
        db = PipelineDB(context.paths.db_path)
        try:
            db.update_review(
                sample_id=sample_id,
                review_status=payload.review_status,
                review_notes=payload.review_notes,
                approved_text=approved_text,
            )
        finally:
            db.close()
        return {"ok": True, "sample_id": sample_id}

    return app
