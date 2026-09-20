"""CVAT annotation integration.

Two routers:

webhook_router  (prefix="/webhook")
  POST /webhook/cvat  — receives CVAT webhook events; no API key required,
                        verified via HMAC-SHA256 X-Hook-Secret header.

cvat_admin_router  (prefix="/admin/cvat", API key required)
  GET  /admin/cvat/frames — list frames pending annotation.
  POST /admin/cvat/task   — register a CVAT task → frame mapping.

Annotation workflow:
  1. GET /admin/cvat/frames?al_tag=needs_annotation  → list of frame paths
  2. Create a CVAT task with those images (manually or via API).
  3. POST /admin/cvat/task  {cvat_task_id, frame_ids}  → store mapping
  4. Annotate frames in CVAT, mark job as completed.
  5. CVAT fires POST /webhook/cvat  → frames updated to al_tag='annotated'.
"""

import json
import os
import tempfile
from typing import Any

import asyncpg
import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from selfsuvis.app.db import get_db_pool
from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.pipeline.core import get_logger, settings
from ss_kit.security import verify_hmac_sha256

logger = get_logger(__name__)


# -- helpers -------------------------------------------------------------------


def _verify_cvat_signature(body: bytes, signature: str) -> bool:
    """Verify HMAC-SHA256 from CVAT's X-Hook-Secret header.

    Fail-closed: returns False (reject) when CVAT_WEBHOOK_SECRET is not configured.
    Set CVAT_WEBHOOK_SECRET to the secret you configured in CVAT's webhook settings.
    """
    secret = settings.CVAT_WEBHOOK_SECRET
    if not secret:
        return False
    return verify_hmac_sha256(secret, body, signature)


async def _frames_for_cvat_task(cvat_task_id: int, pool: asyncpg.Pool) -> list[str]:
    """Return selfsuvis frame_ids registered for the given CVAT task."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT frame_id FROM cvat_tasks WHERE cvat_task_id = $1",
            cvat_task_id,
        )
        return [r["frame_id"] for r in rows]


async def _mark_frames_annotated(
    frame_ids: list[str],
    pool: asyncpg.Pool,
    basename_to_label: dict[str, str] | None = None,
) -> int:
    """Set al_tag='annotated' on the given frames. Returns count of rows updated.

    If basename_to_label is provided, also writes cvat_label by matching
    basename(frame_path) against the dict keys.
    """
    if not frame_ids:
        return 0
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE frames SET al_tag = 'annotated' "
            "WHERE id = ANY($1::text[]) AND al_tag != 'annotated'",
            frame_ids,
        )
        count = int(result.split()[-1])

        if basename_to_label:
            rows = await conn.fetch(
                "SELECT id, frame_path FROM frames WHERE id = ANY($1::text[])",
                frame_ids,
            )
            updates = [
                (basename_to_label[os.path.basename(r["frame_path"])], r["id"])
                for r in rows
                if os.path.basename(r["frame_path"]) in basename_to_label
            ]
            if updates:
                await conn.executemany(
                    "UPDATE frames SET cvat_label = $1 WHERE id = $2",
                    updates,
                )
                logger.debug(
                    "_mark_frames_annotated: stored cvat_label for %d frames", len(updates)
                )

        return count


async def _fetch_cvat_labels(task_id: int) -> dict[str, str]:
    """Fetch per-frame labels from CVAT for a completed task.

    Returns {basename: majority_vote_label}.
    Returns {} (with WARNING) when CVAT_API_TOKEN is empty or the request fails.
    """
    from selfsuvis.pipeline.labeling.cvat import CvatAnnotationParser

    token = settings.CVAT_API_TOKEN
    if not token:
        logger.warning(
            "_fetch_cvat_labels: CVAT_API_TOKEN not set — skipping label fetch for task_id=%s",
            task_id,
        )
        return {}

    url = f"{settings.CVAT_URL}/api/tasks/{task_id}/annotations?format=CVAT+1.1"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers={"Authorization": f"Token {token}"})
    except Exception as exc:
        logger.warning("_fetch_cvat_labels: HTTP request failed task_id=%s err=%s", task_id, exc)
        return {}

    if resp.status_code == 401:
        logger.warning(
            "_fetch_cvat_labels: CVAT returned 401 Unauthorized for task_id=%s — check CVAT_API_TOKEN",
            task_id,
        )
        return {}

    if resp.status_code != 200:
        logger.warning(
            "_fetch_cvat_labels: unexpected status %d for task_id=%s",
            resp.status_code,
            task_id,
        )
        return {}

    try:
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="wb", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        parser = CvatAnnotationParser(tmp_path)
        return dict(parser.frame_labels)
    except Exception as exc:
        logger.warning(
            "_fetch_cvat_labels: failed to parse CVAT XML for task_id=%s err=%s", task_id, exc
        )
        return {}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


async def _maybe_trigger_finetune(pool: asyncpg.Pool) -> None:
    """Enqueue a supervised_finetune job if conditions are met.

    Guards:
    1. SUP_AUTO_TRIGGER must be true.
    2. Total annotated frames >= MIN_ANNOTATED_FRAMES.
    3. New annotated since last retrain >= MIN_NEW_ANNOTATED_SINCE_RETRAIN.
    4. No supervised_finetune job is already queued or running (dedup).

    DB errors are caught and logged so the webhook always returns 200.
    """
    import uuid

    from selfsuvis.pipeline.core.config import settings
    from selfsuvis.pipeline.storage.jobs import create_job

    if not settings.SUP_AUTO_TRIGGER:
        return

    try:
        # Acquire in-process lock to prevent duplicate enqueue from concurrent webhooks
        from selfsuvis.app.state import _finetune_lock

        if _finetune_lock.locked():
            logger.debug("_maybe_trigger_finetune: finetune lock held, skipping")
            return

        async with _finetune_lock:
            async with pool.acquire() as conn:
                # Count total annotated frames
                total_annotated = await conn.fetchval(
                    "SELECT COUNT(*) FROM frames WHERE al_tag = 'annotated'"
                )
                if total_annotated < settings.MIN_ANNOTATED_FRAMES:
                    logger.debug(
                        "_maybe_trigger_finetune: %d annotated < threshold %d",
                        total_annotated,
                        settings.MIN_ANNOTATED_FRAMES,
                    )
                    return

                # Check retrain watermark
                watermark_row = await conn.fetchrow(
                    "SELECT value FROM system_state WHERE key = 'last_retrain_watermark'"
                )
                last_watermark = int(watermark_row["value"]) if watermark_row else 0
                delta = total_annotated - last_watermark
                if delta < settings.MIN_NEW_ANNOTATED_SINCE_RETRAIN:
                    logger.debug(
                        "_maybe_trigger_finetune: delta=%d < min_new=%d (watermark=%d)",
                        delta,
                        settings.MIN_NEW_ANNOTATED_SINCE_RETRAIN,
                        last_watermark,
                    )
                    return

                # Dedup: skip if a finetune job is already queued or running
                existing = await conn.fetchrow(
                    "SELECT id FROM jobs WHERE type = 'supervised_finetune' "
                    "AND status IN ('pending', 'running') LIMIT 1"
                )
                if existing:
                    logger.debug(
                        "_maybe_trigger_finetune: finetune job already active id=%s",
                        existing["id"],
                    )
                    return

                # All guards passed — enqueue
                job_id = uuid.uuid4().hex
                await create_job(conn, job_id, {}, job_type="supervised_finetune")
                logger.info(
                    "_maybe_trigger_finetune: enqueued job_id=%s "
                    "(total_annotated=%d watermark=%d delta=%d)",
                    job_id,
                    total_annotated,
                    last_watermark,
                    delta,
                )

    except Exception as exc:
        logger.warning("_maybe_trigger_finetune failed (non-fatal): %s", exc)


# -- Webhook router (no API key — CVAT uses HMAC secret) ----------------------

webhook_router = APIRouter(prefix="/webhook", tags=["webhook"])


@webhook_router.post("/cvat", summary="Receive CVAT annotation webhook")
async def cvat_webhook(
    request: Request,
    x_hook_secret: str = Header(default=""),
) -> dict[str, Any]:
    """Handle CVAT webhook events and update frame annotation status.

    Triggered by CVAT when a job or task reaches state=completed.
    Marks all frames mapped to that task as al_tag='annotated'.

    Security: verifies HMAC-SHA256 signature in X-Hook-Secret.
    CVAT_WEBHOOK_SECRET must be set; requests with a missing or incorrect
    signature are rejected with 400. Configure the same secret in CVAT:
    Webhooks → Secret → <your CVAT_WEBHOOK_SECRET value>.

    Configure target URL in CVAT: http://<api_host>:8000/webhook/cvat
    """
    body = await request.body()

    if not _verify_cvat_signature(body, x_hook_secret):
        logger.warning("CVAT webhook: invalid HMAC signature from %s", request.client)
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event = payload.get("event", "")
    logger.info("CVAT webhook received: event=%s", event)

    if event in ("update:job", "update:task"):
        obj = payload.get("job") or payload.get("task") or {}
        state = obj.get("state", "")
        if state == "completed":
            # For update:job, task_id is in obj["task_id"].
            # For update:task, the task itself is the obj, so use obj["id"].
            task_id = obj.get("task_id") if event == "update:job" else obj.get("id")
            if task_id is not None:
                pool = get_db_pool(request)
                frame_ids = await _frames_for_cvat_task(int(task_id), pool)
                if frame_ids:
                    basename_to_label = await _fetch_cvat_labels(int(task_id))
                    count = await _mark_frames_annotated(frame_ids, pool, basename_to_label)
                    logger.info(
                        "CVAT webhook: task_id=%s completed → %d frames annotated, %d labels stored",
                        task_id,
                        count,
                        len(basename_to_label),
                    )
                    # Attempt to trigger supervised fine-tuning (non-fatal on error)
                    await _maybe_trigger_finetune(pool)
                    return {"status": "ok", "event": event, "annotated": count}
                else:
                    logger.info(
                        "CVAT webhook: task_id=%s completed but no frame mapping found. "
                        "Register mapping via POST /admin/cvat/task first.",
                        task_id,
                    )

    return {"status": "ok", "event": event, "annotated": 0}


# -- Admin CVAT router (API key required) -------------------------------------

cvat_admin_router = APIRouter(
    prefix="/admin/cvat",
    tags=["admin"],
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)


class CvatFrameItem(BaseModel):
    frame_id: str
    mission_id: str
    frame_path: str
    al_score: float | None
    al_tag: str


class CvatFrameListResponse(BaseModel):
    total: int
    frames: list[CvatFrameItem]


class CvatTaskRegistration(BaseModel):
    cvat_task_id: int
    frame_ids: list[str]


@cvat_admin_router.get(
    "/frames",
    response_model=CvatFrameListResponse,
    summary="List frames pending annotation (for building a CVAT task)",
)
async def cvat_annotation_frames(
    request: Request,
    al_tag: str = "needs_annotation",
    limit: int = 200,
) -> CvatFrameListResponse:
    """Return frames that need annotation, ordered by al_score descending.

    al_tag: 'needs_annotation' | 'novel' | 'any' (both).
    limit: max rows to return (1-5000, default 200).

    Use the returned frame_path list to create a CVAT task, then call
    POST /admin/cvat/task to record the mapping for webhook resolution.
    """
    if al_tag not in ("needs_annotation", "novel", "any"):
        raise HTTPException(
            status_code=422,
            detail="al_tag must be 'needs_annotation', 'novel', or 'any'",
        )
    if not (1 <= limit <= 5000):
        raise HTTPException(status_code=422, detail="limit must be 1–5000")

    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        if al_tag == "any":
            tag_clause = "al_tag IN ('needs_annotation', 'novel')"
            rows = await conn.fetch(
                f"SELECT id, mission_id, frame_path, al_score, al_tag "
                f"FROM frames WHERE {tag_clause} "
                f"ORDER BY al_score DESC NULLS LAST LIMIT $1",
                limit,
            )
            total = await conn.fetchval(f"SELECT COUNT(*) FROM frames WHERE {tag_clause}")
        else:
            rows = await conn.fetch(
                "SELECT id, mission_id, frame_path, al_score, al_tag "
                "FROM frames WHERE al_tag = $1 "
                "ORDER BY al_score DESC NULLS LAST LIMIT $2",
                al_tag,
                limit,
            )
            total = await conn.fetchval("SELECT COUNT(*) FROM frames WHERE al_tag = $1", al_tag)

    frames = [
        CvatFrameItem(
            frame_id=r["id"],
            mission_id=r["mission_id"],
            frame_path=r["frame_path"],
            al_score=r["al_score"],
            al_tag=r["al_tag"],
        )
        for r in rows
    ]
    return CvatFrameListResponse(total=total or 0, frames=frames)


@cvat_admin_router.post(
    "/task",
    summary="Register a CVAT task → frame mapping (enables webhook resolution)",
)
async def register_cvat_task(body: CvatTaskRegistration, request: Request) -> dict[str, Any]:
    """Store the mapping from a CVAT task ID to selfsuvis frame IDs.

    Call this after creating a CVAT task with frames from GET /admin/cvat/frames.
    When CVAT fires a webhook for job completion, this mapping is used to find
    and annotate the corresponding selfsuvis frames.

    Idempotent — duplicate (task_id, frame_id) pairs are silently ignored.
    """
    if not body.frame_ids:
        raise HTTPException(status_code=422, detail="frame_ids must not be empty")
    if len(body.frame_ids) > 5000:
        raise HTTPException(status_code=422, detail="frame_ids must not exceed 5000")

    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO cvat_tasks (cvat_task_id, frame_id) VALUES ($1, $2) "
            "ON CONFLICT DO NOTHING",
            [(body.cvat_task_id, fid) for fid in body.frame_ids],
        )

    logger.info(
        "CVAT task registered: cvat_task_id=%d frame_count=%d",
        body.cvat_task_id,
        len(body.frame_ids),
    )
    return {
        "status": "ok",
        "cvat_task_id": body.cvat_task_id,
        "frame_count": len(body.frame_ids),
    }
