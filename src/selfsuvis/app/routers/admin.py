"""Admin statistics and mission listing endpoints.

GET /admin/stats              — queue depth, job status counts, al_tag distribution.
GET /admin/missions           — list missions with map_status and splat_path(s).
GET /admin/robots             — list distinct robot_ids contributing to the map.
GET /admin/global-maps        — list all global map sites (one entry per geographic site).
GET /admin/export/map-cache   — download NPZ cache of all indexed frames for onboard use.
GET /admin/automation-roi     — annotation frequency, finetune trigger rate, ops time saved.
                                Includes caption_null_rate as a captioner health indicator.
GET /admin/caption-eval       — captioner health: null rate, confidence stats, model breakdown.
POST /admin/reload-model      — hot-swap DINOv3 backbone weights from a checkpoint file.
POST /admin/reembed-all       — enqueue a full re-embedding sweep (all frames → new dino vectors).

al_tag distribution comes from the PostgreSQL frames table.
"""

import glob
import os
import uuid
from pathlib import Path
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from selfsuvis.app.db import get_db_pool
from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.pipeline.core import settings

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)


async def _job_counts(request: Request) -> dict[str, int]:
    """Return counts of jobs by status from PostgreSQL."""
    try:
        pool = get_db_pool(request)
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT status, COUNT(*) AS cnt FROM jobs GROUP BY status")
            counts = {row["status"]: row["cnt"] for row in rows}
    except Exception:
        counts = {}
    return {
        "pending": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "finished": counts.get("finished", 0),
        "error": counts.get("error", 0),
    }


async def _al_tag_counts(request: Request) -> dict[str, int]:
    """Return al_tag distribution from the PostgreSQL frames table.

    Returns zeros until the PostgreSQL migration has been applied and frames are indexed.
    """
    try:
        pool = get_db_pool(request)
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT al_tag, COUNT(*) AS cnt FROM frames GROUP BY al_tag")
            counts = {row["al_tag"]: row["cnt"] for row in rows}
    except Exception:
        counts = {}
    return {
        "needs_annotation": counts.get("needs_annotation", 0),
        "novel": counts.get("novel", 0),
        "none": counts.get("none", 0),
    }


def _discover_splat_paths(mission_id: str) -> list[str]:
    """Return all splat.ply paths for a mission, sorted by scene index.

    Checks for:
      - maps/{mission_id}/scene-*/splat.ply  (multi-scene, chunked)
      - maps/{mission_id}/splat.ply          (single-scene, legacy)
    """
    maps_dir = settings.MAPS_DIR
    chunked = sorted(glob.glob(os.path.join(maps_dir, mission_id, "scene-*", "splat.ply")))
    if chunked:
        return chunked
    single = os.path.join(maps_dir, mission_id, "splat.ply")
    if os.path.isfile(single):
        return [single]
    return []


async def _list_missions(request: Request) -> list[dict[str, Any]]:
    """Return missions from PostgreSQL with splat path discovery."""
    missions = []
    try:
        pool = get_db_pool(request)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, video_id, status, map_status, frame_count, created_at "
                "FROM missions ORDER BY created_at DESC LIMIT 100"
            )
            missions = [dict(r) for r in rows]
    except Exception:
        missions = []

    # Attach splat paths discovered from the filesystem
    for m in missions:
        paths = _discover_splat_paths(m["id"])
        m["splat_paths"] = paths
        m["splat_path"] = paths[0] if paths else None
        m["scene_count"] = len(paths)
    return missions


@router.get("/missions", summary="List missions with map status and splat paths")
async def admin_missions(request: Request) -> list[dict[str, Any]]:
    """Return up to 100 most recent missions with map status and discovered splat.ply paths."""
    return await _list_missions(request)


@router.get("/robots", summary="List distinct robot_ids contributing to the map")
async def admin_robots(request: Request) -> list[str]:
    """Return all distinct robot_ids from the missions table.

    Returns an empty list if the robot_id column has not been added yet
    (migration not yet applied) or if DATABASE_URL is not configured.
    """
    try:
        pool = get_db_pool(request)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT robot_id FROM missions "
                "WHERE robot_id IS NOT NULL ORDER BY robot_id"
            )
            return [row["robot_id"] for row in rows]
    except Exception:
        return []


@router.get("/global-maps", summary="List all global map sites")
async def admin_global_maps(request: Request) -> list[dict[str, Any]]:
    """Return all global_map rows — one entry per geographic site.

    Each entry has: id, origin_lat, origin_lon, origin_alt, splat_path, created_at.
    Returns an empty list if DATABASE_URL is not configured or the table does not exist.
    """
    try:
        from selfsuvis.pipeline.storage.global_maps import list_global_maps

        pool = get_db_pool(request)
        async with pool.acquire() as conn:
            return await list_global_maps(conn)
    except Exception:
        return []


@router.get(
    "/export/map-cache",
    summary="Download NPZ map cache for onboard robot use",
    response_class=Response,
)
def export_map_cache(
    mission_ids: str | None = Query(
        default=None,
        description="Comma-separated mission IDs to include (all if omitted)",
    ),
    lat_min: float | None = Query(default=None, description="GPS bounding box — min latitude"),
    lat_max: float | None = Query(default=None, description="GPS bounding box — max latitude"),
    lon_min: float | None = Query(default=None, description="GPS bounding box — min longitude"),
    lon_max: float | None = Query(default=None, description="GPS bounding box — max longitude"),
) -> Response:
    """Export all indexed frames as a compressed NPZ cache file.

    The robot loads this file pre-flight for local nearest-neighbour search
    without a network round-trip.  NPZ layout:

        clip_vectors : float32 (N, D)   CLIP embeddings
        gps          : float32 (N, 3)   [lat, lon, alt], NaN if unavailable
        enu          : float32 (N, 3)   [tx, ty, tz] metres, NaN if unavailable
        t_sec        : float32 (N,)     frame timestamp in source video
        meta_json    : uint8  (M,)      JSON bytes → list of N {mission_id, frame_path, robot_id}

    Optional query params to narrow the export:
        mission_ids  — comma-separated list of mission IDs
        lat_min / lat_max / lon_min / lon_max — GPS bbox (all four required for bbox filter)
    """
    from selfsuvis.app.state import qdrant_store
    from selfsuvis.pipeline.storage.map_cache import build_map_cache

    parsed_mission_ids = (
        [m.strip() for m in mission_ids.split(",") if m.strip()] if mission_ids else None
    )

    npz_bytes = build_map_cache(
        qdrant_store,
        mission_ids=parsed_mission_ids,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    return Response(
        content=npz_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="map_cache.npz"'},
    )


@router.get("/stats", summary="Queue depth, worker status, and al_tag distribution")
async def admin_stats(request: Request) -> dict[str, Any]:
    """Return operational statistics for the admin dashboard.

    Fields:
        jobs:      {pending, running, finished, error} counts from the job queue.
        al_tags:   {needs_annotation, novel, none} frame counts (PostgreSQL frames table).
        worker_active: true if any jobs are currently in 'running' state.
    """
    jobs = await _job_counts(request)
    al_tags = await _al_tag_counts(request)
    return {
        "jobs": jobs,
        "al_tags": al_tags,
        "worker_active": jobs["running"] > 0,
    }


# -- Automation ROI endpoint --------------------------------------------------
#
# Evidence-based answer to: "is the auto-trigger pipeline worth maintaining?"
#
# Derived entirely from existing jobs + frames tables — no schema changes.
# Key question: annotation frequency per week.
#   < 0.5  → LOW_FREQUENCY   — manual restart may be simpler
#   0.5–2  → MODERATE        — automation pays off within months
#   > 2    → HIGH_FREQUENCY  — automation clearly valuable
#
# Ops time saved = accepted_finetune_count × MANUAL_RESTART_MIN (default 3 min).
# This is the lower bound: excludes time saved by not monitoring for threshold
# crossings and not manually kicking off reembed sweeps.

_MANUAL_RESTART_MINUTES = 3  # estimated time per manual `docker restart api`
_LOW_FREQUENCY_THRESHOLD = 0.5  # annotations/week below which automation may not be worth it
_HIGH_FREQUENCY_THRESHOLD = 2.0  # annotations/week above which automation is clearly valuable
_MIN_OBSERVATION_DAYS = 7  # minimum days before frequency verdict is meaningful


async def _compute_caption_null_rate(conn) -> float | None:
    """Fraction of captionable frames with no caption.

    Excludes frames where caption_skip_reason IS NOT NULL (intentionally skipped).
    Returns None if no captionable frames exist.
    """
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE caption IS NULL AND caption_skip_reason IS NULL)
                AS null_count,
            COUNT(*) FILTER (WHERE caption_skip_reason IS NULL)
                AS captionable_total
        FROM frames
        """
    )
    if row is None or row["captionable_total"] == 0:
        return None
    return row["null_count"] / row["captionable_total"]


async def _fetch_roi_raw_data(conn) -> dict[str, Any]:
    """Collect all raw ROI metrics from the DB in a single connection."""
    import json as _json

    total_annotated = (
        await conn.fetchval("SELECT COUNT(*) FROM frames WHERE al_tag = 'annotated'") or 0
    )
    timeline = await conn.fetchrow(
        "SELECT MIN(created_at) AS first_at, MAX(created_at) AS last_at "
        "FROM frames WHERE al_tag = 'annotated'"
    )
    campaigns = (
        await conn.fetchval(
            "SELECT COUNT(DISTINCT TO_CHAR(created_at, 'YYYY-MM')) "
            "FROM frames WHERE al_tag = 'annotated'"
        )
        or 0
    )
    ft_rows = await conn.fetch(
        "SELECT status, progress_json FROM jobs WHERE type = 'supervised_finetune'"
    )
    ft_triggered = len(ft_rows)
    ft_accepted = sum(
        1
        for r in ft_rows
        if (
            (
                r["progress_json"]
                if isinstance(r["progress_json"], dict)
                else _json.loads(r["progress_json"] or "{}")
            ).get("accepted")
            is True
        )
    )
    reembed_done = (
        await conn.fetchval(
            "SELECT COUNT(*) FROM jobs WHERE type = 'reembed' AND status = 'finished'"
        )
        or 0
    )
    return {
        "total_annotated": total_annotated,
        "first_at": timeline["first_at"] if timeline else None,
        "last_at": timeline["last_at"] if timeline else None,
        "campaigns": campaigns,
        "ft_triggered": ft_triggered,
        "ft_accepted": ft_accepted,
        "reembed_done": reembed_done,
    }


def _compute_roi_verdict(freq_per_week: float | None) -> tuple:
    """Return (verdict, verdict_detail) given the annotation frequency."""
    if freq_per_week is None:
        return (
            "INSUFFICIENT_DATA",
            "Less than 7 days of annotation data. Re-check after 4+ weeks of production use.",
        )
    if freq_per_week < _LOW_FREQUENCY_THRESHOLD:
        return (
            "LOW_FREQUENCY",
            f"Annotations happen ~{freq_per_week:.1f}×/week. "
            "Manual `docker restart api` after each fine-tune may be simpler to maintain "
            "than the auto-trigger pipeline.",
        )
    if freq_per_week < _HIGH_FREQUENCY_THRESHOLD:
        return (
            "MODERATE_FREQUENCY",
            f"Annotations happen ~{freq_per_week:.1f}×/week. "
            "Automation is paying off but the benefit is modest. "
            "Keep the pipeline; revisit if annotation cadence drops.",
        )
    return (
        "HIGH_FREQUENCY",
        f"Annotations happen ~{freq_per_week:.1f}×/week. "
        "Automation is clearly valuable — the compounding improvement loop is active.",
    )


@router.get(
    "/automation-roi", summary="Annotation frequency, finetune trigger rate, ops time saved"
)
async def automation_roi() -> dict[str, Any]:
    """Measure whether the auto-trigger pipeline is paying its maintenance cost.

    All metrics are derived from the jobs and frames tables — no extra
    instrumentation required.  Call this after 4+ weeks of production use.

    Returns:
        total_annotated_frames:       frames with al_tag='annotated' (cumulative)
        annotation_campaigns:         distinct months with at least one annotation
        finetune_jobs_triggered:      supervised_finetune jobs ever enqueued
        finetune_jobs_accepted:       checkpoints that passed the eval gate
        finetune_acceptance_rate:     accepted / triggered (null if no jobs)
        model_reloads:                accepted jobs (each triggers a hot-reload)
        reembed_sweeps:               reembed jobs ever completed
        estimated_ops_minutes_saved:  model_reloads × 3 min manual restart
        first_annotation_at:          ISO timestamp of first annotation (null if none)
        last_annotation_at:           ISO timestamp of most recent annotation (null)
        days_observed:                calendar days since first annotation (null if none)
        annotation_frequency_per_week: annotations per 7-day window (null if <7 days)
        verdict:                      LOW_FREQUENCY | MODERATE_FREQUENCY |
                                      HIGH_FREQUENCY | INSUFFICIENT_DATA
        verdict_detail:               plain-English interpretation
    """
    if not settings.DATABASE_URL:
        return {"error": "DATABASE_URL not configured"}
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            d = await _fetch_roi_raw_data(conn)
            caption_null_rate = await _compute_caption_null_rate(conn)
        finally:
            await conn.close()
    except Exception as exc:
        return {"error": f"DB query failed: {exc}"}

    first_at, last_at = d["first_at"], d["last_at"]
    days_observed: float | None = None
    freq_per_week: float | None = None
    if first_at is not None and last_at is not None:
        days_observed = max((last_at - first_at) / 86400.0, 1.0)
        if days_observed >= _MIN_OBSERVATION_DAYS:
            freq_per_week = d["total_annotated"] / (days_observed / 7.0)

    ft_triggered, ft_accepted = d["ft_triggered"], d["ft_accepted"]
    acceptance_rate: float | None = (ft_accepted / ft_triggered) if ft_triggered else None
    verdict, verdict_detail = _compute_roi_verdict(freq_per_week)

    return {
        "total_annotated_frames": d["total_annotated"],
        "annotation_campaigns": d["campaigns"],
        "finetune_jobs_triggered": ft_triggered,
        "finetune_jobs_accepted": ft_accepted,
        "finetune_acceptance_rate": round(acceptance_rate, 3)
        if acceptance_rate is not None
        else None,
        "model_reloads": ft_accepted,
        "reembed_sweeps_completed": d["reembed_done"],
        "estimated_ops_minutes_saved": ft_accepted * _MANUAL_RESTART_MINUTES,
        "first_annotation_at": first_at,
        "last_annotation_at": last_at,
        "days_observed": round(days_observed, 1) if days_observed is not None else None,
        "annotation_frequency_per_week": round(freq_per_week, 2)
        if freq_per_week is not None
        else None,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "caption_null_rate": round(caption_null_rate, 4) if caption_null_rate is not None else None,
    }


# -- Caption eval endpoint ----------------------------------------------------


@router.get(
    "/caption-eval", summary="Captioner health: null rate, confidence stats, model breakdown"
)
async def caption_eval(request: Request) -> dict[str, Any]:
    """Captioner health metrics derived from the frames table.

    Returns:
        caption_null_rate:    Fraction of captionable frames with caption IS NULL.
                              Excludes frames with caption_skip_reason (intentional skips).
                              0.0 = all frames captioned; 1.0 = none captioned.
        mean_confidence:      Mean caption_confidence across all captioned frames (null if none).
        p95_confidence:       95th-percentile caption_confidence (null if <20 frames).
        total_frames:         Total indexed frames.
        captioned_frames:     Frames with a non-null caption.
        skipped_frames:       Frames with caption_skip_reason IS NOT NULL.
        null_caption_frames:  Frames with caption IS NULL and no skip reason.
        model_breakdown:      Dict of caption_model → frame count (top captioners).
    """
    try:
        pool = get_db_pool(request)
        async with pool.acquire() as conn:
            # Aggregate stats in one query
            agg = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)                                         AS total_frames,
                    COUNT(*) FILTER (WHERE caption IS NOT NULL)     AS captioned_frames,
                    COUNT(*) FILTER (WHERE caption_skip_reason IS NOT NULL) AS skipped_frames,
                    COUNT(*) FILTER (
                        WHERE caption IS NULL AND caption_skip_reason IS NULL
                    )                                               AS null_caption_frames,
                    AVG(caption_confidence)
                        FILTER (WHERE caption_confidence IS NOT NULL) AS mean_confidence,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (
                        ORDER BY caption_confidence
                    ) FILTER (WHERE caption_confidence IS NOT NULL)  AS p95_confidence
                FROM frames
                """
            )

            # Per-model breakdown
            model_rows = await conn.fetch(
                """
                SELECT caption_model, COUNT(*) AS cnt
                FROM frames
                WHERE caption_model IS NOT NULL
                GROUP BY caption_model
                ORDER BY cnt DESC
                LIMIT 20
                """
            )
            model_breakdown = {r["caption_model"]: r["cnt"] for r in model_rows}

            captionable_total = (agg["total_frames"] or 0) - (agg["skipped_frames"] or 0)
            null_count = agg["null_caption_frames"] or 0
            caption_null_rate = (null_count / captionable_total) if captionable_total > 0 else None

    except Exception as exc:
        return {"error": f"DB query failed: {exc}"}

    total = agg["total_frames"] or 0
    captioned = agg["captioned_frames"] or 0
    mean_conf = agg["mean_confidence"]
    p95_conf = agg["p95_confidence"]

    return {
        "caption_null_rate": round(caption_null_rate, 4) if caption_null_rate is not None else None,
        "mean_confidence": round(float(mean_conf), 4) if mean_conf is not None else None,
        "p95_confidence": round(float(p95_conf), 4) if p95_conf is not None else None,
        "total_frames": total,
        "captioned_frames": captioned,
        "skipped_frames": agg["skipped_frames"] or 0,
        "null_caption_frames": null_count,
        "model_breakdown": model_breakdown,
    }


# -- Hot-reload model endpoint ------------------------------------------------


class ReloadModelRequest(BaseModel):
    checkpoint: str | None = None  # path to checkpoint; falls back to active_checkpoint.txt


@router.post("/reload-model", summary="Hot-swap DINOv3 backbone weights")
async def reload_model(body: ReloadModelRequest = ReloadModelRequest()) -> dict[str, Any]:
    """Load a DINOv3 backbone checkpoint and swap the live model reference.

    The swap is GIL-atomic: in-flight inference calls hold their captured
    dino_model reference and complete normally with old weights. Only new
    requests after the swap see the new weights.

    - 400: checkpoint path not found on disk.
    - 400: no checkpoint specified and active_checkpoint.txt is missing.
    - 409: another reload is already in progress (lock is held).
    - 500: checkpoint loaded but weights failed to parse (original model unchanged).
    """
    from fastapi import HTTPException

    import selfsuvis.app.state as state

    if state.dino_model is None:
        raise HTTPException(
            status_code=400, detail="DINO model is not loaded (MODEL_NAME not dinov2/dinov3)"
        )

    # Resolve checkpoint path
    ckpt_path = body.checkpoint
    if not ckpt_path:
        active_txt = Path(settings.SUP_CHECKPOINT_DIR) / "active_checkpoint.txt"
        if not active_txt.exists():
            raise HTTPException(
                status_code=400,
                detail="No checkpoint specified and active_checkpoint.txt not found",
            )
        ckpt_path = active_txt.read_text().strip()

    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise HTTPException(status_code=400, detail=f"Checkpoint not found: {ckpt_path}")

    if state.dino_model_lock.locked():
        raise HTTPException(status_code=409, detail="Reload already in progress")

    async with state.dino_model_lock:
        try:
            state.dino_model.load_backbone_checkpoint(ckpt_path)
            # Write active checkpoint so the next restart picks it up
            active_txt = Path(settings.SUP_CHECKPOINT_DIR) / "active_checkpoint.txt"
            active_txt.parent.mkdir(parents=True, exist_ok=True)
            active_txt.write_text(ckpt_path)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to load checkpoint: {exc}"
            ) from exc

    return {"status": "ok", "checkpoint": ckpt_path}


# -- Re-embed all endpoint ----------------------------------------------------


@router.post("/reembed-all", summary="Enqueue full re-embedding sweep (dino vectors)")
async def reembed_all(request: Request) -> dict[str, Any]:
    """Enqueue a background job to re-embed all indexed frames with the current DINOv3 model.

    Returns the job_id immediately. Monitor progress via GET /jobs/{job_id}.
    Each batch of 256 frames checkpoints last_offset so the sweep is resumable.
    Returns 409 if a reembed job is already pending or running.
    """
    from fastapi import HTTPException

    from selfsuvis.pipeline.storage.jobs import create_job

    job_id = uuid.uuid4().hex
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM jobs WHERE type = 'reembed' "
            "AND status IN ('pending', 'running') LIMIT 1"
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Reembed job already active: {existing['id']}",
            )
        await create_job(conn, job_id, {}, job_type="reembed")

    return {"job_id": job_id}


# -- Reembed status endpoint ---------------------------------------------------


@router.get("/reembed-status", summary="Check whether a reembed sweep is active")
async def reembed_status(request: Request) -> dict[str, Any]:
    """Return whether a re-embedding sweep is currently running.

    Search uses this to suppress dino vector reranking during the sweep window,
    because Qdrant contains a mix of old-model and new-model dino vectors while
    the sweep is in progress — cosine similarity between them is meaningless.

    Returns:
        active: true if a reembed job has status='running'.
        job_id: the running job's id, or null if none.
        frames_reembedded: progress counter from the job payload, or null.
    """
    try:
        pool = get_db_pool(request)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, progress_json FROM jobs "
                "WHERE type = 'reembed' AND status = 'running' "
                "ORDER BY created_at DESC LIMIT 1"
            )
    except Exception:
        return {"active": False, "job_id": None, "frames_reembedded": None}

    if row is None:
        return {"active": False, "job_id": None, "frames_reembedded": None}

    import json as _json

    progress = row["progress_json"] or {}
    if isinstance(progress, str):
        progress = _json.loads(progress)
    return {
        "active": True,
        "job_id": row["id"],
        "frames_reembedded": progress.get("frames_reembedded"),
    }
