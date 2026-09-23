"""Worker jobs for the causal fast profile and the postflight deep profile."""

import time

from selfsuvis.pipeline.analysis4d.profile import default_prompts
from selfsuvis.pipeline.storage.missions import fetch_mission, list_mission_frames
from selfsuvis.pipeline.workflows.analysis4d_profile import (
    frames_from_rows,
    run_deep_profile,
    run_fast_profile,
)
from selfsuvis.worker._run import _run, _update_job_sync
from selfsuvis.worker.handlers.postflight import (
    _finalize_postflight_job_error,
    _finalize_postflight_job_success,
)


def handle_analysis4d_fast_job(job_id: str, payload: dict, pool, logger) -> None:
    """Run the causal fast profile for one indexed mission."""
    _handle(job_id, payload, pool, logger, kind="fast")


def handle_analysis4d_deep_job(job_id: str, payload: dict, pool, logger) -> None:
    """Revise the fast profile. Fast events stay and the deep manifest supersedes them."""
    _handle(job_id, payload, pool, logger, kind="deep")


def _handle(job_id: str, payload: dict, pool, logger, *, kind: str) -> None:
    mission_id = payload.get("mission_id") or payload.get("video_id")
    label = "analysis4d_fast" if kind == "fast" else "postflight_analysis4d_deep"
    if not mission_id:
        _update_job_sync(
            pool,
            job_id,
            status="error",
            error=f"{label} requires mission_id",
            finished_at=time.time(),
        )
        return
    try:

        async def _load():
            async with pool.acquire() as conn:
                mission = await fetch_mission(conn, mission_id)
                rows = await list_mission_frames(conn, mission_id)
                return mission, rows

        mission, rows = _run(_load())
        if mission is None:
            raise LookupError(f"mission not found: {mission_id}")
        frames = frames_from_rows(rows)
        if not frames:
            raise RuntimeError(f"mission {mission_id} has no indexed frames")
        prompts = default_prompts()
        if kind == "fast":
            result = run_fast_profile(mission_id, frames, prompts)
        else:
            result = run_deep_profile(mission_id, frames, prompts)

        async def _finish():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _finalize_postflight_job_success(
                        conn,
                        job_id=job_id,
                        mission_id=mission_id,
                        payload=payload,
                        progress=_progress(mission_id, result),
                        logger=logger,
                    )

        _run(_finish())
        logger.info("%s finished id=%s mission=%s", label, job_id, mission_id)
    except Exception as exc:
        logger.exception("%s failed id=%s error=%s", label, job_id, exc)
        error_message = str(exc)

        async def _mark_error():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _finalize_postflight_job_error(
                        conn,
                        job_id=job_id,
                        mission_id=mission_id,
                        error=error_message,
                    )

        _run(_mark_error())


def _progress(mission_id: str, result) -> dict:
    return {
        "mission_id": mission_id,
        "analysis4d": {
            "profile": result.profile,
            "queue_depth": result.queue_depth,
            "queue_capacity": result.queue_capacity,
            "degradations": list(result.degradations),
            "backlog": list(result.backlog),
            "published_events": len(result.published_event_ids),
            "real_time_factor": result.real_time_factor,
            "replayed": result.replayed,
            "superseded": result.superseded,
        },
    }
