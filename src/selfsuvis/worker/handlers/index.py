"""INDEX job handler."""

import os
import time
import uuid

import selfsuvis.pipeline.storage.processed as processed_db_mod
from selfsuvis.pipeline.analysis4d.profile import scheduled_analysis_jobs
from selfsuvis.pipeline.core import file_sha256, settings
from selfsuvis.pipeline.media import download_url
from selfsuvis.pipeline.storage import update_job
from selfsuvis.pipeline.storage.missions import (
    mark_mission_finished,
    replace_frames,
    upsert_mission,
)
from selfsuvis.pipeline.storage.processed import get_by_hash
from selfsuvis.pipeline.workflows import VideoIndexer
from selfsuvis.worker._run import _run, _update_job_sync
from selfsuvis.worker.handlers.postflight import (
    _enqueue_postflight_jobs,
    _normalize_postflight_job_names,
    _resolve_site_origin,
)


def handle_index_job(job_id: str, payload: dict, job: dict, pool, logger) -> None:
    """Handle an INDEX job (type=None for legacy rows, type='index' for new rows)."""
    logger.info("Index job started id=%s video_id=%s", job_id, payload.get("video_id"))
    indexer = VideoIndexer(enable_tiles=payload.get("enable_tiles", True))

    def progress_cb(progress):
        _update_job_sync(pool, job_id, progress=progress)

    video_path: str | None = None
    try:
        video_id = payload["video_id"]
        video_path = payload.get("video_path")
        url = payload.get("video_url")

        if url and not video_path:
            video_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}.mp4")
            if payload.get("ingest_mode") == "rtsp":
                from selfsuvis.pipeline.media.rtsp_ingest import record_rtsp

                record_rtsp(url, video_path, duration_sec=payload.get("duration_sec"))
            else:
                download_url(url, video_path)

        if not video_path or not os.path.exists(video_path):
            raise RuntimeError("video_path not found")

        size_bytes = os.path.getsize(video_path)
        mtime = os.path.getmtime(video_path)
        file_hash = file_sha256(video_path)
        existing = get_by_hash(file_hash)
        if existing and existing.get("status") == "processed":
            if url and video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError as e:
                    logger.warning(
                        "Could not remove duplicate video file path=%s err=%s",
                        video_path,
                        e,
                    )
            logger.info(
                "Skipping duplicate video_id=%s hash=%s",
                payload.get("video_id"),
                file_hash,
            )
            _update_job_sync(
                pool,
                job_id,
                status="finished",
                progress={
                    "skipped": True,
                    "reason": "duplicate",
                    "video_id": existing.get("video_id"),
                },
                finished_at=time.time(),
            )
            return

        mission_id = payload.get("mission_id") or video_id
        global_map_id, site_enu_origin = _resolve_site_origin(video_path, logger)
        result = indexer.index_video(
            video_path,
            video_id,
            mission_id=mission_id,
            robot_id=settings.ROBOT_ID,
            site_enu_origin=site_enu_origin,
            global_map_id=global_map_id,
            progress_cb=progress_cb,
        )
        result_summary = {k: v for k, v in result.items() if k != "frame_records"}

        async def _persist_index_result():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await upsert_mission(
                        conn,
                        mission_id=mission_id,
                        video_id=video_id,
                        video_path=video_path,
                        job_id=job_id,
                        robot_id=settings.ROBOT_ID,
                        status="indexing",
                        frame_count=result_summary.get("frames", 0),
                        duration_sec=result_summary.get("duration_sec"),
                        gps_origin=result_summary.get("gps_origin"),
                    )
                    await replace_frames(conn, mission_id, result.get("frame_records", []))

        _run(_persist_index_result())

        postflight_jobs = _normalize_postflight_job_names(
            [*(payload.get("postflight_jobs") or []), *scheduled_analysis_jobs(payload)]
        )
        if postflight_jobs:
            payload["postflight_jobs"] = postflight_jobs

        async def _finalize_success():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await processed_db_mod.aupsert(
                        file_hash,
                        video_id,
                        video_path,
                        size_bytes,
                        mtime,
                        "processed",
                        {"url": url},
                        conn=conn,
                    )
                    if postflight_jobs:
                        await mark_mission_finished(
                            conn,
                            mission_id,
                            status="indexing",
                            error=None,
                        )
                        await _enqueue_postflight_jobs(conn, payload, logger)
                    else:
                        await mark_mission_finished(
                            conn,
                            mission_id,
                            status="done",
                            error=None,
                        )
                    await update_job(
                        conn,
                        job_id,
                        status="finished",
                        progress={**(job.get("progress") or {}), **result_summary},
                        finished_at=time.time(),
                    )

        _run(_finalize_success())
        logger.info("Index job finished id=%s video_id=%s", job_id, video_id)

    except Exception as exc:
        logger.exception("Index job failed id=%s error=%s", job_id, exc)
        if video_path and os.path.exists(video_path):
            try:
                size_bytes = os.path.getsize(video_path)
                mtime = os.path.getmtime(video_path)
                file_hash = file_sha256(video_path)
                error_message = str(exc)

                async def _finalize_error():
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            await processed_db_mod.aupsert(
                                file_hash,
                                payload.get("video_id", uuid.uuid4().hex),
                                video_path,
                                size_bytes,
                                mtime,
                                "error",
                                {"error": error_message},
                                conn=conn,
                            )
                            if payload.get("video_id"):
                                await mark_mission_finished(
                                    conn,
                                    payload.get("mission_id") or payload["video_id"],
                                    status="error",
                                    error=error_message,
                                )

                _run(_finalize_error())
            except OSError as e:
                logger.warning(
                    "Could not read video for error record path=%s err=%s",
                    video_path,
                    e,
                )
            except Exception as e:
                logger.warning(
                    "Could not upsert error record for path=%s err=%s",
                    video_path,
                    e,
                )
        _update_job_sync(pool, job_id, status="error", error=str(exc), finished_at=time.time())
