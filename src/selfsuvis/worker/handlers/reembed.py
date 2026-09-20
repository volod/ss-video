"""REEMBED job handler — re-embed all indexed frames with the current DINOv3 model."""

import time

import asyncpg

from selfsuvis.pipeline.core import datetime_to_ts, get_dino_model_name, settings
from selfsuvis.pipeline.storage import update_job
from selfsuvis.pipeline.storage.common import qdrant_id_from_pg
from selfsuvis.pipeline.storage.missions import list_frames_after
from selfsuvis.worker._run import _run
from selfsuvis.worker.gpu import GPULock


async def _load_reembed_cursor(conn, job_id: str) -> tuple:
    """Return (cursor, frames_reembedded) restored from stored job progress."""
    import json as _json

    row = await conn.fetchrow("SELECT progress_json FROM jobs WHERE id = $1", job_id)
    progress: dict = {}
    if row:
        progress = row["progress_json"] or {}
        if isinstance(progress, str):
            progress = _json.loads(progress)
    raw = progress.get("last_cursor")
    cursor = (raw[0], raw[1]) if isinstance(raw, list) and len(raw) == 2 else None
    return cursor, progress.get("frames_reembedded", 0)


def _load_batch_images(batch, logger) -> tuple:
    """Open PIL images for a frame batch, skipping unreadable files.

    Returns (images, valid_rows).
    """
    from PIL import Image as PILImage

    images, valid_rows = [], []
    for frame_row in batch:
        try:
            img = PILImage.open(frame_row["frame_path"]).convert("RGB")
            images.append(img)
            valid_rows.append(frame_row)
        except Exception as exc:
            logger.warning("Reembed: skipping unreadable frame id=%s err=%s", frame_row["id"], exc)
    return images, valid_rows


def _build_reembed_points(valid_rows, dino_vecs, clip_vecs):
    """Assemble Qdrant PointStructs from embedding vectors and frame metadata."""
    from selfsuvis.pipeline.core.optional_deps import require_qdrant_models

    qmodels = require_qdrant_models()

    return [
        qmodels.PointStruct(
            id=qdrant_id_from_pg(row["qdrant_id"]),
            vector={"clip": clip_vecs[i].tolist(), "dino": dino_vecs[i].tolist()},
            payload={"frame_id": row["id"], "mission_id": row["mission_id"]},
        )
        for i, row in enumerate(valid_rows)
    ]


async def _run_reembed(conn, job_id: str, dino, clip, qdrant, batch_size: int, logger) -> int:
    """Iterate all frames in cursor order, re-embed, and checkpoint progress.

    Returns the total number of frames re-embedded.
    """
    cursor, frames_reembedded = await _load_reembed_cursor(conn, job_id)
    logger.info("Reembed job started id=%s resuming_from_cursor=%s", job_id, cursor)

    while True:
        batch = await list_frames_after(conn, cursor, batch_size)
        if not batch:
            break

        images, valid_rows = _load_batch_images(batch, logger)
        if images:
            dino_vecs = dino.encode_images(images)
            clip_vecs = clip.encode_images(images)
            points = _build_reembed_points(valid_rows, dino_vecs, clip_vecs)
            try:
                qdrant.upsert_points(points)
            except Exception as exc:
                logger.error("Reembed: Qdrant upsert failed cursor=%s err=%s", cursor, exc)
                cursor_serial = [datetime_to_ts(cursor[0]), cursor[1]] if cursor else None
                await update_job(
                    conn,
                    job_id,
                    status="error",
                    error=str(exc),
                    progress={"last_cursor": cursor_serial, "frames_reembedded": frames_reembedded},
                    finished_at=time.time(),
                )
                return frames_reembedded
            frames_reembedded += len(valid_rows)

        last_row = batch[-1]
        cursor = (last_row["created_at"], last_row["id"])
        await update_job(
            conn,
            job_id,
            progress={
                "last_cursor": [datetime_to_ts(cursor[0]), cursor[1]],
                "frames_reembedded": frames_reembedded,
            },
        )
        logger.debug("Reembed: cursor=%s frames_reembedded=%d", cursor, frames_reembedded)

    cursor_serial = [datetime_to_ts(cursor[0]), cursor[1]] if cursor else None
    await update_job(
        conn,
        job_id,
        status="finished",
        progress={"last_cursor": cursor_serial, "frames_reembedded": frames_reembedded},
        finished_at=time.time(),
    )
    return frames_reembedded


def handle_reembed_job(job_id: str, payload: dict, conn_url: str, logger) -> None:
    """Re-embed all indexed frames with the current DINOv3 model.

    Processes frames in batches of REEMBED_BATCH_SIZE (default 256).
    Checkpoints last_cursor after each batch so the sweep is resumable.
    """
    from selfsuvis.models.dino_model import DINOEmbedder
    from selfsuvis.models.openclip_model import OpenCLIPEmbedder
    from selfsuvis.pipeline.storage.qdrant import QdrantStore

    try:
        dino_name = get_dino_model_name(settings.MODEL_NAME)
        if dino_name is None:
            raise ValueError(f"Unsupported DINO model family: {settings.MODEL_NAME}")
        dino = DINOEmbedder(dino_name)
        clip = OpenCLIPEmbedder()
        qdrant = QdrantStore(clip_dim=clip.image_dim(), dino_dim=dino.image_dim())

        async def _connect_and_run() -> int:
            conn = await asyncpg.connect(conn_url)
            try:
                return await _run_reembed(
                    conn, job_id, dino, clip, qdrant, settings.REEMBED_BATCH_SIZE, logger
                )
            finally:
                await conn.close()

        with GPULock(job_id, "reembed", conn_url, logger):
            frames_reembedded = _run(_connect_and_run())
        logger.info("Reembed job finished id=%s frames_reembedded=%d", job_id, frames_reembedded)

    except Exception as exc:
        logger.exception("Reembed job failed id=%s error=%s", job_id, exc)
        error_message = str(exc)

        async def _mark_error():
            conn = await asyncpg.connect(conn_url)
            try:
                await update_job(
                    conn, job_id, status="error", error=error_message, finished_at=time.time()
                )
            finally:
                await conn.close()

        _run(_mark_error())
