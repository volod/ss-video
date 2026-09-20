"""FINETUNE job handler — supervised contrastive fine-tuning on CVAT-annotated frames."""

import os
import time

from selfsuvis.pipeline.core import settings, utcnow
from selfsuvis.pipeline.core.manifests import (
    finetune_model_artifact,
    model_manifest_path,
    write_manifest,
)
from selfsuvis.pipeline.storage import update_job
from selfsuvis.worker._run import _run
from selfsuvis.worker.gpu import GPULock

_UPSERT_SYSTEM_STATE_SQL = (
    "INSERT INTO system_state (key, value, updated_at) VALUES ($1, $2, $3) "
    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at"
)

_API_RELOAD_TIMEOUT_SEC = 30


def _write_model_manifest(
    ckpt_path: str,
    model_version_id: str,
    base_model: str,
    result: dict,
    annotation_count: int,
    created_at,
    logger,
) -> None:
    """Write the checkpoint's `model-artifact` manifest next to it; a failure only warns."""
    try:
        manifest = finetune_model_artifact(
            ckpt_path,
            model_version_id=model_version_id,
            base_model=base_model,
            result=result,
            annotation_count=annotation_count,
            created_at=created_at,
        )
        path = write_manifest(manifest, model_manifest_path(ckpt_path))
        logger.info("Finetune manifest written path=%s", path)
    except (OSError, ValueError) as exc:
        logger.warning("Finetune manifest not written for %s: %s", ckpt_path, exc)


async def _persist_finetune_acceptance(
    conn, job_id: str, ckpt_path: str, result: dict, logger, base_model: str | None = None
) -> str:
    """Persist watermark, checkpoint provenance, and mark job finished.

    With `base_model`, also writes the checkpoint's model-artifact manifest.
    Returns the model_version_id assigned to this checkpoint.
    """
    now = utcnow()
    total_annotated = await conn.fetchval("SELECT COUNT(*) FROM frames WHERE al_tag = 'annotated'")
    model_version_id = f"sup_{job_id[:8]}"

    await conn.execute(
        _UPSERT_SYSTEM_STATE_SQL, "last_retrain_watermark", str(total_annotated), now
    )
    await conn.execute(_UPSERT_SYSTEM_STATE_SQL, "active_dino_checkpoint", ckpt_path, now)
    await conn.execute(
        "INSERT INTO model_checkpoints "
        "(checkpoint_path, model_version_id, annotation_count, best_accuracy, "
        " distribution_shift, created_at, notes) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (checkpoint_path) DO NOTHING",
        ckpt_path,
        model_version_id,
        total_annotated,
        result["best_accuracy"],
        result.get("distribution_shift", 0.0),
        now,
        f"finetune job_id={job_id}",
    )
    if base_model:
        _write_model_manifest(
            ckpt_path, model_version_id, base_model, result, total_annotated, now, logger
        )
    settings.MODEL_VERSION_ID = model_version_id
    logger.info(
        "Finetune job id=%s -- provenance registered model_version_id=%s "
        "annotation_count=%d distribution_shift=%.4f",
        job_id,
        model_version_id,
        total_annotated,
        result.get("distribution_shift", 0.0),
    )
    await update_job(
        conn,
        job_id,
        status="finished",
        progress={
            "accepted": True,
            "best_accuracy": result["best_accuracy"],
            "epochs": result["epochs"],
            "checkpoint": ckpt_path,
            "model_version_id": model_version_id,
        },
        finished_at=now,
    )
    return model_version_id


def _hot_reload_model(ckpt_path: str, job_id: str, logger) -> None:
    """Call POST /admin/reload-model to swap DINOv3 weights in the API process."""
    import httpx

    api_base = f"http://localhost:{os.environ.get('API_PORT', '8000')}"
    try:
        resp = httpx.post(
            f"{api_base}/admin/reload-model",
            json={"checkpoint": ckpt_path},
            headers={"X-API-Key": settings.API_KEY},
            timeout=_API_RELOAD_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        logger.info("Finetune job id=%s -- model reloaded checkpoint=%s", job_id, ckpt_path)
    except Exception as exc:
        logger.warning("Finetune job id=%s -- reload HTTP call failed: %s", job_id, exc)


def handle_finetune_job(job_id: str, payload: dict, db_pool, conn_url: str, logger) -> None:
    """Run supervised contrastive fine-tuning on CVAT-annotated frames.

    Expects payload = {} (no fields required; frames fetched from DB via from_db()).
    On success: promotes checkpoint, updates system_state.last_retrain_watermark,
    calls POST /admin/reload-model via HTTP.
    """
    from selfsuvis.pipeline.training.supervised import config_from_settings, run_supervised_finetune

    try:
        cfg = config_from_settings(frames_dir=settings.FRAMES_DIR)
        logger.info("Finetune job started id=%s", job_id)

        async def _mark_running():
            async with db_pool.acquire() as conn:
                await update_job(conn, job_id, status="running", started_at=time.time())

        _run(_mark_running())

        with GPULock(job_id, "supervised_finetune", conn_url, logger):
            result = run_supervised_finetune(cfg)

        if not result["accepted"]:
            logger.info(
                "Finetune job id=%s -- checkpoint rejected (accuracy=%.4f < gate=%.4f)",
                job_id,
                result["best_accuracy"],
                settings.SUP_EVAL_GATE_THRESHOLD,
            )

            async def _mark_rejected():
                async with db_pool.acquire() as conn:
                    await update_job(
                        conn,
                        job_id,
                        status="finished",
                        progress={"accepted": False, "best_accuracy": result["best_accuracy"]},
                        finished_at=time.time(),
                    )

            _run(_mark_rejected())
            return

        ckpt_path = result["path"]
        _hot_reload_model(ckpt_path, job_id, logger)

        async def _finish_accepted():
            async with db_pool.acquire() as conn:
                await _persist_finetune_acceptance(
                    conn, job_id, ckpt_path, result, logger, base_model=cfg.model_name
                )

        _run(_finish_accepted())
        logger.info(
            "Finetune job finished id=%s checkpoint=%s accuracy=%.4f",
            job_id,
            ckpt_path,
            result["best_accuracy"],
        )

    except Exception as exc:
        logger.exception("Finetune job failed id=%s error=%s", job_id, exc)
        error_message = str(exc)

        async def _mark_error():
            async with db_pool.acquire() as conn:
                await update_job(
                    conn, job_id, status="error", error=error_message, finished_at=time.time()
                )

        _run(_mark_error())
