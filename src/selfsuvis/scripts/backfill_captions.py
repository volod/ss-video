#!/usr/bin/env python3
"""Backfill Florence-2 captions for frames that have caption=NULL.

Resume-safe: re-running this script skips frames that already have a caption
OR that have a caption_skip_reason set (e.g. 'file_missing').

Frames whose disk file is missing are marked with caption_skip_reason='file_missing'
so the null-rate metric remains honest and subsequent runs don't re-check them.

Usage:
    python scripts/backfill_captions.py [--skip-qdrant] [--batch-size N]

Options:
    --skip-qdrant     Skip Qdrant set_payload updates (useful when Qdrant is unavailable).
    --batch-size N    Override FLORENCE_BATCH_SIZE for this run (default: from settings).
    --dry-run         Print how many frames need captioning, then exit.
"""

import argparse
import asyncio
import os

import asyncpg
from PIL import Image

from selfsuvis.pipeline.core.env import env_str, load_script_env

load_script_env(anchor_file=__file__)

from selfsuvis.pipeline.core.config import settings  # noqa: E402
from selfsuvis.pipeline.core.logging import get_logger  # noqa: E402
from selfsuvis.pipeline.vision.florence import FlorenceModel  # noqa: E402

logger = get_logger(__name__)

DATABASE_URL = env_str(
    "DATABASE_URL",
    "postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis",
)

_SKIP_FILE_MISSING = "file_missing"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Florence-2 captions for uncaptioned frames."
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip Qdrant set_payload updates.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Florence batch size (default: FLORENCE_BATCH_SIZE from settings).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print frame count and exit without captioning.",
    )
    return parser.parse_args()


async def _count_pending(conn) -> int:
    """Count frames that need captioning (no caption AND no skip reason)."""
    row = await conn.fetchrow(
        "SELECT COUNT(*) FROM frames WHERE caption IS NULL AND caption_skip_reason IS NULL"
    )
    return row[0]


async def _count_skipped(conn) -> int:
    """Count frames permanently skipped (e.g. file_missing)."""
    row = await conn.fetchrow("SELECT COUNT(*) FROM frames WHERE caption_skip_reason IS NOT NULL")
    return row[0]


async def _fetch_pending_batch(conn, after_id: str, limit: int) -> list[dict]:
    """Fetch next batch of frames needing captioning.

    Excludes frames that already have a caption OR a skip reason — both indicate
    prior processing. Only frames with caption IS NULL AND caption_skip_reason IS NULL
    are considered pending.
    """
    rows = await conn.fetch(
        """
        SELECT id, frame_path, qdrant_id
        FROM frames
        WHERE caption IS NULL
          AND caption_skip_reason IS NULL
          AND id > $1
        ORDER BY id
        LIMIT $2
        """,
        after_id,
        limit,
    )
    return [dict(r) for r in rows]


async def _update_frames(conn, updates: list[tuple[str, str, float, str]]) -> None:
    """Bulk-update caption, caption_confidence, caption_model for a batch of frames.

    updates: list of (frame_id, caption, confidence, model_tag)
    """
    await conn.executemany(
        """
        UPDATE frames
        SET caption = $1,
            caption_confidence = $2,
            caption_model = $3,
            updated_at = NOW()
        WHERE id = $4
        """,
        [
            (caption, confidence, model_tag, frame_id)
            for frame_id, caption, confidence, model_tag in updates
        ],
    )


async def _mark_skip_reason(conn, frame_ids: list[str], reason: str) -> None:
    """Set caption_skip_reason for frames that cannot be captioned.

    These frames remain with caption=NULL but are excluded from future backfill
    runs, keeping the null-rate metric honest.
    """
    await conn.executemany(
        """
        UPDATE frames
        SET caption_skip_reason = $1,
            updated_at = NOW()
        WHERE id = $2
        """,
        [(reason, fid) for fid in frame_ids],
    )


def _set_qdrant_payload(
    updates: list[tuple[str, str, float, str]],
    qdrant_id_map: dict,
    skip_qdrant: bool,
) -> None:
    """Push captions to Qdrant for updated frames."""
    if skip_qdrant:
        return

    try:
        from selfsuvis.pipeline.core.optional_deps import require_qdrant_client

        QdrantClient = require_qdrant_client()
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        collection = settings.QDRANT_COLLECTION

        failed = 0
        for frame_id, caption, _, _ in updates:
            qdrant_id = qdrant_id_map.get(frame_id)
            if qdrant_id is None:
                continue
            try:
                client.set_payload(
                    collection_name=collection,
                    payload={"caption": caption},
                    points=[qdrant_id],
                )
            except Exception:
                failed += 1
        if failed:
            logger.warning(
                "Qdrant set_payload: %d/%d calls failed; DB is authoritative.",
                failed,
                len(updates),
            )
    except Exception:
        logger.warning("Qdrant update failed; DB captions are intact.", exc_info=True)


async def run_backfill(args: argparse.Namespace) -> None:
    batch_size = args.batch_size or settings.FLORENCE_BATCH_SIZE

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        pending = await _count_pending(conn)
        already_skipped = await _count_skipped(conn)
        logger.info(
            "Frames pending captioning: %d  (already skipped permanently: %d)",
            pending,
            already_skipped,
        )

        if args.dry_run:
            print(
                f"Dry run: {pending} frames need captioning, {already_skipped} permanently skipped."
            )
            return

        if pending == 0:
            logger.info("All frames already captioned or skipped. Nothing to do.")
            return

        logger.info("Loading Florence-2-large …")
        florence = FlorenceModel()
        model_tag = florence.model_tag

        after_id = ""
        total_captioned = 0
        total_file_missing = 0

        while True:
            rows = await _fetch_pending_batch(conn, after_id, limit=batch_size)
            if not rows:
                break

            # Load PIL images; collect missing-file frame_ids for DB marking
            pil_images: list[Image.Image] = []
            valid_rows: list[dict] = []
            missing_ids: list[str] = []

            for row in rows:
                fp = row["frame_path"]
                if not os.path.exists(fp):
                    logger.warning(
                        "Frame file missing — marking caption_skip_reason='file_missing': "
                        "%s (id=%s)",
                        fp,
                        row["id"],
                    )
                    missing_ids.append(row["id"])
                    continue
                try:
                    pil_images.append(Image.open(fp).convert("RGB"))
                    valid_rows.append(row)
                except Exception:
                    logger.warning(
                        "Could not open frame %s (id=%s) — marking as file_missing.",
                        fp,
                        row["id"],
                        exc_info=True,
                    )
                    missing_ids.append(row["id"])

            # Mark irrecoverable frames so they're excluded from future runs
            if missing_ids:
                await _mark_skip_reason(conn, missing_ids, _SKIP_FILE_MISSING)
                total_file_missing += len(missing_ids)

            if valid_rows:
                try:
                    captions_and_confs = florence.caption_batch(pil_images, batch_size=batch_size)
                except Exception:
                    logger.warning(
                        "Florence batch failed; using empty captions for %d frames.",
                        len(valid_rows),
                        exc_info=True,
                    )
                    captions_and_confs = [("", 0.5)] * len(valid_rows)

                updates = [
                    (row["id"], caption, confidence, model_tag)
                    for row, (caption, confidence) in zip(valid_rows, captions_and_confs)
                ]
                await _update_frames(conn, updates)

                qdrant_id_map = {row["id"]: row["qdrant_id"] for row in valid_rows}
                _set_qdrant_payload(updates, qdrant_id_map, args.skip_qdrant)

                total_captioned += len(valid_rows)
                logger.info(
                    "Progress: %d captioned, %d marked file_missing, %d remaining",
                    total_captioned,
                    total_file_missing,
                    pending - total_captioned - total_file_missing,
                )

            # Advance cursor: use the last row id from original batch (includes all rows)
            after_id = rows[-1]["id"]

        logger.info(
            "Backfill complete: %d frames captioned, %d marked file_missing.",
            total_captioned,
            total_file_missing,
        )
    finally:
        await conn.close()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run_backfill(args))
