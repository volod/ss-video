"""RTSP live-stream caption pipeline (Phase 6).

Consumes frames from a MediaMTX RTSP stream in real time, runs Gemma captioning
on each sampled frame, and writes captions + structured facts to the
``scene_timeline`` PostgreSQL table.

Design principles:
  - **Non-blocking consumer**: if the captioner is behind, frames are silently
    skipped. Latency is more important than completeness.
  - **Configurable sampling rate** (``RTSP_CAPTION_FPS``, default 0.5 fps = one
    caption every 2 s).
  - **Florence fallback**: on Gemma API timeout or service error, falls back to
    Florence-2 captioning for that frame (no structured facts).
  - **Graceful shutdown**: responds to ``SIGINT`` / ``SIGTERM`` or a stop event.

Usage::

    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner
    captioner = RtspCaptioner(
        rtsp_url="rtsp://localhost:8554/drone1",
        mission_id="mission_abc",
        db_pool=pool,        # asyncpg pool
    )
    await captioner.run()    # blocks until stopped or stream ends

    # Or with a stop event:
    stop = asyncio.Event()
    asyncio.create_task(captioner.run(stop_event=stop))
    # ... later:
    stop.set()
"""

import asyncio
import time
from typing import Any

from selfsuvis.pipeline.core import get_logger, settings

logger = get_logger(__name__)


class RtspCaptioner:
    """Non-blocking RTSP stream captioner that writes to scene_timeline.

    Args:
        rtsp_url:   Full RTSP URL (e.g. ``rtsp://localhost:8554/drone1``).
        mission_id: Mission identifier written to scene_timeline rows.
        db_pool:    asyncpg connection pool for scene_timeline writes.
        caption_fps: Override ``RTSP_CAPTION_FPS`` env var.
        florence_fallback: Enable Florence-2 fallback on Gemma timeout.
    """

    def __init__(
        self,
        rtsp_url: str,
        mission_id: str,
        db_pool,
        caption_fps: float | None = None,
        florence_fallback: bool = True,
        on_caption: Any | None = None,
    ) -> None:
        self._rtsp_url = rtsp_url
        self._mission_id = mission_id
        self._db_pool = db_pool
        self._caption_fps = caption_fps if caption_fps is not None else settings.RTSP_CAPTION_FPS
        self._florence_fallback = florence_fallback
        self._on_caption = on_caption
        self._frame_interval_s = 1.0 / max(self._caption_fps, 0.001)

        # Lazy-loaded models
        self._gemma_model = None  # QwenModel (now Gemma-backed)
        self._florence_model = None

        logger.info(
            "RtspCaptioner: url=%s mission=%s fps=%.2f",
            rtsp_url,
            mission_id,
            self._caption_fps,
        )

    # -- Model loading ---------------------------------------------------------

    def _get_gemma_model(self):
        if self._gemma_model is None:
            from selfsuvis.pipeline.vision.qwen import QwenModel

            self._gemma_model = QwenModel()
        return self._gemma_model

    def _get_florence(self):
        if self._florence_model is None:
            from selfsuvis.pipeline.vision.florence import FlorenceModel

            self._florence_model = FlorenceModel()
        return self._florence_model

    # -- Caption dispatch ------------------------------------------------------

    def _caption_frame(self, pil_image) -> dict[str, Any]:
        """Caption a single PIL image. Returns a dict with caption + facts."""
        gemma = self._get_gemma_model()
        if gemma.is_enabled() and gemma.is_healthy():
            result = gemma.extract_frame_facts(pil_image)
            # Gemma returns structured facts; derive a text caption from scene_summary
            if result and not result.get("disabled") and not result.get("timeout"):
                caption = result.get("scene_summary") or ""
                return {"caption": caption, "facts_json": result, "model": "gemma"}

        # Gemma unavailable or timed out → Florence fallback
        if self._florence_fallback:
            try:
                florence = self._get_florence()
                caption = florence.caption(pil_image)
                return {"caption": caption, "facts_json": None, "model": "florence-2"}
            except Exception as exc:
                logger.warning("Florence fallback failed: %s", exc)

        return {"caption": None, "facts_json": None, "model": "none"}

    # -- DB write --------------------------------------------------------------

    async def _write_to_timeline(
        self,
        frame_id: str,
        caption: str | None,
        facts_json: dict[str, Any] | None,
        gps_lat: float | None,
        gps_lon: float | None,
        gps_alt: float | None,
        t_sec: float,
    ) -> None:
        """Insert a row into scene_timeline (non-fatal on error)."""
        import json as _json

        try:
            await self._db_pool.execute(
                """
                INSERT INTO scene_timeline
                    (mission_id, frame_id, gps_lat, gps_lon, gps_alt, t_sec, caption, facts_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                """,
                self._mission_id,
                frame_id,
                gps_lat,
                gps_lon,
                gps_alt,
                t_sec,
                caption,
                _json.dumps(facts_json) if facts_json else None,
            )
        except Exception as exc:
            logger.error("scene_timeline write failed: %s", exc)
        if self._on_caption is not None:
            try:
                await self._on_caption(
                    mission_id=self._mission_id,
                    frame_id=frame_id,
                    caption=caption,
                    facts_json=facts_json,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                    gps_alt=gps_alt,
                    t_sec=t_sec,
                )
            except Exception as exc:
                logger.debug("scene caption publish failed: %s", exc)

    # -- Frame reader ----------------------------------------------------------

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Consume the RTSP stream and caption frames at the configured rate.

        Blocks until the stream ends, an error occurs, or ``stop_event`` is set.

        Args:
            stop_event: When set, the captioner drains the current caption and
                stops gracefully. Pass ``None`` to run until the stream ends.
        """
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "opencv-python is required for RtspCaptioner: pip install opencv-python"
            )

        from PIL import Image as _PIL_Image

        logger.info("RtspCaptioner: opening stream %s", self._rtsp_url)
        cap = cv2.VideoCapture(self._rtsp_url)
        if not cap.isOpened():
            logger.error("RtspCaptioner: could not open RTSP stream %s", self._rtsp_url)
            return

        stream_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # How many source frames to skip between caption samples
        skip_n = max(1, int(round(stream_fps / self._caption_fps)))

        frame_count = 0
        last_caption_time = 0.0
        t_start = time.monotonic()

        logger.info(
            "RtspCaptioner: stream_fps=%.1f caption_fps=%.2f skip_n=%d",
            stream_fps,
            self._caption_fps,
            skip_n,
        )

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    logger.info("RtspCaptioner: stop_event set — shutting down")
                    break

                ret, frame_bgr = cap.read()
                if not ret:
                    logger.info("RtspCaptioner: stream ended or read failed")
                    break

                frame_count += 1

                # Non-blocking skip: only process every skip_n-th frame
                if frame_count % skip_n != 0:
                    continue

                now = time.monotonic()
                # Extra guard: if the last caption was too recent (captioner is slow),
                # skip this frame to avoid backpressure
                if now - last_caption_time < self._frame_interval_s * 0.8:
                    continue
                last_caption_time = now

                t_sec = now - t_start
                frame_id = f"{self._mission_id}_rtsp_{frame_count:08d}"

                # Convert BGR → PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_image = _PIL_Image.fromarray(frame_rgb)

                # Caption (may be slow — offload to thread to keep loop alive)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._caption_frame, pil_image)

                caption = result.get("caption")
                facts_json = result.get("facts_json")
                model = result.get("model", "unknown")

                logger.debug(
                    "RtspCaptioner frame=%d t=%.1fs model=%s caption=%s",
                    frame_count,
                    t_sec,
                    model,
                    (caption or "")[:80],
                )

                # Write to scene_timeline (GPS coords not available from RTSP frames)
                await self._write_to_timeline(
                    frame_id=frame_id,
                    caption=caption,
                    facts_json=facts_json,
                    gps_lat=None,
                    gps_lon=None,
                    gps_alt=None,
                    t_sec=t_sec,
                )

                # Yield to the event loop so other tasks can run
                await asyncio.sleep(0)

        finally:
            cap.release()
            logger.info(
                "RtspCaptioner: stopped. frames_read=%d mission=%s",
                frame_count,
                self._mission_id,
            )
