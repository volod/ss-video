"""Bridge Frigate NVR camera RTSP streams into the selfsuvis MediaMTX + RtspCaptioner pipeline.

On startup the bridge:
  1. Queries the Frigate HTTP API to discover enabled cameras.
  2. Registers each camera's RTSP re-stream path in MediaMTX.
  3. Starts a background RtspCaptioner session per camera so captions and
     structured scene facts land in the scene_timeline PostgreSQL table in
     real time.

The bridge runs a periodic refresh loop so cameras added or disabled in Frigate
are automatically picked up without a restart.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from selfsuvis.pipeline.core.logging import get_logger

from .camera_settings import camera_settings

logger = get_logger(__name__)

_FRIGATE_CAM_RTSP_TEMPLATE = "rtsp://{host}:8554/{camera}"


@dataclass
class _CameraSession:
    camera: str
    rtsp_url: str
    mediamtx_path: str
    session_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FrigateRtspBridge:
    """Register Frigate camera streams in MediaMTX and start live captioning.

    Args:
        mediamtx_client:      MediaMtxClient instance (from app.services.live_streams).
        stream_manager:       RealtimeStreamManager instance.
        db_pool:              asyncpg pool for RtspCaptioner scene_timeline writes.
        refresh_interval_sec: How often to re-query Frigate for camera changes.
    """

    def __init__(
        self,
        mediamtx_client: Any,
        stream_manager: Any,
        db_pool: Any,
        refresh_interval_sec: float = 60.0,
    ) -> None:
        self._mtx = mediamtx_client
        self._mgr = stream_manager
        self._db_pool = db_pool
        self._refresh_sec = refresh_interval_sec
        self._active: dict[str, _CameraSession] = {}
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Start the bridge (blocks until shutdown() is called)."""
        logger.info("FrigateRtspBridge starting, Frigate API=%s", camera_settings.frigate_api_url)
        while not self._stop.is_set():
            try:
                await self._sync_cameras()
            except Exception:
                logger.exception("FrigateRtspBridge: camera sync error")
            try:
                await asyncio.wait_for(asyncio.shield(self._stop.wait()), timeout=self._refresh_sec)
            except asyncio.TimeoutError:
                pass

    async def shutdown(self) -> None:
        self._stop.set()
        for cam_name in list(self._active):
            await self._stop_camera(cam_name)
        logger.info("FrigateRtspBridge stopped (%d sessions closed)", len(self._active))

    def active_cameras(self) -> list[dict[str, Any]]:
        return [
            {
                "camera": s.camera,
                "rtsp_url": s.rtsp_url,
                "session_id": s.session_id,
                "started_at": s.started_at.isoformat(),
            }
            for s in self._active.values()
        ]

    # -- Internal --------------------------------------------------------------

    async def _fetch_cameras(self) -> list[str]:
        """Return names of enabled cameras from the Frigate API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{camera_settings.frigate_api_url.rstrip('/')}/api/cameras"
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                return [name for name, cfg in data.items() if cfg.get("enabled", True)]
        except Exception as exc:
            logger.warning("FrigateRtspBridge: cannot reach Frigate API (%s)", exc)
            return []

    async def _sync_cameras(self) -> None:
        live = set(await self._fetch_cameras())
        current = set(self._active)

        for cam in live - current:
            await self._start_camera(cam)

        for cam in current - live:
            logger.info("FrigateRtspBridge: camera removed or disabled: %s", cam)
            await self._stop_camera(cam)

    async def _start_camera(self, camera: str) -> None:
        frigate_host = httpx.URL(camera_settings.frigate_api_url).host or "frigate"
        rtsp_url = _FRIGATE_CAM_RTSP_TEMPLATE.format(host=frigate_host, camera=camera)
        path_name = f"ssv/{camera}"

        try:
            await self._mtx.ensure_path(path_name=path_name, source_url=rtsp_url)
        except Exception as exc:
            logger.warning(
                "FrigateRtspBridge: MediaMTX path registration failed for %s: %s", camera, exc
            )

        mission_id = f"coop-live-{camera}"
        try:
            info = await self._mgr.start(
                session_id=f"coop-{camera}",
                mission_id=mission_id,
                robot_id=f"frigate:{camera}",
                path_name=path_name,
            )
            session_id = info.get("session_id")
        except Exception as exc:
            logger.warning("FrigateRtspBridge: captioner start failed for %s: %s", camera, exc)
            session_id = None

        self._active[camera] = _CameraSession(
            camera=camera,
            rtsp_url=rtsp_url,
            mediamtx_path=path_name,
            session_id=session_id,
        )
        logger.info("FrigateRtspBridge: started camera=%s session=%s", camera, session_id)

    async def _stop_camera(self, camera: str) -> None:
        session = self._active.pop(camera, None)
        if session and session.session_id:
            try:
                await self._mgr.stop(session.session_id)
            except Exception as exc:
                logger.debug(
                    "FrigateRtspBridge: error stopping session %s: %s", session.session_id, exc
                )
        if session:
            try:
                await self._mtx.delete_path(session.mediamtx_path)
            except Exception:
                pass
