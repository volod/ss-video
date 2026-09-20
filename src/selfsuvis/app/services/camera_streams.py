"""Orchestrate Frigate RTSP bridge and per-camera sound analysis at app startup.

On startup CameraStreamService:
  1. Starts FrigateRtspBridge — discovers Frigate cameras, registers each in
     MediaMTX and launches an RtspCaptioner session writing to scene_timeline.
  2. Starts SoundAnalyzer tasks for each active camera. Acoustic alarms are
     republished as camera-event contracts when a publisher is configured.
"""

import asyncio
from typing import Any

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)


class CameraStreamService:
    """Lifecycle manager for coop realtime stream analysis.

    Args:
        mediamtx_client:   MediaMtxClient from app.services.live_streams.
        stream_manager:    RealtimeStreamManager from app.services.live_streams.
        site_aggregator:   unused; kept for call-site compatibility.
        db_pool:           asyncpg pool for FrigateRtspBridge / scene_timeline.
        event_publisher:   optional VideoContractPublisher for acoustic events.
        enable_sound:      Whether to start SoundAnalyzer tasks per camera.
        refresh_sec:       Camera discovery refresh interval (seconds).
    """

    def __init__(
        self,
        mediamtx_client: Any,
        stream_manager: Any,
        site_aggregator: Any | None = None,
        db_pool: Any | None = None,
        event_publisher: Any | None = None,
        enable_sound: bool = False,
        refresh_sec: float = 60.0,
    ) -> None:
        self._mtx = mediamtx_client
        self._mgr = stream_manager
        self._aggregator = site_aggregator
        self._db_pool = db_pool
        self._publisher = event_publisher
        self._enable_sound = enable_sound
        self._refresh_sec = refresh_sec
        self._tasks: list[asyncio.Task] = []
        self._bridge: Any = None

    async def start(self) -> None:
        """Start the RTSP bridge (and optionally per-camera sound analysis)."""
        try:
            from selfsuvis.pipeline.realtime.rtsp_bridge import FrigateRtspBridge

            self._bridge = FrigateRtspBridge(
                mediamtx_client=self._mtx,
                stream_manager=self._mgr,
                db_pool=self._db_pool,
                refresh_interval_sec=self._refresh_sec,
            )
            bridge_task = asyncio.create_task(self._bridge.start(), name="coop_rtsp_bridge")
            self._tasks.append(bridge_task)
            logger.info("CameraStreamService: RTSP bridge started")

            if self._enable_sound:
                # Give the bridge a moment to discover cameras before launching analyzers
                asyncio.create_task(self._start_sound_analyzers_delayed(), name="coop_sound_init")

        except ImportError as exc:
            logger.warning("CameraStreamService: coop not available (%s)", exc)
        except Exception:
            logger.exception("CameraStreamService: startup error")

    async def shutdown(self) -> None:
        """Cancel all background tasks and shut down the bridge."""
        if self._bridge is not None:
            await self._bridge.shutdown()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("CameraStreamService stopped")

    def active_cameras(self) -> list[dict[str, Any]]:
        """Return active RTSP bridge camera sessions for API status views."""
        if self._bridge is None:
            return []
        return self._bridge.active_cameras()

    async def _start_sound_analyzers_delayed(self) -> None:
        """Wait for bridge to populate cameras, then start SoundAnalyzer per camera."""
        await asyncio.sleep(10.0)
        if self._bridge is None:
            return
        try:
            from selfsuvis.pipeline.realtime.sound_analyzer import SoundAnalyzer

            for cam_info in self._bridge.active_cameras():
                camera = cam_info["camera"]
                rtsp_url = cam_info["rtsp_url"]

                on_obs = self._make_acoustic_callback(camera)
                analyzer = SoundAnalyzer(camera=camera, rtsp_url=rtsp_url, on_observation=on_obs)
                task = asyncio.create_task(analyzer.run(), name=f"coop_sound_{camera}")
                self._tasks.append(task)
                logger.info("CameraStreamService: SoundAnalyzer started for camera=%s", camera)
        except ImportError as exc:
            logger.debug("CameraStreamService: SoundAnalyzer unavailable (%s)", exc)
        except Exception:
            logger.exception("CameraStreamService: error starting sound analyzers")

    def _make_acoustic_callback(self, camera: str):
        async def _cb(observation) -> None:
            if not observation.silence:
                logger.debug(
                    "acoustic [%s] rms=%.1f dB events=%s transcript=%r",
                    camera,
                    observation.rms_db,
                    observation.acoustic_events,
                    (observation.speech_transcript or "")[:80],
                )
            if self._publisher is not None:
                for ae in observation.acoustic_events:
                    try:
                        from selfsuvis.pipeline.realtime.camera_events import CameraEvent

                        synth = CameraEvent(
                            event_id=f"acoustic:{camera}:{ae['event']}",
                            camera=camera,
                            label=ae["event"],
                            score=float(ae.get("energy_ratio", 0.5)),
                            top_score=float(ae.get("energy_ratio", 0.5)),
                            event_type="new",
                            started_at=observation.recorded_at,
                            ended_at=None,
                            has_snapshot=False,
                            has_clip=False,
                            region={},
                            raw={},
                        )
                        await self._publisher.publish_camera_event(synth)
                    except Exception:
                        pass

        return _cb
