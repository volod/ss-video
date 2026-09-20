"""Unit tests for live stream MediaMTX helpers and runtime manager."""

import asyncio
from unittest.mock import patch

import pytest

from selfsuvis.app.services.live_streams import (
    RealtimeStreamManager,
    build_rtsp_stream_url,
    validate_stream_path,
)


class _FakeCaptioner:
    def __init__(self, *, rtsp_url, mission_id, db_pool, caption_fps, on_caption=None):
        self.rtsp_url = rtsp_url
        self.mission_id = mission_id
        self.db_pool = db_pool
        self.caption_fps = caption_fps

    async def run(self, stop_event=None):
        assert stop_event is not None
        await stop_event.wait()


def test_build_rtsp_stream_url_uses_public_base():
    with patch(
        "selfsuvis.app.services.live_streams.settings.MEDIAMTX_PUBLIC_RTSP_BASE_URL",
        "rtsp://stream.example.com:8554",
    ):
        assert (
            build_rtsp_stream_url("live/drone-1", public=True)
            == "rtsp://stream.example.com:8554/live/drone-1"
        )


def test_validate_stream_path_rejects_invalid_chars():
    with pytest.raises(ValueError, match="path_name may contain only"):
        validate_stream_path("live/drone 1")


@pytest.mark.anyio
async def test_realtime_stream_manager_start_and_stop():
    manager = RealtimeStreamManager(db_pool=object())
    with (
        patch("selfsuvis.app.services.live_streams.RtspCaptioner", _FakeCaptioner),
        patch(
            "selfsuvis.app.services.live_streams.settings.MEDIAMTX_RTSP_BASE_URL",
            "rtsp://mediamtx:8554",
        ),
    ):
        started = await manager.start(
            session_id="session-1",
            mission_id="mission-1",
            robot_id="drone-1",
            path_name="live/drone-1",
            caption_fps=1.0,
        )
        assert started["path_name"] == "live/drone-1"
        assert started["rtsp_url"] == "rtsp://mediamtx:8554/live/drone-1"

        await asyncio.sleep(0)
        status = await manager.get("session-1")
        assert status["status"] == "running"

        stopped = await manager.stop("session-1")
        assert stopped["status"] == "stopped"
