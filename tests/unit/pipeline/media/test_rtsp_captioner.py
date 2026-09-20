"""Unit tests for pipeline/rtsp_captioner.py.

All tests mock cv2 and the DB pool — no hardware or live stream required.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_fake_db_pool():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)
    return pool


# ── RtspCaptioner construction ────────────────────────────────────────────────


def test_rtsp_captioner_uses_config_fps(monkeypatch):
    """RtspCaptioner reads RTSP_CAPTION_FPS from settings when not overridden."""
    import selfsuvis.pipeline.media.rtsp_captioner as rc

    monkeypatch.setattr(rc.settings, "RTSP_CAPTION_FPS", 1.0)

    captioner = rc.RtspCaptioner(
        rtsp_url="rtsp://localhost:8554/test",
        mission_id="m1",
        db_pool=_make_fake_db_pool(),
    )
    assert captioner._caption_fps == pytest.approx(1.0)
    assert captioner._frame_interval_s == pytest.approx(1.0)


def test_rtsp_captioner_override_fps():
    """caption_fps kwarg overrides settings."""
    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner

    captioner = RtspCaptioner(
        rtsp_url="rtsp://localhost:8554/test",
        mission_id="m1",
        db_pool=_make_fake_db_pool(),
        caption_fps=2.0,
    )
    assert captioner._caption_fps == pytest.approx(2.0)
    assert captioner._frame_interval_s == pytest.approx(0.5)


# ── caption dispatch ──────────────────────────────────────────────────────────


def test_caption_frame_uses_gemma_when_enabled():
    """_caption_frame returns Gemma facts when QwenModel is enabled and healthy."""
    from PIL import Image

    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner

    captioner = RtspCaptioner("rtsp://x", "m1", _make_fake_db_pool())

    fake_gemma = MagicMock()
    fake_gemma.is_enabled.return_value = True
    fake_gemma.is_healthy.return_value = True
    fake_gemma.extract_frame_facts.return_value = {
        "vehicle_groups": [{"count": 3}],
        "road_condition": "clear",
        "scene_summary": "Three trucks on a clear road.",
    }
    captioner._gemma_model = fake_gemma

    img = Image.new("RGB", (8, 8))
    result = captioner._caption_frame(img)

    assert result["model"] == "gemma"
    assert "Three trucks" in result["caption"]
    assert result["facts_json"] is not None


def test_caption_frame_falls_back_to_florence_on_gemma_timeout():
    """_caption_frame falls back to Florence when Gemma times out."""
    from PIL import Image

    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner

    captioner = RtspCaptioner("rtsp://x", "m1", _make_fake_db_pool(), florence_fallback=True)

    fake_gemma = MagicMock()
    fake_gemma.is_enabled.return_value = True
    fake_gemma.is_healthy.return_value = True
    fake_gemma.extract_frame_facts.return_value = {"timeout": True, "timeout_sec": 60}
    captioner._gemma_model = fake_gemma

    fake_florence = MagicMock()
    fake_florence.caption.return_value = "A dusty road with no vehicles."
    captioner._florence_model = fake_florence

    img = Image.new("RGB", (8, 8))
    result = captioner._caption_frame(img)

    assert result["model"] == "florence-2"
    assert result["caption"] == "A dusty road with no vehicles."
    assert result["facts_json"] is None


def test_caption_frame_falls_back_to_florence_when_gemma_disabled():
    """_caption_frame uses Florence when Gemma is not enabled."""
    from PIL import Image

    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner

    captioner = RtspCaptioner("rtsp://x", "m1", _make_fake_db_pool(), florence_fallback=True)

    fake_gemma = MagicMock()
    fake_gemma.is_enabled.return_value = False
    captioner._gemma_model = fake_gemma

    fake_florence = MagicMock()
    fake_florence.caption.return_value = "Open field."
    captioner._florence_model = fake_florence

    img = Image.new("RGB", (8, 8))
    result = captioner._caption_frame(img)

    assert result["model"] == "florence-2"


def test_caption_frame_returns_none_when_both_fail():
    """_caption_frame returns null caption when both Gemma and Florence fail."""
    from PIL import Image

    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner

    captioner = RtspCaptioner("rtsp://x", "m1", _make_fake_db_pool(), florence_fallback=True)

    fake_gemma = MagicMock()
    fake_gemma.is_enabled.return_value = False
    captioner._gemma_model = fake_gemma

    fake_florence = MagicMock()
    fake_florence.caption.side_effect = RuntimeError("OOM")
    captioner._florence_model = fake_florence

    img = Image.new("RGB", (8, 8))
    result = captioner._caption_frame(img)
    assert result["caption"] is None
    assert result["model"] == "none"


# ── write_to_timeline ─────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_write_to_timeline_calls_db_execute():
    """_write_to_timeline calls pool.execute with the correct SQL parameters."""
    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner

    pool = _make_fake_db_pool()
    captioner = RtspCaptioner("rtsp://x", "m1", pool)

    await captioner._write_to_timeline(
        frame_id="frame_001",
        caption="A convoy.",
        facts_json={"road_condition": "clear"},
        gps_lat=47.0,
        gps_lon=8.0,
        gps_alt=None,
        t_sec=10.5,
    )

    pool.execute.assert_awaited_once()
    call_args = pool.execute.call_args[0]
    assert "scene_timeline" in call_args[0]
    assert "m1" in call_args  # mission_id
    assert "frame_001" in call_args


@pytest.mark.anyio
async def test_write_to_timeline_is_non_fatal_on_error():
    """_write_to_timeline swallows DB errors (non-blocking pipeline)."""
    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner

    pool = _make_fake_db_pool()
    pool.execute = AsyncMock(side_effect=Exception("db down"))
    captioner = RtspCaptioner("rtsp://x", "m1", pool)

    # Should not raise
    await captioner._write_to_timeline(
        frame_id="f1",
        caption=None,
        facts_json=None,
        gps_lat=None,
        gps_lon=None,
        gps_alt=None,
        t_sec=0.0,
    )


# ── run() — stream consumer ───────────────────────────────────────────────────


@pytest.mark.anyio
async def test_run_stops_on_stop_event():
    """run() exits promptly when stop_event is set before it opens the stream."""
    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner

    captioner = RtspCaptioner("rtsp://x", "m1", _make_fake_db_pool())

    stop = asyncio.Event()
    stop.set()  # pre-set before run

    # Mock cv2 so we don't need OpenCV
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.read.return_value = (False, None)  # stream ends immediately
    fake_cap.get.return_value = 25.0

    with (
        patch("cv2.VideoCapture", return_value=fake_cap, create=True),
        patch("cv2.CAP_PROP_FPS", 1, create=True),
    ):
        # Should return without hanging
        await asyncio.wait_for(captioner.run(stop_event=stop), timeout=2.0)

    fake_cap.release.assert_called_once()


@pytest.mark.anyio
async def test_run_handles_missing_cv2():
    """run() raises ImportError with helpful message when cv2 is missing."""

    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner

    captioner = RtspCaptioner("rtsp://x", "m1", _make_fake_db_pool())

    with patch.dict("sys.modules", {"cv2": None}):
        with pytest.raises((ImportError, TypeError)):
            await captioner.run()


@pytest.mark.anyio
async def test_run_logs_warning_on_stream_open_failure():
    """run() returns without error when RTSP stream cannot be opened."""
    from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner

    captioner = RtspCaptioner("rtsp://x", "m1", _make_fake_db_pool())

    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = False
    fake_cap.get.return_value = 25.0

    with (
        patch("cv2.VideoCapture", return_value=fake_cap, create=True),
        patch("cv2.CAP_PROP_FPS", 1, create=True),
    ):
        await captioner.run()  # should return without exception

    fake_cap.release.assert_not_called()  # never opened, so release not called
