"""Unit tests for pipeline/rtsp_ingest.py."""

import subprocess
from unittest.mock import patch

import pytest

from selfsuvis.pipeline.media.rtsp_ingest import record_rtsp, validate_rtsp_url

# ── validate_rtsp_url ─────────────────────────────────────────────────────────


def test_valid_rtsp_url_passes():
    with patch("selfsuvis.pipeline.media.rtsp_ingest.socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        validate_rtsp_url("rtsp://camera.example.com:554/stream")  # no exception


def test_valid_rtmp_url_passes():
    with patch("selfsuvis.pipeline.media.rtsp_ingest.socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        validate_rtsp_url("rtmp://media.example.com/live/cam1")


def test_http_url_rejected():
    with pytest.raises(ValueError, match="scheme must be one of"):
        validate_rtsp_url("http://example.com/stream")


def test_https_url_rejected():
    with pytest.raises(ValueError, match="scheme must be one of"):
        validate_rtsp_url("https://example.com/stream")


def test_missing_hostname_rejected():
    with pytest.raises(ValueError, match="hostname"):
        validate_rtsp_url("rtsp:///stream")


def test_credentials_in_url_rejected():
    with patch("selfsuvis.pipeline.media.rtsp_ingest.socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        with pytest.raises(ValueError, match="credentials"):
            validate_rtsp_url("rtsp://user:pass@camera.example.com/stream")


def test_private_ip_rejected_by_default():
    with patch("selfsuvis.pipeline.media.rtsp_ingest.socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(None, None, None, None, ("192.168.1.100", 0))]
        with pytest.raises(ValueError, match="private"):
            validate_rtsp_url("rtsp://192.168.1.100:554/stream")


def test_loopback_ip_rejected():
    with patch("selfsuvis.pipeline.media.rtsp_ingest.socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(None, None, None, None, ("127.0.0.1", 0))]
        with pytest.raises(ValueError, match="private"):
            validate_rtsp_url("rtsp://localhost:554/stream")


def test_private_ip_allowed_with_flag(monkeypatch):
    from selfsuvis.pipeline.core import config

    monkeypatch.setattr(config.settings, "ALLOW_PRIVATE_URLS", True)
    with patch("selfsuvis.pipeline.media.rtsp_ingest.socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(None, None, None, None, ("192.168.1.100", 0))]
        validate_rtsp_url("rtsp://192.168.1.100:554/stream")  # no exception


def test_dns_failure_raises():
    import socket

    with patch(
        "selfsuvis.pipeline.media.rtsp_ingest.socket.getaddrinfo",
        side_effect=socket.gaierror("NXDOMAIN"),
    ):
        with pytest.raises(ValueError, match="Cannot resolve"):
            validate_rtsp_url("rtsp://does.not.exist/stream")


# ── record_rtsp ───────────────────────────────────────────────────────────────


@patch("selfsuvis.pipeline.media.rtsp_ingest.run_checked")
def test_record_rtsp_calls_ffmpeg(mock_run):
    record_rtsp("rtsp://cam/stream", "/tmp/out.mp4", duration_sec=60)
    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    assert "ffmpeg" in cmd
    assert "rtsp://cam/stream" in cmd
    assert "/tmp/out.mp4" in cmd


@patch("selfsuvis.pipeline.media.rtsp_ingest.run_checked")
def test_record_rtsp_includes_duration(mock_run):
    record_rtsp("rtsp://cam/stream", "/tmp/out.mp4", duration_sec=120)
    cmd = mock_run.call_args.args[0]
    assert "-t" in cmd
    assert "120" in cmd


@patch("selfsuvis.pipeline.media.rtsp_ingest.run_checked")
def test_record_rtsp_no_duration_omits_t_flag(mock_run):
    record_rtsp("rtsp://cam/stream", "/tmp/out.mp4", duration_sec=None)
    cmd = mock_run.call_args.args[0]
    assert "-t" not in cmd


@patch("selfsuvis.pipeline.media.rtsp_ingest.run_checked")
def test_record_rtsp_caps_duration_at_max(mock_run, monkeypatch):
    from selfsuvis.pipeline.core import config

    monkeypatch.setattr(config.settings, "RTSP_MAX_DURATION_SEC", 300)
    record_rtsp("rtsp://cam/stream", "/tmp/out.mp4", duration_sec=9999)
    cmd = mock_run.call_args.args[0]
    t_idx = cmd.index("-t")
    assert int(cmd[t_idx + 1]) == 300


@patch("selfsuvis.pipeline.media.rtsp_ingest.run_checked")
def test_record_rtsp_uses_tcp_transport(mock_run):
    record_rtsp("rtsp://cam/stream", "/tmp/out.mp4")
    cmd = mock_run.call_args.args[0]
    assert "-rtsp_transport" in cmd
    assert "tcp" in cmd


@patch("selfsuvis.pipeline.media.rtsp_ingest.run_checked")
def test_record_rtsp_stream_copy(mock_run):
    """Uses -c copy for no re-encoding."""
    record_rtsp("rtsp://cam/stream", "/tmp/out.mp4")
    cmd = mock_run.call_args.args[0]
    assert "-c" in cmd
    assert "copy" in cmd


@patch(
    "selfsuvis.pipeline.media.rtsp_ingest.run_checked",
    side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
)
def test_record_rtsp_raises_on_ffmpeg_error(mock_run):
    with pytest.raises(subprocess.CalledProcessError):
        record_rtsp("rtsp://cam/stream", "/tmp/out.mp4")
