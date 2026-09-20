"""Unit tests for pipeline.media.download."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

from selfsuvis.pipeline.media.download import download_url


def _make_response(headers=None, iter_content=None):
    """Create a mock response that works as context manager."""
    mock = MagicMock()
    mock.headers = headers or {}
    mock.iter_content = iter_content or (lambda **kw: iter([]))
    mock.raise_for_status = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_download_url_content_length_exceeds_max():
    """When Content-Length exceeds max_bytes, raises ValueError before writing."""
    mock_response = _make_response(headers={"Content-Length": "200"})

    with patch("selfsuvis.pipeline.media.download.safe_request", return_value=mock_response):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            dest = f.name
        try:
            with pytest.raises(ValueError) as exc_info:
                download_url("http://example.com/video.mp4", dest, max_bytes=100)
            assert "Content-Length" in str(exc_info.value)
            assert "exceeds max download size" in str(exc_info.value)
        finally:
            Path(dest).unlink(missing_ok=True)


def test_download_url_stream_exceeds_max():
    """When streamed bytes exceed max_bytes, raises ValueError and cleans up partial file."""
    mock_response = _make_response(iter_content=lambda **kw: iter([b"x" * 5, b"y" * 10]))

    with patch("selfsuvis.pipeline.media.download.safe_request", return_value=mock_response):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            dest = f.name
        with pytest.raises(ValueError) as exc_info:
            download_url("http://example.com/video.mp4", dest, max_bytes=10)
        assert "exceeded max size" in str(exc_info.value)
        # Partial file must have been cleaned up
        assert not Path(dest).exists()


def test_download_url_within_limit():
    """When bytes within limit, download succeeds."""
    content = b"small"
    mock_response = _make_response(
        headers={"Content-Length": str(len(content))},
        iter_content=lambda **kw: iter([content]),
    )

    with patch("selfsuvis.pipeline.media.download.safe_request", return_value=mock_response):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            dest = f.name
        try:
            download_url("http://example.com/video.mp4", dest, max_bytes=100)
            assert Path(dest).read_bytes() == content
        finally:
            Path(dest).unlink(missing_ok=True)


def test_download_url_cleanup_on_http_error():
    """When iter_content raises an exception, the partial file is removed."""

    def _raising_iter(**kw):
        yield b"partial"
        raise requests_lib.RequestException("network error")

    mock_response = _make_response(iter_content=_raising_iter)

    with patch("selfsuvis.pipeline.media.download.safe_request", return_value=mock_response):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            dest = f.name
        with pytest.raises(requests_lib.RequestException):
            download_url("http://example.com/video.mp4", dest, max_bytes=10_000)
        # Partial file must have been cleaned up
        assert not Path(dest).exists()
