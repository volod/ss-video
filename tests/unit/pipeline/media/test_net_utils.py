"""Unit tests for pipeline.net_utils — URL validation and SSRF protection."""

import ipaddress
from unittest.mock import MagicMock, patch

import pytest

from selfsuvis.pipeline.core import config
from selfsuvis.pipeline.media.network import _is_ip_allowed, safe_request, validate_url

# ---------------------------------------------------------------------------
# validate_url
# ---------------------------------------------------------------------------


def test_validate_url_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="scheme"):
        validate_url("ftp://example.com/file")


def test_validate_url_rejects_credentials():
    with pytest.raises(ValueError, match="credentials"):
        validate_url("http://user:pass@example.com/")


def test_validate_url_rejects_single_label_hostname(monkeypatch):
    monkeypatch.setattr(config.settings, "ALLOW_PRIVATE_URLS", False)
    with pytest.raises(ValueError):
        validate_url("http://localhost/")


def test_validate_url_rejects_private_ip(monkeypatch):
    monkeypatch.setattr(config.settings, "ALLOW_PRIVATE_URLS", False)
    with patch(
        "selfsuvis.pipeline.media.network._iter_resolved_ips",
        return_value=[ipaddress.ip_address("192.168.1.1")],
    ):
        with pytest.raises(ValueError, match="not allowed"):
            validate_url("http://example.com/")


# ---------------------------------------------------------------------------
# _is_ip_allowed
# ---------------------------------------------------------------------------


def test_is_ip_allowed_blocks_private(monkeypatch):
    monkeypatch.setattr(config.settings, "ALLOW_PRIVATE_URLS", False)
    assert not _is_ip_allowed(ipaddress.ip_address("192.168.0.1"))
    assert not _is_ip_allowed(ipaddress.ip_address("127.0.0.1"))
    assert not _is_ip_allowed(ipaddress.ip_address("169.254.1.1"))


def test_is_ip_allowed_passes_public(monkeypatch):
    monkeypatch.setattr(config.settings, "ALLOW_PRIVATE_URLS", False)
    assert _is_ip_allowed(ipaddress.ip_address("8.8.8.8"))


def test_is_ip_allowed_passes_all_when_allow_private(monkeypatch):
    monkeypatch.setattr(config.settings, "ALLOW_PRIVATE_URLS", True)
    assert _is_ip_allowed(ipaddress.ip_address("127.0.0.1"))
    assert _is_ip_allowed(ipaddress.ip_address("10.0.0.1"))


# ---------------------------------------------------------------------------
# Post-connect IP validation (DNS rebinding mitigation)
# ---------------------------------------------------------------------------


def _make_mock_response(is_redirect=False):
    resp = MagicMock()
    resp.is_redirect = is_redirect
    resp.is_permanent_redirect = False
    resp.close = MagicMock()
    return resp


def test_post_connect_ip_validation_blocks_private_peer(monkeypatch):
    """safe_request raises ValueError when peer IP is private (DNS rebinding scenario)."""
    monkeypatch.setattr(config.settings, "ALLOW_PRIVATE_URLS", False)
    monkeypatch.setattr(config.settings, "MAX_REDIRECTS", 0)

    mock_resp = _make_mock_response()

    with patch("selfsuvis.pipeline.media.network.validate_url"):
        with patch(
            "selfsuvis.pipeline.media.network._peer_ip",
            return_value=ipaddress.ip_address("192.168.1.1"),
        ):
            with patch("requests.Session.request", return_value=mock_resp):
                with pytest.raises(ValueError, match="Post-connect IP validation failed"):
                    safe_request("GET", "http://example.com/", timeout=5)


def test_post_connect_ip_validation_passes_public_peer(monkeypatch):
    """safe_request succeeds when peer IP is public."""
    monkeypatch.setattr(config.settings, "ALLOW_PRIVATE_URLS", False)
    monkeypatch.setattr(config.settings, "MAX_REDIRECTS", 0)

    mock_resp = _make_mock_response()

    with patch("selfsuvis.pipeline.media.network.validate_url"):
        with patch(
            "selfsuvis.pipeline.media.network._peer_ip",
            return_value=ipaddress.ip_address("8.8.8.8"),
        ):
            with patch("requests.Session.request", return_value=mock_resp):
                result = safe_request("GET", "http://example.com/", timeout=5)
    assert result is mock_resp


def test_post_connect_ip_validation_skipped_when_peer_ip_unavailable(monkeypatch):
    """If _peer_ip returns None (e.g. non-socket transport), no error is raised."""
    monkeypatch.setattr(config.settings, "ALLOW_PRIVATE_URLS", False)
    monkeypatch.setattr(config.settings, "MAX_REDIRECTS", 0)

    mock_resp = _make_mock_response()

    with patch("selfsuvis.pipeline.media.network.validate_url"):
        with patch("selfsuvis.pipeline.media.network._peer_ip", return_value=None):
            with patch("requests.Session.request", return_value=mock_resp):
                result = safe_request("GET", "http://example.com/", timeout=5)
    assert result is mock_resp
