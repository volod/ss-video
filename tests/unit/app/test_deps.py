"""Unit tests for app.deps — API key auth and rate limiting."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import selfsuvis.app.deps as deps_mod
from selfsuvis.app.deps import (
    _MAX_LIMITERS,
    _evict_oldest_limiter,
    rate_limit,
    require_api_key,
)
from selfsuvis.pipeline.core import config

# ---------------------------------------------------------------------------
# require_api_key
# ---------------------------------------------------------------------------


def test_require_api_key_passes_when_no_key_configured(monkeypatch):
    """When API_KEY is empty, require_api_key allows any request."""
    monkeypatch.setattr(config.settings, "API_KEY", "")
    monkeypatch.setattr(config.settings, "API_AUTH_REQUIRED", False)
    # Should not raise
    require_api_key(x_api_key="anything")


def test_require_api_key_accepts_correct_key(monkeypatch):
    """Correct key is accepted."""
    monkeypatch.setattr(config.settings, "API_KEY", "secret123")
    require_api_key(x_api_key="secret123")


def test_require_api_key_rejects_wrong_key(monkeypatch):
    """Wrong key raises 403."""
    monkeypatch.setattr(config.settings, "API_KEY", "secret123")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key="wrong")
    assert exc_info.value.status_code == 403


def test_require_api_key_uses_timing_safe_compare(monkeypatch):
    """keys_match (hmac.compare_digest) is used for timing-safe comparison."""
    monkeypatch.setattr(config.settings, "API_KEY", "secret")
    with patch("ss_kit.security.hmac.compare_digest", return_value=True) as mock_cmp:
        require_api_key(x_api_key="secret")
    mock_cmp.assert_called_once()


def test_require_api_key_rejects_empty_key_when_configured(monkeypatch):
    """Empty submitted key is rejected when API_KEY is set."""
    monkeypatch.setattr(config.settings, "API_KEY", "secret")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key="")
    assert exc_info.value.status_code == 403


def test_require_api_key_rejects_misconfigured_required_auth(monkeypatch):
    """Missing API_KEY is a server misconfiguration when auth is required."""
    monkeypatch.setattr(config.settings, "API_KEY", "")
    monkeypatch.setattr(config.settings, "API_AUTH_REQUIRED", True)
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key="anything")
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# rate limiter — bounded table
# ---------------------------------------------------------------------------


def test_rate_limit_table_bounded(monkeypatch):
    """_limiters never grows beyond _MAX_LIMITERS entries."""
    monkeypatch.setattr(config.settings, "RATE_LIMIT_PER_MIN", 60)
    monkeypatch.setattr(config.settings, "RATE_LIMIT_BURST", 10)

    # Clear the shared dict for a clean test
    deps_mod._limiters.clear()

    for i in range(_MAX_LIMITERS + 5):
        req = MagicMock()
        req.client = MagicMock()
        req.client.host = f"10.0.{i // 256}.{i % 256}"
        monkeypatch.setattr(config.settings, "TRUST_PROXY_HEADERS", False)
        rate_limit(req)

    assert len(deps_mod._limiters) == _MAX_LIMITERS

    deps_mod._limiters.clear()


def test_evict_oldest_limiter_removes_oldest():
    """_evict_oldest_limiter removes the first-inserted key."""
    deps_mod._limiters.clear()
    # Fill to capacity
    for i in range(_MAX_LIMITERS):
        deps_mod._limiters[f"ip_{i}"] = MagicMock()
    _evict_oldest_limiter()
    assert len(deps_mod._limiters) == _MAX_LIMITERS - 1
    assert "ip_0" not in deps_mod._limiters
    deps_mod._limiters.clear()
