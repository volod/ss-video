"""Unit tests for app.routers.cvat — webhook receiver and admin endpoints."""

import asyncio
import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from selfsuvis.app.routers.cvat import (
    CvatTaskRegistration,
    _verify_cvat_signature,
    cvat_annotation_frames,
    cvat_webhook,
    register_cvat_task,
)

_TEST_SECRET = "test-webhook-secret"


class _Request:
    def __init__(self, body: bytes = b"", db_pool: Any = None):
        self._body = body
        self.client = SimpleNamespace(host="test")
        self.app = SimpleNamespace(state=SimpleNamespace(db_pool=db_pool))

    async def body(self) -> bytes:
        return self._body


def _sign(body: bytes, secret: str = _TEST_SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _webhook_payload(event: str, state: str, task_id: int = 1, job_id: int = 10) -> dict[str, Any]:
    if event == "update:job":
        return {"event": event, "job": {"id": job_id, "task_id": task_id, "state": state}}
    if event == "update:task":
        return {"event": event, "task": {"id": task_id, "state": state}}
    return {"event": event}


def run(coro):
    return asyncio.run(coro)


# ── _verify_cvat_signature unit tests ─────────────────────────────────────────


def test_verify_signature_no_secret_rejects():
    """Fail-closed: when CVAT_WEBHOOK_SECRET is empty, reject all requests."""
    with patch("selfsuvis.app.routers.cvat.settings") as m:
        m.CVAT_WEBHOOK_SECRET = ""
        assert _verify_cvat_signature(b"body", "anything") is False


def test_verify_signature_correct():
    secret = "mysecret"
    body = b'{"event":"update:job"}'
    sig = _sign(body, secret)
    with patch("selfsuvis.app.routers.cvat.settings") as m:
        m.CVAT_WEBHOOK_SECRET = secret
        assert _verify_cvat_signature(body, sig) is True


def test_verify_signature_wrong():
    secret = "mysecret"
    body = b'{"event":"update:job"}'
    with patch("selfsuvis.app.routers.cvat.settings") as m:
        m.CVAT_WEBHOOK_SECRET = secret
        assert _verify_cvat_signature(body, "badsignature") is False


# ── cvat_webhook integration tests ────────────────────────────────────────────


@patch("selfsuvis.app.routers.cvat._fetch_cvat_labels", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat._mark_frames_annotated", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat._frames_for_cvat_task", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat._maybe_trigger_finetune", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat.settings")
def test_webhook_job_completed_marks_frames(
    mock_settings, mock_trigger, mock_frames, mock_mark, mock_fetch
):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    mock_frames.return_value = ["f1", "f2", "f3"]
    mock_mark.return_value = 3
    mock_fetch.return_value = {}
    db_pool = object()

    body = json.dumps(_webhook_payload("update:job", "completed", task_id=5)).encode()
    data = run(cvat_webhook(_Request(body, db_pool=db_pool), x_hook_secret=_sign(body)))

    assert data["annotated"] == 3
    mock_frames.assert_awaited_once_with(5, db_pool)
    mock_fetch.assert_awaited_once_with(5)
    mock_mark.assert_awaited_once_with(["f1", "f2", "f3"], db_pool, {})
    mock_trigger.assert_awaited_once_with(db_pool)


@patch("selfsuvis.app.routers.cvat._fetch_cvat_labels", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat._mark_frames_annotated", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat._frames_for_cvat_task", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat._maybe_trigger_finetune", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat.settings")
def test_webhook_task_completed_marks_frames(
    mock_settings, mock_trigger, mock_frames, mock_mark, mock_fetch
):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    mock_frames.return_value = ["fa", "fb"]
    mock_mark.return_value = 2
    mock_fetch.return_value = {}
    db_pool = object()

    body = json.dumps(_webhook_payload("update:task", "completed", task_id=7)).encode()
    data = run(cvat_webhook(_Request(body, db_pool=db_pool), x_hook_secret=_sign(body)))

    assert data["annotated"] == 2
    mock_frames.assert_awaited_once_with(7, db_pool)
    mock_trigger.assert_awaited_once_with(db_pool)


@patch("selfsuvis.app.routers.cvat._frames_for_cvat_task", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat.settings")
def test_webhook_job_not_completed_returns_zero(mock_settings, mock_frames):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    mock_frames.return_value = []

    body = json.dumps(_webhook_payload("update:job", "in_progress", task_id=3)).encode()
    data = run(cvat_webhook(_Request(body), x_hook_secret=_sign(body)))

    assert data["annotated"] == 0
    mock_frames.assert_not_awaited()


@patch("selfsuvis.app.routers.cvat.settings")
def test_webhook_unknown_event_returns_ok(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    body = json.dumps({"event": "create:project"}).encode()
    data = run(cvat_webhook(_Request(body), x_hook_secret=_sign(body)))
    assert data["annotated"] == 0


@patch("selfsuvis.app.routers.cvat.settings")
def test_webhook_invalid_signature_returns_400(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = "secret123"
    body = json.dumps(_webhook_payload("update:job", "completed")).encode()
    with pytest.raises(HTTPException) as exc:
        run(cvat_webhook(_Request(body), x_hook_secret="wrongsig"))
    assert exc.value.status_code == 400


@patch("selfsuvis.app.routers.cvat.settings")
def test_webhook_no_secret_configured_returns_400(mock_settings):
    """Fail-closed: webhook must reject requests when no secret is configured."""
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    body = json.dumps(_webhook_payload("update:job", "completed")).encode()
    with pytest.raises(HTTPException) as exc:
        run(cvat_webhook(_Request(body), x_hook_secret=""))
    assert exc.value.status_code == 400


@patch("selfsuvis.app.routers.cvat._fetch_cvat_labels", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat._mark_frames_annotated", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat._frames_for_cvat_task", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat._maybe_trigger_finetune", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat.settings")
def test_webhook_valid_signature_passes(
    mock_settings, mock_trigger, mock_frames, mock_mark, mock_fetch
):
    secret = "supersecret"
    mock_settings.CVAT_WEBHOOK_SECRET = secret
    mock_frames.return_value = ["f1"]
    mock_mark.return_value = 1
    mock_fetch.return_value = {}
    db_pool = object()

    body = json.dumps(_webhook_payload("update:job", "completed", task_id=2)).encode()
    sig = _sign(body, secret)
    data = run(cvat_webhook(_Request(body, db_pool=db_pool), x_hook_secret=sig))

    assert data["annotated"] == 1
    mock_trigger.assert_awaited_once()


@patch("selfsuvis.app.routers.cvat.settings")
def test_webhook_invalid_json_returns_400(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    body = b"not-json"
    with pytest.raises(HTTPException) as exc:
        run(cvat_webhook(_Request(body), x_hook_secret=_sign(body)))
    assert exc.value.status_code == 400


@patch("selfsuvis.app.routers.cvat._frames_for_cvat_task", new_callable=AsyncMock)
@patch("selfsuvis.app.routers.cvat.settings")
def test_webhook_no_mapping_returns_zero(mock_settings, mock_frames):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    mock_frames.return_value = []
    db_pool = object()

    body = json.dumps(_webhook_payload("update:job", "completed", task_id=99)).encode()
    data = run(cvat_webhook(_Request(body, db_pool=db_pool), x_hook_secret=_sign(body)))
    assert data["annotated"] == 0


@patch("selfsuvis.app.routers.cvat.settings")
def test_register_task_empty_frame_ids_returns_422(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    with pytest.raises(HTTPException) as exc:
        run(register_cvat_task(CvatTaskRegistration(cvat_task_id=1, frame_ids=[]), _Request()))
    assert exc.value.status_code == 422


@patch("selfsuvis.app.routers.cvat.settings")
def test_register_task_too_many_frames_returns_422(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    with pytest.raises(HTTPException) as exc:
        run(
            register_cvat_task(
                CvatTaskRegistration(cvat_task_id=1, frame_ids=[f"f{i}" for i in range(5001)]),
                _Request(),
            )
        )
    assert exc.value.status_code == 422


@patch("selfsuvis.app.routers.cvat.settings")
def test_register_task_no_db_returns_503(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    mock_settings.API_KEY = ""
    mock_settings.DATABASE_URL = ""
    with pytest.raises(HTTPException) as exc:
        run(
            register_cvat_task(
                CvatTaskRegistration(cvat_task_id=1, frame_ids=["f1", "f2"]), _Request()
            )
        )
    assert exc.value.status_code == 503


@patch("selfsuvis.app.routers.cvat.settings")
def test_frames_no_db_returns_empty(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    mock_settings.API_KEY = ""
    mock_settings.DATABASE_URL = ""
    with pytest.raises(HTTPException) as exc:
        run(cvat_annotation_frames(_Request()))
    assert exc.value.status_code == 503


@patch("selfsuvis.app.routers.cvat.settings")
def test_frames_invalid_al_tag_returns_422(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    mock_settings.API_KEY = ""
    mock_settings.DATABASE_URL = "postgresql://x"
    with pytest.raises(HTTPException) as exc:
        run(cvat_annotation_frames(_Request(), al_tag="invalid"))
    assert exc.value.status_code == 422


@patch("selfsuvis.app.routers.cvat.settings")
def test_frames_invalid_limit_returns_422(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = _TEST_SECRET
    mock_settings.API_KEY = ""
    mock_settings.DATABASE_URL = "postgresql://x"
    with pytest.raises(HTTPException) as exc:
        run(cvat_annotation_frames(_Request(), limit=0))
    assert exc.value.status_code == 422
