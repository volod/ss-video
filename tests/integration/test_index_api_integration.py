"""Integration tests for the index API (app/routers/index.py).

Exercises the full HTTP → job-enqueue path using TestClient + mock DB pool.
No live PostgreSQL, Qdrant, or file I/O beyond tmp_path fixtures.

Covers:
- POST /index/video  — multipart upload creates a pending job
- POST /index/url    — URL job carries video_url in payload
- POST /index/rtsp   — RTSP job sets ingest_mode='rtsp'
- POST /index/dir    — directory scan creates one job per video file
- POST /index/precheck       — returns duplicate/new based on file hash
- POST /index/precheck_dir   — batch precheck + optional enqueue
- Validation errors (bad extension, missing params, disallowed path)
"""

import io
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Stub app.state before any router import (avoids torchvision circular init)
_state_stub = MagicMock()
_state_stub.clip_model = MagicMock()
_state_stub.qdrant_store = MagicMock()
_state_stub.dino_model = None
sys.modules.setdefault("selfsuvis.app.state", _state_stub)


# ── App fixture ───────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient for the index router with auth bypassed and DB mocked."""
    from fastapi import FastAPI

    from selfsuvis.app.deps import rate_limit, require_api_key
    from selfsuvis.app.routers.index import router

    app = FastAPI()

    async def _no_auth():
        return "test-key"

    async def _no_rate():
        pass

    app.dependency_overrides[require_api_key] = _no_auth
    app.dependency_overrides[rate_limit] = _no_rate
    app.include_router(router)

    # Point VIDEOS_DIR at tmp_path so uploads land somewhere writable
    import selfsuvis.pipeline.core.config as cfg

    monkeypatch.setattr(cfg.settings, "VIDEOS_DIR", str(tmp_path))
    monkeypatch.setattr(cfg.settings, "ALLOWED_INDEX_PATHS", str(tmp_path))
    monkeypatch.setattr(cfg.settings, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)

    return TestClient(app)


def _make_job_create_mock():
    """Return (mock_pool, recorded_jobs list) pair."""
    jobs = []

    async def _fake_create_job(conn, job_id, payload, job_type="index"):
        jobs.append({"id": job_id, "payload": payload, "type": job_type})

    pool = MagicMock()
    conn_ctx = AsyncMock()
    conn_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
    conn_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = conn_ctx

    return pool, jobs, _fake_create_job


# ── POST /index/video ─────────────────────────────────────────────────────────


def test_upload_video_creates_pending_job(client, tmp_path):
    """Uploading a .mp4 file creates a pending job and returns job_id."""
    fake_video = b"\x00" * 64  # minimal fake mp4 bytes

    with (
        patch("selfsuvis.app.routers.index.create_job") as mock_cj,
        patch("selfsuvis.app.routers.index.get_db_pool") as mock_pool,
    ):
        mock_pool.return_value = AsyncMock()
        mock_cj.return_value = None

        resp = client.post(
            "/index/video",
            files={"file": ("test_video.mp4", io.BytesIO(fake_video), "video/mp4")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert "video_id" in data
    mock_cj.assert_called_once()
    payload_arg = mock_cj.call_args[0][2]  # third positional arg: payload dict
    assert "video_id" in payload_arg


def test_upload_video_rejected_for_non_video_extension(client):
    """Uploading a .txt file returns 400."""
    resp = client.post(
        "/index/video",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_video_no_file_no_path_returns_400(client):
    """No file and no path → 400."""
    resp = client.post("/index/video")
    assert resp.status_code in (400, 422)


# ── POST /index/url ───────────────────────────────────────────────────────────


def test_index_url_creates_job_with_url_in_payload(client):
    """POST /index/url enqueues a job whose payload contains video_url."""
    with (
        patch("selfsuvis.app.routers.index.create_job") as mock_cj,
        patch("selfsuvis.app.routers.index.get_db_pool") as mock_pool,
    ):
        mock_pool.return_value = AsyncMock()
        mock_cj.return_value = None

        resp = client.post(
            "/index/url",
            data={"stream_url": "https://example.com/clip.mp4"},
        )

    assert resp.status_code == 200
    payload = mock_cj.call_args[0][2]
    assert "video_url" in payload
    assert "clip.mp4" in payload["video_url"] or "example.com" in payload["video_url"]


def test_index_url_with_credentials_rejected(client):
    """URL containing basic-auth credentials must be rejected."""
    resp = client.post(
        "/index/url",
        data={"stream_url": "https://user:pass@example.com/clip.mp4"},
    )
    assert resp.status_code == 400


def test_index_url_no_stream_url_returns_422(client):
    resp = client.post("/index/url", data={})
    assert resp.status_code == 422


# ── POST /index/rtsp ──────────────────────────────────────────────────────────


def test_index_rtsp_sets_ingest_mode_in_payload(client):
    """RTSP job payload has ingest_mode='rtsp'."""
    with (
        patch("selfsuvis.app.routers.index.create_job") as mock_cj,
        patch("selfsuvis.app.routers.index.get_db_pool") as mock_pool,
    ):
        mock_pool.return_value = AsyncMock()
        mock_cj.return_value = None

        resp = client.post(
            "/index/rtsp",
            data={"stream_url": "rtsp://cam.local:8554/live"},
        )

    assert resp.status_code == 200
    payload = mock_cj.call_args[0][2]
    assert payload.get("ingest_mode") == "rtsp"
    assert "video_url" in payload


def test_index_rtsp_duration_propagated_to_payload(client):
    """Optional duration_sec is forwarded into the job payload."""
    with (
        patch("selfsuvis.app.routers.index.create_job") as mock_cj,
        patch("selfsuvis.app.routers.index.get_db_pool") as mock_pool,
    ):
        mock_pool.return_value = AsyncMock()
        mock_cj.return_value = None

        resp = client.post(
            "/index/rtsp",
            data={"stream_url": "rtsp://cam.local:8554/live", "duration_sec": "60"},
        )

    assert resp.status_code == 200
    payload = mock_cj.call_args[0][2]
    assert payload.get("duration_sec") == pytest.approx(60.0)


def test_index_rtsp_mission_id_propagated(client):
    """Explicit mission_id is forwarded into the job payload."""
    with (
        patch("selfsuvis.app.routers.index.create_job") as mock_cj,
        patch("selfsuvis.app.routers.index.get_db_pool") as mock_pool,
    ):
        mock_pool.return_value = AsyncMock()
        mock_cj.return_value = None

        resp = client.post(
            "/index/rtsp",
            data={"stream_url": "rtsp://cam.local:8554/live", "mission_id": "patrol-001"},
        )

    assert resp.status_code == 200
    payload = mock_cj.call_args[0][2]
    assert payload.get("mission_id") == "patrol-001"


# ── POST /index/dir ───────────────────────────────────────────────────────────


def test_index_dir_creates_one_job_per_video(client, tmp_path, monkeypatch):
    """Each video file in the allowed directory gets its own job."""
    import selfsuvis.pipeline.core.config as cfg

    monkeypatch.setattr(cfg.settings, "ALLOWED_INDEX_PATHS", str(tmp_path))

    (tmp_path / "a.mp4").write_bytes(b"\x00" * 16)
    (tmp_path / "b.mp4").write_bytes(b"\x00" * 16)
    (tmp_path / "notes.txt").write_text("ignore me")

    created_jobs = []

    async def _fake_cj(conn, job_id, payload, job_type="index"):
        created_jobs.append(payload)

    with (
        patch("selfsuvis.app.routers.index.create_job", side_effect=_fake_cj),
        patch("selfsuvis.app.routers.index.get_db_pool") as mock_pool,
    ):
        mock_pool.return_value = AsyncMock()

        resp = client.post("/index/dir", data={"dir_path": str(tmp_path)})

    assert resp.status_code == 200
    data = resp.json()
    assert "jobs" in data
    assert len(data["jobs"]) == 2


def test_index_dir_disallowed_path_returns_403(client, tmp_path, monkeypatch, tmp_path_factory):
    """A path outside ALLOWED_INDEX_PATHS returns 403."""
    import selfsuvis.pipeline.core.config as cfg

    allowed = tmp_path_factory.mktemp("allowed")
    forbidden = tmp_path_factory.mktemp("forbidden")
    monkeypatch.setattr(cfg.settings, "ALLOWED_INDEX_PATHS", str(allowed))

    resp = client.post("/index/dir", data={"dir_path": str(forbidden)})
    assert resp.status_code == 403


# ── POST /index/precheck ──────────────────────────────────────────────────────


def test_precheck_known_file_returns_duplicate(client, tmp_path):
    """A file with a hash that's already processed returns duplicate status."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\xab" * 64)

    # Simulate the file being in the processed_files cache
    with patch(
        "selfsuvis.app.routers.index.get_by_hash",
        return_value={"video_id": "existing-v1", "status": "processed"},
    ):
        resp = client.post("/index/precheck", data={"file_path": str(video)})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "duplicate"


def test_precheck_new_file_returns_new(client, tmp_path):
    """A file not yet indexed returns 'new' status."""
    video = tmp_path / "fresh.mp4"
    video.write_bytes(b"\xcc" * 64)

    with patch("selfsuvis.app.routers.index.get_by_hash", return_value=None):
        resp = client.post("/index/precheck", data={"file_path": str(video)})

    assert resp.status_code == 200
    assert resp.json()["status"] == "new"


# ── POST /index/precheck_dir ──────────────────────────────────────────────────


def test_precheck_dir_returns_status_per_file(client, tmp_path, monkeypatch):
    """precheck_dir scans the directory and returns one result per video."""
    import selfsuvis.pipeline.core.config as cfg

    monkeypatch.setattr(cfg.settings, "ALLOWED_INDEX_PATHS", str(tmp_path))

    (tmp_path / "a.mp4").write_bytes(b"\x00" * 32)
    (tmp_path / "b.mp4").write_bytes(b"\x11" * 32)

    with patch("selfsuvis.app.routers.index.get_by_hash", return_value=None):
        resp = client.post(
            "/index/precheck_dir",
            data={"dir_path": str(tmp_path), "enqueue": "false"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 2


def test_precheck_dir_enqueue_creates_jobs_for_new_files(client, tmp_path, monkeypatch):
    """precheck_dir with enqueue=true creates jobs for 'new' files."""
    import selfsuvis.pipeline.core.config as cfg

    monkeypatch.setattr(cfg.settings, "ALLOWED_INDEX_PATHS", str(tmp_path))

    (tmp_path / "new_clip.mp4").write_bytes(b"\x22" * 32)

    created_jobs = []

    async def _fake_cj(conn, job_id, payload, job_type="index"):
        created_jobs.append(job_id)

    with (
        patch("selfsuvis.app.routers.index.get_by_hash", return_value=None),
        patch("selfsuvis.app.routers.index.create_job", side_effect=_fake_cj),
        patch("selfsuvis.app.routers.index.get_db_pool") as mock_pool,
    ):
        mock_pool.return_value = AsyncMock()

        resp = client.post(
            "/index/precheck_dir",
            data={"dir_path": str(tmp_path), "enqueue": "true"},
        )

    assert resp.status_code == 200
    assert len(created_jobs) == 1
