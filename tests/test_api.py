import logging
import os
import time

import pytest
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
ASSETS_DIR = os.getenv("ASSETS_DIR", os.path.join(os.path.dirname(__file__), "assets"))
INDEX_DIR_PATH = os.getenv("INDEX_DIR_PATH")
RUN_API_TESTS = os.getenv("RUN_API_TESTS", "").lower() in {"1", "true", "yes"}

if not RUN_API_TESTS:
    pytest.skip(
        "API integration tests disabled; set RUN_API_TESTS=1 to run them", allow_module_level=True
    )

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


def _wait_job(job_id: str, timeout_sec: int = 300):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        resp = requests.get(f"{API_URL}/jobs/{job_id}")
        resp.raise_for_status()
        status = resp.json().get("status")
        if status in {"finished", "error"}:
            return resp.json()
        time.sleep(2)
    raise TimeoutError(f"job timeout: {job_id}")


def _wait_api(timeout_sec: int = 120) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = requests.get(f"{API_URL}/docs")
            if resp.status_code in {200, 404}:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


@pytest.fixture(scope="session", autouse=True)
def wait_for_api():
    if not _wait_api():
        pytest.skip("API not reachable")


def test_index_video_and_query_text():
    green_video = os.path.join(ASSETS_DIR, "vid_green.mp4")
    assert os.path.exists(green_video)

    with open(green_video, "rb") as f:
        resp = requests.post(
            f"{API_URL}/index/video",
            files={"file": ("vid_green.mp4", f, "video/mp4")},
            data={"enable_tiles": "false"},
        )
    resp.raise_for_status()
    job = resp.json()
    job_status = _wait_job(job["job_id"])
    assert job_status["status"] == "finished"
    assert not (job_status.get("progress") or {}).get("skipped")

    resp = requests.post(
        f"{API_URL}/query/text",
        json={"text": "green field"},
        params={"top_k": 5, "search_type": "frame", "enable_rerank": False},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    assert isinstance(results, list)


def test_index_video_and_query_image():
    test_video = os.path.join(ASSETS_DIR, "vid_testsrc.mp4")
    query_img = os.path.join(ASSETS_DIR, "green.png")
    assert os.path.exists(test_video)
    assert os.path.exists(query_img)

    with open(test_video, "rb") as f:
        resp = requests.post(
            f"{API_URL}/index/video",
            files={"file": ("vid_testsrc.mp4", f, "video/mp4")},
            data={"enable_tiles": "true"},
        )
    resp.raise_for_status()
    job = resp.json()
    job_status = _wait_job(job["job_id"])
    assert job_status["status"] == "finished"
    assert not (job_status.get("progress") or {}).get("skipped")

    with open(query_img, "rb") as f:
        resp = requests.post(
            f"{API_URL}/query/image",
            files={"file": ("green.png", f, "image/png")},
            data={
                "top_k": "5",
                "search_type": "both",
                "vector_space": "clip",
                "enable_rerank": "false",
            },
        )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    assert len(results) > 0


def test_precheck_dir_optional_enqueue():
    if not INDEX_DIR_PATH:
        pytest.skip("INDEX_DIR_PATH not set")
    resp = requests.post(
        f"{API_URL}/index/precheck_dir",
        data={"path": INDEX_DIR_PATH, "enqueue": "false"},
    )
    resp.raise_for_status()
    data = resp.json()
    assert "results" in data


def test_index_dir_optional():
    if not INDEX_DIR_PATH:
        pytest.skip("INDEX_DIR_PATH not set")
    resp = requests.post(
        f"{API_URL}/index/dir",
        data={"path": INDEX_DIR_PATH, "enable_tiles": "false"},
    )
    resp.raise_for_status()
    data = resp.json()
    assert "jobs" in data


# --- Tests for refactored functionality ---


def test_health_endpoint():
    """GET /health returns 200 with status ok when Qdrant is connected."""
    resp = requests.get(f"{API_URL}/health")
    resp.raise_for_status()
    data = resp.json()
    assert data["status"] in {"ok", "degraded"}
    assert "qdrant" in data


def test_query_text_missing_body():
    """POST /query/text returns 422 when body is missing or invalid."""
    resp = requests.post(f"{API_URL}/query/text", json={})
    assert resp.status_code == 422


def test_query_text_empty_text():
    """POST /query/text returns 422 when text is empty (min_length=1)."""
    resp = requests.post(f"{API_URL}/query/text", json={"text": ""})
    assert resp.status_code == 422


def test_query_text_invalid_search_type():
    """POST /query/text returns 400 for invalid search_type."""
    resp = requests.post(
        f"{API_URL}/query/text",
        json={"text": "green"},
        params={"search_type": "invalid"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_query_text_invalid_top_k():
    """POST /query/text returns 422 for top_k out of range."""
    resp = requests.post(
        f"{API_URL}/query/text",
        json={"text": "green"},
        params={"top_k": 999},
    )
    assert resp.status_code == 422


def test_query_image_invalid_search_type():
    """POST /query/image returns 400 for invalid search_type."""
    resp = requests.post(
        f"{API_URL}/query/image",
        files={"file": ("img.png", b"x" * 100, "image/png")},
        data={"search_type": "invalid"},
    )
    assert resp.status_code == 400


def test_query_image_invalid_vector_space():
    """POST /query/image returns 400 for invalid vector_space."""
    resp = requests.post(
        f"{API_URL}/query/image",
        files={"file": ("img.png", b"x" * 100, "image/png")},
        data={"vector_space": "invalid"},
    )
    assert resp.status_code == 400


def test_index_video_no_file_no_path():
    """POST /index/video returns 400 when neither file nor path provided."""
    resp = requests.post(f"{API_URL}/index/video", data={})
    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data


def test_job_status_not_found():
    """GET /jobs/{id} returns 404 for unknown job."""
    resp = requests.get(f"{API_URL}/jobs/0123456789abcdef0123456789abcdef")
    assert resp.status_code == 404
    assert "error" in resp.json()


def test_job_status_invalid_id_returns_400():
    """GET /jobs/{id} returns 400 for invalid job_id (non-hex or too long)."""
    resp = requests.get(f"{API_URL}/jobs/invalid-job-id-with-dashes")
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_upload_size_limit_exceeded():
    """POST /index/video returns 413 when upload exceeds MAX_UPLOAD_BYTES."""
    # MAX_UPLOAD_BYTES=150000 in docker-compose.test.yml (allows test videos ~14KB)
    large_content = b"x" * 200000
    resp = requests.post(
        f"{API_URL}/index/video",
        files={"file": ("large.mp4", large_content, "video/mp4")},
        data={"enable_tiles": "false"},
    )
    assert resp.status_code == 413
    assert "error" in resp.json()


def test_path_not_allowed_returns_403():
    """POST /index/dir returns 403 when path is outside ALLOWED_INDEX_PATHS."""
    # ALLOWED_INDEX_PATHS=/app/tests/assets in docker-compose.test.yml
    resp = requests.post(
        f"{API_URL}/index/dir",
        data={"path": "/etc", "enable_tiles": "false"},
    )
    assert resp.status_code == 403
    assert "error" in resp.json()
