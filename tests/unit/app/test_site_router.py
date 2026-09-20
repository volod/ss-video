"""GET /site/cameras on the video API."""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from selfsuvis.app.routers.site import router


def test_site_cameras_lists_sessions() -> None:
    app = FastAPI()
    app.include_router(router)
    streams = MagicMock()
    streams.active_cameras.return_value = [
        {
            "camera": "entrance",
            "started_at": "2026-09-19T08:00:00+00:00",
            "session_id": "coop-entrance",
            "rtsp_url": "rtsp://frigate:8554/entrance",
        }
    ]
    app.state.camera_streams = streams
    with patch("selfsuvis.app.deps.settings") as ms:
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        client = TestClient(app)
        cameras = client.get("/site/cameras")
    assert cameras.status_code == 200
    body = cameras.json()
    assert body["camera_count"] == 1
    assert body["cameras"][0]["camera"] == "entrance"


def test_site_cameras_empty_without_service() -> None:
    app = FastAPI()
    app.include_router(router)
    with patch("selfsuvis.app.deps.settings") as ms:
        ms.API_KEY = ""
        ms.API_AUTH_REQUIRED = False
        client = TestClient(app)
        cameras = client.get("/site/cameras")
    assert cameras.status_code == 200
    assert cameras.json()["camera_count"] == 0
