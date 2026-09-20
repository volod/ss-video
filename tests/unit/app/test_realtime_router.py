"""Unit tests for app/routers/realtime.py."""

from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.app.routers.realtime import router
from tests.support.realtime_db import FakeRealtimeConn as FakeConn
from tests.support.realtime_db import FakeRealtimePool as FakePool


class FakeMediaMtxClient:
    def __init__(self):
        self.created = []
        self.deleted = []

    async def ensure_path(self, path_name: str, *, source_url=None, source_on_demand=False):
        self.created.append(
            {
                "path_name": path_name,
                "source_url": source_url,
                "source_on_demand": source_on_demand,
            }
        )
        return True

    async def list_paths(self):
        return [{"name": "live/drone-a", "ready": True, "bytesReceived": 1234}]

    async def delete_path(self, path_name: str):
        self.deleted.append(path_name)
        return True


class FakeRealtimeStreamManager:
    def __init__(self):
        self.streams: dict[str, dict[str, Any]] = {}

    async def start(self, *, session_id, mission_id, robot_id, path_name, caption_fps=None):
        data = {
            "session_id": session_id,
            "mission_id": mission_id,
            "robot_id": robot_id,
            "path_name": path_name,
            "rtsp_url": f"rtsp://mediamtx:8554/{path_name}",
            "caption_fps": float(caption_fps or 0.5),
            "started_at": "2026-04-25T12:00:00+00:00",
            "status": "running",
            "error": None,
        }
        self.streams[session_id] = data
        return data

    async def list(self):
        return list(self.streams.values())

    async def get(self, session_id: str):
        if session_id not in self.streams:
            raise LookupError("live stream not found")
        return self.streams[session_id]

    async def stop(self, session_id: str):
        stream = await self.get(session_id)
        stream["status"] = "stopped"
        return stream


def _app():
    app = FastAPI()

    async def _no_auth():
        return "test-key"

    async def _no_rate():
        return None

    app.dependency_overrides[require_api_key] = _no_auth
    app.dependency_overrides[rate_limit] = _no_rate
    app.include_router(router)
    return app


@pytest.mark.anyio
async def test_start_state_stop_session():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post(
                "/realtime/session/start",
                json={
                    "robot_id": "drone_a",
                    "mission_id": "mission_1",
                    "sensors": ["camera", "imu", "gps"],
                },
            )
            assert start.status_code == 200
            session_id = start.json()["session_id"]

            state = await client.get(f"/realtime/session/{session_id}/state")
            assert state.status_code == 200
            assert state.json()["status"] == "active"
            assert state.json()["packet_counts"] == {}
            assert state.json()["latest_pose"] is None

            stop = await client.post(f"/realtime/session/{session_id}/stop")
            assert stop.status_code == 200
            assert stop.json()["status"] == "stopped"


@pytest.mark.anyio
async def test_packet_ingest_creates_stub_pose_from_gps():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post("/realtime/session/start", json={"robot_id": "drone_b"})
            session_id = start.json()["session_id"]

            ingest = await client.post(
                f"/realtime/session/{session_id}/packet",
                json={
                    "packets": [
                        {
                            "sensor_type": "gps",
                            "t_device": 12.5,
                            "seq": 7,
                            "payload": {
                                "east": 1.25,
                                "north": -0.5,
                                "up": 10.0,
                                "global_map_id": 3,
                            },
                        },
                        {
                            "sensor_type": "imu",
                            "t_device": 12.52,
                            "seq": 8,
                            "payload": {"yaw": 0.1, "vx": 0.2, "vy": 0.0, "vz": 0.0},
                        },
                    ]
                },
            )
            assert ingest.status_code == 200
            assert ingest.json()["accepted_packets"] == 2
            assert ingest.json()["pose_updated"] is True
            assert ingest.json()["packet_summary"] == {"gps": 1, "imu": 1}

            pose = await client.get(f"/realtime/session/{session_id}/pose/latest")
            assert pose.status_code == 200
            data = pose.json()
            assert data["source"] == "fused_gps_imu"
            assert data["position_enu"] == {"x": 1.25, "y": -0.5, "z": 10.0}
            assert data["tracking_status"] == "ok"
            assert data["global_map_id"] == 3

            state = await client.get(f"/realtime/session/{session_id}/state")
            assert state.status_code == 200
            assert state.json()["packet_counts"] == {"gps": 1, "imu": 1}


@pytest.mark.anyio
async def test_latest_pose_404_when_missing():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post("/realtime/session/start", json={"robot_id": "drone_c"})
            session_id = start.json()["session_id"]
            pose = await client.get(f"/realtime/session/{session_id}/pose/latest")
            assert pose.status_code == 404


@pytest.mark.anyio
async def test_publish_map_tile_and_semantic_observation():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post("/realtime/session/start", json={"robot_id": "drone_d"})
            session_id = start.json()["session_id"]

            tile = await client.post(
                f"/realtime/session/{session_id}/map/tile",
                json={
                    "tile_key": "tile-001",
                    "map_type": "occupancy",
                    "storage_path": "/tmp/tile-001.bin",
                    "resolution_m": 0.2,
                    "bounds": {"min_x": 0, "max_x": 10},
                    "stats": {"occupied": 42},
                    "global_map_id": 9,
                },
            )
            assert tile.status_code == 200
            tile_data = tile.json()
            assert len(tile_data["tiles"]) == 1
            assert tile_data["tiles"][0]["tile_key"] == "tile-001"

            semantic = await client.post(
                f"/realtime/session/{session_id}/semantic",
                json={
                    "class_name": "tree",
                    "confidence": 0.91,
                    "position_enu": {"x": 3.0, "y": 4.0, "z": 5.0},
                    "facts": {"source": "yolo"},
                },
            )
            assert semantic.status_code == 200
            obs = semantic.json()["observations"][0]
            assert obs["class_name"] == "tree"
            assert obs["facts"]["source"] == "yolo"

            latest_map = await client.get(f"/realtime/session/{session_id}/map/latest")
            assert latest_map.status_code == 200
            assert latest_map.json()["tiles"][0]["stats"]["occupied"] == 42

            nearby = await client.get(
                f"/realtime/session/{session_id}/semantic-nearby?class_name=tree"
            )
            assert nearby.status_code == 200
            assert nearby.json()["observations"][0]["class_name"] == "tree"


@pytest.mark.anyio
async def test_integrate_frame_creates_pose_tile_and_semantic():
    conn = FakeConn()
    app = _app()

    def _stub_tile(*, session_id, t_sec, frame_id, map_type="occupancy", **_):
        from selfsuvis.pipeline.realtime.occupancy import default_tile_key

        return {
            "tile_key": default_tile_key(t_sec=t_sec, frame_id=frame_id),
            "map_type": map_type,
            "storage_path": "/dev/null",
            "resolution_m": 0.2,
            "bounds": {},
            "stats": {},
            "global_map_id": None,
        }

    with (
        patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)),
        patch("selfsuvis.app.services.realtime.write_stub_map_tile", side_effect=_stub_tile),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post("/realtime/session/start", json={"robot_id": "drone_frame"})
            session_id = start.json()["session_id"]

            result = await client.post(
                f"/realtime/session/{session_id}/frame/integrate",
                json={
                    "frame_id": "f1",
                    "t_sec": 1.5,
                    "image_path": "/tmp/frame1.jpg",
                    "packets": [
                        {
                            "sensor_type": "gps",
                            "t_device": 1.5,
                            "payload": {"east": 5.0, "north": 6.0, "up": 1.0},
                        }
                    ],
                    "semantic_observations": [{"class_name": "tree", "confidence": 0.7}],
                },
            )
            assert result.status_code == 200
            data = result.json()
            assert data["pose_updated"] is True
            assert data["tile"]["tile_key"] == "frame-f1"
            assert data["semantic_count"] == 1


@pytest.mark.anyio
async def test_realtime_stats_endpoint():
    conn = FakeConn()
    app = _app()

    with (
        patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)),
        patch(
            "selfsuvis.app.routers.realtime.collect_realtime_stats",
            return_value={
                "pose_backend": "vins_fusion",
                "occupancy_backend": "nvblox",
                "pose": {"configured": True, "ready": True},
                "occupancy": {"configured": True, "tiles": 3},
            },
        ),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/realtime/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["pose_backend"] == "vins_fusion"
            assert data["occupancy_backend"] == "nvblox"


@pytest.mark.anyio
async def test_realtime_backends_endpoint():
    conn = FakeConn()
    app = _app()

    with (
        patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)),
        patch(
            "selfsuvis.app.routers.realtime.list_realtime_backends",
            return_value={
                "selected": {
                    "pose_backend": "vins_fusion",
                    "occupancy_backend": "voxblox",
                },
                "pose_backends": {
                    "vins_fusion": {
                        "name": "vins_fusion",
                        "role": "pose",
                        "provider": "sidecar",
                        "open_source": True,
                        "service_name": "realtime-vins-fusion",
                        "api_url": "http://realtime-vins-fusion:8101",
                        "env_image_var": "REALTIME_VINS_FUSION_IMAGE",
                        "default_image": "",
                        "hardware_profile": "cpu_or_gpu",
                        "required_modalities": ["camera", "imu"],
                        "recommended_modalities": ["camera", "imu", "gps"],
                        "pros": [
                            "Strong visual-inertial pose estimation on drone-class RGB + IMU feeds."
                        ],
                        "cons": ["Needs reliable camera/IMU calibration."],
                        "integration_doc": "docs/runbooks/realtime-sidecars/vins-fusion.md",
                        "notes": "RGB + IMU + GPS visual-inertial fusion sidecar.",
                    }
                },
                "occupancy_backends": {
                    "voxblox": {
                        "name": "voxblox",
                        "role": "occupancy",
                        "provider": "sidecar",
                        "open_source": True,
                        "service_name": "realtime-voxblox",
                        "api_url": "http://realtime-voxblox:8101",
                        "env_image_var": "REALTIME_VOXBLOX_IMAGE",
                        "default_image": "",
                        "hardware_profile": "cpu",
                        "required_modalities": ["pose", "depth"],
                        "recommended_modalities": ["pose", "depth", "camera"],
                        "pros": ["CPU-friendly volumetric mapping option."],
                        "cons": ["Lower throughput than nvblox on dense depth streams."],
                        "integration_doc": "docs/runbooks/realtime-sidecars/voxblox.md",
                        "notes": "CPU occupancy mapping sidecar.",
                    }
                },
            },
        ),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/realtime/backends")
            assert resp.status_code == 200
            data = resp.json()
            assert data["selected"]["pose_backend"] == "vins_fusion"
            assert data["pose_backends"]["vins_fusion"]["service_name"] == "realtime-vins-fusion"
            assert data["occupancy_backends"]["voxblox"]["hardware_profile"] == "cpu"


@pytest.mark.anyio
async def test_finalize_session_creates_mission_and_optional_job():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post(
                "/realtime/session/start",
                json={"robot_id": "drone_e", "mission_id": "mission_live"},
            )
            session_id = start.json()["session_id"]

            finalize = await client.post(
                f"/realtime/session/{session_id}/finalize",
                json={"recording_path": "/.data/videos/live.mp4", "enqueue_index_job": True},
            )
            assert finalize.status_code == 200
            data = finalize.json()
            assert data["mission_id"] == "mission_live"
            assert data["enqueued_index_job"] is True
            assert data["job_id"]
            assert "mission_live" in conn.missions
            assert data["job_id"] in conn.jobs


@pytest.mark.anyio
async def test_live_stream_start_list_and_stop():
    conn = FakeConn()
    app = _app()
    app.state.mediamtx_client = FakeMediaMtxClient()
    app.state.realtime_stream_manager = FakeRealtimeStreamManager()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post(
                "/realtime/streams",
                json={
                    "robot_id": "drone_live",
                    "path_name": "live/drone-a",
                    "mission_id": "mission_live_rtsp",
                    "caption_fps": 1.0,
                },
            )
            assert start.status_code == 200
            data = start.json()
            session_id = data["session_id"]
            assert data["publish_url"] == "rtsp://localhost:8554/live/drone-a"
            assert data["analysis"]["status"] == "running"

            listed = await client.get("/realtime/streams")
            assert listed.status_code == 200
            listed_data = listed.json()
            assert listed_data["streams"][0]["session_id"] == session_id
            assert listed_data["mediamtx_paths"][0]["name"] == "live/drone-a"

            stopped = await client.post(
                f"/realtime/streams/{session_id}/stop",
                json={"delete_path": True},
            )
            assert stopped.status_code == 200
            assert stopped.json()["deleted_path"] is True
