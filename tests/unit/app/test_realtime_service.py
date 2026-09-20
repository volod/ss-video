"""Unit tests for app.services.realtime workflow methods."""

from unittest.mock import AsyncMock, patch

import pytest

from selfsuvis.app.services import realtime as realtime_service
from selfsuvis.pipeline.realtime import packet_store


@pytest.mark.anyio
async def test_start_realtime_session_creates_profile_and_session():
    conn = object()
    with (
        patch.object(realtime_service, "new_session_id", return_value="session-123"),
        patch.object(
            realtime_service, "create_robot_session", new_callable=AsyncMock
        ) as create_session,
    ):
        result = await realtime_service.start_realtime_session(
            conn,
            robot_id="drone_a",
            mission_id="mission_a",
            sensors=["gps", "imu", "gps"],
        )

    create_session.assert_awaited_once()
    assert result == {
        "session_id": "session-123",
        "robot_id": "drone_a",
        "mission_id": "mission_a",
        "sensor_profile": {
            "sensors": ["gps", "imu"],
            "sensor_count": 2,
            "capabilities": {
                "gps": ["position", "velocity", "global_reference"],
                "imu": ["orientation", "acceleration", "angular_velocity", "velocity"],
            },
        },
        "status": "active",
    }


@pytest.mark.anyio
async def test_ingest_realtime_packets_persists_packets_and_stub_pose():
    conn = object()
    with (
        patch.object(
            packet_store,
            "fetch_realtime_state",
            new_callable=AsyncMock,
            return_value={"session": {"id": "s1"}},
        ),
        patch.object(
            packet_store, "insert_sensor_packets", new_callable=AsyncMock
        ) as insert_packets,
        patch.object(packet_store, "insert_realtime_pose", new_callable=AsyncMock) as insert_pose,
    ):
        result = await realtime_service.ingest_realtime_packets(
            conn,
            session_id="s1",
            packets=[
                {"sensor_type": "gps", "t_device": 10.0, "payload": {"east": 1.0, "north": 2.0}},
                {
                    "sensor_type": "imu",
                    "t_device": 10.1,
                    "payload": {"yaw": 0.1, "vx": 0.4, "vy": 0.0},
                },
            ],
        )

    insert_packets.assert_awaited_once()
    insert_pose.assert_awaited_once()
    kwargs = insert_pose.await_args.kwargs
    assert kwargs["source"] == "fused_gps_imu"
    assert kwargs["position_enu"] == {"x": 1.0, "y": 2.0, "z": 0.0}
    assert result == {
        "session_id": "s1",
        "accepted_packets": 2,
        "packet_summary": {"gps": 1, "imu": 1},
        "pose_updated": True,
    }


@pytest.mark.anyio
async def test_ingest_realtime_packets_uses_pose_sidecar_when_configured():
    conn = object()

    class _PoseClient:
        is_configured = True

        async def estimate_pose(self, *, session_id: str, packets):
            assert session_id == "s1"
            assert len(packets) == 1
            return {
                "source": "vins_fusion",
                "t_sec": 5.0,
                "position_enu": {"x": 9.0, "y": 8.0, "z": 7.0},
                "orientation_quat": None,
                "velocity_enu": None,
                "covariance": {"trace": 0.1},
                "tracking_status": "ok",
                "global_map_id": 4,
            }

    with (
        patch.object(
            packet_store,
            "fetch_realtime_state",
            new_callable=AsyncMock,
            return_value={"session": {"id": "s1"}},
        ),
        patch.object(packet_store, "insert_sensor_packets", new_callable=AsyncMock),
        patch.object(packet_store, "insert_realtime_pose", new_callable=AsyncMock) as insert_pose,
        patch.object(packet_store.settings, "REALTIME_POSE_BACKEND", "vins_fusion"),
        patch.object(packet_store, "RealtimePoseClient", return_value=_PoseClient()),
    ):
        result = await realtime_service.ingest_realtime_packets(
            conn,
            session_id="s1",
            packets=[
                {"sensor_type": "gps", "t_device": 5.0, "payload": {"east": 1.0, "north": 2.0}}
            ],
        )

    assert result["pose_updated"] is True
    assert insert_pose.await_args.kwargs["source"] == "vins_fusion"


@pytest.mark.anyio
async def test_ingest_realtime_packets_skips_pose_when_gps_and_imu_are_stale():
    conn = object()
    with (
        patch.object(
            packet_store,
            "fetch_realtime_state",
            new_callable=AsyncMock,
            return_value={"session": {"id": "s1"}},
        ),
        patch.object(packet_store, "insert_sensor_packets", new_callable=AsyncMock),
        patch.object(packet_store, "insert_realtime_pose", new_callable=AsyncMock) as insert_pose,
        patch.object(packet_store.settings, "REALTIME_MAX_SENSOR_LAG_MS", 100),
    ):
        result = await realtime_service.ingest_realtime_packets(
            conn,
            session_id="s1",
            packets=[
                {"sensor_type": "gps", "t_device": 10.0, "payload": {"east": 1.0, "north": 2.0}},
                {"sensor_type": "imu", "t_device": 10.5, "payload": {"yaw": 0.1}},
            ],
        )

    insert_pose.assert_not_awaited()
    assert result["pose_updated"] is False


@pytest.mark.anyio
async def test_ingest_realtime_packets_missing_session_raises():
    conn = object()
    with patch.object(
        packet_store, "fetch_realtime_state", new_callable=AsyncMock, return_value=None
    ):
        with pytest.raises(LookupError, match="session not found"):
            await realtime_service.ingest_realtime_packets(
                conn,
                session_id="missing",
                packets=[
                    {"sensor_type": "gps", "t_device": 1.0, "payload": {"east": 1, "north": 2}}
                ],
            )


@pytest.mark.anyio
async def test_publish_and_fetch_map_tiles_delegate_to_storage():
    conn = object()
    rows = [{"tile_key": "tile-1", "map_type": "occupancy"}]
    with (
        patch.object(
            realtime_service,
            "fetch_realtime_state",
            new_callable=AsyncMock,
            return_value={"session": {"id": "s1"}},
        ),
        patch.object(realtime_service, "upsert_map_tile", new_callable=AsyncMock) as upsert_tile,
        patch.object(
            realtime_service, "list_map_tiles", new_callable=AsyncMock, return_value=rows
        ) as list_tiles,
    ):
        await realtime_service.publish_map_tile(
            conn,
            session_id="s1",
            tile={"tile_key": "tile-1", "storage_path": "/tmp/t.bin"},
        )
        result = await realtime_service.fetch_map_tiles(
            conn, session_id="s1", map_type="occupancy", limit=5
        )

    upsert_tile.assert_awaited_once()
    list_tiles.assert_awaited_once_with(conn, "s1", map_type="occupancy", limit=5)
    assert result == rows


@pytest.mark.anyio
async def test_publish_semantic_observation_validates_and_delegates():
    conn = object()
    with (
        patch.object(
            realtime_service,
            "fetch_realtime_state",
            new_callable=AsyncMock,
            return_value={"session": {"id": "s1"}},
        ),
        patch.object(
            realtime_service, "insert_semantic_observation", new_callable=AsyncMock
        ) as insert_obs,
    ):
        await realtime_service.publish_semantic_observation(
            conn,
            session_id="s1",
            observation={"class_name": "Tree", "confidence": 0.8},
        )

    insert_obs.assert_awaited_once()
    kwargs = insert_obs.await_args.kwargs
    assert kwargs["session_id"] == "s1"
    assert kwargs["class_name"] == "tree"
    assert kwargs["confidence"] == 0.8


@pytest.mark.anyio
async def test_fetch_semantic_observations_missing_session_raises():
    conn = object()
    with patch.object(
        realtime_service, "fetch_realtime_state", new_callable=AsyncMock, return_value=None
    ):
        with pytest.raises(LookupError, match="session not found"):
            await realtime_service.fetch_semantic_observations(conn, session_id="missing")


@pytest.mark.anyio
async def test_finalize_realtime_session_creates_job_when_requested():
    conn = object()
    with (
        patch.object(
            realtime_service,
            "fetch_realtime_state",
            new_callable=AsyncMock,
            return_value={"session": {"robot_id": "drone_a", "mission_id": "mission_live"}},
        ),
        patch.object(realtime_service, "upsert_mission", new_callable=AsyncMock) as upsert_mission,
        patch.object(realtime_service, "create_job", new_callable=AsyncMock) as create_job,
        patch.object(
            realtime_service, "stop_robot_session", new_callable=AsyncMock
        ) as stop_session,
        patch.object(
            realtime_service, "list_realtime_frames", new_callable=AsyncMock, return_value=[]
        ),
        patch.object(
            realtime_service,
            "summarize_realtime_session",
            new_callable=AsyncMock,
            return_value={
                "packet_counts": {"camera": 5},
                "duration_sec": 12.0,
                "latest_pose": {"position_enu_json": {"x": 1.0, "y": 2.0, "z": 3.0}},
            },
        ),
        patch.object(realtime_service.uuid, "uuid4") as uuid4_mock,
    ):
        uuid4_mock.return_value.hex = "job-123"
        result = await realtime_service.finalize_realtime_session(
            conn,
            session_id="s1",
            recording_path="/data/live.mp4",
            enqueue_index_job=True,
        )

    assert upsert_mission.await_count == 2
    create_job.assert_awaited_once()
    stop_session.assert_awaited_once_with(conn, "s1")
    assert result == {
        "session_id": "s1",
        "mission_id": "mission_live",
        "job_id": "job-123",
        "status": "stopped",
        "enqueued_index_job": True,
    }
    payload = create_job.await_args.args[2]
    assert payload["postflight_jobs"] == [
        "postflight_mapping",
        "postflight_semantic_graph",
    ]


def test_list_realtime_backends_reports_selected_and_catalog():
    with (
        patch.object(realtime_service.settings, "REALTIME_POSE_BACKEND", "orbslam3"),
        patch.object(realtime_service.settings, "REALTIME_OCCUPANCY_BACKEND", "voxblox"),
    ):
        result = realtime_service.list_realtime_backends()

    assert result["selected"] == {
        "pose_backend": "orbslam3",
        "occupancy_backend": "voxblox",
    }
    assert result["pose_backends"]["orbslam3"]["service_name"] == "realtime-orbslam3"
    assert result["occupancy_backends"]["voxblox"]["hardware_profile"] == "cpu"


@pytest.mark.anyio
async def test_finalize_realtime_session_without_recording_does_not_enqueue():
    conn = object()
    with (
        patch.object(
            realtime_service,
            "fetch_realtime_state",
            new_callable=AsyncMock,
            return_value={"session": {"robot_id": "drone_a", "mission_id": None}},
        ),
        patch.object(realtime_service, "upsert_mission", new_callable=AsyncMock) as upsert_mission,
        patch.object(realtime_service, "create_job", new_callable=AsyncMock) as create_job,
        patch.object(realtime_service, "stop_robot_session", new_callable=AsyncMock),
        patch.object(
            realtime_service, "list_realtime_frames", new_callable=AsyncMock, return_value=[]
        ),
        patch.object(
            realtime_service,
            "summarize_realtime_session",
            new_callable=AsyncMock,
            return_value={"packet_counts": {}, "duration_sec": None, "latest_pose": None},
        ),
    ):
        result = await realtime_service.finalize_realtime_session(
            conn,
            session_id="s1",
            recording_path=None,
            enqueue_index_job=True,
        )

    upsert_mission.assert_awaited_once()
    create_job.assert_not_called()
    assert result["mission_id"] == "realtime-s1"
    assert result["job_id"] is None
    assert result["enqueued_index_job"] is False


@pytest.mark.anyio
async def test_integrate_realtime_frame_writes_stub_tile_and_semantics():
    conn = object()
    with (
        patch.object(
            realtime_service,
            "fetch_realtime_state",
            new_callable=AsyncMock,
            return_value={"session": {"id": "s1"}, "latest_pose": None},
        ),
        patch.object(
            realtime_service, "insert_sensor_packets", new_callable=AsyncMock
        ) as insert_packets,
        patch.object(
            realtime_service, "insert_realtime_pose", new_callable=AsyncMock
        ) as insert_pose,
        patch.object(realtime_service, "upsert_map_tile", new_callable=AsyncMock) as upsert_tile,
        patch.object(
            realtime_service, "insert_semantic_observation", new_callable=AsyncMock
        ) as insert_semantic,
        patch.object(
            realtime_service, "upsert_realtime_frame", new_callable=AsyncMock
        ) as upsert_realtime_frame,
        patch.object(realtime_service.settings, "REALTIME_OCCUPANCY_BACKEND", "stub"),
        patch.object(
            realtime_service,
            "write_stub_map_tile",
            return_value={
                "tile_key": "frame-f1",
                "map_type": "occupancy",
                "storage_path": "/tmp/frame-f1.json",
                "resolution_m": 0.2,
                "bounds": {},
                "stats": {"occupied": 3},
                "global_map_id": None,
            },
        ),
    ):
        result = await realtime_service.integrate_realtime_frame(
            conn,
            session_id="s1",
            frame_id="f1",
            t_sec=3.0,
            image_path="/tmp/f1.jpg",
            packets=[
                {"sensor_type": "gps", "t_device": 3.0, "payload": {"east": 1.0, "north": 2.0}}
            ],
            semantic_observations=[{"class_name": "tree", "confidence": 0.8}],
        )

    insert_packets.assert_awaited_once()
    insert_pose.assert_awaited_once()
    upsert_tile.assert_awaited_once()
    insert_semantic.assert_awaited_once()
    upsert_realtime_frame.assert_awaited_once()
    assert result["semantic_count"] == 1
    assert result["tile"]["tile_key"] == "frame-f1"
