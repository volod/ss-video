import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from selfsuvis.realtime import bridge_runtime


class _AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCtx(self._conn)


@pytest.mark.anyio
async def test_packet_publisher_auto_creates_session_and_flushes():
    conn = object()
    pool = FakePool(conn)
    with (
        patch.object(
            bridge_runtime, "fetch_realtime_state", new_callable=AsyncMock, return_value=None
        ),
        patch.object(
            bridge_runtime, "create_robot_session", new_callable=AsyncMock
        ) as create_session,
        patch.object(
            bridge_runtime, "ingest_realtime_packets", new_callable=AsyncMock
        ) as ingest_packets,
    ):
        publisher = bridge_runtime.RealtimePacketPublisher(
            pool,
            session_id="session-1",
            robot_id="robot-1",
            mission_id="mission-1",
            sensors=["gps", "imu"],
            auto_create_session=True,
            batch_size=8,
            flush_interval_sec=0.01,
            log_every_n_packets=0,
        )
        await publisher.start()
        await publisher.publish(
            {"sensor_type": "gps", "t_device": 1.0, "payload": {"east": 1.0, "north": 2.0}}
        )
        await publisher.publish({"sensor_type": "imu", "t_device": 1.0, "payload": {"yaw": 0.1}})
        await publisher.shutdown()

    create_session.assert_awaited_once()
    ingest_packets.assert_awaited_once()
    assert ingest_packets.await_args.kwargs["session_id"] == "session-1"
    assert len(ingest_packets.await_args.kwargs["packets"]) == 2


def test_ros_navsatfix_message_to_dict_projects_relative_to_origin():
    msg = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=500_000_000)),
        latitude=50.0001,
        longitude=19.0002,
        altitude=120.0,
    )
    result = bridge_runtime.ros_navsatfix_message_to_dict(
        "/gps/fix",
        msg,
        {"lat": 50.0, "lon": 19.0, "altitude": 100.0},
    )
    assert result["topic"] == "/gps/fix"
    assert result["t_device"] == 10.5
    assert result["payload"]["east"] > 0.0
    assert result["payload"]["north"] > 0.0
    assert result["payload"]["up"] == 20.0


def test_ros_fluid_pressure_message_to_dict_converts_pressure_to_altitude():
    msg = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
        fluid_pressure=89_874.6,
    )
    result = bridge_runtime.ros_fluid_pressure_message_to_dict("/barometer", msg)
    assert result["topic"] == "/barometer"
    assert result["payload"]["pressure_pa"] == 89_874.6
    assert result["payload"]["altitude"] == pytest.approx(1000.0, abs=50.0)


def test_mavsdk_position_velocity_to_message_converts_ned_to_enu():
    sample = SimpleNamespace(
        position=SimpleNamespace(north_m=4.0, east_m=3.0, down_m=2.0),
        velocity=SimpleNamespace(north_m_s=1.5, east_m_s=2.5, down_m_s=0.75),
        timestamp_us=123456789,
    )
    message = bridge_runtime.mavsdk_position_velocity_to_message(sample)
    assert message["message_type"] == "GLOBAL_POSITION_INT"
    assert message["north"] == 4.0
    assert message["east"] == 3.0
    assert message["up"] == -2.0
    assert message["vx"] == 150.0
    assert message["vy"] == 250.0
    assert message["vz"] == -75.0


@pytest.mark.anyio
async def test_build_runtime_from_settings_uses_ros_bridge_when_selected():
    pool = FakePool(object())

    class FakeSource:
        async def run(self, on_message):
            await on_message({"topic": "/imu", "t_device": 1.0, "payload": {"yaw": 0.1}})
            raise asyncio.CancelledError

    with patch.object(bridge_runtime.settings, "REALTIME_BRIDGE_BACKEND", "ros"):
        runtime = bridge_runtime.build_runtime_from_settings(
            backend="ros", db_pool=pool, source=FakeSource()
        )

    assert isinstance(runtime._bridge, bridge_runtime.RosTopicBridge)


def test_bridge_runtime_config_from_settings_collects_runtime_knobs():
    with (
        patch.object(bridge_runtime.settings, "REALTIME_BRIDGE_BACKEND", "mavsdk"),
        patch.object(bridge_runtime.settings, "REALTIME_BRIDGE_SESSION_ID", "session-a"),
        patch.object(bridge_runtime.settings, "REALTIME_BRIDGE_ROBOT_ID", "robot-a"),
        patch.object(bridge_runtime.settings, "REALTIME_BRIDGE_MISSION_ID", "mission-a"),
        patch.object(bridge_runtime.settings, "REALTIME_BRIDGE_SENSORS", "gps,imu"),
        patch.object(bridge_runtime.settings, "REALTIME_BRIDGE_AUTO_CREATE_SESSION", True),
        patch.object(bridge_runtime.settings, "REALTIME_PACKET_BATCH_SIZE", 32),
        patch.object(bridge_runtime.settings, "REALTIME_BRIDGE_FLUSH_INTERVAL_SEC", 0.25),
        patch.object(bridge_runtime.settings, "REALTIME_BRIDGE_RECONNECT_SEC", 5.0),
        patch.object(bridge_runtime.settings, "REALTIME_BRIDGE_LOG_EVERY_N_PACKETS", 100),
    ):
        config = bridge_runtime.bridge_runtime_config_from_settings()

    assert config == bridge_runtime.BridgeRuntimeConfig(
        backend="mavsdk",
        session_id="session-a",
        robot_id="robot-a",
        mission_id="mission-a",
        sensors=["gps", "imu"],
        auto_create_session=True,
        batch_size=32,
        flush_interval_sec=0.25,
        reconnect_sec=5.0,
        log_every_n_packets=100,
    )
