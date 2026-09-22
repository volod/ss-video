"""Runtime bridge daemons for MAVSDK and ROS telemetry ingestion."""

import argparse
import asyncio
import math
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

import asyncpg

from selfsuvis.pipeline.core import get_logger, settings
from selfsuvis.pipeline.media.drone_bridge import DroneTelemetryBridge
from selfsuvis.pipeline.media.ros_bridge import RosTopicBridge
from selfsuvis.pipeline.realtime import build_sensor_profile, ingest_realtime_packets
from selfsuvis.pipeline.storage import create_robot_session, fetch_realtime_state

logger = get_logger(__name__)


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _sensor_set(value: str) -> list[str]:
    sensors = _csv_list(value)
    return sensors or ["gps", "imu", "barometer", "magnetometer"]


def _attr(value: Any, *names: str) -> Any:
    current = value
    for name in names:
        if current is None:
            return None
        current = getattr(current, name, None)
    return current


def _stamp_to_sec(stamp: Any) -> float:
    sec = _attr(stamp, "sec")
    nanosec = _attr(stamp, "nanosec")
    if sec is None:
        sec = _attr(stamp, "secs")
        nanosec = _attr(stamp, "nsecs")
    if sec is None:
        return 0.0
    return float(sec) + float(nanosec or 0) / 1_000_000_000.0


def _ecef_like_local_xy(
    origin_lat: float, origin_lon: float, lat: float, lon: float
) -> tuple[float, float]:
    lat0 = math.radians(origin_lat)
    d_lat = math.radians(lat - origin_lat)
    d_lon = math.radians(lon - origin_lon)
    earth_radius_m = 6_378_137.0
    north = d_lat * earth_radius_m
    east = d_lon * earth_radius_m * math.cos(lat0)
    return east, north


def ros_imu_message_to_dict(topic: str, msg: Any) -> dict[str, Any]:
    return {
        "topic": topic,
        "t_device": _stamp_to_sec(_attr(msg, "header", "stamp")),
        "payload": {
            "orientation_quat": {
                "x": float(_attr(msg, "orientation", "x") or 0.0),
                "y": float(_attr(msg, "orientation", "y") or 0.0),
                "z": float(_attr(msg, "orientation", "z") or 0.0),
                "w": float(_attr(msg, "orientation", "w") or 1.0),
            },
            "angular_velocity": {
                "x": float(_attr(msg, "angular_velocity", "x") or 0.0),
                "y": float(_attr(msg, "angular_velocity", "y") or 0.0),
                "z": float(_attr(msg, "angular_velocity", "z") or 0.0),
            },
            "acceleration": {
                "x": float(_attr(msg, "linear_acceleration", "x") or 0.0),
                "y": float(_attr(msg, "linear_acceleration", "y") or 0.0),
                "z": float(_attr(msg, "linear_acceleration", "z") or 0.0),
            },
        },
    }


def ros_navsatfix_message_to_dict(
    topic: str, msg: Any, origin: dict[str, float] | None
) -> dict[str, Any]:
    lat = float(getattr(msg, "latitude"))
    lon = float(getattr(msg, "longitude"))
    altitude = float(getattr(msg, "altitude", 0.0) or 0.0)
    payload: dict[str, Any] = {
        "lat": lat,
        "lon": lon,
        "altitude": altitude,
    }
    if origin is not None:
        east, north = _ecef_like_local_xy(origin["lat"], origin["lon"], lat, lon)
        payload["east"] = east
        payload["north"] = north
        payload["up"] = altitude - origin.get("altitude", 0.0)
    return {
        "topic": topic,
        "t_device": _stamp_to_sec(_attr(msg, "header", "stamp")),
        "payload": payload,
    }


def ros_fluid_pressure_message_to_dict(topic: str, msg: Any) -> dict[str, Any]:
    pressure_pa = float(getattr(msg, "fluid_pressure", 0.0) or 0.0)
    sea_level_pa = 101_325.0
    altitude_m = 0.0
    if pressure_pa > 0.0:
        altitude_m = 44_330.0 * (1.0 - (pressure_pa / sea_level_pa) ** 0.1903)
    return {
        "topic": topic,
        "t_device": _stamp_to_sec(_attr(msg, "header", "stamp")),
        "payload": {
            "altitude": altitude_m,
            "pressure_pa": pressure_pa,
        },
    }


def ros_magnetic_field_message_to_dict(topic: str, msg: Any) -> dict[str, Any]:
    x = float(_attr(msg, "magnetic_field", "x") or 0.0)
    y = float(_attr(msg, "magnetic_field", "y") or 0.0)
    heading = math.atan2(y, x) if x or y else 0.0
    return {
        "topic": topic,
        "t_device": _stamp_to_sec(_attr(msg, "header", "stamp")),
        "payload": {
            "heading": heading,
            "field": {"x": x, "y": y, "z": float(_attr(msg, "magnetic_field", "z") or 0.0)},
        },
    }


def ros_image_message_to_dict(topic: str, msg: Any) -> dict[str, Any]:
    header = _attr(msg, "header")
    return {
        "topic": topic,
        "t_device": _stamp_to_sec(_attr(header, "stamp")),
        "payload": {
            "frame_id": str(_attr(header, "frame_id") or ""),
            "width": int(getattr(msg, "width", 0) or 0),
            "height": int(getattr(msg, "height", 0) or 0),
            "encoding": str(getattr(msg, "encoding", "") or ""),
        },
    }


def mavsdk_position_velocity_to_message(sample: Any) -> dict[str, Any]:
    position = getattr(sample, "position", sample)
    velocity = getattr(sample, "velocity", sample)
    down_m = float(_attr(position, "down_m") or 0.0)
    vz = float(_attr(velocity, "down_m_s") or 0.0)
    return {
        "message_type": "GLOBAL_POSITION_INT",
        "timestamp": float(getattr(sample, "timestamp_us", 0) or 0) or time.time(),
        "north": float(_attr(position, "north_m") or 0.0),
        "east": float(_attr(position, "east_m") or 0.0),
        "up": -down_m,
        "vx": float(_attr(velocity, "north_m_s") or 0.0) * 100.0,
        "vy": float(_attr(velocity, "east_m_s") or 0.0) * 100.0,
        "vz": -vz * 100.0,
    }


def mavsdk_attitude_euler_to_message(sample: Any) -> dict[str, Any]:
    return {
        "message_type": "ATTITUDE",
        "timestamp": float(getattr(sample, "timestamp_us", 0) or 0) or time.time(),
        "roll": math.radians(float(getattr(sample, "roll_deg", 0.0) or 0.0)),
        "pitch": math.radians(float(getattr(sample, "pitch_deg", 0.0) or 0.0)),
        "yaw": math.radians(float(getattr(sample, "yaw_deg", 0.0) or 0.0)),
    }


def mavsdk_heading_to_message(sample: Any) -> dict[str, Any]:
    heading_deg = float(getattr(sample, "heading_deg", sample) or 0.0)
    return {
        "message_type": "VFR_HUD",
        "timestamp": float(getattr(sample, "timestamp_us", 0) or 0) or time.time(),
        "heading": math.radians(heading_deg),
    }


@dataclass
class BridgeRuntimeConfig:
    backend: str
    session_id: str
    robot_id: str
    mission_id: str | None
    sensors: list[str]
    auto_create_session: bool
    batch_size: int
    flush_interval_sec: float
    reconnect_sec: float
    log_every_n_packets: int


class RealtimePacketPublisher:
    def __init__(
        self,
        db_pool: Any,
        *,
        session_id: str,
        robot_id: str,
        mission_id: str | None,
        sensors: Iterable[str],
        auto_create_session: bool,
        batch_size: int,
        flush_interval_sec: float,
        log_every_n_packets: int,
    ) -> None:
        self._db_pool = db_pool
        self._session_id = session_id
        self._robot_id = robot_id
        self._mission_id = mission_id
        self._sensors = list(sensors)
        self._auto_create_session = auto_create_session
        self._batch_size = max(1, int(batch_size))
        self._flush_interval_sec = max(0.05, float(flush_interval_sec))
        self._log_every_n_packets = max(0, int(log_every_n_packets))
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._published_packets = 0
        self._flush_count = 0

    async def start(self) -> None:
        await self._ensure_session()
        self._task = asyncio.create_task(
            self._flush_loop(), name=f"realtime_bridge_flush:{self._session_id}"
        )

    async def publish(self, packet: dict[str, Any]) -> None:
        await self._queue.put(packet)

    async def shutdown(self) -> None:
        await self._queue.put(None)
        if self._task is not None:
            await self._task
            self._task = None

    def stats(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            "published_packets": self._published_packets,
            "flush_count": self._flush_count,
            "queue_size": self._queue.qsize(),
        }

    async def _ensure_session(self) -> None:
        async with self._db_pool.acquire() as conn:
            state = await fetch_realtime_state(conn, self._session_id)
            if state is not None:
                return
            if not self._auto_create_session:
                raise LookupError(f"realtime session not found: {self._session_id}")
            await create_robot_session(
                conn,
                session_id=self._session_id,
                robot_id=self._robot_id,
                mission_id=self._mission_id,
                sensor_profile=build_sensor_profile(self._sensors),
            )
            logger.info(
                "Realtime bridge session created: session=%s robot=%s mission=%s",
                self._session_id,
                self._robot_id,
                self._mission_id or "",
            )

    async def _flush_loop(self) -> None:
        batch: list[dict[str, Any]] = []
        while True:
            try:
                packet = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval_sec)
            except TimeoutError:
                packet = ...

            if packet is None:
                if batch:
                    await self._flush(batch)
                return
            if packet is not ...:
                batch.append(packet)
            if batch and (packet is ... or len(batch) >= self._batch_size):
                await self._flush(batch)

    async def _flush(self, batch: list[dict[str, Any]]) -> None:
        packets = list(batch)
        batch.clear()
        if not packets:
            return
        async with self._db_pool.acquire() as conn:
            await ingest_realtime_packets(conn, session_id=self._session_id, packets=packets)
        self._published_packets += len(packets)
        self._flush_count += 1
        if self._log_every_n_packets and self._published_packets % self._log_every_n_packets == 0:
            logger.info(
                "Realtime bridge published %d packets into session=%s",
                self._published_packets,
                self._session_id,
            )


class MavsdkTelemetrySource:
    def __init__(
        self,
        *,
        system_address: str,
        server_address: str,
        server_port: int,
        connect_timeout_sec: float,
        system_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._system_address = system_address
        self._server_address = server_address
        self._server_port = int(server_port)
        self._connect_timeout_sec = float(connect_timeout_sec)
        self._system_factory = system_factory

    async def run(self, on_message: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        system = self._create_system()
        await self._connect(system)
        telemetry = getattr(system, "telemetry", None)
        if telemetry is None:
            raise RuntimeError("MAVSDK system has no telemetry interface")
        tasks = []
        if hasattr(telemetry, "position_velocity_ned"):
            tasks.append(
                asyncio.create_task(self._pump_position(telemetry, on_message), name="mavsdk_pos")
            )
        if hasattr(telemetry, "attitude_euler"):
            tasks.append(
                asyncio.create_task(self._pump_attitude(telemetry, on_message), name="mavsdk_att")
            )
        if hasattr(telemetry, "heading"):
            tasks.append(
                asyncio.create_task(
                    self._pump_heading(telemetry, on_message), name="mavsdk_heading"
                )
            )
        if not tasks:
            raise RuntimeError("No supported MAVSDK telemetry streams are available")
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _create_system(self) -> Any:
        if self._system_factory is not None:
            return self._system_factory()
        try:
            from mavsdk import System
        except ImportError as exc:
            raise RuntimeError(
                "mavsdk is not installed; install it in the realtime bridge image"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self._server_address:
            kwargs["mavsdk_server_address"] = self._server_address
            kwargs["port"] = self._server_port
        return System(**kwargs)

    async def _connect(self, system: Any) -> None:
        await system.connect(system_address=self._system_address)
        core = getattr(system, "core", None)
        if core is None or not hasattr(core, "connection_state"):
            return

        async def _wait_connected() -> None:
            async for state in core.connection_state():
                if getattr(state, "is_connected", False):
                    return

        await asyncio.wait_for(_wait_connected(), timeout=self._connect_timeout_sec)
        logger.info("MAVSDK bridge connected: %s", self._system_address)

    async def _pump_position(
        self,
        telemetry: Any,
        on_message: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        async for sample in telemetry.position_velocity_ned():
            await on_message(mavsdk_position_velocity_to_message(sample))

    async def _pump_attitude(
        self,
        telemetry: Any,
        on_message: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        async for sample in telemetry.attitude_euler():
            await on_message(mavsdk_attitude_euler_to_message(sample))

    async def _pump_heading(
        self,
        telemetry: Any,
        on_message: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        async for sample in telemetry.heading():
            await on_message(mavsdk_heading_to_message(sample))


class RosTelemetrySource:
    def __init__(
        self,
        *,
        imu_topic: str,
        gps_topic: str,
        barometer_topic: str,
        mag_topic: str,
        camera_topic: str,
        domain_id: str,
    ) -> None:
        self._imu_topic = imu_topic
        self._gps_topic = gps_topic
        self._barometer_topic = barometer_topic
        self._mag_topic = mag_topic
        self._camera_topic = camera_topic
        self._domain_id = domain_id
        self._gps_origin: dict[str, float] | None = None

    async def run(self, on_message: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from sensor_msgs.msg import FluidPressure, Image, Imu, MagneticField, NavSatFix
        except ImportError as exc:
            raise RuntimeError(
                "rclpy and sensor_msgs are not installed in the realtime bridge image"
            ) from exc

        if self._domain_id:
            os.environ["ROS_DOMAIN_ID"] = self._domain_id
        rclpy.init(args=None)
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        class BridgeNode(Node):
            def __init__(self, outer: "RosTelemetrySource") -> None:
                super().__init__("selfsuvis_realtime_ros_bridge")
                if outer._imu_topic:
                    self.create_subscription(
                        Imu,
                        outer._imu_topic,
                        lambda msg: queue.put_nowait(
                            ros_imu_message_to_dict(outer._imu_topic, msg)
                        ),
                        10,
                    )
                if outer._gps_topic:
                    self.create_subscription(
                        NavSatFix,
                        outer._gps_topic,
                        lambda msg: queue.put_nowait(outer._gps_msg(outer._gps_topic, msg)),
                        10,
                    )
                if outer._barometer_topic:
                    self.create_subscription(
                        FluidPressure,
                        outer._barometer_topic,
                        lambda msg: queue.put_nowait(
                            ros_fluid_pressure_message_to_dict(outer._barometer_topic, msg)
                        ),
                        10,
                    )
                if outer._mag_topic:
                    self.create_subscription(
                        MagneticField,
                        outer._mag_topic,
                        lambda msg: queue.put_nowait(
                            ros_magnetic_field_message_to_dict(outer._mag_topic, msg)
                        ),
                        10,
                    )
                if outer._camera_topic:
                    self.create_subscription(
                        Image,
                        outer._camera_topic,
                        lambda msg: queue.put_nowait(
                            ros_image_message_to_dict(outer._camera_topic, msg)
                        ),
                        10,
                    )

        node = BridgeNode(self)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_task = asyncio.create_task(self._spin_ros(executor, rclpy), name="ros_spin")
        try:
            while True:
                message = await queue.get()
                if message is None:
                    return
                await on_message(message)
        finally:
            spin_task.cancel()
            await asyncio.gather(spin_task, return_exceptions=True)
            executor.remove_node(node)
            node.destroy_node()
            rclpy.shutdown()

    async def _spin_ros(self, executor: Any, rclpy_module: Any) -> None:
        while rclpy_module.ok():
            executor.spin_once(timeout_sec=0.2)
            await asyncio.sleep(0)

    def _gps_msg(self, topic: str, msg: Any) -> dict[str, Any]:
        if self._gps_origin is None:
            self._gps_origin = {
                "lat": float(getattr(msg, "latitude")),
                "lon": float(getattr(msg, "longitude")),
                "altitude": float(getattr(msg, "altitude", 0.0) or 0.0),
            }
        return ros_navsatfix_message_to_dict(topic, msg, self._gps_origin)


class RealtimeBridgeRuntime:
    def __init__(
        self,
        *,
        config: BridgeRuntimeConfig,
        publisher: RealtimePacketPublisher,
        source: Any,
    ) -> None:
        self._config = config
        self._publisher = publisher
        self._source = source
        if config.backend == "ros":
            self._bridge = RosTopicBridge(on_packet=publisher.publish)
        else:
            self._bridge = DroneTelemetryBridge(on_packet=publisher.publish)

    async def run_forever(self) -> None:
        await self._publisher.start()
        try:
            while True:
                try:
                    await self._source.run(self._bridge.ingest_message)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Realtime bridge backend failed: %s", self._config.backend)
                    await asyncio.sleep(self._config.reconnect_sec)
        finally:
            await self._publisher.shutdown()


def bridge_runtime_config_from_settings(*, backend: str | None = None) -> BridgeRuntimeConfig:
    backend_name = (backend or settings.REALTIME_BRIDGE_BACKEND or "mavsdk").strip().lower()
    return BridgeRuntimeConfig(
        backend=backend_name,
        session_id=settings.REALTIME_BRIDGE_SESSION_ID,
        robot_id=settings.REALTIME_BRIDGE_ROBOT_ID,
        mission_id=settings.REALTIME_BRIDGE_MISSION_ID or None,
        sensors=_sensor_set(settings.REALTIME_BRIDGE_SENSORS),
        auto_create_session=settings.REALTIME_BRIDGE_AUTO_CREATE_SESSION,
        batch_size=settings.REALTIME_PACKET_BATCH_SIZE,
        flush_interval_sec=settings.REALTIME_BRIDGE_FLUSH_INTERVAL_SEC,
        reconnect_sec=settings.REALTIME_BRIDGE_RECONNECT_SEC,
        log_every_n_packets=settings.REALTIME_BRIDGE_LOG_EVERY_N_PACKETS,
    )


def bridge_source_from_settings(*, backend: str, source: Any = None) -> Any:
    if source is not None:
        return source
    if backend == "ros":
        return RosTelemetrySource(
            imu_topic=settings.REALTIME_ROS_IMU_TOPIC,
            gps_topic=settings.REALTIME_ROS_GPS_TOPIC,
            barometer_topic=settings.REALTIME_ROS_BAROMETER_TOPIC,
            mag_topic=settings.REALTIME_ROS_MAG_TOPIC,
            camera_topic=settings.REALTIME_ROS_CAMERA_TOPIC,
            domain_id=settings.REALTIME_ROS_DOMAIN_ID,
        )
    return MavsdkTelemetrySource(
        system_address=settings.REALTIME_MAVSDK_SYSTEM_ADDRESS,
        server_address=settings.REALTIME_MAVSDK_SERVER_ADDRESS,
        server_port=settings.REALTIME_MAVSDK_SERVER_PORT,
        connect_timeout_sec=settings.REALTIME_MAVSDK_CONNECT_TIMEOUT_SEC,
    )


def build_runtime_from_settings(
    *,
    backend: str | None = None,
    db_pool: Any,
    source: Any = None,
) -> RealtimeBridgeRuntime:
    config = bridge_runtime_config_from_settings(backend=backend)
    publisher = RealtimePacketPublisher(
        db_pool,
        session_id=config.session_id,
        robot_id=config.robot_id,
        mission_id=config.mission_id,
        sensors=config.sensors,
        auto_create_session=config.auto_create_session,
        batch_size=config.batch_size,
        flush_interval_sec=config.flush_interval_sec,
        log_every_n_packets=config.log_every_n_packets,
    )
    source = bridge_source_from_settings(backend=config.backend, source=source)
    return RealtimeBridgeRuntime(config=config, publisher=publisher, source=source)


async def _amain(backend: str | None = None) -> None:
    if not settings.DATABASE_URL:
        raise RuntimeError("DATABASE_URL must be configured for realtime bridge runtimes")
    db_pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL, min_size=1, max_size=2, timeout=10
    )
    runtime = build_runtime_from_settings(backend=backend, db_pool=db_pool)
    try:
        await runtime.run_forever()
    finally:
        await db_pool.close()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a realtime telemetry bridge runtime")
    parser.add_argument(
        "--backend",
        choices=["mavsdk", "ros"],
        default=None,
        help="Telemetry source backend. Defaults to REALTIME_BRIDGE_BACKEND.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        asyncio.run(_amain(backend=args.backend))
    except KeyboardInterrupt:
        logger.info("Realtime bridge stopped by user")
        return 0
    return 0


__all__ = [
    "BridgeRuntimeConfig",
    "MavsdkTelemetrySource",
    "RealtimeBridgeRuntime",
    "RealtimePacketPublisher",
    "RosTelemetrySource",
    "bridge_runtime_config_from_settings",
    "bridge_source_from_settings",
    "build_runtime_from_settings",
    "main",
    "mavsdk_attitude_euler_to_message",
    "mavsdk_heading_to_message",
    "mavsdk_position_velocity_to_message",
    "ros_fluid_pressure_message_to_dict",
    "ros_image_message_to_dict",
    "ros_imu_message_to_dict",
    "ros_magnetic_field_message_to_dict",
    "ros_navsatfix_message_to_dict",
]
