# Realtime Bridge Runtimes

`selfsuvis` now ships project-owned telemetry bridge daemons for live MAVSDK and ROS ingestion.

## Purpose

Use these runtimes when telemetry arrives as a live serial or network feed and must be written into a realtime session continuously.

They own:

- transport connection
- session bootstrap
- packet normalization
- batched persistence into `sensor_packets`
- automatic pose updates through the existing realtime ingest path

## Deployment surfaces

Package entrypoint:

```bash
ssv-realtime-bridge --backend mavsdk
ssv-realtime-bridge --backend ros
```

Shell wrapper:

```bash
./scripts/ssv/ssv-realtime-bridge.sh --backend mavsdk
```

Docker compose module:

```bash
docker compose -f docker/core/docker-compose.yml -f docker/realtime/docker-compose.realtime-bridge.yml up -d realtime-mavsdk-bridge
docker compose -f docker/core/docker-compose.yml -f docker/realtime/docker-compose.realtime-bridge.yml up -d realtime-ros-bridge
```

Start only the bridge you actually use.

## Session ownership

By default the bridge auto-creates the target realtime session:

- `REALTIME_BRIDGE_SESSION_ID`
- `REALTIME_BRIDGE_ROBOT_ID`
- `REALTIME_BRIDGE_MISSION_ID`
- `REALTIME_BRIDGE_AUTO_CREATE_SESSION=true`

Packets are flushed in batches using:

- `REALTIME_PACKET_BATCH_SIZE`
- `REALTIME_BRIDGE_FLUSH_INTERVAL_SEC`

## MAVSDK runtime

Use when the autopilot is exposed through MAVSDK and emits NED position / velocity plus attitude telemetry.

Key env vars:

- `REALTIME_BRIDGE_BACKEND=mavsdk`
- `REALTIME_MAVSDK_SYSTEM_ADDRESS=udp://:14540`
- `REALTIME_MAVSDK_SERVER_ADDRESS`
- `REALTIME_MAVSDK_SERVER_PORT`

The runtime subscribes to:

- `position_velocity_ned`
- `attitude_euler`
- `heading`

## ROS runtime

Use when the vehicle stack publishes ROS topics and the bridge container can join the ROS domain.

Key env vars:

- `REALTIME_BRIDGE_BACKEND=ros`
- `REALTIME_ROS_DOMAIN_ID`
- `REALTIME_ROS_IMU_TOPIC`
- `REALTIME_ROS_GPS_TOPIC`
- `REALTIME_ROS_BAROMETER_TOPIC`
- `REALTIME_ROS_MAG_TOPIC`
- `REALTIME_ROS_CAMERA_TOPIC`

The runtime currently subscribes to:

- `sensor_msgs/Imu`
- `sensor_msgs/NavSatFix`
- `sensor_msgs/FluidPressure`
- `sensor_msgs/MagneticField`
- `sensor_msgs/Image`

`NavSatFix` messages are projected to a local ENU-like frame relative to the first GPS fix seen by the bridge.

## Dependency boundary

The base repo does not force-install `mavsdk`, `rclpy`, or ROS message packages.

That is intentional:

- API and batch pipeline installs stay lightweight
- bridge containers can carry transport-specific dependencies separately

If a required library is missing, the runtime fails with a targeted startup error instead of breaking the main application.

## What this replaces

Before this change the repo only had normalization helpers:

- `pipeline/media/mavlink.py`
- `pipeline/media/drone_bridge.py`
- `pipeline/media/ros_bridge.py`

Those helpers still exist, but they are now backed by deployable bridge runtimes instead of being dead-end utility code.
