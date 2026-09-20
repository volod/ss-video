# Realtime Reference Sidecar

`docker/realtime/docker-compose.realtime.yml` deploys the project-owned reference realtime service.

## Purpose

Use this service when you need:

- local API bring-up
- contract testing against `/estimate_pose`, `/integrate_frame`, `/stats`
- a no-extra-dependency fallback for CI or laptop development

Do not treat it as a production SLAM or occupancy engine.

## What it does

- estimates a fused pose from normalized `gps`, `imu`, `barometer`, `magnetometer` packets
- writes stub map tiles under `.data/maps/realtime/<session>/...`
- exposes the same HTTP contract used by external realtime engines

## Image

Builds from:

- `docker/realtime/Dockerfile.realtime_reference`

Starts:

- `selfsuvis.mapper.realtime_main:app`

## Run

```bash
docker compose -f docker/core/docker-compose.yml -f docker/realtime/docker-compose.realtime.yml up -d realtime-reference
```

## Select it

```bash
export REALTIME_POSE_BACKEND=stub
export REALTIME_OCCUPANCY_BACKEND=stub
```

## Limits

- no loop closure
- no visual feature tracking
- no LiDAR odometry
- no TSDF / ESDF fusion
- intended for validation, not field-grade mapping
