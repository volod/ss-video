# LIO-SAM Sidecar

## Use when

Choose `LIO-SAM` when the drone stack includes:

- LiDAR
- IMU
- optional GPS prior

This is the most robust pose sidecar in the current OSS set when the environment is weak-texture or visually unstable.

## Strengths

- robust in scenes where visual SLAM degrades
- good long-flight mapping posture
- strong fit for LiDAR-equipped drones

## Weaknesses

- LiDAR integration raises hardware and calibration cost
- clock alignment is critical

## Select it

```bash
export REALTIME_POSE_BACKEND=liosam
export REALTIME_LIOSAM_API_URL=http://realtime-liosam:8101
```

## Deploy it

```bash
export REALTIME_LIOSAM_IMAGE=registry.example/lio-sam-sidecar:latest
docker compose -f docker/core/docker-compose.yml -f docker/realtime/docker-compose.realtime-engines.yml up -d realtime-liosam
```

## Integration contract

The sidecar must expose:

- `POST /estimate_pose`
- `GET /stats`
- `GET /health`

Required inputs:

- LiDAR packets or scans
- IMU packets

Recommended inputs:

- GPS prior for global alignment

Expected output:

- `pose.source="liosam"`
- ENU pose
- orientation and velocity
- pose quality / tracking metadata

## Operational notes

- enforce LiDAR/IMU timestamp discipline
- validate scan rate assumptions against the actual sensor
- pair with `nvblox` when dense GPU mapping is available
