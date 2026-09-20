# VINS-Fusion Sidecar

## Use when

Choose `VINS-Fusion` when the drone provides:

- RGB camera
- IMU
- optional GPS prior

This is the default pose estimator for camera-first drone flights.

## Strengths

- strong visual-inertial pose quality for RGB + IMU
- practical default for flight stacks without LiDAR
- mature open-source baseline

## Weaknesses

- calibration quality matters
- time sync drift will degrade pose quickly
- low-texture scenes can still produce drift

## Select it

```bash
export REALTIME_POSE_BACKEND=vins_fusion
export REALTIME_VINS_FUSION_API_URL=http://realtime-vins-fusion:8101
```

## Deploy it

```bash
export REALTIME_VINS_FUSION_IMAGE=registry.example/vins-fusion-sidecar:latest
docker compose -f docker/core/docker-compose.yml -f docker/realtime/docker-compose.realtime-engines.yml up -d realtime-vins-fusion
```

`docker/realtime/docker-compose.realtime-engines.yml` expects an image that already wraps `VINS-Fusion` behind the `selfsuvis` realtime HTTP contract.

## Integration contract

The sidecar must expose:

- `POST /estimate_pose`
- `GET /stats`
- `GET /health`

Expected input signals:

- camera frames or frame references
- IMU packets
- optional GPS priors

Expected output:

- `pose.source="vins_fusion"`
- `position_enu`
- `orientation_quat`
- `velocity_enu`
- covariance / tracking status metadata

## Operational notes

- keep camera and IMU timestamps aligned
- freeze configuration per airframe
- validate with replay traces before live use
