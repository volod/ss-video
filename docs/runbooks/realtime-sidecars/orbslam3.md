# ORB-SLAM3 Sidecar

## Use when

Choose `ORB-SLAM3` when:

- the deployment is camera-first
- IMU may exist but LiDAR does not
- you need a fallback to `VINS-Fusion`

## Strengths

- flexible camera modes
- useful fallback when visual-only or visual-inertial SLAM is enough
- broad operator familiarity

## Weaknesses

- motion blur and repeated textures can hurt tracking
- relocalization tuning is operationally heavier than the stub path

## Select it

```bash
export REALTIME_POSE_BACKEND=orbslam3
export REALTIME_ORBSLAM3_API_URL=http://realtime-orbslam3:8101
```

## Deploy it

```bash
export REALTIME_ORBSLAM3_IMAGE=registry.example/orbslam3-sidecar:latest
docker compose -f docker/core/docker-compose.yml -f docker/realtime/docker-compose.realtime-engines.yml up -d realtime-orbslam3
```

## Integration contract

The sidecar must expose:

- `POST /estimate_pose`
- `GET /stats`
- `GET /health`

Recommended inputs:

- monocular or stereo camera feed
- optional IMU packets

Expected output:

- `pose.source="orbslam3"`
- ENU pose and orientation
- tracking status that surfaces relocalization failures

## Operational notes

- validate the camera mode used by the chosen build
- keep vocabulary and map assets versioned with the airframe configuration
