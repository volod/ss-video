# Realtime Sidecar Selection

This guide covers how to choose realtime pose and occupancy sidecars for `selfsuvis`.

## What is deployed

There are now two separate deployment surfaces:

- `docker/realtime/docker-compose.realtime.yml`
  - project-owned reference service
  - good for local API bring-up and contract validation
  - not a production SLAM or volumetric mapping engine
- `docker/realtime/docker-compose.realtime-engines.yml`
  - open-source sidecars only
  - limited to `VINS-Fusion`, `ORB-SLAM3`, `LIO-SAM`, `nvblox`, `voxblox`
  - requires engine-specific image references via env vars

## How selection works

`selfsuvis` selects sidecars through runtime env vars:

```bash
export REALTIME_POSE_BACKEND=vins_fusion
export REALTIME_OCCUPANCY_BACKEND=nvblox
export REALTIME_VINS_FUSION_API_URL=http://realtime-vins-fusion:8101
export REALTIME_NVBLOX_API_URL=http://realtime-nvblox:8101
```

The backend catalog is visible through `GET /realtime/backends`. Health and current selection are visible through `GET /realtime/stats`.

## Quick pick matrix

| Need | Pose sidecar | Occupancy sidecar | Why |
|---|---|---|---|
| RGB + IMU drone with no LiDAR | `VINS-Fusion` | `voxblox` or `nvblox` | Best default for visual-inertial flight stacks |
| Camera-only SLAM fallback | `ORB-SLAM3` | `voxblox` or `nvblox` | Good when IMU integration is partial or delayed |
| LiDAR-equipped drone | `LIO-SAM` | `nvblox` | Most robust path for weak-texture or long missions |
| CPU-only mapper host | `VINS-Fusion` or `ORB-SLAM3` | `voxblox` | Avoids GPU dependency |
| GPU mapping host | `VINS-Fusion` or `LIO-SAM` | `nvblox` | Best dense mapping throughput |

## Pros and cons

### `VINS-Fusion`

Pros:
- Strong open-source baseline for RGB + IMU flight pose.
- Good default when GPS is intermittent but camera and IMU are calibrated.

Cons:
- Sensitive to calibration and time alignment quality.
- Not ideal for feature-poor scenes without additional priors.

### `ORB-SLAM3`

Pros:
- Flexible camera modes.
- Practical fallback when LiDAR is absent and camera SLAM is acceptable.

Cons:
- More fragile under heavy blur, low texture, or repeated patterns.
- Needs vocabulary and relocalization care in operations.

### `LIO-SAM`

Pros:
- Best fit when LiDAR is available.
- More robust than visual-only estimators in vegetation, dusk, dust, and repetitive scenes.

Cons:
- Higher integration cost.
- Requires tight LiDAR/IMU synchronization discipline.

### `nvblox`

Pros:
- Best dense mapping option when CUDA GPU is available.
- Good for planning stacks that want TSDF / ESDF products.

Cons:
- GPU dependency raises deployment cost.
- More sensitive to depth noise and bandwidth spikes.

### `voxblox`

Pros:
- CPU-friendly volumetric mapping.
- Easier to deploy on lightweight edge hosts.

Cons:
- Lower throughput than `nvblox`.
- Resolution tuning matters more on weak CPUs.

## Recommended combinations

### Small RGB + IMU drone

- `REALTIME_POSE_BACKEND=vins_fusion`
- `REALTIME_OCCUPANCY_BACKEND=voxblox`

### Survey drone with LiDAR

- `REALTIME_POSE_BACKEND=liosam`
- `REALTIME_OCCUPANCY_BACKEND=nvblox`

### Laptop integration and CI

- `REALTIME_POSE_BACKEND=stub`
- `REALTIME_OCCUPANCY_BACKEND=stub`

Use the reference sidecar only for bring-up, replay development, and API verification. Switch to OSS sidecars before operational validation.

## Compose usage

Reference service:

```bash
docker compose -f docker/core/docker-compose.yml -f docker/realtime/docker-compose.realtime.yml up -d realtime-reference
```

OSS sidecars:

```bash
export REALTIME_VINS_FUSION_IMAGE=registry.example/vins-fusion-sidecar:latest
export REALTIME_NVBLOX_IMAGE=registry.example/nvblox-sidecar:latest
docker compose -f docker/core/docker-compose.yml -f docker/realtime/docker-compose.realtime-engines.yml up -d \
  realtime-vins-fusion realtime-nvblox
```

Only start the sidecars you actually select. Do not run all five by default.
