# Realtime And Maps

Live streams and 3D maps are part of this repository. They are optional: ingest
and search work without them.

## MediaMTX

MediaMTX is the media edge: RTSP/RTMP publishers, upstream proxies, path control
from `POST /realtime/streams`. Recordings land under `$DATA_DIR/mediamtx/` when
enabled. Reference: [streaming-mediamtx.md](../reference/streaming-mediamtx.md).

RtspCaptioner sessions write live captions into `scene_timeline` so operators
can search a live path the same way they search an uploaded file.

## Frigate

`make frigate-up` starts Frigate from `docker/core` (profile `frigate`).
`CameraStreamService` discovers cameras, registers `ssv/{camera}` paths in
MediaMTX, and starts captioner sessions. `GET /site/cameras` stays on the video
API. Helpers: `scripts/ssv/ssv-camera.sh`, `add_camera.sh`.

## Pose and occupancy bridges

`ssv-realtime-bridge` adapts ROS/MAVLink-style traces into realtime ingest
without making one SLAM engine mandatory. Compose files live under
`docker/realtime/`. Start with
[realtime-sidecar-selection.md](../runbooks/realtime-sidecar-selection.md), then
the engine-specific runbook (VINS-Fusion, ORB-SLAM3, LIO-SAM, nvblox, voxblox).

## ICP mapper

`src/selfsuvis/mapper/` is a separate container (no GPU). The worker can attach
a mission splat to the global map when ICP converges. ENU origins must not mix
across sites. Postflight job `POSTFLIGHT_MAPPING` runs 3D mapping after index.

Focus: a first mission bootstraps the global map; a second mission with
converged ICP updates the fused splat; a diverged ICP must not replace the map.

Next: [fusion-rt and site](05_fusion_rt_and_site.md).
