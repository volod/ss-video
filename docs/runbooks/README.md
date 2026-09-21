# Runbooks

Operations runbooks for models used by the video indexer and for realtime pose sidecars.

## Indexer models

| Runbook | Use |
|---|---|
| [clip-dino.md](clip-dino.md) | CLIP + DINOv3 embeddings |
| [florence-2.md](florence-2.md) | Florence-2 captioning |
| [whisper-asr.md](whisper-asr.md) | Whisper ASR |
| [ocr.md](ocr.md) | OCR text extraction |
| [depth.md](depth.md) | Monocular depth |
| [detection-hf.md](detection-hf.md) | HF object detection |
| [yolo-sam.md](yolo-sam.md) | YOLO11 + SAM2/3 |
| [four-d-models.md](four-d-models.md) | 4D keyframes, Grounding DINO, and track propagation |
| [four-d-geometry.md](four-d-geometry.md) | 4D depth, boxes, and masked appearance |

Research-pipeline models (Gemma, Qwen, UniDrive, world-model, RF-DETR tracking) live in
[volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.2.0`.

## Realtime mapping sidecars

| Runbook | Scope |
|---|---|
| [realtime-sidecar-selection.md](realtime-sidecar-selection.md) | Sidecar selection matrix |
| [realtime-bridge-runtimes.md](realtime-bridge-runtimes.md) | Telemetry bridge daemons |
| [realtime-reference-sidecar.md](realtime-reference-sidecar.md) | Project-owned reference service |
| [realtime-sidecars/vins-fusion.md](realtime-sidecars/vins-fusion.md) | `VINS-Fusion` pose sidecar |
| [realtime-sidecars/orbslam3.md](realtime-sidecars/orbslam3.md) | `ORB-SLAM3` pose sidecar |
| [realtime-sidecars/lio-sam.md](realtime-sidecars/lio-sam.md) | `LIO-SAM` pose sidecar |
| [realtime-sidecars/nvblox.md](realtime-sidecars/nvblox.md) | `nvblox` occupancy sidecar |
| [realtime-sidecars/voxblox.md](realtime-sidecars/voxblox.md) | `voxblox` occupancy sidecar |
