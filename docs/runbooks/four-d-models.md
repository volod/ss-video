# 4D keyframe and track models

Prompted 2D tracks for one mission. Heavy grounding runs on narrative-changing
keyframes. A lightweight tracker carries identities between those frames. This path
does not train models and does not load every candidate at once.

Current behavior:
[production server](../impl/current/production-server.md#four-dimensional-keyframes-and-tracks).
Configuration:
[configuration](../reference/configuration.md#4d-keyframes-and-tracks).

## Pin

The reference-host pin is Grounding DINO tiny plus kinematic masks. Counting is off.

| Field | Value |
| --- | --- |
| Grounding | `grounding_dino` |
| Model id | `IDEA-Research/grounding-dino-tiny` |
| Revision | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| Weights | `sha256:1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` |
| Mask | `kinematic` |
| Count | off |
| License | Apache-2.0 |
| VRAM budget | 8 GiB peak allocation |
| Latency budget | 1.0 s per keyframe after one warmup forward |

Weights live under `$DATA_DIR/hf-cache`. The loader sets `HF_HOME` and
`HUGGINGFACE_HUB_CACHE` only when those variables are unset. The digest is the
sha256 of `model.safetensors`, which is the Hugging Face blob name the snapshot
symlink points at.

Measured on this host (NVIDIA GeForce RTX 4060 Ti, 16380 MiB) during
`python -m selfsuvis.pipeline.analysis4d.track_benchmark`: latency
`0.13969442749023436` s, peak allocation `938438144` bytes. The report is
`$DATA_DIR/analysis/_benchmark/tracks-report.json`.

## Candidates left out

The benchmark probes these in order and keeps the first grounding provider and the
first mask propagator that pass the license, memory, and latency gates.

| Provider | Role | This host |
| --- | --- | --- |
| `grounding_dino` | grounding | pinned |
| `sam3` | joint alternative | license unset; `sam3` does not import |
| `rf_detr` | grounding | `rfdetr` does not import |
| `yolo_sam` | grounding | `ultralytics` does not import |
| `sam2` | mask | `sam2` does not import |
| `kinematic` | mask | pinned; no neural weights |
| `countgd` | count only | not in ss-perception `v0.2.0` |

CountGD output is never a persistent identity. SAM 3, if a later host accepts its
license and it passes the gate, replaces Grounding DINO plus a separate mask model.
It is not an extra production pass. YOLO and RF-DETR remain the indexer paths
documented in ss-fusion; they are not loaded for 4D tracks.

A whole-task no-go applies only when no provider passes the gates. This host has a
pin, so later 4D tasks stay on the plan.

## What a mission run writes

`selfsuvis.pipeline.workflows.analysis4d_tracks.run_mission_tracks` writes
`$DATA_DIR/analysis/<mission_id>/4d/`. The worker does not call it yet.

- `tracks.jsonl` holds fast and deep observations. Ids are prefixed (`fast-`, `deep-`) and the combined log is time-ordered.
- `masks/<track_id>/<milliseconds>.json` stores an 8x8 kinematic mask reference.
- `track-audit.json` lists keyframe reasons, memory resets, and count disagreements.
- `gaps.jsonl` records quality failures and coalesced frames. An empty file is valid when nothing was skipped.

Forced keyframe reasons are `stream_start`, `scene_cut`, `prompt_change`,
`calibration_change`, `max_gap`, `track_birth`, `track_death`, `low_association`,
and `count_disagreement`. A cut or a calibration change ends live tracks and records
a memory reset. The deep pass may link the next birth to the ended track
(`camera_cut` or `occlusion`) without changing the old observation. The fast pass
does not write that link.

## Benchmark

```bash
python -m selfsuvis.pipeline.analysis4d.track_benchmark
python -m selfsuvis.pipeline.analysis4d.track_benchmark --skip-gpu
```

`--skip-gpu` scores the fixture corpus with scripted detections. The default command
also probes CUDA and checks the cached weights against the pin. It replaces
`$DATA_DIR/analysis/_benchmark/tracks/` on each run. Pass means:

- every annotated `truth.json` keyframe boundary has a selected keyframe within 0.5 s
- both profiles are timestamp-monotonic
- a fixed seed repeats the same track ids
- the occlusion/cut fixture does not keep one id across the cut (`silent_id_reuse` is false)
- the live pin is `grounding_dino` plus `kinematic`, and the cached blob matches the table above
