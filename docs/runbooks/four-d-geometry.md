# 4D geometry

Back-projects prompted tracks into the mission frame and keeps a versioned
appearance prototype per track. Relative depth is never stored as meters.
Dynamic objects are not merged into the static cloud.

Current behavior:
[production server](../impl/current/production-server.md#four-dimensional-geometry).
Configuration:
[configuration](../reference/configuration.md#4d-geometry).
Indexer depth, which this path does not replace:
[depth](depth.md).

## Pin

Production loads one relative depth model, one metric depth model, and one
appearance model. ZoeDepth is measured and not loaded on the mission path.

| Field | Value |
| --- | --- |
| Relative depth | `depth-anything/Depth-Anything-V2-Small-hf` |
| Relative revision | `5426e4f0f36572d16453bbda7a8389317b1bef99` |
| Relative weights | `sha256:3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70` |
| Metric depth | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` |
| Metric revision | `fd2c22027eaf20374204f14099b8341e1925ad39` |
| Metric weights | `sha256:ad065c77a7421ca55159a1f0db9433397a607690f2d76bb8a6fc54b1be7a3124` |
| Appearance | `dinov3_vitb14` (`dinov2_vitb14_reg4_pretrain.pth`) |
| Appearance weights | `sha256:73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` |
| Preprocessing | `analysis4d-masked-dino-v1` |
| License | Apache-2.0 |
| VRAM budget | 8 GiB peak allocation |
| Latency budget | 1.0 s after one warmup forward |

Weights live under `$DATA_DIR/hf-cache` when `HF_HOME` and
`HUGGINGFACE_HUB_CACHE` are unset. The loader also recognizes a snapshot that
is already in `~/.cache/huggingface/hub`. The digest is the sha256 of
`model.safetensors` (the Hugging Face blob name).

`HF_TOKEN` from `.env` is passed to Hugging Face downloads. It is not printed.
A gated, private, or unknown repository fails the probe with a sentence that
names `HF_TOKEN`: empty when the token is unset, and not visible when the
configured token was sent and the repository still cannot be read. That probe
detail fails the geometry benchmark. The old id `Intel/zoedepth-nk` is not a
repository. The NYU+KITTI checkpoint is `Intel/zoedepth-nyu-kitti`.

Measured on this host (NVIDIA GeForce RTX 4060 Ti, 16380 MiB):

| Provider | Latency | Peak allocation | Gate |
| --- | --- | --- | --- |
| Depth Anything V2 Small | `0.0068689918518066405` s | `125814784` bytes | pinned |
| Depth Anything V2 Metric Outdoor Small | `0.0066085438728332516` s | `125814784` bytes | pinned |
| ZoeDepth NYU+KITTI | `0.06188822555541992` s | `982449664` bytes | cached, not pinned |
| `dinov3_vitb14` | `0.007149727821350098` s | `364275712` bytes | pinned |

ZoeDepth revision `f364d4c7936e91f465abba182208dd68142bf0ca`, weights
`sha256:c5494fa0938f18d71e215e245472470c3aefebd7b434abd89750e5ae4008e2dc`.
Outdoor small is the metric pin because it meets the same gate at lower latency
and memory. Metric3D is not installed in ss-perception `v0.2.0`. Normals are
depth gradients rotated into the mission frame. `perspective_fields` is not
installed. The scripted fallback fills roll, pitch, and field of view only.

## Scale and boxes

`metric_scale` is `metric` only when the depth model claims meters and the
camera has a `calibration_id`. Relative depth stays `relative` when the pose is
metric. Missing pose or intrinsics is `unavailable`.

Boxes are gravity-aligned (mission ENU, gravity +Z). A track is dynamic when
its image speed exceeds 0.15 normalized units per second. Dynamic points are
fused over a one-second window and omitted from the static cloud.

Association of appearance prototypes requires the same model id, weights digest,
dimension, and preprocessing id. A mismatch is `IncompatibleEmbedding`.

## What a mission run writes

`selfsuvis.pipeline.workflows.analysis4d_geometry.run_mission_geometry` reads
`tracks.jsonl` and writes under `$DATA_DIR/analysis/<mission_id>/4d/`. The
fast profile calls `run_keyframe_geometry` when `dense_geometry` is admitted,
which is the path used by the `analysis4d_fast` worker job. The track file
bytes are unchanged. A keyframe with depth and no calibration still stores a
box with `metric_scale` `unavailable`. Those numbers are not meters, the
bundle speed check ignores them, and they stay out of the static cloud.

- `geometry/<track_id>/<milliseconds>.json` is one `ss-video.geometry-sample.v1`.
- `embeddings/<track_id>/` holds `ss-video.track-embedding.v1` prototypes.
- A failed depth provider adds `provider_unavailable` and writes no samples.

## Benchmark

```bash
python -m selfsuvis.pipeline.analysis4d.geometry_benchmark
python -m selfsuvis.pipeline.analysis4d.geometry_benchmark --skip-gpu
```

The report is `$DATA_DIR/analysis/_benchmark/geometry-report.json`
(`ss-video.geometry-benchmark.v1`). `--skip-gpu` skips neural probes and still
scores the synthetic camera in `tests/assets/synthetic-camera/scene.json`.

Pinned tolerances: reprojection at most 0.05 px, box center and extent at most
0.02 m, yaw at most 1 degree, solid-center bias at most 0.05 m. The reference
run measured reprojection `2.096581030415516e-15` px, center and extent and yaw
`0`, and solid-center bias `0.004218628535996772` m. Missing calibration stayed
`relative`. Incompatible embeddings were rejected. A failing depth provider left
the track file unchanged. Dynamic points stayed out of the static cloud.
