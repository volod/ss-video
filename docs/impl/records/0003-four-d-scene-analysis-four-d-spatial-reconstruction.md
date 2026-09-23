# 4D spatial reconstruction

## Task and scope

- Id / capability: `four-d-spatial-reconstruction` / `four-d-scene-analysis`
- State: accepted
- Source: plan task at `168f567`. This change is depth, masked appearance, and gravity-aligned boxes.
- Plan counts at start: 4 agent, 0 human; next eligible `four-d-spatial-reconstruction`.
- Accepted task:

```markdown
#### four-d-spatial-reconstruction

Current depth summaries and frame-level DINO vectors cannot back-project masks, maintain
instance-level appearance, or express uncertainty-aware 3D boxes in a mission frame.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#3d-reconstruction-and-persistent-features)
- Agent status: RUN NEEDED
- Research: yes
- Dependencies: `four-d-keyframes-and-tracks`. Blocks: `four-d-scene-graph`.
- User-visible outcome: Each eligible track has relative or metric 3D observations, persistent appearance descriptors, oriented boxes, and explicit calibration/scale confidence instead of implied precision.
- Scope boundary: Integrate depth/normal and masked DINOv3 provider outputs with ss-mapping transforms, Perspective Fields calibration fallback, robust point filtering, motion-aware box fitting, and uncertainty propagation; do not infer metric scale from relative depth alone or merge dynamic objects into a static map.
- Data and artifact paths: `src/selfsuvis/pipeline/analysis4d/`, pinned `ss-perception`/`ss-mapping` packages, `tests/assets/analysis4d/`, `$DATA_DIR/analysis/<mission_id>/4d/geometry/`, and `$DATA_DIR/analysis/<mission_id>/4d/embeddings/`.
- Execution path: Evaluate Depth Anything-style relative depth and Metric3D-style metric depth/normals behind one contract, validate camera/pose transforms and reprojection, pool masked DINO features into versioned track prototypes, and fit gravity-aligned boxes with covariance and residuals.
- Acceptance gates: `make test-unit`, `make test-ci`, and the declared GPU geometry benchmark pass; synthetic-camera fixtures meet pinned reprojection and box-error tolerances, incompatible embedding versions cannot associate, missing calibration yields `relative` or `unavailable` rather than metric output, and provider failure preserves the 2D track graph.
- Documentation target: [Production server](current/production-server.md), [Data/config](current/data-config.md), [Depth runbook](../runbooks/depth.md), and a new 4D geometry runbook.
```

- Amendments: none.

## Implementation

`pipeline/analysis4d/camera.py` uses the ss-mapping camera convention (`X_cam = R @ X_world + t`). `resolve_scale` keeps relative depth relative. Metric output requires a metric depth claim and a `calibration_id`. Missing pose or intrinsics is `unavailable`. Perspective Fields fills roll, pitch, and field of view and does not create metric scale. The `perspective_fields` package is not installed; the scripted fallback is what the tests exercise.

`pipeline/analysis4d/reconstruct.py` filters points per axis, fits a gravity-aligned box, and keeps dynamic points out of `StaticCloud`. Normals are depth gradients rotated by the pose. Metric3D is not in ss-perception `v0.2.0`.

`pipeline/analysis4d/appearance.py` pools a mask over DINO patch tokens. Association requires the same model id, weights digest, dimension, and preprocessing `analysis4d-masked-dino-v1`.

`pipeline/analysis4d/geometry_providers.py` loads Depth Anything V2 Small, Depth Anything V2 Metric Outdoor Small, and the project's `dinov3_vitb14` alias. ZoeDepth is probed at `Intel/zoedepth-nyu-kitti` (the id `Intel/zoedepth-nk` is not a repository). Downloads pass `HF_TOKEN` from settings. A gated, private, or unknown repository sets an explicit `HF_TOKEN` failure. That failure fails the geometry benchmark. The token value is not logged. The metric pin is outdoor small, which meets the gate with less latency and memory than ZoeDepth. Both weight sets are cached under `$DATA_DIR/hf-cache`.

`workflows/analysis4d_geometry.py` `run_mission_geometry` writes `geometry/` and `embeddings/` and does not change `tracks.jsonl`. A depth failure records `provider_unavailable`.

Current-state pages: [Production server](../current/production-server.md#four-dimensional-geometry), [Data/config](../current/data-config.md#four-dimensional-analysis-artifacts), [Configuration](../../reference/configuration.md#4d-geometry), [Depth](../../runbooks/depth.md), and the [4D geometry runbook](../../runbooks/four-d-geometry.md).

## Acceptance evidence

| Gate | Exact command, test or artifact | Result and limit |
| --- | --- | --- |
| Unit suite | `make test-unit` | 645 passed, 5 skipped |
| CI suite | `make test-ci` | 554 passed, 5 deselected |
| GPU benchmark | `python -m selfsuvis.pipeline.analysis4d.geometry_benchmark` | `passed=true`, `failures=[]`. Relative and metric Depth Anything plus `dinov3_vitb14` match the pin and `meets_gate`. ZoeDepth NYU+KITTI available, latency `0.06188822555541992` s, peak `982449664` bytes, not pinned. |
| Synthetic camera | same report, `tests/assets/synthetic-camera/scene.json` | reprojection `2.096581030415516e-15` px (max 0.05), center and extent `0` m (max 0.02), yaw `0` deg (max 1), solid-center bias `0.004218628535996772` m (max 0.05) |
| Embedding versions | `pytest tests/unit/pipeline/analysis4d/test_geometry_recon.py` | incompatible model id, revision, dimension, or preprocessing cannot associate |
| Missing calibration | same report, `missing_calibration_scale` | `relative` |
| Provider failure | same report, `tracks_preserved_on_provider_failure` | `true`; dynamic points excluded from the static cloud |

Runtime report: `$DATA_DIR/analysis/_benchmark/geometry-report.json`. Host: NVIDIA GeForce RTX 4060 Ti, 16380 MiB.

## Audit handoff

`none identified`. Reviewed scope: camera and scale gate, box fit, masked DINO prototypes, Hugging Face token failures, geometry writer, synthetic fixture, and the geometry benchmark.

## Close or resume

The declared gates for this task passed. Capability `four-d-scene-analysis` stays `planned` (three tasks remain). Plan counts after: 3 agent, 0 human; next eligible `four-d-scene-graph`.
