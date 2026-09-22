# Keyframe geometry

## Task and scope

- Id / capability: `four-d-keyframe-geometry` / `four-d-scene-analysis`
- State: accepted
- Source: plan task at `21ebf6b`. The working tree already held uncommitted analyze and profile-orchestration edits. This change calls keyframe depth from the fast profile.
- Plan counts at start: 2 agent, 0 human; statuses `CLEAR=1`, `RUN NEEDED=1`; next eligible `four-d-keyframe-geometry`.
- Accepted task:

```markdown
#### four-d-keyframe-geometry

A file analysis writes 2D tracks from the decoded frames. Verified region and count
events still need stored geometry, and the profile does not call the depth pass
when a GPU slot is free.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#3d-reconstruction-and-persistent-features)
- Agent status: RUN NEEDED
- Dependencies: `four-d-profile-orchestration`.
- User-visible outcome: Operators reviewing a video see depth-backed boxes for kept keyframes, explicit unavailable scale when calibration is missing, and accepted events that cite those boxes.
- Scope boundary: Call the existing geometry workflow from the fast profile when `dense_geometry` is admitted; read optional region boxes from a mission file; do not invent metric scale and do not enable a remote VLM.
- Data and artifact paths: `src/selfsuvis/pipeline/workflows/analysis4d_profile.py`, `src/selfsuvis/pipeline/workflows/analysis4d_geometry.py`, and `$DATA_DIR/analysis/<mission_id>/4d/geometry/`.
- Execution path: Run `python -m selfsuvis.pipeline.analysis4d.analyze` on a short clip with a free GPU slot and confirm geometry samples, or an explicit `provider_unavailable` degradation, plus tracks that survive a full queue.
- Acceptance gates: `make test-unit` passes; a CUDA run writes at least one geometry sample or records `provider_unavailable`; `metric_scale` stays `unavailable` without calibration; a saturated queue still keeps tracks and writes a gap record.
- Documentation target: [Production server](current/production-server.md) and the [profile orchestration runbook](../runbooks/four-d-profile-orchestration.md).
```

- Amendments: none.

## Implementation

`workflows/analysis4d_profile.py` calls `run_keyframe_geometry` after tracks when `dense_geometry` is admitted. Admission still requires a free GPU slot and a queue that did not coalesce. A depth exception records `provider_unavailable` on stage `dense_geometry` and keeps the tracks. The loaded grounding model is dropped before depth weights load. Default VLM and review providers stay `unavailable`.

`workflows/analysis4d_geometry.py` `run_keyframe_geometry` opens images only for `track-audit.json` keyframes that also have an active track timestamp. The default depth provider is the pinned relative Depth Anything V2 Small adapter. Each timestamp is estimated once. When scale resolves to `unavailable` and a depth map exists, the sample is still written: `metric_scale` is `unavailable`, `calibration_id` is empty, `scale_confidence` is 0, and the box is a nominal pinhole ray with depth divided by its median. Those numbers are not meters and they are not added to the static cloud. A calibrated camera still uses the existing metric or relative path.

Optional region boxes are `ss-video.region-boxes.v1` at `$DATA_DIR/analysis/<mission_id>/regions.json`, or `<video-stem>.regions.json` beside the video. `analyze` copies the sibling file when the mission file is absent. Count event ids slug spaces so a prompt such as `graphics card` becomes `evt-count-graphics-card-...`. The summary keeps the prompt text.

Current-state pages: [Production server](../current/production-server.md#four-dimensional-geometry), [profile orchestration](../current/production-server.md#profile-orchestration), [Configuration](../../reference/configuration.md#profile-orchestration), the [profile orchestration runbook](../../runbooks/four-d-profile-orchestration.md), and the [geometry runbook](../../runbooks/four-d-geometry.md).

## Acceptance evidence

| Gate | Exact command, test or artifact | Result and limit |
| --- | --- | --- |
| Unit tests | `make test-unit` | 692 passed, 5 skipped |
| CUDA file analysis | `.venv/bin/python -m selfsuvis.pipeline.analysis4d.analyze .data/tmp/nar-short.mp4 --fps 1 --prompts "graphics card,ship" --mission-id mission-keyframe-geometry` | 4 frames, 4 tracks, 8 geometry samples, 3 accepted `count_changed` events. Device `NVIDIA GeForce RTX 4060 Ti`. Cold-start real-time factor about 3.6 because both models loaded inside the timed pass. That is not the 15-minute scheduler gate. |
| Missing calibration | Same CUDA run and `test_missing_calibration_keeps_depth_boxes_unavailable` | Every sample `metric_scale` is `unavailable`, `calibration_id` is null, manifest frame is `mission_enu` / `unavailable`. Events cite `geometry/<track_id>/<milliseconds>.json`. |
| Saturated queue | `test_saturated_queue_keeps_tracks_without_depth` | 12 frames, capacity 3, slots 1. Depth is not called. Tracks remain. Gaps include `queue_coalesce`. `dense_geometry` is shed with `budget_shed`. |
| Depth failure | `test_failed_depth_keeps_tracks` | `provider_unavailable`, tracks kept, manifest scale `unavailable` |

Runtime artifacts: `$DATA_DIR/analysis/mission-keyframe-geometry/4d/` and `$DATA_DIR/analysis/mission-keyframe-geometry/summary.json`. Summary degradations: `geometry:calibration_missing`, `geometry:pose_missing`, `scene_graph:provider_unavailable`, `strict_verifier:provider_unavailable`.

## Audit handoff

`none identified`. Reviewed scope: keyframe admission, unavailable-scale boxes kept out of the static cloud, optional region file, count-event id tokens, and the default VLM and review providers left off.

## Close or resume

The declared gates for this task passed. Capability `four-d-scene-analysis` stays `shipped`. Plan counts after: 1 agent, 0 human; next eligible `four-d-verified-event-delivery`.
