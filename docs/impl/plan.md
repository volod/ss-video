# Implementation Plan (forward work)

Forward-only: this file describes work that remains. Available behavior belongs in
[current-state documentation](current.md). Product behavior belongs in the
[specification](../design/spec.md).

Every task serves a capability from the [capability registry](../design/spec.md#capability-registry).
Task fields are defined in the [planning workflow](../guide/planning-workflow.md).

This repository is ss-video. It pins `ss-perception`, `ss-mapping`, `ss-fusion`, and
`fusion-rt` from [volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.2.0`, and
`ss-common` from [volod/ss-common](https://github.com/volod/ss-common) tag `v0.2.1`.

## Agent Implementation Tasks

### Near-real-time 4D scene analysis -- `four-d-scene-analysis`

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

#### four-d-verified-event-delivery

Accepted 4D events stay in `published-events.jsonl`. The site correlator does not
receive those envelopes.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#processing-profiles-and-data-flow)
- Agent status: CLEAR
- Dependencies: `four-d-profile-orchestration`.
- User-visible outcome: Each newly accepted event is handed to the existing video MQTT contract publisher, and a replay does not send it again.
- Scope boundary: Reuse `pipeline/realtime/contract_publisher.py` and the `verified-scene-event` payload. Do not add or change a fusion-rt correlation rule. Do not publish rejected or uncertain rows.
- Data and artifact paths: `src/selfsuvis/pipeline/analysis4d/publish.py`, `src/selfsuvis/pipeline/realtime/contract_publisher.py`, and `$DATA_DIR/analysis/<mission_id>/4d/published-events.jsonl`.
- Execution path: Publish one accepted envelope through the existing MQTT client seam, skip an id already in the ledger, and leave rejected timeline rows unsent.
- Acceptance gates: `make test-unit` passes; a repeated event id produces one publish call; rejected and uncertain rows produce none; the fusion rule set is unchanged.
- Documentation target: [Operations](../operations/operations.md) and the [profile orchestration runbook](../runbooks/four-d-profile-orchestration.md).

## Human-Assisted Tasks

No human-lane tasks remain.

## Future-task candidates

Not scheduled. Promote an id into a lane only after the [specification](../design/spec.md)
has a matching capability row. Study notes:
[learning path 07](../learning_path/07_future_directions.md).

### This repository (`video-search`)

- `query-eval-harness` -- pinned query set and recall@k on indexed missions
- `cvat-active-learning-loop` -- accepted labels into re-embed or fine-tune with a measurable search lift
- `live-caption-quality` -- RtspCaptioner gates on live RTSP (drift, silence, failover)
- `cross-view-retrieval` -- same-object retrieval across cameras at one site
- `mission-bundle-ingest` -- operator UX for `mission-bundle` manifests
- `global-map-multi-site` -- persist ICP global maps without mixing ENU origins

### Research pipeline ([ss-fusion plan](https://github.com/volod/ss-fusion/blob/v0.2.0/docs/impl/plan.md))

- kernel follow-ups: `kernel-streaming-mode`, `kernel-video-index-spec`, `kernel-benchmark-harness`, `kernel-native-edge`, `ss-kernel-extraction`

### Sensor mesh ([ss-sens plan](https://github.com/volod/ss-sens/blob/v0.1.0/docs/impl/plan.md))

- `lorawan-fuota` -- fleet OTA without WiFi
- `mcuboot-secure-boot` -- STM32 signed boot
- `rust-sdr-dsp`
- `hil-runner` -- self-hosted hardware-in-the-loop CI after the field pilot
- `sens-name-cleanup` -- remaining `COOP_*` env vars (package/CLIs already use ss-sens names)

### Shared contracts

- `odcs-kit` -- contract tooling shared with fl-op
