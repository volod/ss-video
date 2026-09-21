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

#### four-d-keyframes-and-tracks

Current adaptive sampling is pairwise and current SAM use refines sampled-frame boxes; it cannot
preserve promptable instance identities through long occlusion while bounding heavy-model cost.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#dynamic-keyframes-and-2d-perception)
- Agent status: RUN NEEDED
- Research: yes
- Dependencies: `four-d-contracts-and-benchmark`. Blocks: `four-d-spatial-reconstruction`, `four-d-profile-orchestration`.
- User-visible outcome: A mission produces auditable 2D tracks for text- or exemplar-prompted objects, with heavy grounding limited to narrative-changing keyframes and lightweight propagation between them.
- Scope boundary: Implement the bounded MaxInfo-style selector, trigger policy, provider interfaces, track lifecycle, mask references, and Grounding DINO/CountGD plus SAM 2 or SAM 3 integration through a pinned ss-perception release or sidecar; do not train models in this repository, claim CountGD detections as persistent identities, or run all candidate providers in production.
- Data and artifact paths: `src/selfsuvis/pipeline/workflows/`, `src/selfsuvis/pipeline/analysis4d/`, `tests/assets/analysis4d/`, `$DATA_DIR/analysis/<mission_id>/4d/tracks.jsonl`, and `$DATA_DIR/analysis/<mission_id>/4d/masks/`.
- Execution path: Benchmark candidate provider combinations against the existing YOLO+SAM and RF-DETR path, pin the selected artifact/config, run the fast causal pass and the deep forward/backward pass, and record forced-keyframe reasons, memory resets, gaps, and count-vs-track disagreement.
- Acceptance gates: `make test-unit`, `make test-ci`, and the declared GPU benchmark pass; selected keyframes cover every annotated event boundary in the fixture corpus, track output is timestamp-monotonic and deterministic under a fixed seed, occlusion/cut fixtures do not silently reuse ids, and a documented no-go is valid if no provider meets license, memory, quality, and latency gates and dependent tasks are re-scoped or removed before this task closes.
- Documentation target: [Production server](current/production-server.md), [Configuration](../reference/configuration.md), and a new 4D model runbook.

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

#### four-d-scene-graph

The current semantic environment graph is a mission summary; it does not preserve temporal edge
validity, graph edits, action intervals, or superseded identities.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#temporal-3d-scene-graph-and-narrative)
- Agent status: RUN NEEDED
- Research: yes
- Dependencies: `four-d-spatial-reconstruction`. Blocks: `four-d-strict-verifier-and-qa`.
- User-visible outcome: Operators can inspect how objects, relations, and actions evolved over time and trace every narrative event back to graph deltas and observations.
- Scope boundary: Merge 2D tracks and eligible 3D geometry into append-only temporal nodes/edges, implement deterministic relations and event reduction, and integrate one schema-constrained compact VLM proposal provider; SceneGraphVLM-style SFT/GRPO training and dataset work remain in ss-fusion.
- Data and artifact paths: `src/selfsuvis/pipeline/analysis4d/`, `src/selfsuvis/worker/handlers/`, `tests/assets/analysis4d/`, `$DATA_DIR/analysis/<mission_id>/4d/graph-deltas.jsonl`, and `$DATA_DIR/analysis/<mission_id>/4d/proposals.jsonl`.
- Execution path: Add a postflight graph job and a causal graph reducer, materialize current state from graph deltas, serialize region/geometry/trajectory context for the selected VLM provider, and preserve proposed, corrected, rejected, and superseded claims.
- Acceptance gates: `make test-unit`, `make test-ci`, and the pinned scene-graph benchmark pass; replaying deltas is deterministic, graph intervals and references are valid, deterministic relation precision improves over the current YOLO semantic-graph baseline, and an unavailable VLM yields a useful deterministic graph with an explicit degradation record.
- Documentation target: [Production server](current/production-server.md), [API reference](../reference/api.md), and a new temporal scene-graph runbook.

#### four-d-strict-verifier-and-qa

No current gate prevents a VLM-proposed spatial relation or action from becoming unsupported
narrative output, and Video-QA answers are not compiled from verified graph state.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#strict-verifier)
- Agent status: RUN NEEDED
- Dependencies: `four-d-scene-graph`. Blocks: `four-d-profile-orchestration`.
- User-visible outcome: Timeline events and auto-generated spatial Video-QA are schema-valid, evidence-linked, geometrically checked, and explicit about rejection or uncertainty.
- Scope boundary: Implement deterministic validators, provider-neutral multimodal review with local Qwen-VL and optional strict-JSON remote providers, claim resolution/audit, executable graph-query QA generation, persistence, and read APIs; do not let VLM confidence override reliable geometry or send rejected/uncertain claims to fusion-rt.
- Data and artifact paths: `src/selfsuvis/pipeline/analysis4d/`, `src/selfsuvis/app/routers/`, `src/selfsuvis/pipeline/storage/`, `tests/assets/analysis4d/`, `$DATA_DIR/analysis/<mission_id>/4d/timeline.json`, and `$DATA_DIR/analysis/<mission_id>/4d/qa.jsonl`.
- Execution path: Validate strict schemas, recompute geometry and temporal constraints, submit only unresolved semantic claims with positive and counter-evidence, execute accepted graph programs for answers, and add mission timeline/QA endpoints with evidence retrieval.
- Acceptance gates: `make test-unit`, `make test-ci`, API schema export, and the contradiction/QA benchmark pass; deep-profile relation precision is at least 0.90, false acceptance is at most 0.02 on the pinned contradiction suite, every accepted event and answer resolves to evidence, and timeout/refusal/malformed model output fails closed without blocking deterministic results.
- Documentation target: [Production server](current/production-server.md), [API reference](../reference/api.md), and a new strict-verifier/Video-QA runbook.

#### four-d-profile-orchestration

The component stages need an end-to-end budget controller, observable degradation, and a safe
handoff from verified video events to site correlation.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#processing-profiles-and-data-flow)
- Agent status: RUN NEEDED
- Dependencies: `four-d-keyframes-and-tracks`, `four-d-strict-verifier-and-qa`.
- User-visible outcome: Operators can select fast or deep analysis, see backlog and degraded stages, receive bounded-latency verified events, and later inspect higher-quality superseding results.
- Scope boundary: Add profile configuration, GPU/resource admission, bounded queues, stage telemetry, gap/coalescing records, restart/idempotency behavior, UI/job status, an ss-common ODCS contract, and verified event publication; do not change fusion-rt correlation policy or make remote VLM use the default.
- Data and artifact paths: `src/selfsuvis/worker/`, `src/selfsuvis/pipeline/workflows/`, `src/selfsuvis/pipeline/realtime/`, `src/selfsuvis/app/`, `src/selfsuvis/ui/`, `tests/assets/analysis4d/`, and `$DATA_DIR/analysis/<mission_id>/4d/`.
- Execution path: Wire causal fast and postflight deep DAGs, prioritize track continuity over optional stages, publish only accepted versioned event envelopes, run a 15-minute load fixture on declared reference hardware, and verify restart replay plus fast-to-deep supersession.
- Acceptance gates: `make ci` and the declared load/integration run pass; fast-profile real-time factor is at most 1.0 with bounded queue growth, p95 verified-event lag is at most 3 seconds for configured fast events, every discarded interval has a gap record, deep results supersede rather than overwrite fast results, and disabling the capability preserves current video-search behavior and performance.
- Documentation target: [Production server](current/production-server.md), [Configuration](../reference/configuration.md), [Operations](../operations/operations.md), and a new 4D operations runbook.

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
