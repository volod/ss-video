# ss-video Specification

## Purpose

ss-video is the video ingest and search service for outdoor autonomy. It indexes mission
video, stores frame metadata and embeddings, and answers text, image, scene, and pose
queries. Import package `selfsuvis`.

Perception, mapping, and the research pipeline live in
[volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.1.0`. Site sensors live in
[volod/ss-sens](https://github.com/volod/ss-sens). Ingress and SSO live in
[volod/ss-control](https://github.com/volod/ss-control). Shared contracts live in
[volod/ss-common](https://github.com/volod/ss-common).

This document owns product behavior, boundaries, evaluations, and the
[capability registry](#capability-registry). The [forward plan](../impl/plan.md) owns work that
remains; [current implementation](../impl/current.md) owns what exists. The specification is
living; see [Extending this specification](#extending-this-specification).

## Design principles

- **Single-domain service.** This repository owns video ingest, search, missions, labeling,
  and the operator UI. It does not own training, sensor decoding, or incident fusion.
- **Pin heavy code.** `ss-perception` and `ss-mapping` come from ss-fusion git tags. fusion-rt
  may run as a compose sidecar from the same tag.
- **Contracts before code.** Cross-service messages are ODCS contracts in ss-common.
- **Evidence over prose.** Every capability declares an evaluation and a valid negative result.

## Target architecture

| Repository | Owns |
| --- | --- |
| `ss-video` (this repository) | ingest (upload, URL, directory, RTSP), MediaMTX, Frigate, captioning, embeddings and Qdrant, query API, missions, robot pose bridges, global maps, UI, CVAT labeling, re-embedding |
| `ss-fusion` | research steps, perception and mapping packages, fusion-rt |
| `ss-sens` | LoRaWAN, MQTT mesh, device registry |
| `ss-control` | reverse proxy, identity, site CA |
| `ss-common` | ODCS contracts and `ss_kit` |

**Interfaces.** REST behind the ss-control proxy; MQTT `ss/v1/...` with mTLS for camera and
scene events; model artifacts pinned from ss-fusion.

## Video search

Ingest mission video, embed frames (CLIP and DINO), enrich them (captions, ASR, OCR, depth,
detection), store metadata in PostgreSQL and vectors in Qdrant, and answer text, image, scene,
and pose queries.

**Boundary.** Training, distillation, and the 36-step research run are ss-fusion. Incident
correlation is fusion-rt (sidecar). Sensor decoding is ss-sens.

**Evaluation.** Docker integration suite and unit tests over app, worker, and storage.
Valid negative result: if a model extra is missing, that enricher degrades and the rest of
ingest continues.

Current state: [production server](../impl/current/production-server.md).

## Four-dimensional video scene analysis

Operators need more than frame retrieval: they need to know which persistent entities were present,
where they moved in a common coordinate frame, how their relationships changed, and which changes
constitute an event of interest. The planned `four-d-scene-analysis` capability turns monocular video,
pose, and optional calibration into a provenance-bearing temporal scene graph, an event timeline, and
spatial Video-QA records. Here, 4D means 3D state evolving over time; it does not imply metrically
accurate reconstruction when metric scale or camera calibration is unavailable.

The capability extends the existing adaptive sampling, CLIP/DINO embeddings, depth, YOLO+SAM,
Gemma-directed tracking, semantic graph, `scene_timeline`, and postflight job seams. It does not
replace frame search or fusion-rt incident correlation.

### Ownership and trust boundary

| Component | Responsibility |
| --- | --- |
| `ss-video` | Scheduling, resource budgets, mission artifacts, graph/event/QA persistence, query APIs, UI, provider-neutral verifier orchestration |
| `ss-perception` in ss-fusion | Grounding, segmentation/tracking, depth/normal, and instance-feature model adapters |
| `ss-mapping` in ss-fusion | Camera/pose transforms, point-cloud fusion, uncertainty propagation, and 3D box fitting |
| ss-fusion research pipeline | Training or adapting graph/VLM models, including SFT/GRPO, and publishing evaluated artifacts |
| fusion-rt | Consuming only verified event envelopes for site correlation; it must not turn raw VLM proposals into incidents |

Model training does not run in the ss-video worker. Heavy implementations arrive through a pinned
ss-fusion release or an explicitly configured sidecar. Every model artifact records model id,
revision, weights digest, license decision, prompt/config digest, and preprocessing version. A remote
VLM is opt-in because frames may leave the deployment boundary and incur variable cost and latency.
Publishing verified events outside ss-video additionally requires a versioned ODCS contract in
ss-common; internal artifact schemas are not a substitute for that cross-service contract.

### Processing profiles and data flow

Both profiles emit the same versioned contracts so a fast result can be superseded, not silently
overwritten, by a deep result.

```text
decoded frames + timestamps + pose/calibration
        |
        +--> cheap quality/motion/scene-cut gate --> lightweight track propagation
        |                                             |
        +--> chunked diversity selector --------------+--> selected keyframes
                                                        |
                    text/exemplar prompts --> grounding + counting
                                                        |
                                           temporal mask tracker
                                                        |
                     depth + normals + camera model + masked DINO features
                                                        |
                                      2D/3D track association and OBBs
                                                        |
                                      append-only 4D scene-graph deltas
                                                        |
                               VLM relation/action/event proposals
                                                        |
                    deterministic geometry + VLM strict verifier
                                                        |
                           verified timeline + spatial Video-QA
                                                        |
                                 search/UI; verified events -> fusion-rt
```

**Fast profile.** Decode continuously, run a lightweight motion/box/mask tracker on every decoded or
rate-limited frame, and reserve expensive models for selected frames. The selector combines the
current blur/motion/histogram/SSIM/embedding-drift gates with a bounded, chunked
[MaxInfo](https://arxiv.org/abs/2502.03183)-style maximum-volume diversity pass. Scene cuts, track
birth/death, low association confidence, prompt changes, count disagreement, and a maximum time gap
force a keyframe. MaxInfo is not treated as a fully online algorithm: the fast path uses bounded
chunks and a deterministic deadline, while the deep path may reselect globally. Heavy grounding is
periodic or trigger-driven; temporal masks and kinematic state propagate between grounding calls.
The scheduler sheds optional VLM and dense-geometry work before dropping track updates.

**Deep profile.** Run postflight or below-real-time processing with global keyframe reselection,
multi-scale grounding, forward/backward mask tracking, denser depth and geometry, cross-cut re-ID,
graph refinement, and stronger VLM verification. It may revise fast-profile identities and event
boundaries, but every revision links to the superseded record.

Each stage reports queue delay, inference time, processed/skipped frames, trigger reason, model
provenance, and degradation flags. Backpressure is explicit: bounded queues may coalesce redundant
keyframes, never reorder timestamps, and must emit a gap record when evidence is discarded.

### Dynamic keyframes and 2D perception

The pipeline uses interchangeable providers selected by a pinned benchmark rather than requiring all
models on every frame.

| Candidate | Value and intended use | Constraint |
| --- | --- | --- |
| MaxInfo-style selection | Preserves embedding diversity within a fixed VLM/keyframe budget and complements the current pairwise drift gate | Global MaxInfo is postflight; the live path must use bounded chunks, forced event triggers, and a heartbeat so a stable but important scene is not lost |
| [Grounding DINO](https://arxiv.org/abs/2303.05499) | Open-set boxes from operator text/referring-expression prompts; initializes tracks and recovers lost targets | Run on selected frames or recovery triggers; normalize phrases into the mission ontology and preserve the raw phrase |
| [CountGD](https://arxiv.org/abs/2407.04619) | Text/exemplar-conditioned open-world counts for crowded, small, or repeated objects | It is a count/localization expert, not an identity tracker; disagreement with unique track counts creates an uncertainty signal rather than duplicate nodes |
| [SAM 2](https://arxiv.org/abs/2408.00714) | Prompted masks plus streaming memory for fast temporal mask propagation and occlusion recovery | Memory is scoped by mission, camera, and track epoch and is reset at hard cuts or calibration changes |
| [SAM 3](https://arxiv.org/abs/2511.16719) | Concept-prompted detection, segmentation, and tracking in one provider | Evaluate as an alternative to Grounding DINO plus SAM 2, not an unconditional second pass; adoption is gated by license, memory, latency, and domain accuracy |

The common grounding result contains `prompt_id`, raw and normalized labels, box, mask reference,
confidence, negative prompts/exemplars, model provenance, and timestamp. Track association combines
mask IoU, lightweight motion prediction, class compatibility, masked appearance similarity, and 3D
gating when available. A track can be `tentative`, `confirmed`, `occluded`, `lost`, or `ended`.
Re-identification across long occlusions or camera cuts creates an auditable identity link; it never
rewrites the source observations.

### 3D reconstruction and persistent features

1. A depth provider offers relative depth (for example
   [Depth Anything V2](https://arxiv.org/abs/2406.09414)) or metric depth and surface normals (for
   example [Metric3D v2](https://arxiv.org/abs/2404.15506)). Intrinsics, lens distortion, pose, mask,
   and valid depth pixels are required to back-project an instance point cloud. Relative-depth output
   stays scale-free unless aligned with SLAM, known camera height, ground plane, GPS/IMU, or another
   metric observation.
2. Masked dense [DINOv3](https://arxiv.org/abs/2508.10104) features are pooled into per-observation
   descriptors and a time-decayed track prototype. They are one association signal, not proof of
   identity. The existing frame-level Qdrant DINO vector remains a retrieval feature; instance
   descriptors use a separately versioned space so model changes cannot mix incompatible vectors.
3. [Perspective Fields](https://arxiv.org/abs/2212.03239) may estimate or validate roll, pitch, and
   field of view when metadata is absent. It does not directly produce global scene normals. Surface
   normals come from a depth-normal model or depth gradients, then transform through the calibrated
   camera pose into the mission frame. Gravity and a robust ground plane regularize orientation.
4. An oriented 3D bounding box is fit from uncertainty-filtered instance points, using gravity-aligned
   robust PCA or minimum-volume fitting. Each box carries coordinate frame, center, extent,
   orientation quaternion, covariance/residuals, observation count, and `metric_scale` status.
   Dynamic objects are fused over short motion-compensated windows; static and dynamic points are not
   merged under the same assumption.

The system must say `relative`, `metric`, or `unavailable` for scale and must not label an estimated
box "physically accurate" without calibration and an evaluation residual. If pose, scale, or depth is
unreliable, the graph retains 2D/ordinal relations and marks metric predicates `uncertain`.

### Temporal 3D scene graph and narrative

The graph is append-only and time-indexed. Nodes represent persistent object tracks, agents, regions,
and places. Observation records attach masks, boxes, instance embeddings, pose, and uncertainty to a
node at a timestamp. Edges are typed subject-predicate-object claims with `[start_sec, end_sec)`,
coordinate frame, confidence, source, evidence ids, and verification status. Graph deltas add, end,
correct, or supersede nodes and edges; the current graph is a materialized view over those deltas.

Deterministic predicates include containment, intersection, distance bands, support/contact,
left/right/above/below in a declared frame, visibility/occlusion, and relative motion. A compact VLM
may propose open-vocabulary attributes, actions, and relations from selected keyframes, region crops,
track histories, and serialized geometry. The event reducer converts verified graph edits into
intervals such as `entered_region`, `left_region`, `picked_up`, `put_down`, `approached`,
`count_changed`, or a configured domain event, with participants and before/after state.

The learned component is inspired, not assumed to be a drop-in dependency, by:

- [SceneGraphVLM](https://arxiv.org/abs/2605.13667) (the likely intended reference rather than
  "SceneGraphLM"), whose compact structured output and SFT-then-GRPO hallucination-aware training are
  candidates for an ss-fusion model artifact;
- [SpatialRGPT](https://arxiv.org/abs/2406.01584), which motivates region prompts and explicit depth
  context for spatial reasoning;
- [VLM-3R](https://arxiv.org/abs/2505.20279), which motivates geometry tokens and evaluation of
  changing spatial relations, without replacing explicit metric geometry;
- [Synthetic Visual Genome 2 / TRaSER](https://arxiv.org/abs/2602.23543), which motivates
  trajectory-aligned and temporal-window representations rather than independent frame graphs; and
- [SG-Ego / GLEN](https://arxiv.org/abs/2607.02425), which motivates describing narrative as
  interpretable graph edits aligned with activities.

SFT/GRPO training, reward design, and dataset construction remain in ss-fusion. ss-video consumes a
schema-constrained inference adapter and compares it with deterministic and simpler baselines before
adoption. Recent preprints are design evidence, not production guarantees.

### Strict verifier

The verifier separates proposal from acceptance:

1. Validate JSON Schema, ids, units, coordinate frames, timestamp ordering, and referential
   integrity. Unknown fields or unversioned model output fail closed.
2. Recompute all geometric predicates from stored masks, poses, point sets, and 3D boxes with explicit
   tolerances and uncertainty. Examples include box intersection/containment, signed axis ordering,
   support-plane contact, distance bands, depth ordering, reprojection residual, and feasible
   velocity/acceleration. Mutually exclusive edges and non-overlapping lifetimes are rejected.
3. Ask a schema-capable multimodal provider to judge only semantic claims not settled by geometry,
   using evidence crops/keyframes, track ids, deterministic measurements, and counter-evidence. A
   local Qwen-VL-family provider is the offline option. A GPT-6 Astra provider uses image inputs and
   strict JSON Schema structured output through the Responses API; it receives selected images or a
   contact sheet because the model does not accept video input. See the official
   [model capabilities](https://developers.openai.com/api/docs/models/gpt-6-astra) and
   [structured-output guide](https://developers.openai.com/api/docs/guides/structured-outputs).
4. Resolve each claim to `accepted`, `rejected`, `uncertain`, or `corrected`. Geometry wins for a
   measurable geometric predicate when its calibration/uncertainty gate passes. Otherwise the claim
   stays uncertain; VLM confidence cannot manufacture metric evidence. Corrections create a new
   claim linked to the original, and all rejected proposals remain in the audit artifact.

The VLM is never the only evidence for identity continuity, counts, metric distance, intersection,
or event time. Provider timeout, refusal, malformed output, or disagreement is a valid negative
result and cannot block deterministic graph production.

### Timeline and Video-QA contract

Artifacts live under `$DATA_DIR/analysis/<mission_id>/4d/`: `manifest.json`, `tracks.jsonl`,
`graph-deltas.jsonl`, `proposals.jsonl`, `timeline.json`, `qa.jsonl`, plus referenced masks and
geometry. PostgreSQL stores queryable event/edge/QA metadata and artifact references; large masks,
point clouds, and dense embeddings remain in the artifact store. The minimum timeline representation
is:

```json
{
  "schema_version": "ss-video.scene-timeline.v1",
  "mission_id": "mission-42",
  "profile": "fast",
  "coordinate_frame": {
    "name": "mission_enu",
    "metric_scale": "metric",
    "calibration_id": "cal-7"
  },
  "model_manifest_ref": "manifest.json",
  "degradations": [],
  "events": [
    {
      "event_id": "evt-19",
      "type": "entered_region",
      "summary": "track-8 entered loading-zone",
      "start_sec": 12.4,
      "end_sec": 14.1,
      "participants": ["track-8", "region-loading-zone"],
      "state_delta_refs": ["delta-103", "delta-104"],
      "location": {
        "frame": "mission_enu",
        "center_m": [4.2, -1.1, 0.7],
        "covariance_diag": [0.08, 0.09, 0.16]
      },
      "confidence": 0.94,
      "verification": {
        "status": "accepted",
        "rules": ["lifetime_overlap", "obb_region_intersection", "reprojection"],
        "vlm_claim_ref": "proposal-88",
        "reasons": []
      },
      "evidence": [
        {
          "frame_id": "mission-42:17:12400",
          "t_sec": 12.4,
          "track_ids": ["track-8"],
          "mask_ref": "masks/track-8/12400.rle",
          "geometry_ref": "geometry/track-8/12400.json"
        }
      ],
      "supersedes": null
    }
  ],
  "qa_pairs": [
    {
      "qa_id": "qa-31",
      "type": "spatial",
      "question": "Which tracked object entered the loading zone after 12 seconds?",
      "answer": {"kind": "track_ref", "value": "track-8", "unit": null},
      "graph_program": "entered(?track, region-loading-zone, after=12.0)",
      "interval_sec": [12.0, 15.0],
      "evidence_event_ids": ["evt-19"],
      "verification_status": "accepted"
    }
  ]
}
```

QA generation follows the executable-ground-truth idea in
[SPRITE](https://arxiv.org/abs/2512.16237): generate or select a graph query, execute it against only
accepted graph state, reject ambiguous or empty answers, then optionally let a VLM paraphrase the
question. The answer is produced by the graph program, not by the VLM. Spatial, temporal, count,
state-change, and evidence-localization questions are allowed; causal questions require an explicit
causal relation from independent evidence and are otherwise excluded.

### Evaluation and valid negative results

The pinned evaluation corpus includes camera cuts, long occlusion, crowded repeated objects, moving
camera, weak/absent calibration, relative-depth-only clips, contradictory VLM proposals, and empty
scenes. It reports keyframe event coverage, grounding AP/recall, count MAE, mask J/F, HOTA/IDF1 and
identity switches, metric or scale-aligned depth error, 3D box IoU/center error, relation precision
and recall, event F1 and temporal IoU, verifier false-accept/false-reject rates, QA accuracy, stage
latency, real-time factor, peak VRAM, queue depth, and artifact size.

Acceptance requires:

- no regression in current video-search evaluation when the capability is disabled;
- fast profile real-time factor <= 1.0 on declared reference hardware for a 15-minute stream, bounded
  queue growth, p95 verified-event lag <= 3 seconds for configured fast-path event types, and no
  skipped-frame interval without a gap record;
- deep profile relation precision >= 0.90 and verifier false acceptance <= 0.02 on the pinned
  contradiction suite; and
- every accepted event and QA answer resolves to stored evidence and reproducible deterministic
  checks, with metric claims disabled when calibration or scale gates fail.

Thresholds for tracking, geometry, and event recall are recorded per dataset after the baseline run
and must improve over the existing YOLO+SAM semantic-graph baseline; they may not be selected on the
test split. A valid negative result is a 2D/ordinal graph with explicit degradation reasons when
models, calibration, pose, or metric scale are missing. Empty timelines and zero QA pairs are valid
when no claim passes verification. Fabricated metric geometry, silently bridged evidence gaps, and
unverified incident publication are not valid degradation modes.

## Capability Registry

| # | Capability | Status | How it is evaluated | Implementation |
| --- | --- | --- | --- | --- |
| 1 | `video-search` | shipped | Docker integration suite and unit tests over app, worker, storage | [Production server](../impl/current/production-server.md) |
| 2 | `four-d-scene-analysis` | planned | Pinned 4D mission corpus: track/geometry/relation/event/QA quality, strict-verifier contradiction suite, provenance audit, and fast-profile latency/backpressure benchmark | [Implementation plan](../impl/plan.md) |

## Extending this specification

A capability gap is a product discovery, not an automatic refusal and not permission for silent
scope growth. Use this lifecycle in order:

1. State the problem in operator or domain terms.
2. Amend the owning section of this specification, including what the capability does not do.
3. Declare the measurement, acceptance signal, and valid negative result before implementation.
4. Add a `planned` registry row with that evaluation.
5. Put tasks under the capability in the implementation line; every task declares `Serves`.
6. Build and evaluate, document available behavior under current state, remove finished plan
   scope, and change the registry row to `shipped` with its implementation link.

## Specification and plan integrity

The registry and the [implementation plan](../impl/plan.md) are two views of one product:

- every task serves a registered capability and sits in its capability group;
- every capability declares an evaluation;
- every planned capability has at least one open task;
- every shipped capability links to current-state documentation;
- groups follow registry order in each task lane;
- every task declares the fields required by the [planning workflow](../guide/planning-workflow.md).
