# Future Directions

Open engineering problems, not a description of current behavior. Read this
after you can explain one indexed mission from Postgres, Qdrant, and (if used)
fusion-rt incidents. Scheduled work lives in the [plan](../impl/plan.md).
Unscheduled ids live under [Future-task candidates](../impl/plan.md#future-task-candidates).

## Already in this service

Track-aware SSL, ONNX export, and the 36-step runner are ss-fusion. This
repository already has production ingest, dual embeddings, enrichments, CVAT
hooks, re-embed jobs, MediaMTX, Frigate camera discovery, pose bridges, ICP
maps, and a fusion-rt sidecar. Do not re-propose those as new capabilities.

## 1. Cross-view and cross-modal retrieval

**Gap.** CLIP/DINO retrieval is per-frame in one mission. Frames from two
cameras that observe the same object at the same time are not pulled together.
GPS-aligned revisits (same waypoint, different altitude or lighting) are not
trained as positives.

**Why it matters.** Operators ask "show this vehicle from the other tower" and
"show this pad last week at dusk". That is location and identity persistence,
not only object similarity.

**Candidate ids:** `cross-view-retrieval` (this repo); `cross-modal-temporal-ssl`
(ss-fusion).

**Study first:** CLIP and MoCo papers in [06_study_resources.md](06_study_resources.md);
calibration notes in ss-fusion
[sensor fusion fundamentals](https://github.com/volod/ss-fusion/blob/v0.2.0/docs/learning_path/03_sensor_fusion_fundamentals.md).

## 2. Query evaluation harness

**Gap.** Search quality is judged by hand. There is no pinned query set, no
recall@k, no caption-null rate gate on ingest beyond ad hoc admin endpoints.

**Candidate id:** `query-eval-harness`.

## 3. Live caption quality

**Gap.** RtspCaptioner writes `scene_timeline` but has no acceptance metric for
drift, silence, or model failover on a live path.

**Candidate id:** `live-caption-quality`.

## 4. CVAT loop into re-embed

**Gap.** Webhooks and AL tagging exist. Closing the loop (accept labels ->
re-embed or supervised fine-tune -> measurable search lift) is still operator
choreography.

**Candidate id:** `cvat-active-learning-loop`.

## 5. Global threat persistence

**Gap.** fusion-rt incidents are per process/database. A sector that was hot two
days ago is not a first-class prior for a new mission. Rolling site state in
ss-sens covers minutes, not weeks.

**Candidate id:** `global-threat-persistence` (sidecar + storage design; ss-fusion
and ss-sens share the threat vocabulary).

## 6. Calibration and contradiction

**Gap.** Camera vs IMU "are we moving?" and camera vs RF bearing disagreements
are logs, not estimated clock offsets or boresight residuals.

**Candidate id:** `calibration-contradiction-handling` (ss-fusion math; this repo
would only persist and display the result).

## 7. Environmental fields

**Gap.** RF, gas, and acoustic readings are points. A GP field would give
spatial exceedance probability.

**Candidate id:** `environmental-field-models` (ss-fusion / ss-sens).

## 8. Time-varying 3D scene understanding

**Gap.** The indexer can keep diverse frames, estimate depth, refine detections with SAM, run
directed tracking, and build a semantic environment graph. The `four-d-scene-analysis` capability
now also writes versioned tracks, masked geometry, interval-bearing graph edges, a verified
timeline, and published accepted events (`pipeline/analysis4d/`, corpus `analysis4d-v1`).

**Why it matters.** An operator asking "what changed, where, and which evidence proves it?" needs
track continuity, coordinate-frame discipline, graph edits, and evidence-linked events. Independent
frame captions cannot reliably answer that question, especially through occlusion or a camera cut.

**Shipped capability:** `four-d-scene-analysis`. Read the
[specification](../design/spec.md#four-dimensional-video-scene-analysis) and the
[production server](../impl/current/production-server.md#profile-orchestration).
Two follow-ups are scheduled: [keyframe geometry](../impl/plan.md#four-d-keyframe-geometry)
so a real file can produce depth-backed events, and
[verified event delivery](../impl/plan.md#four-d-verified-event-delivery) so accepted
envelopes reach the existing MQTT publisher.

**Study first:** distinguish relative from metric depth; camera calibration from surface-normal
estimation; a VLM proposal from a verified graph edge; and online bounded keyframe selection from a
global postflight selection. The capability deliberately keeps learned perception and training in
ss-fusion while ss-video owns scheduling, evidence persistence, query, and operator-visible quality.

## Pre-extension checklist

Before adding runtime behavior:

- Which stored fields are evidence vs model guesses?
- Which enrichers degraded or skipped?
- Do scene and pose queries cite enough independent evidence?
- Does the map origin match the GPS claims in the UI?
- If fusion-rt is up, do incidents cite camera events this API actually published?

If you cannot answer from artifacts, stay in chapters 01-05. Adding a model
before the evidence path is understood makes the system harder to debug.

Sibling research themes (kernel extraction, LoRaWAN FUOTA) are listed as
candidates on the plan and owned by ss-fusion or ss-sens.
