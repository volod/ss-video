# Production Server

FastAPI API + PostgreSQL-backed async worker + Streamlit UI. Ingests mission video,
embeds frames (CLIP + DINOv3), enriches them (Florence-2 captions, ASR, OCR, depth,
detection), stores metadata in PostgreSQL and vectors in Qdrant, and answers text,
image, scene, and pose queries in real time.

Start with `make up` (compose files under `docker/core/`).

## Module map

| Path | Role |
| --- | --- |
| `src/selfsuvis/app/main.py` | FastAPI app assembly, lifespan services, security middleware |
| `src/selfsuvis/app/routers/` | `admin`, `analysis4d`, `cvat`, `health`, `index`, `jobs`, `query`, `realtime`, `robot`, `scene`, `site` (`GET /site/cameras` only) |
| `src/selfsuvis/app/services/` | `search`, `live_streams`, `camera_streams`, `realtime`, `upload_utils`, `form_templates` |
| `src/selfsuvis/app/deps.py` | API-key auth (timing-safe compare), bounded rate limiting |
| `selfsuvis.fusion_rt.app` (package `fusion-rt` from [volod/ss-fusion](https://github.com/volod/ss-fusion) `v0.2.0`) | fusion-rt FastAPI app: `/api/v1/*`, `/site/state`, `/site/threat`, `/site/synthesis`, `WS /site/stream` |
| `src/selfsuvis/worker/` | Job consumer; `gpu.py` advisory GPU semaphore; `_run.py` persistent event loop |
| `src/selfsuvis/worker/handlers/` | `index`, `finetune`, `reembed`, `postflight` job handlers |
| `src/selfsuvis/ui/` | Streamlit app (`app.py`, `pages/`, `components/`) |
| `src/selfsuvis/pipeline/` | Video remainder: workflows, realtime, ICP mapper, video media/storage |
| `src/selfsuvis/pipeline/analysis4d/` | Versioned 4D contracts, keyframes, tracks, depth, appearance, the temporal scene graph, and the strict verifier |
| `src/selfsuvis/pipeline/workflows/analysis4d_tracks.py` | Fast causal and deep forward/backward 4D track passes |
| `src/selfsuvis/pipeline/workflows/analysis4d_geometry.py` | Back-projected geometry samples and masked appearance prototypes |
| `src/selfsuvis/pipeline/workflows/analysis4d_graph.py` | Postflight temporal scene graph from tracks and geometry |
| `src/selfsuvis/pipeline/workflows/analysis4d_verify.py` | Strict verifier and graph-program Video-QA |
| [volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.2.0` | Perception, mapping, research pipeline, fusion-rt |
| `src/selfsuvis/realtime/` | SLAM/pose bridge runtime + adapters (`pose`, `occupancy`, `registry`) |
| `src/selfsuvis/mapper/` | ICP fusion service (separate container, no GPU) |

## Job lifecycle

1. `/index/video`, `/index/url`, or `/index/dir` creates a PostgreSQL job row.
2. The worker claims the job (`worker/handlers/index.py` -> `VideoIndexer`).
3. Frames are sampled, quality-filtered, embedded (CLIP + DINO named vectors).
4. Core enrichments run: Florence captions, Whisper ASR, OCR, depth, detection.
5. Optional stages when enabled: YOLO+SAM semantic environment graph,
   Gemma-directed tracking (SAM prompts + RF-DETR), Qwen VLM frame reasoning,
   UniDriveVLA expert pass (stored in `frame_facts_json["unidrive_vla"]`).
6. Metadata -> PostgreSQL; vectors -> Qdrant; optional postflight jobs
   (`POSTFLIGHT_MAPPING`, `POSTFLIGHT_SEMANTIC_GRAPH`) run 3D mapping and graphs.

Job types: `INDEX`, `SUPERVISED_FINETUNE`, `REEMBED`, `POSTFLIGHT_MAPPING`,
`POSTFLIGHT_SEMANTIC_GRAPH`, `POSTFLIGHT_SCENE_GRAPH` -- one handler module per type under
`worker/handlers/`. `POSTFLIGHT_SCENE_GRAPH` reads an existing 4D artifact directory. It is
not in the default mapping chain. An accepted
fine-tuning checkpoint also gets a `model-artifact` manifest next to it
([manifests](data-config.md#manifests)).

## Query surface

| Endpoint | Mechanism |
| --- | --- |
| `POST /query/text` | OpenCLIP text embedding vs Qdrant vectors |
| `POST /query/image` | Image embedding, optional DINO vector space |
| `POST /query/scene` | PostgreSQL filtering over `frame_facts_json`, optional CLIP rerank |
| `POST /query/pose` | GPS/ENU spatial filter + vector ranking; accepts robot advisory context |

## fusion-rt (`fusion-rt` package)

Site-operations process, compose service `fusion-rt` (host `8001` -> container `8000`),
image `selfsuvis-fusion-rt:local` from `docker/core/Dockerfile.fusion_rt`
(`install-python.sh --runtime`, no torch). Command
`uvicorn selfsuvis.fusion_rt.app:app`. The Streamlit UI uses `FUSION_RT_URL`
(default `http://fusion-rt:8000`) for `/api/v1/*`.

- `POST /api/v1/events/{modality}` -- normalized sensor event ingest (`EventEnvelope`).
- `GET /api/v1/site/state` -- DB-backed site snapshot (zones + incidents).
- `incidents.py` -- list/detail/ack/dismiss/notes/search/export.
- `rules.py` -- fusion rule CRUD (which event combinations escalate).
- `zones.py` -- zone CRUD + history.
- `GET /api/v1/events/stream` -- SSE push of incident notifications.
- `GET /site/state`, `/site/threat`, `/site/synthesis`, `WS /site/stream`.

MQTT: fusion-rt consumes contract `sensor-event`, `sensor-state`, `camera-event`,
and `scene-caption` topics and persists camera plus sensor events into
`site_events` so the correlator can open incidents. The video API publishes
`camera-event` and `scene-caption` (`pipeline/realtime/contract_publisher.py`)
and still consumes Frigate MQTT. OpenAPI specs:
`docs/api/video-openapi.json` and `docs/api/fusion-rt-openapi.json`
(`make export-openapi`). Integration: `tests/test_fusion_rt.py` via
`make test` (full stack, including the fusion-rt sidecar) or `make test-no-gpu`.

## Realtime layer

- **MediaMTX** is the media edge: accepts RTSP/RTMP publishers, proxies upstream
  sources, and is controlled by the API via `/realtime/streams`.
- **RtspCaptioner** sessions write live captions to `scene_timeline`.
- **Bridge runtimes** (`ssv-realtime-bridge`): pose and occupancy adapters
  replay or bridge ROS/MAVLink-style traces into realtime ingestion without making
  any single SLAM engine mandatory. Compose files under `docker/realtime/`.
- **ss-sens integration**: fusion-rt `FusionContractConsumer`
  (`fusion_rt/mqtt_consumer.py`) subscribes to contract sensor and camera topics;
  `CombinedSiteSnapshot` and `SensorEventIngestor` feed `/site/state`,
  `/site/threat`, and `/site/synthesis`. The video API keeps `GET /site/cameras`
  from `CameraStreamService`. `app/services/camera_streams.py` discovers Frigate
  cameras, registers `ssv/{camera}` paths in MediaMTX, and starts captioner
  sessions. The mesh itself is
  [volod/ss-sens](https://github.com/volod/ss-sens) tag `v0.1.0`.

## State stores

- **PostgreSQL**: video database `selfsuvis` holds `jobs`, `missions`,
  `frames`, `processed_files`, `change_detections`, `global_map` + mapping tables,
  CVAT/automation state (`cvat_tasks`, `system_state`, `gpu_jobs`), model provenance,
  and 4D query metadata (`analysis4d_runs`, `analysis4d_events`, `analysis4d_edges`,
  `analysis4d_qa`).
  Fusion database `selfsuvis_fusion` holds `sensor_keys`, `site_events`, `zones`,
  `fusion_rules`, `incidents`, `incident_notes`. The video API applies the video
  schema and opens `app.state.db_pool`; fusion-rt applies the fusion schema and
  opens its own `app.state.db_pool`. The worker applies video only. `ssv-migrate`
  (`python -m selfsuvis.scripts.migrate_postgres --owner all`) remains the CLI.
  Details: [data-config.md](data-config.md).
- **Qdrant**: frame/tile points with named vectors (CLIP + DINO)
  and payloads: type, mission/robot ids, timestamps, GPS + ENU, model provenance.
  The stack runs Qdrant `v1.19.1` with `qdrant-client>=1.19.1,<2` so search uses
  `/collections/{name}/points/query` (`QdrantStore.search` -> `query_points`).

## Security posture

Fail-closed auth when secrets are missing, timing-safe API-key check, fail-closed
empty `ALLOWED_INDEX_PATHS`, DNS-rebinding peer-IP validation, bounded rate-limit
table, security headers middleware, CVAT webhook HMAC-SHA256 signatures, SHA-256
`stable_point_id`. The shared primitives live in `ss_kit.security` / `ss_kit.web`
and are re-exported from `app/deps.py`, `app/main.py`, and `pipeline/core/utils.py`
([kit](https://github.com/volod/ss-common/blob/v0.2.1/docs/impl/current/kit.md)).
Details: `docs/reference/configuration.md` (security section).

## Four-dimensional analysis contracts

Internal schemas for persistent tracks, graph deltas, proposals, verified timelines, and
spatial QA live in `pipeline/analysis4d/`. They are not an ss-common ODCS contract and
are not published to fusion-rt. Artifact files stay under
`$DATA_DIR/analysis/<mission_id>/4d/` (`worker/artifacts.py` `analysis_artifact_dir`).
JSONL streams are append-only. `timeline.json` and `manifest.json` are replaced only
when the new file names the previous digest in `supersedes_sha256`; the previous file
is copied under `history/`.

PostgreSQL rows in `analysis4d_*` are a query index of one manifest. The run id is the
manifest sha256 hex. Replacing a run that another run already supersedes is rejected.
Large masks and geometry samples stay on disk.

The fixture benchmark does not load a model:

```bash
python -m selfsuvis.pipeline.analysis4d.benchmark
```

It scores the pinned corpus `analysis4d-v1` under `tests/assets/analysis4d/` and writes
`$DATA_DIR/analysis/_benchmark/report.json` (`ss-video.analysis4d-benchmark.v1`). An empty
verified timeline (no events, QA pairs, or tracks) is a valid result. Tracking scores are
single-threshold HOTA and majority-vote IDF1 on the fixtures, not a TrackEval run. The
report's real-time factor and peak VRAM describe this contract runner; the 15-minute
fast-profile gate belongs to a later task. Record:
[0001-four-d-scene-analysis-four-d-contracts-and-benchmark](../records/0001-four-d-scene-analysis-four-d-contracts-and-benchmark.md).

## Four-dimensional keyframes and tracks

`workflows/analysis4d_tracks.py` `run_mission_tracks` writes one mission directory. It is
not yet a worker job; profile admission and the indexer DAG are a later task. The call
runs a fast causal pass and a deep forward/backward pass into the same artifact set.
Heavy grounding runs only on selected keyframes. Between keyframes the tracker
propagates boxes. The pinned mask propagator is kinematic (box motion, no neural
weights). Production loads that one grounding provider and that one mask propagator.

The pinned grounding model is `IDEA-Research/grounding-dino-tiny` at revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`, weights
`sha256:1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`. CountGD is
not loaded. A count, when a caller supplies one, is an audit signal and never a track
id. YOLO+SAM, RF-DETR, SAM 2, and SAM 3 stay off this path on the reference host.

`track-audit.json` (`ss-video.track-audit.v1`) records keyframe reasons, memory resets,
and count-vs-track disagreements. The fast pass does not link identities. The deep pass
may set `identity_link` on a new track after occlusion or a camera cut; it does not
rewrite the earlier observation. A long gap or a cut ends the live id.

Keyframe and track gate:

```bash
python -m selfsuvis.pipeline.analysis4d.track_benchmark
```

The report is `$DATA_DIR/analysis/_benchmark/tracks-report.json`
(`ss-video.track-benchmark.v1`). `--skip-gpu` checks the fixture corpus without loading
weights. Model choice, budgets, and cache layout:
[4D model runbook](../../runbooks/four-d-models.md). Record:
[0002-four-d-scene-analysis-four-d-keyframes-and-tracks](../records/0002-four-d-scene-analysis-four-d-keyframes-and-tracks.md).

## Four-dimensional geometry

`workflows/analysis4d_geometry.py` `run_mission_geometry` reads an existing
`tracks.jsonl` and writes geometry samples plus masked appearance prototypes. It
does not rewrite the track file. A depth provider that fails to load, or that
returns no map, records `provider_unavailable` and leaves the 2D tracks in place.
The worker does not call this pass yet.

Each sample is `ss-video.geometry-sample.v1`. `metric_scale` is `metric` only when
the depth provider claims meters and the camera has a `calibration_id`. Relative
depth stays `relative` even when the pose is metric. A missing pose or missing
intrinsics is `unavailable`. Perspective Fields, when present, fills roll, pitch,
and field of view. It does not invent metric scale or normals. The
`perspective_fields` package is not installed on the reference host.

Boxes are gravity-aligned in the mission ENU frame (gravity +Z). Dynamic tracks,
those whose image speed exceeds 0.15 normalized units per second, are fused over
a one-second window and are not inserted into the static cloud. Normals come from
depth gradients and are rotated by the camera pose. Metric3D is not installed in
ss-perception `v0.2.0`.

Appearance prototypes are `ss-video.track-embedding.v1`. Association requires the
same model id, weights digest, dimension, and preprocessing
`analysis4d-masked-dino-v1`. A mismatch raises `IncompatibleEmbedding`.

The pinned depth pair is Depth Anything V2 Small (relative) and Depth Anything V2
Metric Outdoor Small. ZoeDepth NYU+KITTI also meets the memory and latency gate
and stays an evaluated candidate. Gated Hugging Face repositories use `HF_TOKEN`
from `.env`. A missing or rejected token fails the geometry benchmark with a
detail that names `HF_TOKEN`. The token value is not logged.

```bash
python -m selfsuvis.pipeline.analysis4d.geometry_benchmark
```

The report is `$DATA_DIR/analysis/_benchmark/geometry-report.json`
(`ss-video.geometry-benchmark.v1`). Pins, the synthetic camera, and the cache:
[4D geometry runbook](../../runbooks/four-d-geometry.md). Record:
[0003-four-d-scene-analysis-four-d-spatial-reconstruction](../records/0003-four-d-scene-analysis-four-d-spatial-reconstruction.md).

## Temporal scene graph

`workflows/analysis4d_graph.py` `run_mission_graph` reads `tracks.jsonl` and
`geometry/`, then appends `graph-deltas.jsonl`. Replaying those deltas in
`(t_sec, delta_id)` order is the current graph. The worker job is
`postflight_scene_graph`. It does not replace the YOLO semantic environment
graph and it does not publish events to fusion-rt.

Relation intervals are half-open. A predicate is written when it stops holding,
or one second after the last sample when it holds through the end of the
mission. Accepted edges use the deterministic predicates in the contract.
Metric predicates (`distance_band`, `supports`, `contacts`) are omitted unless
the manifest scale is `metric`. Identity links add a new node whose delta
`supersedes` the earlier node. The earlier track file is not rewritten.

The event reducer turns those edits into `entered_region`, `left_region`,
`approached`, `put_down`, `picked_up`, and `count_changed`. Each accepted event
points at the delta and a geometry sample.

`pipeline/analysis4d/vlm.py` is the schema-constrained proposal adapter.
`ANALYSIS4D_VLM_PROVIDER` defaults to `unavailable`. SmolVLM
(`HuggingFaceTB/SmolVLM-256M-Instruct`) is optional. SceneGraphVLM training
stays in ss-fusion. A missing model, a refusal, or malformed JSON records
`provider_unavailable` and keeps the deterministic graph. Proposals stay in
`proposals.jsonl` with status `uncertain`, `rejected`, or `corrected`. A
correction sets `supersedes`. VLM claims are not accepted timeline events.

Read routes, all API-key protected:

- `GET /analysis/{mission_id}/4d/graph` with optional `t_sec`
- `GET /analysis/{mission_id}/4d/deltas`
- `GET /analysis/{mission_id}/4d/proposals`

```bash
python -m selfsuvis.pipeline.analysis4d.graph_benchmark
```

The report is `$DATA_DIR/analysis/_benchmark/graph-report.json`
(`ss-video.scene-graph-benchmark.v1`). On the pinned scene the deterministic
relation precision is 1.0 and the YOLO semantic-graph baseline is 0.0, because
that baseline emits undirected `near` edges without these predicates or
intervals. An unavailable VLM still yields 19 accepted edges and
`provider_unavailable`. Details:
[temporal scene-graph runbook](../../runbooks/four-d-scene-graph.md). Record:
[0004-four-d-scene-analysis-four-d-scene-graph](../records/0004-four-d-scene-analysis-four-d-scene-graph.md).

## Strict verifier and Video-QA

`workflows/analysis4d_verify.py` `run_mission_verify` reads the mission 4D
directory and supersedes `timeline.json`. It appends `qa.jsonl` and, when a
claim's status changes, a proposal row whose `supersedes` points at the
original. The worker job is `postflight_strict_verifier`. It is not in the
default postflight chain. Rejected and uncertain claims stay in the audit log.
`pipeline/storage/analysis4d.py` `publishable_metadata` is the only row set
that may be handed to fusion-rt, and this job does not call fusion-rt.

Geometry is recomputed before any reviewer runs. A measurable predicate whose
calibration gate passes is accepted or rejected from the stored boxes. Model
confidence cannot override that result. A high residual or a missing metric
scale leaves the claim `uncertain`. Semantic claims that geometry does not
settle go to a reviewer with measurements and counter-evidence. An action is
accepted only when an accepted deterministic event already covers its
interval. Identity, counts, metric distance, intersection, and event time are
not taken from the reviewer alone. Timeout, refusal, and malformed output
record `provider_unavailable` and keep the deterministic timeline.

`ANALYSIS4D_REVIEW_PROVIDER` defaults to `unavailable`. `qwen` loads
`Qwen/Qwen2.5-VL-3B-Instruct`. `remote` calls the Responses API with strict
JSON and image inputs. The remote model name defaults to `gpt-6-astra`.
Neither model is the default.

Read routes, all API-key protected:

- `GET /analysis/{mission_id}/4d/timeline`
- `GET /analysis/{mission_id}/4d/qa`
- `GET /analysis/{mission_id}/4d/evidence` with `event_id` or `qa_id`

```bash
python -m selfsuvis.pipeline.analysis4d.verifier_benchmark
```

The report is `$DATA_DIR/analysis/_benchmark/verifier-report.json`
(`ss-video.strict-verifier-benchmark.v1`). On the pinned contradiction suite
false acceptance is 0.0. Deep-profile relation precision on the pinned scene
is 1.0. Every accepted event and answer cites evidence. Details:
[strict verifier runbook](../../runbooks/four-d-strict-verifier.md). Record:
[0005-four-d-scene-analysis-four-d-strict-verifier-and-qa](../records/0005-four-d-scene-analysis-four-d-strict-verifier-and-qa.md).

## Design decisions

Load-bearing choices: single SQL store, dual embeddings, Florence-2,
pycolmap+nerfstudio, PostgreSQL-based active tagging, MediaMTX, Qdrant named
vectors, FastAPI+worker queue, optional ss-sens MQTT, graceful degradation,
realtime mapping as optional sidecars.
