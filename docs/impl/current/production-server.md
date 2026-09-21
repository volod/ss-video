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
| `src/selfsuvis/app/routers/` | `admin`, `cvat`, `health`, `index`, `jobs`, `query`, `realtime`, `robot`, `scene`, `site` (`GET /site/cameras` only) |
| `src/selfsuvis/app/services/` | `search`, `live_streams`, `camera_streams`, `realtime`, `upload_utils`, `form_templates` |
| `src/selfsuvis/app/deps.py` | API-key auth (timing-safe compare), bounded rate limiting |
| `selfsuvis.fusion_rt.app` (package `fusion-rt` from [volod/ss-fusion](https://github.com/volod/ss-fusion) `v0.2.0`) | fusion-rt FastAPI app: `/api/v1/*`, `/site/state`, `/site/threat`, `/site/synthesis`, `WS /site/stream` |
| `src/selfsuvis/worker/` | Job consumer; `gpu.py` advisory GPU semaphore; `_run.py` persistent event loop |
| `src/selfsuvis/worker/handlers/` | `index`, `finetune`, `reembed`, `postflight` job handlers |
| `src/selfsuvis/ui/` | Streamlit app (`app.py`, `pages/`, `components/`) |
| `src/selfsuvis/pipeline/` | Video remainder: workflows, realtime, ICP mapper, video media/storage |
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
`POSTFLIGHT_SEMANTIC_GRAPH` -- one handler module per type under `worker/handlers/`. An accepted
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
  CVAT/automation state (`cvat_tasks`, `system_state`, `gpu_jobs`), model provenance.
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

## Design decisions

Load-bearing choices: single SQL store, dual embeddings, Florence-2,
pycolmap+nerfstudio, PostgreSQL-based active tagging, MediaMTX, Qdrant named
vectors, FastAPI+worker queue, optional ss-sens MQTT, graceful degradation,
realtime mapping as optional sidecars.
