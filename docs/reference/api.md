# API

Video (ss-video) routes honor `API_KEY` when configured and are subject to rate limiting.
`GET /health` and `GET /index/form` are the only public video endpoints by default.
Site-operations routes (`/api/v1/*`, `/site/state`, `/site/threat`, `/site/synthesis`)
are served by fusion-rt; see `docs/api/fusion-rt-openapi.json` (host port 8001).
The committed video spec is `docs/api/video-openapi.json`.

## Public routes

### `GET /health`

Returns service health and Qdrant connectivity.

### `GET /index/form`

Minimal HTML upload form for manual indexing tests.

## Indexing routes

### `POST /index/video`

Index a single video from:

- uploaded `file`
- allowed local `path`

Form fields:

- `enable_tiles=true|false`

Returns `{"video_id", "job_id"}`.

### `POST /index/url`

Index a remote video by URL. Returns `{"video_id", "job_id"}`.

### `POST /index/dir`

Index all allowed video files under a directory. Returns a list of enqueued jobs.

### `POST /index/precheck`

Deduplication check for one `file`, `path`, or `url` before enqueueing.

### `POST /index/precheck_dir`

Directory-wide precheck. Can optionally enqueue new items with `enqueue=true`.

## Job route

### `GET /jobs/{job_id}`

Returns job `status`, `type`, `progress`, timestamps, and `error`.

## Search routes

### `POST /query/text`

OpenCLIP text search across frames and/or tiles.

Query params:

- `top_k`
- `search_type=both|frame|tile`
- `enable_rerank=true|false`

Body:

```json
{"text": "query string"}
```

### `POST /query/image`

Image-to-frame/tile search.

Form fields:

- `file`
- `top_k`
- `search_type=both|frame|tile`
- `vector_space=clip|dino`
- `enable_rerank=true|false`

### `POST /query/scene`

Structured scene query over `frames.frame_facts_json`.

Supported filters:

- `text`
- `vehicle_count_min`, `vehicle_count_max`
- `road_condition`
- `gps_bbox`
- `time_range`
- `top_k`

### `POST /query/pose`

Robot advisory query against indexed frame locations.

Supports either:

- GPS query: `lat`, `lon`, optional `alt`
- ENU query: `tx`, `ty`, `tz`

Optional filters:

- `robot_ids`
- `global_map_id`
- `radius_m`
- `top_k`

## Admin routes

All `/admin/*` routes require the API key.

### `GET /admin/stats`

Queue depth, worker state, and active-learning tag counts.

### `GET /admin/missions`

Recent missions with map status and discovered splat paths.

### `GET /admin/robots`

Distinct `robot_id` values seen in missions.

### `GET /admin/global-maps`

Current `global_map` rows.

### `GET /admin/export/map-cache`

Exports a compressed NPZ cache for robot-side lookup. Supports mission and GPS filtering.

### `GET /admin/automation-roi`

Automation metrics for annotation and fine-tuning loops.

### `GET /admin/caption-eval`

Captioner health summary, including null rate and model breakdown.

### `POST /admin/reload-model`

Hot-swaps DINO weights from a checkpoint path.

### `POST /admin/reembed-all`

Enqueues a full DINO re-embedding sweep.

## CVAT integration routes

### `GET /admin/cvat/frames`

Returns frames pending annotation, filtered by `al_tag`.

### `POST /admin/cvat/task`

Registers a CVAT task ID to selfsuvis frame IDs.

### `POST /webhook/cvat`

Consumes CVAT webhook events, marks mapped frames annotated, and may enqueue supervised fine-tuning.

## Realtime routes

### `POST /realtime/session/start`

Creates a sensor-oriented realtime session in PostgreSQL.

### `POST /realtime/session/{session_id}/packet`

Ingests realtime packets such as GPS / IMU samples into the current session.

### `GET /realtime/session/{session_id}/state`

Returns realtime session metadata, packet counts, and the latest fused pose if available.

### `GET /realtime/session/{session_id}/pose/latest`

Returns the latest fused pose for a realtime session.

### `POST /realtime/session/{session_id}/map/tile`

Publishes or updates a realtime map tile entry.

### `GET /realtime/session/{session_id}/map/latest`

Returns recent realtime map tiles.

### `POST /realtime/session/{session_id}/semantic`

Publishes a semantic observation for a realtime session.

### `GET /realtime/session/{session_id}/semantic-nearby`

Returns recent semantic observations for a realtime session.

### `POST /realtime/session/{session_id}/stop`

Stops a sensor-oriented realtime session.

### `POST /realtime/session/{session_id}/finalize`

Finalizes a realtime session and can optionally enqueue downstream indexing.

## MediaMTX-backed live stream routes

### `POST /realtime/streams`

Creates or reuses a MediaMTX path, starts a realtime session, and launches background RTSP frame analysis.

Body fields:

- `robot_id`
- `mission_id`
- `path_name`
- `sensors`
- `caption_fps`
- optional `source_url`
- optional `source_on_demand`

Returns:

- realtime `session_id`
- `publish_url`
- `read_url`
- whether the MediaMTX path was newly created
- current background analysis status

### `GET /realtime/streams`

Returns:

- active SelfSuvis live stream runtimes
- current MediaMTX path list from the control API

### `GET /realtime/streams/{session_id}`

Returns one active or completed live stream runtime entry.

### `POST /realtime/streams/{session_id}/stop`

Stops a live stream runtime. With `{"delete_path": true}`, also deletes the MediaMTX path.

For deployment and publish examples, see [MediaMTX streaming](streaming-mediamtx.md).

---
[← Develop](../guide/develop.md) | [UI →](ui.md)
