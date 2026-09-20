# Quick Start — Production (Docker)

Deploy SelfSuvis using Docker — the recommended path for production use. No host Python installation required.

---

## Prerequisites

**Required:**
- Git
- Docker Engine >= 24 with Compose v2

**GPU (optional but recommended):**
- NVIDIA Container Toolkit — run `sudo ./scripts/install/install_nvidia_docker.sh` if it is not installed

---

## 1. Clone and enter the repo

```bash
git clone <repo-url>
cd selfsuvis
```

---

## 2. Configure environment

Generate a repo-root `.env` for production first:

```bash
python -m selfsuvis.scripts.generate_env --env prod
```

Then set the values that have no default:

```
API_KEY=<choose-a-secret>
ALLOWED_INDEX_PATHS=/app/.data/videos
```

The API now fails closed in production: when `APP_ENV=prod`, startup raises an error if `API_KEY` is empty unless you explicitly override `API_AUTH_REQUIRED=false`.

**Optional — CVAT annotation webhook** (required if you use CVAT for supervised fine-tuning):

```
CVAT_WEBHOOK_SECRET=<openssl rand -hex 32>
CVAT_API_TOKEN=<your-cvat-api-token>
```

`CVAT_WEBHOOK_SECRET` must match the secret you enter in CVAT's webhook settings (see [Configuring the CVAT webhook](#configuring-the-cvat-webhook) below). When it is unset the webhook endpoint rejects all incoming requests.

Edit `.env` directly:

```bash
$EDITOR .data/.env
```

---

## 3. Start the stack

```bash
make up
```

This builds images and starts `postgres`, `qdrant`, `api`, `worker`, `ui`, `nginx`, and `mediamtx`. Wait until you see `api-1 | Application startup complete`. The API and worker apply the idempotent Postgres schema before they serve or poll jobs.

MediaMTX is started with RTSP, RTMP, HLS/WebRTC ports and its internal control API enabled. The API container talks to MediaMTX over the compose network through `http://mediamtx:9997`; that control port is not published on the host.

---

## 4. Optional: run the database migration from the CLI

Schema bootstrap also runs inside the API and worker on startup. Use the CLI when you want to apply it without starting those services:

```bash
python -m selfsuvis.scripts.migrate_postgres
```

If you do not have Python on the host, run it inside the api container:

```bash
docker exec -it docker-api-1 python -m selfsuvis.scripts.migrate_postgres
```

---

## 5. Open the UI

```
http://localhost:8501
```

Upload a video or provide a URL. Run text or image queries once indexing finishes.

---

## Useful make targets

| Command | Action |
|---|---|
| `make up` | Build and start all containers |
| `make down` | Stop all containers |
| `make logs` | Stream last 100 lines and follow |
| `make test-unit` | Run unit tests (no Docker required) |
| `make lint` | ruff check + format check + spec-plan and doc-link checks |

---

## Running the pipeline

Once all services are up, the indexing pipeline runs automatically in the background worker whenever a job is enqueued. Here are the three ways to trigger it.

### Option A — UI (easiest)

Open `http://localhost:8501`. Use the upload widget to submit a video file or paste a URL. The worker picks it up immediately; progress is shown on the jobs page.

### Option B — API upload

```bash
curl -X POST http://localhost:8000/index/video \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/mission.mp4"
```

Response returns a `job_id`. Poll for status:

```bash
curl http://localhost:8000/jobs/<job_id> -H "X-API-Key: $API_KEY"
```

### Option C — index a local directory

Set `ALLOWED_INDEX_PATHS` in `.env` to the directory you want to expose, then:

```bash
curl -X POST http://localhost:8000/index/dir \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"path": "/abs/path/to/videos"}'
```

### What happens during indexing

1. **Frame extraction** — two passes: dense frames for SfM (pycolmap pose estimation), sparse keyframes for search
2. **Captioning** — Florence-2 generates a text caption per keyframe
3. **Embedding** — OpenCLIP (and optionally DINOv3) encodes each keyframe -> stored in Qdrant
4. **Active learning tagging** — frames with high uncertainty get `al_tag=needs_annotation` for future fine-tuning
5. **Change detection** — GPS-overlapping frames from earlier missions are compared; results saved to `change_detections`
6. **Report** — HTML mission summary written to `.data/reports/<mission_id>/summary.html`

Full logs appear in the worker terminal. Job status transitions: `pending -> running -> finished` (or `error`).

### Querying after indexing

**Text search:**

```bash
curl -X POST http://localhost:8000/query/text \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "damaged road surface", "top_k": 10}'
```

**Image search:**

```bash
curl -X POST http://localhost:8000/query/image \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/reference.jpg" \
  -F "top_k=10"
```

Both endpoints return a ranked list of matching frames with paths, captions, and scores. The UI exposes both via the search bar and image-upload widget.

### Live drone stream ingestion

Create a managed live stream and attach the realtime caption pipeline:

```bash
curl -X POST http://localhost:8000/realtime/streams \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "drone-1",
    "mission_id": "mission-live-drone-1",
    "path_name": "live/drone-1",
    "caption_fps": 1.0
  }'
```

The response returns the `publish_url` / `read_url`. Push a drone or test video feed into MediaMTX with RTSP or RTMP. Example with ffmpeg:

```bash
ffmpeg -re -i /path/to/drone.mp4 -c copy -f rtsp rtsp://localhost:8554/live/drone-1
```

To have MediaMTX pull an upstream RTSP / RTMP source instead of publishing into it, pass `source_url` when creating the stream:

```bash
curl -X POST http://localhost:8000/realtime/streams \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "drone-2",
    "path_name": "live/drone-2",
    "source_url": "rtsp://camera.example.com:554/stream"
  }'
```

List or stop live streams:

```bash
curl http://localhost:8000/realtime/streams -H "X-API-Key: $API_KEY"
curl -X POST http://localhost:8000/realtime/streams/<session_id>/stop \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"delete_path": true}'
```

---

## Configuring the CVAT webhook

Skip this section if you are not using CVAT for annotation-driven fine-tuning.

### 1. Generate a secret and add it to `.env`

```bash
echo "CVAT_WEBHOOK_SECRET=$(openssl rand -hex 32)" >> .data/.env
```

Restart the stack so the API picks it up:

```bash
make down && make up
```

### 2. Register the webhook in CVAT

1. Open CVAT at `http://localhost:8091`.
2. Click your avatar (top-right) → **Settings** → **Webhooks** → **Create webhook**.
3. Fill in the form:

   | Field | Value |
   |---|---|
   | **Target URL** | `http://api:8000/webhook/cvat` (inside Docker) or `http://<host>:8000/webhook/cvat` |
   | **Secret** | The value you set for `CVAT_WEBHOOK_SECRET` |
   | **Events** | Check **Job updated** and **Task updated** |
   | **Content type** | `application/json` |

4. Save.

CVAT will HMAC-SHA256-sign every POST using that secret in the `X-Hook-Secret` header. The API verifies the signature before processing any event.

### 3. Verify

Annotate one frame in CVAT and mark the job as completed. The API logs should show:

```
CVAT webhook received: event=update:job
CVAT webhook: task_id=<N> completed → N frames annotated
```

---

## Next steps

- [Configuration](../reference/configuration.md) — full env var reference and security settings
- [Data layout](../reference/data_layout.md) — where files are written, sensor sidecars, output artifacts
- [API reference](../reference/api.md) — HTTP endpoints including the robot pose API
- [Troubleshooting](../operations/troubleshooting.md) — common errors and fixes
