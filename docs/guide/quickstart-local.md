# Quick Start — Local Service Setup

Set up the ss-video API, worker, and UI for local development with hot-reload. This covers the Docker-backed service stack only.

The research pipeline (`ssv --mode local`) lives in
[volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.2.0`.

---

## Prerequisites

**Required:**
- Git
- Docker Engine >= 24 with Compose v2
- Python 3.11
- ffmpeg, libgl1 (`sudo ./scripts/install/install_system_deps.sh --with-python`)

**GPU (optional but recommended):**
- NVIDIA Container Toolkit — run `sudo ./scripts/install/install_nvidia_docker.sh` if it is not installed

---

## 1. Install system dependencies and create the venv

```bash
sudo ./scripts/install/install_system_deps.sh --with-python
make venv
```

---

## 2. Configure environment

Generate `.env` at the project root. The generator auto-detects your GPU and RAM, picks appropriate models, and prints exactly which sidecar commands to run.

**Quick (non-interactive) — use detected hardware defaults:**

```bash
make env
```

**Interactive — choose sidecar backend, profile, models:**

```bash
make env-interactive
```

Both commands write to `repo_root/.data/.env`. The config loader reads it after the packaged `selfsuvis/env/dev.env` defaults from ss-perception, so `DATABASE_URL`, `QDRANT_HOST=localhost`, `DEVICE`, and sidecar URLs are all set for you.

After generation, set the remaining values:

```bash
$EDITOR .data/.env
```

```
API_KEY=<choose-a-secret>   # leave empty for unauthenticated dev use
HF_TOKEN=hf_xxx             # optional — gated HuggingFace models only
```

`ALLOWED_INDEX_PATHS` is pre-set to `./.data/videos` by the generator. Drop mission videos there and the path-based indexing API will accept them. Change it if your videos live elsewhere.

---

## 3. Start sidecars

The generator prints the exact commands at the end of its output. Refer to those, or use the patterns below based on what you chose.

**Ollama** (default — models pulled automatically):

```bash
ollama serve                         # keep running in a terminal
ollama pull <GEMMA_API_MODEL>        # value from .env, e.g. gemma4:e4b
ollama pull <REASONING_MODEL>        # value from .env, e.g. deepseek-r1:14b
```

**vLLM** (if chosen for Qwen or Gemma — each in its own terminal):

Use a separate project-local environment so vLLM's Torch dependencies do not
replace the pipeline's CUDA build:

```bash
uv venv .data/venvs/vllm --python 3.11
UV_CACHE_DIR=.data/.cache/uv uv pip install \
  --python .data/venvs/vllm/bin/python vllm --torch-backend=auto
```

```bash
# Qwen visual model (port 8010)
.data/venvs/vllm/bin/vllm serve \
  <QWEN_MODEL> --port 8010 --max-model-len 8192

# Gemma (port 8000) — only if GEMMA_API_BACKEND=vllm
.data/venvs/vllm/bin/vllm serve \
  <GEMMA_API_MODEL> --port 8000 --max-model-len 8192
```

Replace `<GEMMA_API_MODEL>` / `<QWEN_MODEL>` / `<REASONING_MODEL>` with the values written to `.env`.

For a single 16 GiB GPU, serve only one VLM at a time. Stop a local vLLM
server before a CUDA vision or training step; unlike Ollama, its OpenAI API
does not evict the model on a `keep_alive=0` request. The local `run-full`
path verifies Ollama unloads between CUDA steps.

The `owl10/UniDriveVLA_Nusc_Base_Stage3` repository contains a custom `.pt`
checkpoint, not a Transformers/vLLM chat model. It cannot be served by the
generic UniDrive sidecar client. For ordinary video input, use
`RUN_ARGS=--no-unidrive` until a dedicated UniDrive inference bridge is
configured for its required sensor inputs.

---

## 4. Start backing services only

```bash
docker compose -f docker/core/docker-compose.yml up -d postgres qdrant
```

---

## 5. Run the database migration (first time only)

```bash
APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.migrate_postgres
```

---

## 6. Start each service in a separate terminal

```bash
# Terminal 1 — API (hot-reload enabled)
APP_ENV=dev .venv/bin/uvicorn selfsuvis.app.main:app \
  --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Worker
APP_ENV=dev .venv/bin/python -m selfsuvis.worker

# Terminal 3 — UI
APP_ENV=dev .venv/bin/python -m selfsuvis.ui \
  --server.address 0.0.0.0 --server.port 8501
```

---

## Default service URLs

| Service | URL |
|---|---|
| UI | http://localhost:8501 |
| API | http://localhost:8000 |
| fusion-rt | http://localhost:8001 |
| API docs | http://localhost:8000/docs |
| fusion-rt docs | http://localhost:8001/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Nginx static server | http://localhost:8080 |

---

## Running the API pipeline

Once services are up, the indexing pipeline runs automatically in the background worker. Trigger jobs via the UI at `http://localhost:8501` or via API:

```bash
curl -X POST http://localhost:8000/index/video \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/mission.mp4"
```

See the [Production Quick Start](quickstart-production.md) for full pipeline and querying details.

---

## Optional: MQTT from ss-sens

Clone [volod/ss-sens](https://github.com/volod/ss-sens) and start its stack there, then point this API at localhost MQTT and Frigate:

```bash
APP_ENV=dev \
COOP_MQTT_HOST=localhost \
COOP_MQTT_PORT=1883 \
COOP_MQTT_TLS=false \
COOP_FRIGATE_API_URL=http://localhost:8971 \
.venv/bin/uvicorn selfsuvis.app.main:app \
  --reload --host 0.0.0.0 --port 8000
APP_ENV=dev \
COOP_MQTT_HOST=localhost \
COOP_MQTT_PORT=1883 \
COOP_MQTT_TLS=false \
.venv/bin/uvicorn selfsuvis.fusion_rt.app:app \
  --reload --host 0.0.0.0 --port 8001
```

---

## Next steps

- [Configuration](../reference/configuration.md) — full env var reference and security settings
- [Data layout](../reference/data_layout.md) — where files are written
- [API reference](../reference/api.md) — HTTP endpoints including the robot pose API
- [Troubleshooting](../operations/troubleshooting.md) — common errors and fixes
