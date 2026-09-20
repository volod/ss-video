# Setup

## Docker stack

Prerequisites:

- Docker with Compose support
- Optional GPU support via NVIDIA Container Toolkit

Start the main stack:

```bash
make up
```

`make up` creates writable `data/` and `cache/` directories, then starts `postgres`, `qdrant`, `api`, `fusion-rt`, `worker`, `ui`, `nginx`, `mediamtx`, and any default compose services. The video API applies the video Postgres schema on startup; fusion-rt applies the fusion schema; the worker applies video only (`ssv-migrate` / `python -m selfsuvis.scripts.migrate_postgres` remains the CLI).

`mediamtx` is configured from `config/mediamtx/mediamtx.yml` with a publisher-friendly default path policy plus the internal control API on `:9997`. The compose stack publishes RTSP on `8554`, RTMP on `1935`, and HLS/WebRTC ports for live feeds.

For the full live-stream and MediaMTX operator guide, see [MediaMTX streaming](../reference/streaming-mediamtx.md).

Optional helpers:

- `make cvat-up` to start CVAT services
- `make logs` to follow stack logs
- `make down` to stop the stack

If GPU containers fail to start, install the toolkit with `sudo ./scripts/install/install_nvidia_docker.sh` or use CPU-only workflows where possible.

## Local development setup

Install host dependencies:

```bash
sudo ./scripts/install/install_system_deps.sh --with-python
make venv
```

`make venv` installs the project from `pyproject.toml` and uses its optional
dependency groups as the single source of truth for Python requirements.

Then start the services you need. Typical split:

```bash
docker compose -f docker/core/docker-compose.yml up -d postgres qdrant redis
python -m selfsuvis.scripts.migrate_postgres
.venv/bin/uvicorn selfsuvis.app.main:app --reload --host 0.0.0.0 --port 8000
.venv/bin/uvicorn selfsuvis.fusion_rt.app:app --reload --host 0.0.0.0 --port 8001
.venv/bin/python -m selfsuvis.worker
python -m selfsuvis.ui --server.address 0.0.0.0 --server.port 8501
```

## Default URLs

- UI: `http://localhost:8501`
- API: `http://localhost:8000`
- fusion-rt: `http://localhost:8001`
- Qdrant: `http://localhost:6333`
- Nginx static server: `http://localhost:8080`
- SuperSplat viewer: `http://localhost:8090`
- CVAT when enabled: `http://localhost:8091`

---
[← Overview](../README.md) | [Develop →](develop.md)
