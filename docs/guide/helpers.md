# Helpers

## Install scripts (run with sudo when noted)
- `./scripts/install/install_system_deps.sh` — ffmpeg, OpenCV deps (Linux). Add `--with-python` for Python/venv.
- `./scripts/install/install_nvidia_docker.sh` — NVIDIA Container Toolkit for Docker GPU
- `./scripts/install/install_requirements.sh` — install Python deps from `pyproject.toml` extras into a venv (called by `make venv`)
- `ssv-env` — generate a resource-aware root `.env` from packaged presets

## Pre-download weights for offline use
```bash
python -m selfsuvis.scripts.prepare_models
DOWNLOAD_DINO=true DINO_MODEL=dinov2_vitb14 python -m selfsuvis.scripts.prepare_models
```

## Batch index a directory
```bash
curl -s \
  -F "path=/path/to/video_dir" \
  -F "enable_tiles=true" \
  http://localhost:8000/index/dir | python -m json.tool
```

## Index a URL
```bash
curl -s \
  -F "url=https://example.com/video.mp4" \
  -F "enable_tiles=true" \
  http://localhost:8000/index/url | python -m json.tool
```

## Watch a job
```bash
JOB_ID=<job_id>
while true; do
  STATUS="$(curl -s http://localhost:8000/jobs/${JOB_ID})"
  echo "$STATUS" | python -m json.tool
  STATE="$(printf '%s' "$STATUS" | python -c 'import json,sys; print(json.load(sys.stdin).get("status",""))')"
  [[ "$STATE" == "finished" || "$STATE" == "error" ]] && break
  sleep 2
done
```

## Precheck (avoid double load)
```bash
curl -s -F "file=@/path/to/video.mp4" \
  http://localhost:8000/index/precheck | python -m json.tool

curl -s -F "path=/path/to/video.mp4" \
  http://localhost:8000/index/precheck | python -m json.tool

curl -s -F "url=https://example.com/video.mp4" \
  http://localhost:8000/index/precheck | python -m json.tool
```

## Precheck directory (optionally enqueue new)
```bash
curl -s \
  -F "path=/path/to/video_dir" \
  http://localhost:8000/index/precheck_dir | python -m json.tool

curl -s \
  -F "path=/path/to/video_dir" \
  -F "enqueue=true" \
  -F "enable_tiles=true" \
  http://localhost:8000/index/precheck_dir | python -m json.tool
```

## Reset Qdrant collection
```bash
./scripts/ssv/ssv-reset-qdrant.sh
```

---
[← UI](../reference/ui.md) | [Configuration →](../reference/configuration.md)
