# Architecture

ss-video is the production video server. It shares Postgres and Qdrant with an optional
fusion-rt sidecar, and consumes MQTT camera events from ss-sens.

```text
Frigate / MediaMTX / uploads
        |
        v
ss-video API + worker + UI ----[REST / Qdrant]---- Client / Robot
        |
        +-- fusion-rt sidecar (published ss-fusion package)
        |
        MQTT camera-event / scene-caption
```

## Repository structure

```text
src/selfsuvis/
  app/          FastAPI routers, request dependencies, and API services
  pipeline/     indexing workflows, realtime, ICP mapper, video media/storage
  worker/       PostgreSQL-backed async job worker
  realtime/     SLAM bridge runtimes
  ui/           Streamlit operator UI
  mapper/       ICP fusion service
  scripts/      ssv-env, ssv-migrate, quality checks
docker/         compose files and container definitions
tests/          unit, integration, assets, and shared test helpers
docs/           specification, current implementation, guides, runbooks, learning path
```

Video Settings come from `selfsuvis.pipeline.core.config` (ss-perception package).
FusionSettings come from `selfsuvis.fusion_rt.config` when the sidecar is installed.

### Test structure

`tests/unit/` mirrors `src/selfsuvis/` where practical. Integration tests run in Docker
(`make test` / `make test-no-gpu`).
