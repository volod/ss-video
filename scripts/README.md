# Scripts

Project scripts are organized under `scripts/` subdirectories. The `scripts/` root is intentionally kept minimal.

## Canonical commands

- `install/install_system_deps.sh` — install host system packages for local development
- `install/install_requirements.sh` — install Python dependencies into an existing virtualenv
- `install/install_nvidia_docker.sh` — install NVIDIA Container Toolkit for Docker
- `ssv/ssv-reset-qdrant.sh` — delete the configured Qdrant collection
- `ssv/ssv-realtime-bridge.sh` — run the packaged MAVSDK or ROS realtime telemetry bridge runtime
- `ssv/ssv-utilyze.sh` — run Utilyze with project defaults (if installed)

## Camera helpers

- `ssv/ssv-camera.sh` -- update the Frigate camera config and optionally restart Frigate
- `ssv/ssv-test-usb-cameras.sh` -- inspect V4L2 devices and optionally test capture
- `ssv/add_camera.sh` -- wrapper that execs `ssv-camera.sh`

The sensor mesh lives in [volod/ss-sens](https://github.com/volod/ss-sens) tag `v0.1.0`.

## Project utility

- `project/project-package.sh` — create a tarball of the current repo while excluding generated secrets and runtime data
- `project/seed_test_events.sh` — seed test zones and events (development helper)

## Shared shell helpers

Shell entrypoints reuse `common.sh` for:

- project-root resolution
- `.data/.env` loading
- `$DATA_DIR` resolution
- consistent logging and fatal errors
- runtime `PUID` and `PGID`
- package-backed Python module execution via the project venv or `python3`

New shell scripts should source `scripts/shared/common.sh` instead of duplicating path or environment bootstrapping logic.
