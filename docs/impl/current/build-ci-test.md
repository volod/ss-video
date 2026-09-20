# Build, CI, and Test

## Python environments

- `make venv` -- `uv venv .venv` + `scripts/install/install_requirements.sh vision,dev .venv`;
  installs `selfsuvis` and pins ss-fusion packages (`ss-perception`, `ss-mapping`, `ss-fusion`,
  `fusion-rt`) plus `ss-common` at git tag `v0.1.0`. Prompts recreate/update when
  `.venv` exists. `make venv-cuda` forces CUDA wheels when `nvidia-smi` is absent.
- Torch is intentionally **not** a pyproject dependency: `install_requirements.sh`
  selects the CUDA/CPU wheel index against detected host hardware (AGENTS.md rule).
- Extras in `pyproject.toml`: `vision` (full CUDA stack: torchvision, open-clip,
  ultralytics, xformers, SAM2/3, rfdetr, gsplat, pycolmap), `mapper` (CPU ICP
  service), `dev` (pytest, ruff). Base dependencies include `aiomqtt` for the
  site MQTT consumer. The sensor mesh extra and console scripts live in
  [volod/ss-sens](https://github.com/volod/ss-sens) tag `v0.1.0`.
- Separate venv: `.venv` (video API plus ss-fusion git-tag packages). The model lab lives in
  [volod/ss-mlab](https://github.com/volod/ss-mlab) tag `v0.1.0`.

## Heavy native builds

- Parallelism is always capped via the canonical helpers -- main/xformers:
  `MAX_JOBS=$(.venv/bin/python -m ss_kit max-jobs)` (`ss-kit max-jobs`;
  `src/selfsuvis/scripts/shell_helpers.py max-jobs` re-exports it).
  Never inline the formula.
- Compiled wheels (flash-attn, vllm forks, xformers) are cached under
  `.data/wheels/<package>_<abi-key>/` -- the single shared cache root.
- `make venv-rebuild-xformers` rebuilds xformers for the detected compute capability.

## Docker composition

| Compose file | Stack |
| --- | --- |
| `docker/core/docker-compose.yml` (+ `.override`, `.no-gpu`) | api, worker, ui, qdrant, mapper; Frigate on profile `frigate`. Qdrant image is `qdrant/qdrant:v1.19.1` (`docker/qdrant/Dockerfile.qdrant`); Python `qdrant-client>=1.19.1,<2` |
| [ss-control compose](https://github.com/volod/ss-control/blob/v0.1.0/docker-compose.yml) | Site control plane (Caddy 80/443, Authelia, step-ca) |
| [ss-sens compose](https://github.com/volod/ss-sens/blob/v0.1.0/docker/ss-sens/docker-compose.ss-sens.yml) | IoT edge stack (profiles: lorawan, edge; metrics moved to ss-control) |
| `docker/realtime/*.yml` | MediaMTX, SLAM engines, bridge runtimes |
| `docker/cvat/docker-compose.cvat.yml` | Annotation service |
| [ss-fusion vLLM compose](https://github.com/volod/ss-fusion/blob/v0.1.0/docker/vllm/docker-compose.vllm.yml) | Reasoning/vision sidecar |
| `docker/test/docker-compose.test.yml` | Integration test harness (`Dockerfile.tests`) |

Docker images select GPU/CUDA targets from the build host configuration
(AGENTS.md rule). `UID`/`GID` are injected so bind mounts stay user-owned.
Images that `pip install` this project use `python:3.11-slim` and install `git` plus
`ca-certificates` in the builder so `pip` can fetch ss-common and ss-fusion from git tags
(ss-common requires Python 3.11). `docker/install-python.sh` installs ss-fusion packages
from `git+https://github.com/volod/ss-fusion.git@v0.1.0#subdirectory=` (pip cannot parse
uv `file:` path pins). The same
script sets `MAX_JOBS` from `python -m ss_kit max-jobs` (`CMAKE_BUILD_PARALLEL_LEVEL`
and `NINJAFLAGS` match) so ninja/cmake/CUDA compiles are not single-threaded. Vision
images install torch from `download.pytorch.org/whl/$TORCH_CUDA_INDEX` (Makefile
exports the host venv extra, for example `cu128`) and pin that version so pip does not
upgrade to an unrelated PyPI torch. `make venv` reuses
`.data/wheels/flash-attn_<torch_cu_sm_nvcc>/` when the ABI key matches. Vision image
builds bind-mount that directory as Compose `additional_contexts.wheels` (Makefile
exports `WHEELS_DIR`; `data-dirs` / `test-dirs` create it). The install script
installs a wheel only when torch in the image matches the key. The empty Dockerfile
`wheels` stage is the default when that context is omitted (plain `docker build`). The
tests image is an HTTP client (pytest, requests, httpx, aiomqtt, ss-common mqtt extra).
fusion-rt uses `docker/core/Dockerfile.fusion_rt` (`install-python.sh --runtime`: no torch).

## Tests

- `make test-ci` -- GitHub unit job. Collects `tests/unit/` with `--ci-light`: skips
  paths in `tests/support/ci_light.py` (torch, OpenCV, ffmpeg, ONNX, model loads)
  and deselects `slow` / `heavy` / `benchmark` / `gpu` / `integration` / `load`.
- `make test-unit` -- full host unit tests, no services (`tests/unit/` mirrors
  `src/selfsuvis/`; fakes and factories in `tests/support/`). Needs the local
  venv with vision deps. Mission-bundle goldens that call `ffprobe` are `heavy`.
- `make test-heavy` -- only `heavy` / `slow` / `benchmark` / `gpu` unit tests.
- `make test` / `make test-no-gpu` -- full integration in Docker (api + fusion-rt +
  worker + qdrant + mosquitto + tests container, `INDEX_DIR_PATH=/app/tests/assets`,
  `RUN_API_TESTS=1`). `DATA_DIR` is passed as an absolute path so compose bind-mounts
  hit project `.data/` (not `docker/core/.data`). `test-dirs` also creates
  `$DATA_DIR/redis`. The video API applies the video schema, fusion-rt applies the
  fusion schema, and the worker applies video only, so a fresh volume can serve
  `/index/video` and `/api/v1/incidents`. The tests image runs `tests/test_api.py`
  and `tests/test_fusion_rt.py`. Relative test-overlay paths resolve from
  `docker/core/` (first `-f` file); mosquitto config is
  `docker/test/mosquitto.conf` mounted as `../test/mosquitto.conf`. Compose uses
  isolated `$DATA_DIR/postgres-test` and `$DATA_DIR/qdrant-test`
  volumes (`make test-reset-state` wipes them first) so leftover `processed_files`
  rows cannot skip indexing after a Qdrant wipe. Qdrant is `v1.19.1` with
  `qdrant-client>=1.19.1,<2`.
- Markers: `slow`, `heavy`, `benchmark`, `gpu`, `integration`, `load` (locust file in ss-sens tests).
- `make lint` -- `ruff check` + `ruff format --check` (line length 100,
  rules E/F/W/I/UP; uses `.venv/bin/ruff` when present), then `make lint-imports` (import-linter layers, empty allowlist),
  `make lint-spec-plan` and `make lint-doc-links` (see
  [current.md](../current.md#documentation-checks) and
  [module ownership](../current.md#ownership-prefixes)). Pyright configured `basic` against
  `.venv`.
- Model-loading unit tests (for example `tests/unit/pipeline/vision/test_florence_model.py`)
  resolve and cache weights under `$DATA_DIR` and download them on first use. `make test-unit`
  exports `DATA_DIR=.data` unless the shell sets it; a direct `pytest` call takes `DATA_DIR` from
  `.env` instead, and fails with a cache permission error or `OSError` when that path is an
  unmounted drive.

## CI (GitHub Actions)

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `.github/workflows/ci.yml` | push to `main`, PR | `make lint`, `make test-ci` (light unit tests, no torch/opencv/ffmpeg) |
| `.github/workflows/openapi.yml` | PR touching `src/selfsuvis/app/**`, `docs/api/**`, this workflow, or `pyproject.toml` | OpenAPI spec diff gate (`make export-openapi` artifacts `docs/api/video-openapi.json` and `docs/api/fusion-rt-openapi.json`) |

The OpenAPI job installs the package with `--no-deps` so it does not pull torch or
transformers, then the lean third-party set needed to import `selfsuvis.app.main` and
`selfsuvis.fusion_rt.app` (FastAPI, pydantic, sse-starlette, asyncpg, python-multipart,
httpx, numpy, pillow, requests, scipy, pyyaml) plus
`ss-common[web,mqtt] @ git+https://github.com/volod/ss-common.git@v0.1.0`
(the same tag as the lint job). A missing `ss_kit` import is a failed install, not a
spec drift.
| `.github/workflows/claude-code-review.yml` | PR open/sync | Automated review |
| `.github/workflows/claude.yml` | issue/PR comments | Interactive agent |

## Known gaps (drive the forward plan)

- No unified aggregate beyond `make ci` (lint + test-unit) and `make ci-github`
  (lint + test-ci). GitHub runs the `ci-github` pair as two jobs.
- No firmware, Go, or FPGA toolchain anywhere in the build system yet; the
  cross-stack build/CI work is specified in the ss-sens plan
  ([ss-sens plan](https://github.com/volod/ss-sens/blob/v0.1.0/docs/impl/plan.md)).
