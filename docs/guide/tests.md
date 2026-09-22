# Tests

## Unit tests (no Docker)
```bash
make test-ci      # GitHub CI subset (no torch/cv2/ffmpeg)
make test-unit    # full suite; needs the local venv with vision deps
make test-heavy   # heavy/slow/benchmark/gpu only
# or
pytest tests/unit/ -v
pytest tests/unit/ -q --ci-light
```

Uses `.venv` if present. GitHub Actions runs `make test-ci`. Skips cv2-dependent tests if numpy/opencv mismatch:
```bash
make test-unit-no-cv2
```

### Unit test layout

`tests/unit/` mirrors `src/selfsuvis/` where practical:

```text
tests/unit/
  app/
  models/
  pipeline/
    analysis/
    analysis4d/
    core/
    mapping/
    media/
    realtime/
    storage/
    training/
    vision/
    workflows/
  scripts/
  worker/
```

Reusable test helpers live in `tests/support/`. Keep `conftest.py` focused on pytest
fixture wiring; move reusable fake DB pools, mock rows, factories, and helper classes
into `tests/support/` when they are not fixture-specific.

Most unit tests should sit under the package area they cover. One exception remains on
purpose:

```text
tests/unit/test_multisite_enu.py
```

That file is intentionally kept flat because it exercises storage, worker, and app
behavior together. Cross-cutting tests can stay at `tests/unit/` root when forcing them
under one subsystem would make ownership less clear.

## Integration tests (Docker)
```bash
make test          # with GPU
make test-no-gpu   # without GPU (NVIDIA Container Toolkit not required)
```

Runs `test-dirs` and `test-reset-state` first so `postgres-test` / `qdrant-test` are empty and owned by you. Then starts api, worker, qdrant, and a tests container. Uses `docker/test/docker-compose.test.yml` with `ALLOWED_INDEX_PATHS=/app/tests/assets` and `MAX_UPLOAD_BYTES=150000`. Qdrant `v1.19.1` and `qdrant-client>=1.19.1,<2`.

Directory-indexing tests:
```bash
make test-dir
```
Same as `make test`; `INDEX_DIR_PATH` is set for dir tests.

## Assets
```
./tests/assets/            # small test videos and reference image
./tests/assets/analysis4d/ # pinned 4D contract corpus (analysis4d-v1)
./tests/assets/synthetic-camera/ # pinned camera and box for the geometry gate
```

The contract benchmark is `python -m selfsuvis.pipeline.analysis4d.benchmark`.
The keyframe and track benchmark is `python -m selfsuvis.pipeline.analysis4d.track_benchmark`
([4D model runbook](../runbooks/four-d-models.md)).
The geometry benchmark is `python -m selfsuvis.pipeline.analysis4d.geometry_benchmark`
([4D geometry runbook](../runbooks/four-d-geometry.md)).
The 15-minute profile load benchmark is `python -m selfsuvis.pipeline.analysis4d.profile_benchmark`
([profile orchestration runbook](../runbooks/four-d-profile-orchestration.md)).

## Integration test coverage
- Health, index (video/url/dir), precheck, precheck_dir, jobs, query (image/text)
- Validation, errors (400, 403, 404, 413), job_id validation

## Lint
```bash
make lint
```
Runs `ruff check`, `ruff format --check`, `make lint-spec-plan`, and `make lint-doc-links`.
Install ruff first (e.g. `pip install ruff`).

---
[← Licensing](../reference/licensing.md) | [README ↑](../../README.md)
