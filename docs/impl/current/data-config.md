# Data Layout and Configuration

## Settings ownership

ss-video and ss-fusion each own a `KitSettings` subclass. Env var names are unchanged.
The `selfsuvis.config` facade is gone; import the owner directly.

```python
from selfsuvis.pipeline.core.config import settings, validate_settings
from selfsuvis.fusion_rt.config import fusion_settings, validate_fusion_settings
from selfsuvis.pipeline.realtime.camera_settings import camera_settings
from selfsuvis.realtime.config import settings as realtime_settings
```

- `settings` (`VideoSettings` / `Settings`) -- video API, worker, Qdrant, `DATABASE_URL`,
  perception and robot-realtime knobs (`selfsuvis.pipeline.core.config` in ss-perception). Subclasses
  `ss_kit.settings.KitSettings` (`fields()`, `masked()`, `data_dir()`). `PROJECT_ROOT` is
  the nearest ancestor with `pyproject.toml` and `AGENTS.md`, or the process working
  directory when the package is installed without those markers (Docker runtime images).
- `fusion_settings` -- correlator, webhook retry, drone-audio adapter paths, probabilistic
  fusion knobs, and `FUSION_DATABASE_URL` (`selfsuvis.fusion_rt.config`). Also
  `KitSettings`. When `FUSION_DATABASE_URL` is unset it is derived by replacing the
  database name on `DATABASE_URL` with `selfsuvis_fusion`.
- `camera_settings` -- Frigate / MediaMTX camera runtime (`COOP_FRIGATE_*` env vars).
- `realtime_settings` -- `src/selfsuvis/realtime/config.py`.
- ss-sens settings live in `ss_sens.config.settings` (`COOP_MQTT_*`, `COOP_SITE_ID`, HTTP bind)
  in [volod/ss-sens](https://github.com/volod/ss-sens) and are not imported by this package.

RF analysis and training checkpoint paths stay on video Settings so perception modules
do not import `fusion_rt`. Call `validate_settings()` and `validate_fusion_settings()`
at process start.

## PostgreSQL databases

Video and fusion may share one Postgres instance and use separate databases:

| Owner | Env var | Default database | Schema module | Tables |
| --- | --- | --- | --- | --- |
| ss-video | `DATABASE_URL` | `selfsuvis` | `pipeline/storage/migrate_video.py` | `jobs`, `processed_files`, `missions`, `frames`, mapping/realtime/CVAT tables, `scene_timeline`, `analysis4d_runs`, `analysis4d_events`, `analysis4d_edges`, `analysis4d_qa` |
| ss-fusion | `FUSION_DATABASE_URL` | `selfsuvis_fusion` | `fusion_rt/migrate.py` | `sensor_keys`, `site_events`, `zones`, `fusion_rules`, `incidents`, `incident_notes` |

`ssv-migrate` (`python -m selfsuvis.scripts.migrate_postgres`) applies both owners by
default (`--owner video|fusion|all`). The video API applies the video schema and holds
`app.state.db_pool`. fusion-rt applies the fusion schema and holds its own
`app.state.db_pool` on `FUSION_DATABASE_URL`. The worker applies only the video schema.
`ensure_database` creates `selfsuvis_fusion` when the volume already exists
(Docker `initdb.d` runs only on an empty data directory). Fresh volumes also run
`docker/core/postgres-init/01-create-fusion-db.sql`.

A grep gate (`tests/unit/scripts/quality/test_migration_owners.py`) fails if a migration
module names a table owned by the other service.

### Dump and restore (existing deployments)

This split does not move rows. If a volume already has fusion tables inside the video
database, dump them and restore into `selfsuvis_fusion`:

```bash
pg_dump -d selfsuvis -t sensor_keys -t site_events -t zones -t fusion_rules \
  -t incidents -t incident_notes | psql -d selfsuvis_fusion
```

Then drop those tables from `selfsuvis` once the fusion service is serving from the
new database. New installs skip this.

## Shared runtime helpers

This repository pins published ss-common (`ss-common @ git+https://github.com/volod/ss-common.git@v0.2.1`
in `pyproject.toml`, `[tool.uv.sources]` tag `v0.2.1`). Docker builders install `git` and
`ca-certificates` so `pip` can fetch that URL, and use `python:3.11-slim` because ss-common
requires Python 3.11.

| Concern | Implementation | Re-export |
| --- | --- | --- |
| `.env` dialect and typed getters | `ss_kit.env` | `pipeline/core/env.py` |
| Layered `.env`, `KitSettings`, `mask_secret` | `ss_kit.settings` | video `pipeline/core/config/`, fusion `fusion_rt/config.py`; `mask_secret` from `_helpers.py` |
| `$DATA_DIR` against the project root | `ss_kit.paths` | `pipeline/core/utils.py` (`ensure_dir`) |
| Compact ASCII logging, silent at import | `ss_kit.logging` | `pipeline/core/logging.py` (adds analytics) |
| API key, HMAC, rate limit, path allowlist, `stable_point_id` | `ss_kit.security` / `ss_kit.web` | `app/deps.py`, `pipeline/core/utils.py` |
| JSONL and HTTP sidecars | `ss_kit.sidecar` | `pipeline/core/sidecars.py` |
| `MAX_JOBS` for heavy compiles | `ss_kit.hw` (`ss-kit max-jobs`) | `scripts/shell_helpers.py` |

Empty `ALLOWED_INDEX_PATHS` disables path endpoints (fail-closed). Plan and doc-link checks in
`src/selfsuvis/scripts/quality/` re-export `ss_kit.quality`. The lint interpreter (CI job and
local `.venv`) must have ss-common installed.

## Environment generation

- `make env` / `make env-interactive` run `ssv-env`
  (`selfsuvis.scripts.generate_env`): packaged presets (`src/selfsuvis/env/*.env`,
  ss-perception `selfsuvis/env/*.env`, `src/selfsuvis/realtime/env/`) + detected hardware
  (GPU/RAM) -> a project-root `.env` with labeled `# ss-video` and `# ss-fusion`
  sections (including `FUSION_DATABASE_URL`). This is the standard bootstrap;
  `.env.example` documents the surface. ss-sens presets live in
  [ss-sens env/](https://github.com/volod/ss-sens/tree/v0.1.0/src/ss_sens/env)
  and are applied with `make` / `ss-sens-env.sh` in that repository.
- Model selection honors `auto` values resolved by
  `pipeline/vision/registry.resolve_model_id` against the model catalog
  (`docs/reference/model-catalog.md` pointer into ss-fusion).

## Correlator seed rules

Canonical seed YAML is package data at
`selfsuvis.fusion_rt` package data `fusion_rules.yaml` (installed via
`[tool.setuptools.package-data]`). The correlator loads it with
`importlib.resources` when `fusion_rules` is empty.

## Data layout rules (enforced by AGENTS.md)

- All runtime data lives under `$DATA_DIR` (default `.data/`), namespaced by
  module: postgres/qdrant/videos dirs created by
  `make data-dirs`. Never a module-local `.data/` inside `src/`.
- `.data/wheels/` is reserved exclusively for compiled wheel artifacts, keyed by
  ABI dimensions (e.g. `flash-attn_torch2.9.1_cu128_sm89_nvcc126`). Host `make venv`
  and vision Docker images install a matching flash-attn wheel from that cache
  before compiling. Compose passes `$DATA_DIR/wheels` as BuildKit named context
  `wheels` (bind-mounted at `/tmp/wheels`). `.data` stays dockerignored; wheels
  are never copied into `docker/` or committed (`*.whl` is gitignored like `*.pt`).
- No hardcoded absolute paths in committed code; resolve from the project root
  and honor `.env` / `DATA_DIR`. Caches derive under `$DATA_DIR`
  (e.g. `.data/uv-cache/`).
- ss-sens bind mounts (`$DATA_DIR/coop/...`) are created by
  [ss-sens-data-dirs.sh](https://github.com/volod/ss-sens/blob/v0.1.0/scripts/ss-sens/ss-sens-data-dirs.sh)
  before first start.

## Four-dimensional analysis artifacts

Versioned 4D outputs are separate from the ss-common `mission-bundle` and `model-artifact`
manifests. Schema `ss-video.analysis4d-manifest.v1` is internal to ss-video.
`pipeline/storage/analysis4d.py` materializes query rows; the files remain the audit copy.
Layout and the benchmark commands are in
[production-server.md](production-server.md#four-dimensional-analysis-contracts).
Keyframe selection and the pinned grounding model are in the
[4D model runbook](../../runbooks/four-d-models.md). Records:
[0001-four-d-scene-analysis-four-d-contracts-and-benchmark](../records/0001-four-d-scene-analysis-four-d-contracts-and-benchmark.md),
[0002-four-d-scene-analysis-four-d-keyframes-and-tracks](../records/0002-four-d-scene-analysis-four-d-keyframes-and-tracks.md),
[0003-four-d-scene-analysis-four-d-spatial-reconstruction](../records/0003-four-d-scene-analysis-four-d-spatial-reconstruction.md),
[0004-four-d-scene-analysis-four-d-scene-graph](../records/0004-four-d-scene-analysis-four-d-scene-graph.md),
[0005-four-d-scene-analysis-four-d-strict-verifier-and-qa](../records/0005-four-d-scene-analysis-four-d-strict-verifier-and-qa.md),
[0006-four-d-scene-analysis-four-d-profile-orchestration](../records/0006-four-d-scene-analysis-four-d-profile-orchestration.md).
Accepted events are also copied into `published-events.jsonl` as `event-envelope`
1.0.0 rows. The payload contract is `contracts/odcs/verified-scene-event.odcs.yaml`.

```text
$DATA_DIR/analysis/<mission_id>/4d/
  manifest.json
  tracks.jsonl
  graph-deltas.jsonl
  proposals.jsonl
  timeline.json
  qa.jsonl
  gaps.jsonl          optional; required when a stage skips frames
  orchestration-state.json  profile digest, queue depth, and replay flags
  published-events.jsonl    accepted event envelopes, append-only
  track-audit.json    keyframe reasons, memory resets, count disagreements
  geometry/           one JSON sample per track and timestamp
  embeddings/         masked appearance prototypes, one directory per track
  masks/              mask artifacts referenced by evidence
  history/            previous timeline or manifest, named by digest prefix
$DATA_DIR/analysis/_benchmark/report.json
$DATA_DIR/analysis/_benchmark/tracks-report.json
$DATA_DIR/analysis/_benchmark/geometry-report.json
$DATA_DIR/analysis/_benchmark/graph-report.json
$DATA_DIR/analysis/_benchmark/verifier-report.json
$DATA_DIR/analysis/_benchmark/profile-report.json
$DATA_DIR/hf-cache/   Hugging Face weights when HF_HOME is unset
```

`truth.json` is evaluation-only and is not listed in the runtime manifest. The pinned
corpus that the benchmark scores is `tests/assets/analysis4d/` (`analysis4d-v1`).

## Manifests

Two file manifests follow ss-common contracts
([manifest contracts](https://github.com/volod/ss-common/blob/v0.2.1/docs/impl/current/contracts.md#manifest-contracts)).
Builders return plain dicts in the canonical wire form, so the generated models validate them
unchanged. This repository pins ss-common; unit tests import `ss_contracts.models` and compare
against `tests/assets/contracts/golden/` (snapshot of the v0.2.1 goldens).
Runtime builders still return dicts rather than constructing those models.

| Manifest | Builder | Written by | Location |
| --- | --- | --- | --- |
| `mission-bundle` | `pipeline/media/mission_bundle.py`: `build_mission_bundle`, `write_mission_bundle` | callers (no export command yet) | `mission.json` at the bundle root |
| `model-artifact` | `pipeline/core/manifests.py`: `build_model_artifact`, `finetune_model_artifact` | the FINETUNE handler on acceptance; `ssv_vdp/scripts/export_onnx.py` for every ONNX file | `<model file>.manifest.json` next to the file |

- A bundle lists each video (sha256, size, and ffprobe duration, fps, size) and its sidecars:
  `<stem>.<kind>.jsonl` beside the video (what `pipeline.core.sidecars` reads) or under
  `sensors/` at the bundle root, plus a DJI `<stem>.srt` as kind `gps`. `row_count` is what the
  loaders read; `t_start_sec`/`t_end_sec` come from the `t` or `timestamp` key. The time base is
  `media`. `origin` is the first GPS fix of the first video (srt first, then the ISO 6709 atom),
  equal to the origin platform fusion builds (tested); a `GPS_SIDECAR_PATH` override is not
  consulted, since a bundle names its own files. `platform.robot_id` defaults to `robot_0`, as
  the `missions` table does.
- Paths are POSIX, relative to the manifest's directory; files outside it or under a dot
  directory are rejected. `derived_from` may climb out with `../`. Timestamps are UTC with `Z`.
- The FINETUNE handler records `model_version_id`, the resolved base model, the annotated-frame
  count, and `best_accuracy`, `distribution_shift`, `epochs`; a failed manifest write only
  warns. `export_onnx.py` records the hub model the weights were actually loaded from (on this
  host `dinov3_vitb14` resolves to `dinov2_vitb14_reg`), the parity `onnx_max_abs_diff`, opset,
  image size, and, for `--quantize`, an `int8_static` manifest derived from the float file.
- Tests: `tests/unit/contracts/test_manifest_contracts.py` (12).

## Secrets

- Managed per `docs/reference/secrets-management.md` (video/fusion/realtime) and
  [ss-control secrets](https://github.com/volod/ss-control/blob/v0.1.0/docs/reference/secrets-management.md)
  (Authelia, site CA, Grafana).
- Production auth fails closed when required secrets are missing; API keys are
  compared timing-safe; empty `ALLOWED_INDEX_PATHS` disables all path endpoints.
- Key env groups: `API_KEY`, `COOP_MQTT_*`, `REASONING_API_URL` /
  `REASONING_MODEL` / `REASONING_TIMEOUT_SEC`, ChirpStack `CHIRPSTACK_API_SECRET`,
  Mosquitto users/ACLs ([aclfile](https://github.com/volod/ss-sens/blob/v0.1.0/config/ss-sens/mosquitto/aclfile),
  [ss-sens-mqtt-users.sh](https://github.com/volod/ss-sens/blob/v0.1.0/scripts/ss-sens/ss-sens-mqtt-users.sh)).

## Reference docs

`docs/reference/configuration.md` (all env vars),
`docs/reference/data_layout.md` (output tree), `docs/reference/model-catalog.md`
(VRAM budgets and model options).
