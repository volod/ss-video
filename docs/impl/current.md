# Current Implementation

ss-video is the production video ingest and search server. Perception and mapping come from
ss-fusion git tags. fusion-rt may run as a compose sidecar from the same tag.

## Topic Map

| Need | Read |
| --- | --- |
| API, worker, UI, storage, query surface, realtime bridges | [current/production-server.md](current/production-server.md) |
| Environments, Docker, tests, CI | [current/build-ci-test.md](current/build-ci-test.md) |
| Config, Postgres, `.env`, `$DATA_DIR` | [current/data-config.md](current/data-config.md) |

Research pipeline: [volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.2.0`.
Shared contracts: [volod/ss-common](https://github.com/volod/ss-common) tag `v0.2.1`.
Sensor mesh: [volod/ss-sens](https://github.com/volod/ss-sens) tag `v0.1.0`.
Site control: [volod/ss-control](https://github.com/volod/ss-control) tag `v0.1.0`.

## Scope of these documents

- **Current behavior only.** Forward work lives in [plan.md](plan.md). Product intent lives
  in the [specification](../design/spec.md).
- Topic files cite real module paths and commands.

## Documentation checks

`make lint-spec-plan` and `make lint-doc-links` must report no findings. `make plan-status`
prints the next eligible task per lane.

### Ownership prefixes

Python modules under `src/` plus installed ss-fusion packages belong to one group. Longest
prefix wins. Layer order, highest to lowest: ss-video, ss-fusion, ss-mapping, ss-perception.

| Prefix | Group |
| --- | --- |
| `ssv_vdp` | ss-fusion |
| `selfsuvis.fusion_rt` | ss-fusion |
| `selfsuvis.pipeline.fusion` | ss-fusion |
| `selfsuvis.pipeline.training` | ss-fusion |
| `selfsuvis.pipeline.analysis` | ss-fusion |
| `selfsuvis.analytics` | ss-fusion |
| `selfsuvis.visualization` | ss-fusion |
| `selfsuvis.scripts.prepare_models` | ss-fusion |
| `selfsuvis.scripts.sensors` | ss-fusion |
| `selfsuvis.scripts.scenetok_server` | ss-fusion |
| `selfsuvis.scripts.add_sensor_key` | ss-fusion |
| `selfsuvis.scripts.seed_test_events` | ss-fusion |
| `selfsuvis.pipeline.storage.elastic` | ss-fusion |
| `selfsuvis.pipeline.mapping.icp` | ss-video |
| `selfsuvis.pipeline.mapping.mapper` | ss-video |
| `selfsuvis.pipeline.mapping` | ss-mapping |
| `selfsuvis.scripts.generate_test_splat` | ss-mapping |
| `selfsuvis.pipeline.core` | ss-perception |
| `selfsuvis.pipeline.vision` | ss-perception |
| `selfsuvis.pipeline.labeling` | ss-perception |
| `selfsuvis.models` | ss-perception |
| `selfsuvis.pipeline.media.frames` | ss-perception |
| `selfsuvis.pipeline.media.ffmpeg` | ss-perception |
| `selfsuvis.pipeline.media.gps` | ss-perception |
| `selfsuvis.pipeline.media.dedup` | ss-perception |
| `selfsuvis.pipeline.media.heuristics` | ss-perception |
| `selfsuvis.pipeline.media.audio` | ss-perception |
| `selfsuvis.pipeline.media.fs_common` | ss-perception |
| `selfsuvis.pipeline.media.subprocess_common` | ss-perception |
| `selfsuvis.pipeline.storage.vector_store` | ss-perception |
| `selfsuvis.pipeline.storage.qdrant` | ss-perception |
| `selfsuvis.pipeline.storage.recent_index` | ss-perception |
| `selfsuvis.pipeline.storage.common` | ss-perception |
| `selfsuvis` | ss-video |
