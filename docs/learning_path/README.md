# Learning Path

Study guides for operators and developers of the video ingest and search service.
Product intent lives in the [specification](../design/spec.md). What exists lives in
[current implementation](../impl/current.md). How to run the stack lives in the
[guide](../guide/README.md).

This path explains **why** each stage exists and what to inspect. It is not the
36-step research runner. That runner, and its original chapter files, live in
[volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.1.0`. IoT mesh deep
dives live in [volod/ss-sens](https://github.com/volod/ss-sens). See
[siblings.md](siblings.md) for the full chapter map.

## Start here

| # | Document | Purpose |
| --- | --- | --- |
| -- | [overview.md](overview.md) | Reading order and how the three views (indexer, research runner, live site) relate |
| 01 | [01_runtime_and_study.md](01_runtime_and_study.md) | How to study the repo: layers, questions, first artifacts |
| 02 | [02_ingest_and_search.md](02_ingest_and_search.md) | Upload, jobs, Qdrant, query API, UI |
| 03 | [03_indexer_enrichments.md](03_indexer_enrichments.md) | Frames, CLIP/DINO, Florence, ASR, OCR, depth, detection |
| 04 | [04_realtime_and_maps.md](04_realtime_and_maps.md) | MediaMTX, Frigate, pose bridges, ICP mapper |
| 05 | [05_fusion_rt_and_site.md](05_fusion_rt_and_site.md) | fusion-rt sidecar, MQTT camera events, site APIs |
| 06 | [06_study_resources.md](06_study_resources.md) | Papers and books behind embeddings, SfM, and retrieval |
| 07 | [07_future_directions.md](07_future_directions.md) | Open themes; maps onto [plan candidates](../impl/plan.md#future-task-candidates) |
| -- | [siblings.md](siblings.md) | ss-fusion and ss-sens learning-path URLs |

Runbooks for the indexer models are under [docs/runbooks/](../runbooks/README.md).
