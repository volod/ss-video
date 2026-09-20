# Overview

ss-video indexes mission video and answers text, image, scene, and pose queries.
Three execution views share models and contracts but are not the same program:

| View | Where it runs | What to read |
| --- | --- | --- |
| Production indexer | this repository: API + worker + UI | this learning path, then [production server](../impl/current/production-server.md) |
| Research local runner | [volod/ss-fusion](https://github.com/volod/ss-fusion) `ssv --mode local` | [ss-fusion learning path](https://github.com/volod/ss-fusion/blob/v0.1.0/docs/learning_path/README.md) |
| Live site mesh | [volod/ss-sens](https://github.com/volod/ss-sens) plus the fusion-rt sidecar here | chapter [05](05_fusion_rt_and_site.md) and [ss-sens getting started](https://github.com/volod/ss-sens/blob/v0.1.0/docs/ss-sens/getting-started.md) |

Treat those as three views of one system, not as contradictions.

## Recommended reading order

1. [Quick start](../guide/quickstart.md) -- get API, worker, UI, and Qdrant running.
2. [Runtime and study](01_runtime_and_study.md) -- five layers and what to inspect.
3. [Ingest and search](02_ingest_and_search.md) -- jobs and query surface.
4. [Indexer enrichments](03_indexer_enrichments.md) -- what each model writes.
5. [Realtime and maps](04_realtime_and_maps.md) -- live streams and ICP.
6. [fusion-rt and site](05_fusion_rt_and_site.md) -- incidents and MQTT.
7. [Study resources](06_study_resources.md) -- foundations, then one runbook.
8. [Future directions](07_future_directions.md) -- only after you can explain one indexed mission from artifacts.

If you were sent here by an older `docs/learning_path/00_*.md` filename, use this
page as the entry point. Research-pipeline chapter numbers 00-20 now live in
ss-fusion ([siblings.md](siblings.md)).
