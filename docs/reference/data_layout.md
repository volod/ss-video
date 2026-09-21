# Data Layout

By default the repository writes runtime data and model caches under `./.data`.

```text
.data/
  videos/         stored video inputs
  mediamtx/       MediaMTX recordings and stream-side assets
  frames/         extracted keyframes
  tiles/          tile crops used for retrieval
  audio/          temporary audio extracted for ASR
  maps/           ICP and splat outputs
  qdrant/         Qdrant volume data (`make up`)
  qdrant-test/    Qdrant volume for `make test`
  postgres/       PostgreSQL volume data (`make up`)
  postgres-test/  PostgreSQL volume for `make test`
  analysis/       4D analysis artifacts and the fixture benchmark report

  .cache/
    torch/
    open_clip/
    huggingface/
```

Common map layouts:

```text
.data/maps/{mission_id}/splat.ply
.data/maps/{mission_id}/scene-0/splat.ply
.data/maps/{mission_id}/colmap/
```

MediaMTX-related runtime data is stored under:

```text
.data/mediamtx/
  ...             MediaMTX-mounted runtime directory; recordings land here when enabled
```

Integration tests use `.data/` and `.data/cache_test/` directories.

Schema state lives in two PostgreSQL databases on the compose instance: video
(`DATABASE_URL` / `selfsuvis`) and fusion (`FUSION_DATABASE_URL` / `selfsuvis_fusion`).
The API applies both on startup; the worker applies video only.
`ssv-migrate` (`python -m selfsuvis.scripts.migrate_postgres`) is the CLI
(`--owner video|fusion|all`). See [data-config.md](../impl/current/data-config.md).

Four-dimensional analysis writes one directory per mission. Query metadata is in the
video database (`analysis4d_runs`, `analysis4d_events`, `analysis4d_edges`,
`analysis4d_qa`). The benchmark report is not a mission artifact.

```text
.data/analysis/<mission_id>/4d/manifest.json
.data/analysis/<mission_id>/4d/tracks.jsonl
.data/analysis/<mission_id>/4d/graph-deltas.jsonl
.data/analysis/<mission_id>/4d/proposals.jsonl
.data/analysis/<mission_id>/4d/timeline.json
.data/analysis/<mission_id>/4d/qa.jsonl
.data/analysis/_benchmark/report.json
```

See [Four-dimensional analysis artifacts](../impl/current/data-config.md#four-dimensional-analysis-artifacts).

A mission bundle's manifest `mission.json` (contract `mission-bundle`) lists videos
with their digests, the `media` time base, and the GPS origin. See
[manifests](../impl/current/data-config.md#manifests).

---
[Performance →](performance.md)
