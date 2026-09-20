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

A mission bundle's manifest `mission.json` (contract `mission-bundle`) lists videos
with their digests, the `media` time base, and the GPS origin. See
[manifests](../impl/current/data-config.md#manifests).

---
[Performance →](performance.md)
