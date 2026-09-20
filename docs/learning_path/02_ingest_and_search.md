# Ingest And Search

Production indexing is a PostgreSQL job plus a worker, not `ssv --mode local`.

Current behavior: [production server](../impl/current/production-server.md).
HTTP details: [API reference](../reference/api.md). UI: [ui.md](../reference/ui.md).

## How a video enters the system

1. `POST /index/video`, `/index/url`, or `/index/dir` creates a job row.
2. The worker claims the job (`worker/handlers/index.py` -> `VideoIndexer`).
3. Frames are sampled and quality-filtered.
4. CLIP and DINOv3 embeddings go to Qdrant as named vectors.
5. Enrichments write metadata to PostgreSQL (`frame_facts_json` and related columns).
6. Optional postflight jobs run mapping or a semantic graph.

Other job types: `SUPERVISED_FINETUNE`, `REEMBED`, `POSTFLIGHT_MAPPING`,
`POSTFLIGHT_SEMANTIC_GRAPH`. Fine-tune checkpoints get a `model-artifact` manifest
([data-config manifests](../impl/current/data-config.md#manifests)).

`ALLOWED_INDEX_PATHS` is fail-closed: empty means directory indexing is refused.
API keys use a timing-safe compare. See [configuration](../reference/configuration.md).

## Query surface

| Endpoint | Mechanism |
| --- | --- |
| `POST /query/text` | OpenCLIP text embedding vs Qdrant `clip` (or configured space) |
| `POST /query/image` | Image embedding, optional DINO space |
| `POST /query/scene` | PostgreSQL filters on `frame_facts_json`, optional CLIP rerank |
| `POST /query/pose` | GPS/ENU spatial filter + vector ranking; robot advisory context |

CLIP is trained on image-text pairs: use it for language queries ("military vehicle").
DINOv3 is vision-only self-supervision: use it for "find visually similar frames".
Qdrant holds both as named vectors. Embeddings are L2-normalized; search uses cosine.

If you change `MODEL_NAME` or the CLIP checkpoint, reset the collection
(`scripts/ssv/ssv-reset-qdrant.sh`) or you mix incompatible vectors.

## What a human should verify

- The job finished without a silent skip (`processed_files` hash short-circuit).
- A text query that matches the video returns those frames near the top.
- A scene query on a caption or OCR token you can see in a frame actually hits.
- Pose query with a known GPS sample from the sidecar returns nearby frames.

## Common failure modes

- Qdrant schema mismatch after a model swap -- mixed embeddings, random-looking search.
- GPU OOM in the embedder -- worker retries on CPU or the job errors; check VRAM.
- Empty `ALLOWED_INDEX_PATHS` -- `/index/dir` refused by design.
- Missing vision extra -- that enricher degrades; search still runs on whatever vectors exist.

Next: [indexer enrichments](03_indexer_enrichments.md).
