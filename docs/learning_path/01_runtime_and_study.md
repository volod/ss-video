# Runtime And Study Guide

The goal is not only to run `make up` once, but to understand why each stage
exists and how evidence moves through the stack.

## Who this is for

- understand the repo well enough to modify ingest, search, or the worker
- debug a bad index job without guessing
- explain the production path to another engineer or operator

If you only want to start containers, use the [quick start](../guide/quickstart.md).
If you want the 36-step research runner, clone
[volod/ss-fusion](https://github.com/volod/ss-fusion).

## Source of truth

| Role | Path |
| --- | --- |
| HTTP API | `src/selfsuvis/app/` |
| Job consumer | `src/selfsuvis/worker/` then `pipeline/workflows/` (indexer) |
| Query and UI | `src/selfsuvis/app/routers/query.py`, `src/selfsuvis/ui/` |
| Embeddings and vision | ss-perception package (`selfsuvis.pipeline.vision`, `selfsuvis.models`) |
| fusion-rt sidecar | published `fusion-rt` package, compose service `fusion-rt` |
| Settings | `selfsuvis.pipeline.core.config` (ss-perception) |

The learning path is the conceptual path. The indexer is the execution path.
Some conceptual steps are optional or grouped in code.

## Five layers

1. **Input and memory** -- frames, CLIP/DINO embeddings, Qdrant retrieval
2. **Evidence extraction** -- captions, speech, OCR, depth, detections
3. **Live ingest** -- MediaMTX, Frigate, pose/occupancy bridges
4. **Maps** -- ICP mapper, global map tables, optional postflight mapping
5. **Site sidecar** -- fusion-rt incidents, MQTT camera events, `/site/*`

Physical sensor decoding (RF, LoRaWAN, thermal sidecars) is ss-sens and
ss-fusion, not this service.

## How to study

Do not start by reading every file. Use this order:

1. Index one short video (`POST /index/video` or the UI).
2. Read [architecture](../design/architecture.md) and [production server](../impl/current/production-server.md).
3. Open the deep-dive for the layer you care about.
4. Open code only after you know the question.

Good questions: what evidence is created, what is stored (Postgres vs Qdrant),
what later query depends on it, what can fail silently, what a human should
verify.

Weak questions: what every line does, which model is newest, memorizing env
vars before a first run.

## What to inspect after a real index job

1. Job row in Postgres (`jobs`) -- status, error, progress
2. Frame rows -- timestamps, GPS if present, `frame_facts_json`
3. Qdrant points -- named vectors `clip` and `dino`, payloads
4. Captions / ASR / OCR fields used by `POST /query/scene`
5. Optional: mapper splat path, realtime stream rows, fusion-rt incidents

If a model extra is missing, that enricher degrades and the rest of ingest
continues. That is the specified valid negative result for `video-search`.

## Practical rules

- Study outputs before internals.
- Compare neighboring stages, not isolated models.
- Separate representation problems (bad embeddings) from query problems (wrong filter).
- Separate "this module exists in a sibling package" from "this stage is enabled in my `.env`".
- When modalities disagree, check timestamps and coordinate frames before blaming the model.

## Code-reading order

1. `src/selfsuvis/app/main.py`
2. `src/selfsuvis/worker/handlers/index.py`
3. indexer workflow under `src/selfsuvis/pipeline/workflows/`
4. one file from this directory
5. the runbook under `docs/runbooks/` for the model you are changing

Next: [ingest and search](02_ingest_and_search.md).
