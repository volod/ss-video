# ss-video Specification

## Purpose

ss-video is the video ingest and search service for outdoor autonomy. It indexes mission
video, stores frame metadata and embeddings, and answers text, image, scene, and pose
queries. Import package `selfsuvis`.

Perception, mapping, and the research pipeline live in
[volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.1.0`. Site sensors live in
[volod/ss-sens](https://github.com/volod/ss-sens). Ingress and SSO live in
[volod/ss-control](https://github.com/volod/ss-control). Shared contracts live in
[volod/ss-common](https://github.com/volod/ss-common).

This document owns product behavior, boundaries, evaluations, and the
[capability registry](#capability-registry). The [forward plan](../impl/plan.md) owns work that
remains; [current implementation](../impl/current.md) owns what exists. The specification is
living; see [Extending this specification](#extending-this-specification).

## Design principles

- **Single-domain service.** This repository owns video ingest, search, missions, labeling,
  and the operator UI. It does not own training, sensor decoding, or incident fusion.
- **Pin heavy code.** `ss-perception` and `ss-mapping` come from ss-fusion git tags. fusion-rt
  may run as a compose sidecar from the same tag.
- **Contracts before code.** Cross-service messages are ODCS contracts in ss-common.
- **Evidence over prose.** Every capability declares an evaluation and a valid negative result.

## Target architecture

| Repository | Owns |
| --- | --- |
| `ss-video` (this repository) | ingest (upload, URL, directory, RTSP), MediaMTX, Frigate, captioning, embeddings and Qdrant, query API, missions, robot pose bridges, global maps, UI, CVAT labeling, re-embedding |
| `ss-fusion` | research steps, perception and mapping packages, fusion-rt |
| `ss-sens` | LoRaWAN, MQTT mesh, device registry |
| `ss-control` | reverse proxy, identity, site CA |
| `ss-common` | ODCS contracts and `ss_kit` |

**Interfaces.** REST behind the ss-control proxy; MQTT `ss/v1/...` with mTLS for camera and
scene events; model artifacts pinned from ss-fusion.

## Video search

Ingest mission video, embed frames (CLIP and DINO), enrich them (captions, ASR, OCR, depth,
detection), store metadata in PostgreSQL and vectors in Qdrant, and answer text, image, scene,
and pose queries.

**Boundary.** Training, distillation, and the 36-step research run are ss-fusion. Incident
correlation is fusion-rt (sidecar). Sensor decoding is ss-sens.

**Evaluation.** Docker integration suite and unit tests over app, worker, and storage.
Valid negative result: if a model extra is missing, that enricher degrades and the rest of
ingest continues.

Current state: [production server](../impl/current/production-server.md).

## Capability Registry

| # | Capability | Status | How it is evaluated | Implementation |
| --- | --- | --- | --- | --- |
| 1 | `video-search` | shipped | Docker integration suite and unit tests over app, worker, storage | [Production server](../impl/current/production-server.md) |

## Extending this specification

A capability gap is a product discovery, not an automatic refusal and not permission for silent
scope growth. Use this lifecycle in order:

1. State the problem in operator or domain terms.
2. Amend the owning section of this specification, including what the capability does not do.
3. Declare the measurement, acceptance signal, and valid negative result before implementation.
4. Add a `planned` registry row with that evaluation.
5. Put tasks under the capability in the implementation line; every task declares `Serves`.
6. Build and evaluate, document available behavior under current state, remove finished plan
   scope, and change the registry row to `shipped` with its implementation link.

## Specification and plan integrity

The registry and the [implementation plan](../impl/plan.md) are two views of one product:

- every task serves a registered capability and sits in its capability group;
- every capability declares an evaluation;
- every planned capability has at least one open task;
- every shipped capability links to current-state documentation;
- groups follow registry order in each task lane;
- every task declares the fields required by the [planning workflow](../guide/planning-workflow.md).
