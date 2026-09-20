# fusion-rt And Site APIs

Incident correlation is the fusion-rt sidecar (package `fusion-rt` from
ss-fusion), not video source. Compose service `fusion-rt` listens on host
8001. The Streamlit UI uses `FUSION_RT_URL`.

## What the sidecar owns

- `POST /api/v1/events/{modality}` -- `EventEnvelope` ingest
- `GET /api/v1/site/state` -- zones and incidents
- incident list/detail/ack/dismiss/notes/search/export
- fusion rule CRUD
- zone CRUD and history
- SSE `GET /api/v1/events/stream`
- `GET /site/state`, `/site/threat`, `/site/synthesis`, `WS /site/stream`

OpenAPI: `docs/api/fusion-rt-openapi.json`. Integration test:
`tests/test_fusion_rt.py` via `make test`.

The video API still owns `GET /site/cameras`. It publishes `camera-event` and
`scene-caption` (`pipeline/realtime/contract_publisher.py`) and consumes Frigate
MQTT.

## MQTT

fusion-rt subscribes to contract topics `sensor-event`, `sensor-state`,
`camera-event`, and `scene-caption`. LoRaWAN decoding and the device registry
are [volod/ss-sens](https://github.com/volod/ss-sens). Two Postgres databases
share the compose instance: video (`selfsuvis`) and fusion (`selfsuvis_fusion`).
Migrate with `python -m selfsuvis.scripts.migrate_postgres --owner all`.

## Why this is a separate process

Video ingest can run when MQTT is down. Site incidents can run when the
indexer is idle. Do not import fusion correlator code into the video worker;
pin the published package.

Deep dive for the mesh itself:
[ss-sens learning path 16](https://github.com/volod/ss-sens/blob/v0.1.0/docs/learning_path/16_coop_pilot_iot_edge_monitoring.md).

Next: [study resources](06_study_resources.md).
