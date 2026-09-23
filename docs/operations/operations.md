# Operations Guide — Site State API v1

## Single-worker deployment constraint

The correlator and webhook retry tasks run as `asyncio.create_task` inside the
fusion-rt FastAPI lifespan. This means they run **once per uvicorn process**.

**Do not use `--workers N` (N > 1).** Multi-worker deployment causes each
process to start its own correlator, which leads to duplicate incidents being
inserted for the same sensor event window.

```bash
# Correct — single worker (default)
uvicorn selfsuvis.fusion_rt.app:app --host 0.0.0.0 --port 8000

# Wrong — causes duplicate incidents
uvicorn selfsuvis.fusion_rt.app:app --host 0.0.0.0 --port 8000 --workers 4
```

If horizontal scaling is needed in the future, move to a separate worker process
with Redis pub/sub for SSE fan-out (deferred, see Phase 3A architecture decision).

## Provisioning sensor keys

Sensor keys are stored as SHA-256 hashes. The raw key is shown once at provision
time and is never stored.

```bash
./scripts/add_sensor_key.sh --sensor-id cam-north-01
# or with explicit scopes:
./scripts/add_sensor_key.sh --sensor-id cam-north-01 --scopes ingest
```

Send the key as an HTTP header on every ingest request:
```
X-Sensor-Key: <raw key>
```

If no sensor_keys rows exist, the ingest endpoint falls back to the site-level
`API_KEY` (X-Api-Key header). Once the first sensor key is provisioned, the
fallback is disabled and all ingest requests must use X-Sensor-Key.

## Fusion rules

Rules are stored in PostgreSQL (`fusion_rules` table). The correlator
auto-seeds from `fusion-rt` package data `selfsuvis/fusion_rt/data/fusion_rules.yaml` at startup
if the table is empty.

To add a rule via API:
```bash
curl -X POST http://localhost:8001/api/v1/rules \
  -H "X-Api-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"rule_id":"drone-test","label":"Drone test","modalities":["camera","audio"],"window_s":60,"min_confidence":0.7,"enabled":true}'
```

## Webhook alerts

Set `WEBHOOK_ALERT_URL` to receive incident alerts. Optional `WEBHOOK_SECRET`
enables HMAC-SHA256 signature verification.

```bash
WEBHOOK_ALERT_URL=https://your-endpoint/alerts
WEBHOOK_SECRET=your-secret  # optional
```

Verify the signature:
```python
import hmac, hashlib

expected = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
assert hmac.compare_digest(f"sha256={expected}", request.headers["X-SelfSuvis-Signature"])
```

Failed deliveries are retried with `[0, 5, 30]` second backoff. After 3 failures
the alert is moved to the dead-letter queue (`fusion:alert:dlq` Redis list).

Inspect DLQ:
```bash
redis-cli LRANGE fusion:alert:dlq 0 -1
```

Replay a DLQ item:
```bash
redis-cli RPOPLPUSH fusion:alert:dlq fusion:alert:retry
```

## DroneAudioAdapter

The drone audio ONNX model is a **training output** from the SV-21 pipeline.
It does not exist in the repo by default. Train first, then configure:

```bash
DRONE_AUDIO_MODEL_PATH=/path/to/drone_audio_cnn.onnx
DRONE_AUDIO_WATCH_DIR=/path/to/wav/watch/dir
```

The adapter polls `DRONE_AUDIO_WATCH_DIR` every 5 s for `*.wav` files.
Processed files are moved to `DRONE_AUDIO_WATCH_DIR/processed/`.

## Verified 4D events

Accepted 4D events are written as ss-common `event-envelope` 1.0.0 lines in
`$DATA_DIR/analysis/<mission_id>/4d/published-events.jsonl`. The payload is
`verified-scene-event` 1.0.0. Each new line is handed to
`pipeline/realtime/contract_publisher.py`, which publishes it on

```text
ss/v1/site/{site_id}/zone/{zone_id}/event/video_4d
```

`site_id` is `COOP_SITE_ID` (default `local`). `zone_id` is `ANALYSIS4D_ZONE_ID`
(default `site`). QoS is 1. An event id already in the ledger is left there and
is not sent again. Rejected and uncertain timeline rows stay in `timeline.json`.
A broker refusal still appends the ledger line, and the later replay skips that
id. The packaged fusion-rt seed `fusion_rules.yaml` is unchanged. fusion-rt
still subscribes to `sensor-event`, `sensor-state`, `camera-event`, and
`scene-caption`. Profile selection, queue pressure, and replay are in the
[profile orchestration runbook](../runbooks/four-d-profile-orchestration.md).

## OpenAPI spec

The OpenAPI specs are tracked in `docs/api/video-openapi.json` and
`docs/api/fusion-rt-openapi.json`. CI fails if a PR adds or changes routes
without updating the matching spec.

Before any PR that touches video or fusion-rt routes:
```bash
make export-openapi
git add docs/api/video-openapi.json docs/api/fusion-rt-openapi.json
```
