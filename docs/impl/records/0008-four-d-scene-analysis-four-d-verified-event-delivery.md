# Verified event delivery

## Task and scope

- Id / capability: `four-d-verified-event-delivery` / `four-d-scene-analysis`
- State: accepted
- Source: plan task at `835d0ae`. The working tree already held uncommitted keyframe-geometry edits.
- Plan counts at start: 1 agent, 0 human; status `CLEAR=1`; next eligible `four-d-verified-event-delivery`.
- Accepted task:

```markdown
#### four-d-verified-event-delivery

Accepted 4D events stay in `published-events.jsonl`. The site correlator does not
receive those envelopes.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#processing-profiles-and-data-flow)
- Agent status: CLEAR
- Dependencies: `four-d-profile-orchestration`.
- User-visible outcome: Each newly accepted event is handed to the existing video MQTT contract publisher, and a replay does not send it again.
- Scope boundary: Reuse `pipeline/realtime/contract_publisher.py` and the `verified-scene-event` payload. Do not add or change a fusion-rt correlation rule. Do not publish rejected or uncertain rows.
- Data and artifact paths: `src/selfsuvis/pipeline/analysis4d/publish.py`, `src/selfsuvis/pipeline/realtime/contract_publisher.py`, and `$DATA_DIR/analysis/<mission_id>/4d/published-events.jsonl`.
- Execution path: Publish one accepted envelope through the existing MQTT client seam, skip an id already in the ledger, and leave rejected timeline rows unsent.
- Acceptance gates: `make test-unit` passes; a repeated event id produces one publish call; rejected and uncertain rows produce none; the fusion rule set is unchanged.
- Documentation target: [Operations](../operations/operations.md) and the [profile orchestration runbook](../runbooks/four-d-profile-orchestration.md).
```

- Amendments: none.

## Implementation

`pipeline/analysis4d/publish.py` still appends `published-events.jsonl` for `accepted` rows only. Each new envelope is passed to `deliver_verified_envelopes` in `pipeline/realtime/contract_publisher.py`. An id already in the ledger is omitted, so a second `publish_accepted` and a profile replay make no further publish call.

`VideoContractPublisher.publish_verified_event` validates the payload as `verified-scene-event` 1.0.0 and sends the ss-common `event-envelope` through the existing `_send` seam. The topic is `ss/v1/site/{site_id}/zone/{zone_id}/event/video_4d`. The worker and `analyze` use a one-shot client that drops `COOP_MQTT_CLIENT_ID` and uses the topic-map QoS. A connected API publisher keeps using its live client. A broker refusal is logged and the ledger line is still written, so a later run skips that id.

Rejected and uncertain rows are dropped before the publisher is called. The publisher also returns without sending when `verification_status` is anything other than `accepted`. The packaged `fusion_rules.yaml` seed stays `rules: []`. fusion-rt's MQTT subscriptions stay `sensor-event`, `sensor-state`, `camera-event`, and `scene-caption`.

Current-state pages: [Operations](../../operations/operations.md#verified-4d-events), [profile orchestration](../current/production-server.md#profile-orchestration), [Configuration](../../reference/configuration.md#profile-orchestration), and the [profile orchestration runbook](../../runbooks/four-d-profile-orchestration.md).

## Acceptance evidence

| Gate | Exact command, test or artifact | Result and limit |
| --- | --- | --- |
| Unit tests | `make test-unit` | 698 passed, 5 skipped |
| Repeated event id | `test_repeated_event_id_produces_one_publish_call` | One `publish_verified_events` call with `evt-keep`. The second call returns no ids and does not publish. The ledger contains that id once. |
| Rejected and uncertain | `test_rejected_and_uncertain_rows_are_not_published` | The publish call contains only the accepted id. A timeline of rejected plus uncertain rows produces no publish call and no ledger line. |
| MQTT client seam | `test_verified_event_uses_the_mqtt_client_seam`, `test_one_shot_client_publishes_without_the_api_client_id` | Connected seam publishes `ss/v1/site/site-a/zone/yard/event/video_4d` once. Rejected and uncertain envelopes in the same batch are omitted. The one-shot client uses QoS 1, timeout 2s, and omits the API client id. |
| Broker refusal | `test_unreachable_broker_still_writes_the_ledger` | `ConnectionRefusedError` still appends the ledger. The following call publishes nothing. |
| Fusion rules | `test_fusion_rule_seed_is_unchanged` | Packaged `fusion_rules.yaml` `rules` is `[]` |

This task does not load a model. The gate is the unit suite above.

## Audit handoff

`none identified`. Reviewed scope: accepted-only MQTT handoff, ledger idempotency, broker refusal still ledgers, and the fusion-rt seed rules plus subscription list left unchanged.

## Close or resume

The declared gates for this task passed. Capability `four-d-scene-analysis` stays `shipped`. Plan counts after: 0 agent, 0 human; no agent-lane tasks remain.
