# Profile orchestration

## Task and scope

- Id / capability: `four-d-profile-orchestration` / `four-d-scene-analysis`
- State: accepted
- Source: plan task at `b6779b9`. This change is profile admission, verified-event publication, and the 15-minute load gate.
- Plan counts at start: 1 agent, 0 human; next eligible `four-d-profile-orchestration`.
- Accepted task:

```markdown
#### four-d-profile-orchestration

The component stages need an end-to-end budget controller, observable degradation, and a safe
handoff from verified video events to site correlation.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#processing-profiles-and-data-flow)
- Agent status: RUN NEEDED
- Dependencies: `four-d-keyframes-and-tracks`, `four-d-strict-verifier-and-qa`.
- User-visible outcome: Operators can select fast or deep analysis, see backlog and degraded stages, receive bounded-latency verified events, and later inspect higher-quality superseding results.
- Scope boundary: Add profile configuration, GPU/resource admission, bounded queues, stage telemetry, gap/coalescing records, restart/idempotency behavior, UI/job status, an ss-common ODCS contract, and verified event publication; do not change fusion-rt correlation policy or make remote VLM use the default.
- Data and artifact paths: `src/selfsuvis/worker/`, `src/selfsuvis/pipeline/workflows/`, `src/selfsuvis/pipeline/realtime/`, `src/selfsuvis/app/`, `src/selfsuvis/ui/`, `tests/assets/analysis4d/`, and `$DATA_DIR/analysis/<mission_id>/4d/`.
- Execution path: Wire causal fast and postflight deep DAGs, prioritize track continuity over optional stages, publish only accepted versioned event envelopes, run a 15-minute load fixture on declared reference hardware, and verify restart replay plus fast-to-deep supersession.
- Acceptance gates: `make ci` and the declared load/integration run pass; fast-profile real-time factor is at most 1.0 with bounded queue growth, p95 verified-event lag is at most 3 seconds for configured fast events, every discarded interval has a gap record, deep results supersede rather than overwrite fast results, and disabling the capability preserves current video-search behavior and performance.
- Documentation target: [Production server](current/production-server.md), [Configuration](../reference/configuration.md), [Operations](../operations/operations.md), and a new 4D operations runbook.
```

- Amendments: none. The versioned event contract is authored in this repository as `verified-scene-event` 1.0.0 because pinned ss-common `v0.2.1` has no such contract. Publication uses the existing ss-common `event-envelope` 1.0.0. No fusion rule was added.

## Implementation

`pipeline/analysis4d/budget.py` is the bounded queue and GPU-slot admission. Coalesced frames become `queue_coalesce` gaps. Optional stages `dense_geometry`, `vlm`, and `review` shed with `budget_shed` under pressure or when `ANALYSIS4D_GPU_SLOTS` is 0. Tracks always run.

`pipeline/analysis4d/profile.py` reads `ANALYSIS4D_PROFILE`. The default `off` schedules no 4D job, so indexing stays on the video-search path. `fast` schedules `analysis4d_fast`. `deep` also schedules `postflight_analysis4d_deep`.

`workflows/analysis4d_profile.py` runs the fast DAG in chunks and the deep revision after it. Restart with the same input digest returns the stored result. Deep keeps fast event ids, sets `supersedes` on new events, and records the fast manifest digest.

`pipeline/analysis4d/publish.py` appends `published-events.jsonl` for `accepted` rows only. The payload model is generated from `contracts/odcs/verified-scene-event.odcs.yaml`. The envelope modality is `video_4d`. Default VLM and review providers stay `unavailable`.

The index form, Streamlit index page, and `GET /analysis/{mission_id}/4d/status` expose the profile, queue, degradations, backlog, and published ids.

Current-state pages: [Production server](../current/production-server.md#profile-orchestration), [Configuration](../../reference/configuration.md#profile-orchestration), [Operations](../../operations/operations.md#verified-4d-events), [API](../../reference/api.md#analysis-routes), and the [profile orchestration runbook](../../runbooks/four-d-profile-orchestration.md).

## Acceptance evidence

| Gate | Exact command, test or artifact | Result and limit |
| --- | --- | --- |
| CI | `make ci` | 673 passed, 5 skipped. Ruff, import-linter, spec-plan, and doc-links reported no findings. |
| API schema export | `make export-openapi` | `docs/api/video-openapi.json` includes `/analysis/{mission_id}/4d/status` and `analysis_profile` on the index routes |
| 15-minute load | `python -m selfsuvis.pipeline.analysis4d.profile_benchmark` | `passed=true`, `failures=[]`, `duration_sec=900`, `frame_count=1800`, `real_time_factor=9.86e-05` (limit at most 1.0), `p95_lag_sec=0.000394` (limit at most 3), `queue_depth=4` at `queue_capacity=4`, `queue_gap_count=900`, `replayed=true`, `supersedes_sha256` set, `reference_device=NVIDIA GeForce RTX 4060 Ti` |
| Disabled capability | `ANALYSIS4D_PROFILE` default `off` and `test_off_profile_schedules_nothing` | no 4D job is scheduled |

Runtime report: `$DATA_DIR/analysis/_benchmark/profile-report.json`.

## Audit handoff

`none identified`. Reviewed scope: queue coalescing, GPU admission, restart digest, fast-to-deep supersession, accepted-only publication, fusion-rt rule set left unchanged, and the default `off` profile.

## Close or resume

The declared gates for this task passed. Capability `four-d-scene-analysis` is `shipped`. Plan counts after: 0 agent, 0 human; no agent-lane tasks remain.
