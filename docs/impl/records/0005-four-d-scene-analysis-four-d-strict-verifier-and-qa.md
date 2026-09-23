# Strict verifier and Video-QA

## Task and scope

- Id / capability: `four-d-strict-verifier-and-qa` / `four-d-scene-analysis`
- State: accepted
- Source: plan task at `f3e373b`. This change is the strict verifier and graph-program Video-QA.
- Plan counts at start: 2 agent, 0 human; next eligible `four-d-strict-verifier-and-qa`.
- Accepted task:

```markdown
#### four-d-strict-verifier-and-qa

No current gate prevents a VLM-proposed spatial relation or action from becoming unsupported
narrative output, and Video-QA answers are not compiled from verified graph state.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#strict-verifier)
- Agent status: RUN NEEDED
- Dependencies: `four-d-scene-graph`. Blocks: `four-d-profile-orchestration`.
- User-visible outcome: Timeline events and auto-generated spatial Video-QA are schema-valid, evidence-linked, geometrically checked, and explicit about rejection or uncertainty.
- Scope boundary: Implement deterministic validators, provider-neutral multimodal review with local Qwen-VL and optional strict-JSON remote providers, claim resolution/audit, executable graph-query QA generation, persistence, and read APIs; do not let VLM confidence override reliable geometry or send rejected/uncertain claims to fusion-rt.
- Data and artifact paths: `src/selfsuvis/pipeline/analysis4d/`, `src/selfsuvis/app/routers/`, `src/selfsuvis/pipeline/storage/`, `tests/assets/analysis4d/`, `$DATA_DIR/analysis/<mission_id>/4d/timeline.json`, and `$DATA_DIR/analysis/<mission_id>/4d/qa.jsonl`.
- Execution path: Validate strict schemas, recompute geometry and temporal constraints, submit only unresolved semantic claims with positive and counter-evidence, execute accepted graph programs for answers, and add mission timeline/QA endpoints with evidence retrieval.
- Acceptance gates: `make test-unit`, `make test-ci`, API schema export, and the contradiction/QA benchmark pass; deep-profile relation precision is at least 0.90, false acceptance is at most 0.02 on the pinned contradiction suite, every accepted event and answer resolves to evidence, and timeout/refusal/malformed model output fails closed without blocking deterministic results.
- Documentation target: [Production server](current/production-server.md), [API reference](../reference/api.md), and a new strict-verifier/Video-QA runbook.
```

- Amendments: none.

## Implementation

`pipeline/analysis4d/verifier.py` recomputes deterministic predicates from stored geometry before any reviewer runs. A passed calibration gate accepts or rejects the claim. Residual above 0.5 m, or a metric predicate without metric scale, stays `uncertain`. Model confidence does not keep a contradicted predicate. Counts come from active tracks. Identity requires an identity link. An action is accepted only when an accepted deterministic event already covers its interval.

`pipeline/analysis4d/review.py` is the provider-neutral reviewer. The default is `unavailable`. `qwen` loads `Qwen/Qwen2.5-VL-3B-Instruct`. `remote` calls the Responses API with a strict JSON schema and image inputs (`gpt-6-astra` by default). Timeout, refusal, and malformed output raise `ReviewError`. Semantic claims then stay `uncertain` and the deterministic timeline is kept.

`pipeline/analysis4d/qa.py` executes graph programs against accepted events and edges. Empty, ambiguous, and causal programs are dropped. The answer is the program result.

`workflows/analysis4d_verify.py` `run_mission_verify` supersedes `timeline.json`, appends `qa.jsonl`, and appends a superseding proposal when the resolved status differs. The worker type is `postflight_strict_verifier`. It is not in the default postflight chain. `publishable_metadata` keeps only `accepted` rows for a future fusion-rt handoff. This job does not call fusion-rt.

Read routes: `GET /analysis/{mission_id}/4d/timeline`, `/qa`, and `/evidence`.

Current-state pages: [Production server](../current/production-server.md#strict-verifier-and-video-qa), [API](../../reference/api.md#analysis-routes), [Configuration](../../reference/configuration.md#strict-verifier-and-video-qa), and the [strict verifier runbook](../../runbooks/four-d-strict-verifier.md).

## Acceptance evidence

| Gate | Exact command, test or artifact | Result and limit |
| --- | --- | --- |
| Unit suite | `make test-unit` | 668 passed, 5 skipped |
| CI suite | `make test-ci` | 577 passed, 5 deselected |
| API schema export | `make export-openapi` | `docs/api/video-openapi.json` includes `/analysis/{mission_id}/4d/timeline`, `/qa`, and `/evidence` |
| Contradiction/QA benchmark | `python -m selfsuvis.pipeline.analysis4d.verifier_benchmark --probe-gpu` | `passed=true`, `failures=[]`, `false_accept_rate=0.0`, `fixture_false_accept_rate=0.0`, `relation_precision=1.0`, `accepted_events_with_evidence=true`, `accepted_answers_with_evidence=true`, `qa_matches_program=true`, `fail_closed=true`, `uncertainty_gate=true` |
| Deep-profile relation precision | same report, `relation_precision` | 1.0, limit at least 0.90 |
| False acceptance | same report, `false_accept_rate` | 0.0 on 50 false geometric claims, limit at most 0.02 |
| Evidence | same report | every accepted event and answer cites evidence |
| Fail closed | same report, `fail_closed` | timeout, refusal, and malformed review kept the deterministic `left_of` edge and left the semantic claim `uncertain` |
| Qwen-VL probe | same report, `qwen_probe` | `parsed`. The probe is not required for `passed` |

Runtime report: `$DATA_DIR/analysis/_benchmark/verifier-report.json`. The Qwen-VL probe loaded `Qwen/Qwen2.5-VL-3B-Instruct` and returned schema-valid JSON.

## Audit handoff

`none identified`. Reviewed scope: geometry override, uncertainty gate, review fail-closed, graph-program QA, publishable filter, worker job, and the timeline/QA/evidence routes.

## Close or resume

The declared gates for this task passed. Capability `four-d-scene-analysis` stays `planned` (one task remains). Plan counts after: 1 agent, 0 human; next eligible `four-d-profile-orchestration`.
