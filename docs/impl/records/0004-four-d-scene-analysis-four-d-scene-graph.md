# Temporal scene graph

## Task and scope

- Id / capability: `four-d-scene-graph` / `four-d-scene-analysis`
- State: accepted
- Source: plan task at `5795bc3`. This change is the append-only temporal graph.
- Plan counts at start: 3 agent, 0 human; next eligible `four-d-scene-graph`.
- Accepted task:

```markdown
#### four-d-scene-graph

The current semantic environment graph is a mission summary; it does not preserve temporal edge
validity, graph edits, action intervals, or superseded identities.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#temporal-3d-scene-graph-and-narrative)
- Agent status: RUN NEEDED
- Research: yes
- Dependencies: `four-d-spatial-reconstruction`. Blocks: `four-d-strict-verifier-and-qa`.
- User-visible outcome: Operators can inspect how objects, relations, and actions evolved over time and trace every narrative event back to graph deltas and observations.
- Scope boundary: Merge 2D tracks and eligible 3D geometry into append-only temporal nodes/edges, implement deterministic relations and event reduction, and integrate one schema-constrained compact VLM proposal provider; SceneGraphVLM-style SFT/GRPO training and dataset work remain in ss-fusion.
- Data and artifact paths: `src/selfsuvis/pipeline/analysis4d/`, `src/selfsuvis/worker/handlers/`, `tests/assets/analysis4d/`, `$DATA_DIR/analysis/<mission_id>/4d/graph-deltas.jsonl`, and `$DATA_DIR/analysis/<mission_id>/4d/proposals.jsonl`.
- Execution path: Add a postflight graph job and a causal graph reducer, materialize current state from graph deltas, serialize region/geometry/trajectory context for the selected VLM provider, and preserve proposed, corrected, rejected, and superseded claims.
- Acceptance gates: `make test-unit`, `make test-ci`, and the pinned scene-graph benchmark pass; replaying deltas is deterministic, graph intervals and references are valid, deterministic relation precision improves over the current YOLO semantic-graph baseline, and an unavailable VLM yields a useful deterministic graph with an explicit degradation record.
- Documentation target: [Production server](current/production-server.md), [API reference](../reference/api.md), and a new temporal scene-graph runbook.
```

- Amendments: none.

## Implementation

`pipeline/analysis4d/graph.py` merges tracks and geometry into append-only nodes and edges. Replay sorts by `(t_sec, delta_id)`. An identity link supersedes the earlier node and does not rewrite the track file. Relation intervals are half-open. Metric predicates stay off unless the manifest scale is `metric`.

`workflows/analysis4d_graph.py` `run_mission_graph` appends `graph-deltas.jsonl` and `proposals.jsonl`, then supersedes `timeline.json` and `manifest.json`. The worker type is `postflight_scene_graph`. It is not in the default mapping chain.

`pipeline/analysis4d/vlm.py` parses a JSON list of claims. Unknown fields fail closed. A deterministic predicate that contradicts geometry is stored as `rejected`. Corrected claims keep `supersedes`. The default provider is `unavailable`, which records `provider_unavailable` and still writes the deterministic graph. `smolvlm` is the optional compact adapter (`HuggingFaceTB/SmolVLM-256M-Instruct`). SceneGraphVLM training stays in ss-fusion. VLM claims are not accepted timeline events.

Read routes: `GET /analysis/{mission_id}/4d/graph` (optional `t_sec`), `/deltas`, and `/proposals`.

Current-state pages: [Production server](../current/production-server.md#temporal-scene-graph), [API](../../reference/api.md#analysis-routes), [Configuration](../../reference/configuration.md#4d-scene-graph), and the [temporal scene-graph runbook](../../runbooks/four-d-scene-graph.md).

## Acceptance evidence

| Gate | Exact command, test or artifact | Result and limit |
| --- | --- | --- |
| Unit suite | `make test-unit` | 654 passed, 5 skipped |
| CI suite | `make test-ci` | 563 passed, 5 deselected |
| Scene-graph benchmark | `python -m selfsuvis.pipeline.analysis4d.graph_benchmark` | `passed=true`, `failures=[]`, `relation_precision=1.0`, `yolo_relation_precision=0.0`, `deterministic_edge_count=19`, `replay_deterministic=true`, `intervals_valid=true`, `vlm_degraded=true`, degradations `provider_unavailable` |
| Replay | same report, `replay_deterministic` | true |
| Intervals and references | same report, `intervals_valid` | true (`validate_bundle`) |
| Relation precision vs YOLO | same report | 1.0 > 0.0. YOLO emits `near` only |
| Unavailable VLM | same report, `vlm_degraded` | true; 19 accepted edges remain |

Runtime report: `$DATA_DIR/analysis/_benchmark/graph-report.json`. Host: NVIDIA GeForce RTX 4060 Ti. SmolVLM weights are not in the local cache, so the required gate is the unavailable provider.

## Audit handoff

`none identified`. Reviewed scope: causal reducer, delta replay, identity supersession, event intervals, proposal statuses, worker job, and the graph read routes.

## Close or resume

The declared gates for this task passed. Capability `four-d-scene-analysis` stays `planned` (two tasks remain). Plan counts after: 2 agent, 0 human; next eligible `four-d-strict-verifier-and-qa`.
