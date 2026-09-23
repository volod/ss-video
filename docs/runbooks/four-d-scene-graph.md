# Temporal scene graph

Turns 2D tracks and 3D geometry into an append-only graph. Operators can read
how objects, relations, and actions changed, and can trace each event to a
delta and a geometry sample.

Current behavior:
[production server](../impl/current/production-server.md#temporal-scene-graph).
Configuration:
[configuration](../reference/configuration.md#4d-scene-graph).
Routes:
[API](../reference/api.md#analysis-routes).

The YOLO semantic environment graph is unchanged. It remains a mission summary
of `near` edges. This graph does not replace it and does not publish to
fusion-rt.

## What a mission run writes

`selfsuvis.pipeline.workflows.analysis4d_graph.run_mission_graph` reads
`tracks.jsonl` and `geometry/` under `$DATA_DIR/analysis/<mission_id>/4d/`.
The worker job type is `postflight_scene_graph`. It is not chained after
mapping.

The run appends `graph-deltas.jsonl` and, when a provider returns claims,
`proposals.jsonl`. It replaces `timeline.json` and `manifest.json` only by
naming the previous digest in `supersedes_sha256`. `tracks.jsonl` is not
rewritten. A second run that already has a `scene_graph` stage returns the
existing log.

## Replay

Deltas sort by `(t_sec, delta_id)`. `add` inserts a node or an edge.
`supersede` inserts the new node and drops the named earlier delta from the
current view. The earlier delta stays in the file. The read API is
`GET /analysis/{mission_id}/4d/graph`. `t_sec` keeps edges whose interval
contains that time. `GET /analysis/{mission_id}/4d/deltas` returns the log.

An identity link does not edit the old track observation. The new track's node
delta sets `supersedes` to the linked node's add delta.

## Relations and events

Predicates are the contract set: `contains`, `intersects`, `left_of`,
`right_of`, `above`, `below`, `supports`, `contacts`, `occludes`,
`distance_band`, `visible`, and `relative_motion`. They are recomputed with
`relation_holds`. An accepted edge's interval must contain a geometry sample
for both endpoints, and that sample must still satisfy the predicate.

Intervals are half-open. A predicate that stops is closed at the next
co-timestamp. A predicate that holds through the last sample is closed one
second later (`HOLD_TAIL_SEC`). Symmetric predicates (`intersects`,
`contacts`, `distance_band`) are stored once, with the lower node id as the
subject. Distance bands are stored only when the manifest scale is `metric`.
The first matching band of `[0, 2]`, `[2, 8]`, and `[8, 32]` meters is kept.
`visible` is a track-to-region edge while the track is tentative, confirmed,
or occluded.

Events come from those edges:

| Event | When |
| --- | --- |
| `entered_region` | A region `contains` edge starts |
| `left_region` | That track has a later sample outside the interval |
| `approached` | A `distance_band` with max 2 m starts |
| `put_down` | A `supports` edge starts |
| `picked_up` | That support ends and the object has moved |
| `count_changed` | The set of active tracks for one label changes |

Each accepted event stores the delta id and a geometry path.

## Proposals

`ANALYSIS4D_VLM_PROVIDER` defaults to `unavailable`. The deterministic graph is
still written, and the manifest records `provider_unavailable`.

`smolvlm` loads `HuggingFaceTB/SmolVLM-256M-Instruct` and must return a JSON
list of claims. Unknown fields fail closed. A deterministic predicate that
contradicts stored geometry is stored as `rejected`. Other claims stay
`uncertain` until a later verifier. A `corrected` claim sets `supersedes` and
leaves the earlier row in the log. SceneGraphVLM SFT and GRPO training stay in
ss-fusion. This service does not train that model.

## Benchmark

```bash
python -m selfsuvis.pipeline.analysis4d.graph_benchmark
```

The report is `$DATA_DIR/analysis/_benchmark/graph-report.json`
(`ss-video.scene-graph-benchmark.v1`). `--probe-gpu` attempts SmolVLM. That
probe is not required for `passed`.

On the pinned scene (two objects inside a region, metric calibration
`cal-bench`):

| Field | Result |
| --- | --- |
| `relation_precision` | 1.0 |
| `yolo_relation_precision` | 0.0 |
| `deterministic_edge_count` | 19 |
| `replay_deterministic` | true |
| `intervals_valid` | true |
| `vlm_degraded` | true |

The YOLO baseline is `build_semantic_environment_graph` on the same labels and
times. Its edges are `near` and do not match the pinned predicates, so its
precision on this relation set is 0. The 4D edges match the predicates
`relation_holds` accepts, including their intervals.
