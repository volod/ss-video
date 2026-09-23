# Strict verifier and Video-QA

Checks timeline claims before they are treated as narrative, and writes
spatial Video-QA from accepted graph state.

Current behavior:
[production server](../impl/current/production-server.md#strict-verifier-and-video-qa).
Configuration:
[configuration](../reference/configuration.md#strict-verifier-and-video-qa).
Routes:
[API](../reference/api.md#analysis-routes).

The YOLO semantic environment graph is unchanged. This job does not publish
rejected or uncertain claims to fusion-rt.

## What a mission run writes

`selfsuvis.pipeline.workflows.analysis4d_verify.run_mission_verify` reads the
mission directory under `$DATA_DIR/analysis/<mission_id>/4d/`. The worker job
type is `postflight_strict_verifier`. It is not in the default postflight
chain. Run the scene-graph job first when the directory has no deltas yet.

The run supersedes `timeline.json` and `manifest.json` by naming the previous
digest in `supersedes_sha256`. It appends `qa.jsonl`. When a claim's resolved
status differs from the stored row, it appends a proposal whose `supersedes`
points at the original. The original row stays in `proposals.jsonl`.
`tracks.jsonl` is not rewritten. A second run that already has a
`strict_verifier` stage returns the existing timeline.

## Resolution order

1. Schema, ids, intervals, and coordinate frame are checked. Unknown fields
   fail closed.
2. Deterministic predicates are recomputed from stored boxes. The calibration
   gate must pass: metric predicates need `metric` scale and a calibration id,
   and a residual above 0.5 m stays `uncertain`. When the gate passes, geometry
   accepts or rejects the claim. A model status of `accepted` does not keep a
   contradicted predicate.
3. Counts are taken from active tracks. Identity needs an identity link.
   Event time is not taken from the reviewer.
4. Only the remaining semantic claims are sent to a reviewer, with
   measurements and counter-evidence. An action is accepted only when an
   accepted deterministic event already covers that interval.

`ANALYSIS4D_REVIEW_PROVIDER` defaults to `unavailable`. Unsettled claims stay
`uncertain`, and the manifest records `provider_unavailable`. `qwen` loads
`Qwen/Qwen2.5-VL-3B-Instruct`. `remote` uses the Responses API, strict JSON,
and image inputs. The remote model defaults to `gpt-6-astra`. Timeout,
refusal, and malformed output fail closed and leave the deterministic
timeline in place.

## Video-QA

Questions are graph programs executed on accepted events and edges:

| Program | Answer |
| --- | --- |
| `entered(?track, REGION, after=T)` | The one track that entered `REGION` at or after `T` |
| `count(LABEL, at=T)` | The participant count of the matching `count_changed` event |
| `left_of(?subject, OBJECT)` | The one subject with an accepted `left_of` edge |
| `picked_up(?track, OTHER)` and the same shape for `put_down`, `left_region`, `approached` | The one track in that event |
| `evidence(EVENT_ID)` | The stored geometry or mask path |

An empty or ambiguous program is dropped. Causal programs are not generated.
Every accepted answer cites an evidence event. Zero QA rows is valid when no
program has one answer.

## Read API

- `GET /analysis/{mission_id}/4d/timeline`
- `GET /analysis/{mission_id}/4d/qa`
- `GET /analysis/{mission_id}/4d/evidence?event_id=` or `qa_id=`

Evidence retrieval returns the cited geometry or mask document when the file
is present.

## Benchmark

```bash
python -m selfsuvis.pipeline.analysis4d.verifier_benchmark
```

The report is `$DATA_DIR/analysis/_benchmark/verifier-report.json`
(`ss-video.strict-verifier-benchmark.v1`). `--probe-gpu` loads Qwen-VL when
the weights are available. The pass/fail gate does not require that probe.

On the pinned contradiction suite (50 false geometric claims plus the
fixture's `proposal-false`) and the deep-profile scene:

| Field | Result |
| --- | --- |
| `false_accept_rate` | 0.0 |
| `fixture_false_accept_rate` | 0.0 |
| `relation_precision` | 1.0 |
| `accepted_events_with_evidence` | true |
| `accepted_answers_with_evidence` | true |
| `qa_matches_program` | true |
| `fail_closed` | true |
| `uncertainty_gate` | true |
| `qwen_probe` | `parsed` with `--probe-gpu` |

False acceptance is at most 0.02. Deep-profile relation precision is at
least 0.90. A residual of 2 m keeps a true `left_of` claim `uncertain`.
