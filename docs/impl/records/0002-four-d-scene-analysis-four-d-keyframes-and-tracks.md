# 4D keyframes and tracks

## Task and scope

- Id / capability: `four-d-keyframes-and-tracks` / `four-d-scene-analysis`
- State: accepted
- Source: plan task at `c476653`. This change is the keyframe selector, track lifecycle, and provider pin.
- Plan counts at start: 5 agent, 0 human; next eligible `four-d-keyframes-and-tracks`.
- Accepted task:

```markdown
#### four-d-keyframes-and-tracks

Current adaptive sampling is pairwise and current SAM use refines sampled-frame boxes; it cannot
preserve promptable instance identities through long occlusion while bounding heavy-model cost.

- Serves: `four-d-scene-analysis` -- [Specification section](../design/spec.md#dynamic-keyframes-and-2d-perception)
- Agent status: RUN NEEDED
- Research: yes
- Dependencies: `four-d-contracts-and-benchmark`. Blocks: `four-d-spatial-reconstruction`, `four-d-profile-orchestration`.
- User-visible outcome: A mission produces auditable 2D tracks for text- or exemplar-prompted objects, with heavy grounding limited to narrative-changing keyframes and lightweight propagation between them.
- Scope boundary: Implement the bounded MaxInfo-style selector, trigger policy, provider interfaces, track lifecycle, mask references, and Grounding DINO/CountGD plus SAM 2 or SAM 3 integration through a pinned ss-perception release or sidecar; do not train models in this repository, claim CountGD detections as persistent identities, or run all candidate providers in production.
- Data and artifact paths: `src/selfsuvis/pipeline/workflows/`, `src/selfsuvis/pipeline/analysis4d/`, `tests/assets/analysis4d/`, `$DATA_DIR/analysis/<mission_id>/4d/tracks.jsonl`, and `$DATA_DIR/analysis/<mission_id>/4d/masks/`.
- Execution path: Benchmark candidate provider combinations against the existing YOLO+SAM and RF-DETR path, pin the selected artifact/config, run the fast causal pass and the deep forward/backward pass, and record forced-keyframe reasons, memory resets, gaps, and count-vs-track disagreement.
- Acceptance gates: `make test-unit`, `make test-ci`, and the declared GPU benchmark pass; selected keyframes cover every annotated event boundary in the fixture corpus, track output is timestamp-monotonic and deterministic under a fixed seed, occlusion/cut fixtures do not silently reuse ids, and a documented no-go is valid if no provider meets license, memory, quality, and latency gates and dependent tasks are re-scoped or removed before this task closes.
- Documentation target: [Production server](current/production-server.md), [Configuration](../reference/configuration.md), and a new 4D model runbook.
```

- Amendments: none.

## Implementation

`pipeline/analysis4d/keyframes.py` selects keyframes with a bounded MaxInfo residual over frame embeddings. Forced triggers (`stream_start`, `scene_cut`, `prompt_change`, `calibration_change`, `max_gap`, and later tracker reasons) may exceed the per-chunk budget. The fast profile budgets inside each chunk. The deep profile uses the same budget times the number of chunks. Histogram, SSIM, and embedding-drift thresholds match the indexer defaults (0.25, 0.25, 0.15). Time orders ties; the seed is only the last tie-break.

`pipeline/analysis4d/tracks.py` keeps tentative, confirmed, occluded, lost, and ended states. A long gap or a camera cut ends the live id and the next observation is a new id. The deep pass may set `identity_link` (`occlusion` or `camera_cut`) on the birth observation. The fast pass does not. Count-only hits are ignored. A count that disagrees with the number of tracks is recorded and does not create tracks.

`pipeline/analysis4d/providers.py` defines grounding, count, and mask interfaces. Production loads one grounding provider. The pin is `IDEA-Research/grounding-dino-tiny` revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, weights `sha256:1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`, plus kinematic masks. CountGD is not in ss-perception `v0.2.0` and is not pinned. SAM 3 is an alternative joint provider; its license is unset, so it cannot pass the gate. YOLO+SAM, RF-DETR, and SAM 2 do not import on this host, so the benchmark records them unavailable and does not load their weights. The direct Transformers zero-shot API is used. The Hugging Face object-detection pipeline does not support this model.

`workflows/analysis4d_tracks.py` `run_mission_tracks` writes the fast and deep passes into `$DATA_DIR/analysis/<mission_id>/4d/`. The worker job does not call it yet; admission and the indexer DAG stay in `four-d-profile-orchestration`. `track-audit.json` records keyframe reasons, memory resets, and count disagreements. Those disagreements are also folded into the stage `trigger_reason`. A Grounding DINO load failure sets `failed` and the manifest records `provider_unavailable`. Mask references are 8x8 kinematic rasters under `masks/`.

Current-state pages: [Production server](../current/production-server.md#four-dimensional-keyframes-and-tracks), [Configuration](../../reference/configuration.md#4d-keyframes-and-tracks), [Data/config](../current/data-config.md#four-dimensional-analysis-artifacts), and the [4D model runbook](../../runbooks/four-d-models.md).

## Acceptance evidence

| Gate | Exact command, test or artifact | Result and limit |
| --- | --- | --- |
| Unit suite | `make test-unit` | 634 passed, 5 skipped |
| CI suite | `make test-ci` | 543 passed, 5 deselected |
| GPU benchmark | `python -m selfsuvis.pipeline.analysis4d.track_benchmark` | `passed=true`, `failures=[]`, `keyframe_event_coverage=1.0`, `monotonic=true`, `seed_stable=true`, `silent_id_reuse=false`. Pin `grounding_dino+kinematic`, `model_gate=pass`. Grounding DINO latency `0.13969442749023436` s, peak allocation `938438144` bytes, under 1.0 s and 8 GiB. Cached revision and weights digest match the pin. |
| Event-boundary coverage | same report, fixture corpus `tests/assets/analysis4d/` | coverage 1.0 on fast and deep |
| Fixed seed | `pytest tests/unit/pipeline/analysis4d/test_tracks.py` | pass, including the seed-stability case |
| Occlusion and cut | occlusion-cut bundle written by the benchmark | deep observation at the cut links to the earlier id with reason `camera_cut`; fast has no `identity_link`; the same id is not reused across the cut |
| Provider no-go | probes in `$DATA_DIR/analysis/_benchmark/tracks-report.json` | CountGD, SAM 3, RF-DETR, YOLO+SAM, and SAM 2 miss the gate. Grounding DINO passes, so dependent tasks stay on the plan. |

Runtime report: `$DATA_DIR/analysis/_benchmark/tracks-report.json`. Host: NVIDIA GeForce RTX 4060 Ti, 16380 MiB.

## Audit handoff

`none identified`. Reviewed scope: keyframe selector, tracker, provider pin, fast and deep writer, fixture track benchmark, and the unit tests that cover them.

## Close or resume

The declared gates for this task passed. Capability `four-d-scene-analysis` stays `planned` (four tasks remain). Plan counts after: 4 agent, 0 human; next eligible `four-d-spatial-reconstruction`.
