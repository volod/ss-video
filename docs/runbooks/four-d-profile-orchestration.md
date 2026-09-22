# Profile orchestration

Selects a fast or deep 4D analysis, keeps track continuity when the queue is
full, and publishes only accepted events.

Current behavior:
[production server](../impl/current/production-server.md#profile-orchestration).
Configuration:
[configuration](../reference/configuration.md#profile-orchestration).
Routes:
[API](../reference/api.md#analysis-routes).

`ANALYSIS4D_PROFILE` defaults to `off`. Indexing then does not enqueue a 4D
job, and video search is unchanged. A remote VLM is not enabled.

## What a mission run writes

`selfsuvis.pipeline.workflows.analysis4d_profile.run_fast_profile` reads indexed
frames and writes the mission directory under
`$DATA_DIR/analysis/<mission_id>/4d/`. The worker job type is `analysis4d_fast`.
`run_deep_profile` is `postflight_analysis4d_deep`. Neither job is in the
default postflight chain.

The fast pass writes `tracks.jsonl`, `gaps.jsonl` when frames are coalesced,
`timeline.json`, `manifest.json`, `orchestration-state.json`, and
`published-events.jsonl`. The deep pass keeps the fast event ids, appends
superseding events, and replaces `manifest.json` only after copying the
previous file under `history/` and setting `supersedes_sha256` to the fast
digest.

A second run with the same frame times, quality flags, and prompt ids returns
the stored result and does not append another ledger line.

## Admission

Frames enter a bounded queue (`ANALYSIS4D_QUEUE_CAPACITY`, default 32). The
queue never reorders timestamps. When it is full it drops a redundant frame and
records a `queue_coalesce` gap. Scene-cut, calibration, and prompt frames are
kept.

`dense_geometry`, `vlm`, and `review` wait for a GPU slot
(`ANALYSIS4D_GPU_SLOTS`, default 1). Pressure, or zero slots, sheds those
stages with `budget_shed`. Tracks still run. `ANALYSIS4D_VLM_PROVIDER` and
`ANALYSIS4D_REVIEW_PROVIDER` remain `unavailable` unless set.

## Publication

Only `accepted` rows are published. Each ledger line is an ss-common
`event-envelope` 1.0.0. The payload is `verified-scene-event` 1.0.0
(`contracts/odcs/verified-scene-event.odcs.yaml`). The envelope modality is
`video_4d`. Default fusion-rt rules match camera plus audio, so this publish
path does not change correlation policy and does not add a fusion rule.

Rejected and uncertain timeline rows stay in `timeline.json` and are absent
from the ledger. Publishing the same event id again is a no-op.

## Status

`GET /analysis/{mission_id}/4d/status` returns profile, queue depth, queue
capacity, degradations, backlog, and published event ids. The index form and
the Streamlit index page send `analysis_profile`. A finished job stores the
same fields under `progress.analysis4d`.

## Load benchmark

The fixture is 15 minutes at 2 frames per second with scripted grounding, so
the gate measures the scheduler rather than Grounding DINO. Lag for each
accepted fast event is that chunk's compute time shared across the configured
event types (`entered_region`, `left_region`, `count_changed`). Real-time
factor is elapsed compute divided by media duration.

```bash
python -m selfsuvis.pipeline.analysis4d.profile_benchmark
```

The report is `$DATA_DIR/analysis/_benchmark/profile-report.json`
(`ss-video.profile-benchmark.v1`). `passed` requires real-time factor at most
1.0, p95 lag at most 3 seconds, queue depth at most the configured capacity,
a gap record for every coalesced span, a fast replay, and a deep manifest
that supersedes the fast digest without dropping fast event ids. The report
records the CUDA device name from torch.

## Analyze a file

```bash
.venv/bin/python -m selfsuvis.pipeline.analysis4d.analyze /path/to/video.mp4
```

Requires `ffmpeg`. Samples at 1 fps unless `--fps` is set. Prompts default to
`person,vehicle`. The summary lists track labels and accepted events. Geometry
samples are absent until the keyframe-geometry task lands, so a clip can finish
with tracks and zero region events. That is a valid result. Artifacts:

```text
$DATA_DIR/analysis/<mission_id>/summary.json
$DATA_DIR/analysis/<mission_id>/4d/tracks.jsonl
$DATA_DIR/analysis/<mission_id>/4d/timeline.json
```
