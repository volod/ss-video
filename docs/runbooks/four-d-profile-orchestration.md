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
`published-events.jsonl`. Each new ledger line is published on the
event-envelope MQTT topic. The deep pass keeps the fast event ids, appends
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
(`contracts/odcs/verified-scene-event.odcs.yaml`). `VideoContractPublisher`
sends that envelope on

```text
ss/v1/site/{site_id}/zone/{zone_id}/event/video_4d
```

`site_id` comes from `COOP_SITE_ID`. `zone_id` comes from `ANALYSIS4D_ZONE_ID`.
The modality is `video_4d`. A one-shot client is used from the worker and from
`python -m selfsuvis.pipeline.analysis4d.analyze`. That client omits
`COOP_MQTT_CLIENT_ID` and leaves the API publisher session in place. QoS
follows the committed `event-envelope` topic entry (1).

```bash
mosquitto_sub -h "${COOP_MQTT_HOST:-localhost}" -p "${COOP_MQTT_PORT:-1883}" \
  -t 'ss/v1/site/+/zone/+/event/video_4d' -v
```

Rejected and uncertain timeline rows stay in `timeline.json`. They are absent
from the ledger and from MQTT. An event id already in the ledger stays
unpublished on later runs. When the broker refuses the connection, the ledger
line is still appended and the warning `verified event MQTT delivery failed`
is logged. A later run skips that id.

The packaged fusion-rt seed `selfsuvis/fusion_rt/data/fusion_rules.yaml` stays
as shipped. fusion-rt still subscribes to `sensor-event`, `sensor-state`,
`camera-event`, and `scene-caption`. This publish path leaves that subscription
list and the seed rules in place.

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
`person,vehicle`. The summary lists track labels, accepted events, geometry
sample count, `metric_scale`, and degradations.

A free GPU slot and a queue that did not coalesce run keyframe depth. Samples
land in `geometry/`. Without a camera calibration id every sample is
`metric_scale` `unavailable` and the manifest stays `unavailable`. The box is a
camera ray with depth divided by its median. It is not meters. A depth load
failure records `provider_unavailable` and keeps the tracks. A saturated queue
sheds `dense_geometry` with `budget_shed`, keeps the tracks, and writes a
`queue_coalesce` gap.

Optional regions use `ss-video.region-boxes.v1`:

```json
{
  "schema_version": "ss-video.region-boxes.v1",
  "regions": [
    {
      "node_id": "region-yard",
      "label": "yard",
      "center_m": [0.0, 0.0, 1.0],
      "extent_m": [4.0, 4.0, 4.0]
    }
  ]
}
```

Place that file at `$DATA_DIR/analysis/<mission_id>/regions.json`, or as
`<video-stem>.regions.json` beside the video. `analyze` copies the sibling file
when the mission file is absent. A prompt may contain spaces. Event ids replace
separators with hyphens. The summary keeps the prompt text.

Artifacts:

```text
$DATA_DIR/analysis/<mission_id>/summary.json
$DATA_DIR/analysis/<mission_id>/4d/tracks.jsonl
$DATA_DIR/analysis/<mission_id>/4d/timeline.json
$DATA_DIR/analysis/<mission_id>/4d/geometry/
```
