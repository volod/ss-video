# ss-video

Video ingest and search for outdoor autonomy. Import package `selfsuvis`.

Pins [volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.2.0` (`ss-perception`,
`ss-mapping`, `ss-fusion`, `fusion-rt`) and [volod/ss-common](https://github.com/volod/ss-common)
tag `v0.2.1`.

| Sibling | Role |
| --- | --- |
| [volod/ss-fusion](https://github.com/volod/ss-fusion) | Research pipeline, perception/mapping packages, fusion-rt |
| [volod/ss-sens](https://github.com/volod/ss-sens) | IoT sensor mesh |
| [volod/ss-control](https://github.com/volod/ss-control) | Site ingress, SSO, and CA |
| [volod/ss-mlab](https://github.com/volod/ss-mlab) | nanochat and sslm |
| [volod/ss-common](https://github.com/volod/ss-common) | Contracts and `ss_kit` |

```bash
make up          # api + fusion-rt + worker + ui + qdrant
```

## Quick start

`make venv` creates the project environment. When `ffmpeg` is missing it offers
to install system packages. It also asks for `HF_TOKEN` when `.env` does not
have one. Each run below checks both again before it starts. The first run of
a model downloads its weights.

### Quick run

Sample one video and write frame descriptions. This is the short file pipeline.

```bash
make venv
make run VIDEO=/path/to/video.mp4
```

```bash
.venv/bin/ssv --mode file --input /path/to/video.mp4
```

Output goes to `$DATA_DIR/local_runs/<video-name>/`. Extra `ssv` flags go in
`RUN_ARGS`.

### 4D analysis

Ground one video, then write tracks and a scene timeline.

```bash
make analyze VIDEO=/path/to/video.mp4
```

```bash
.venv/bin/python -m selfsuvis.pipeline.analysis4d.analyze /path/to/video.mp4
```

Prints track labels, accepted events, geometry sample count, and the artifact
directory `$DATA_DIR/analysis/<mission_id>/4d/`. A copy of that summary is
`$DATA_DIR/analysis/<mission_id>/summary.json`. Sampling defaults to 1 frame
per second with prompts `person,vehicle`. When a GPU slot is free, kept
keyframes get depth-backed boxes. Without calibration, `metric_scale` stays
`unavailable`.

```bash
make analyze VIDEO=/path/to/video.mp4 ANALYZE_ARGS="--profile deep"
```

`--profile deep` adds the postflight revision. `--fps` and `--prompts` change
the sample rate and the grounding text.

### Full run

Run the long local pipeline on one video: perception, mapping, and captions.

```bash
make run-full VIDEO=/path/to/video.mp4
```

```bash
.venv/bin/ssv --mode local --video /path/to/video.mp4
```

Extra `ssv` flags go in `RUN_ARGS`. The run writes under
`$DATA_DIR/local_runs/`.

## Docs

See [docs](docs/README.md), the [specification](docs/design/spec.md), and
[current implementation](docs/impl/current.md).
