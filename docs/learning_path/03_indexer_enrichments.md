# Indexer Enrichments

These stages are the production equivalent of ss-fusion perception steps 1-8.
The worker runs them through ss-perception modules. Model ops live in
[runbooks](../runbooks/README.md).

Everything later (scene query, robot pose, optional tracking) is built on this
evidence. If you undersample frames you miss events; if you oversample, every
model costs more.

## 1. Frame extraction

Turn a compressed video into timestamped JPEGs. FFmpeg handles codec quirks,
variable frame rate, and B-frames; OpenCV/PIL timestamps are often wrong.

Focus: FPS vs scene change rate; `t_sec` alignment with ASR and GPS; motion blur
at high speed. Failure: container with no timestamps, missing HEVC, corrupt first
keyframe.

## 2. CLIP and DINOv3 vectors

Encode each frame into one or two spaces and upsert Qdrant points with payload
(`frame_path`, `t_sec`, mission ids, later captions). Runbook:
[clip-dino.md](../runbooks/clip-dino.md).

Focus: when CLIP retrieval beats DINO and the reverse. Failure: OOM, wrong
`OPENCLIP_PRETRAINED`, mixed collections after a model change.

## 3. Optional scene context (Gemma)

Production may sample frames for scene type, change detection, and clustering
when the Gemma extra is enabled. This is a domain hint for later VLMs, not the
retrieval backbone. Research-path detail:
[ss-fusion step 3](https://github.com/volod/ss-fusion/blob/v0.2.0/docs/learning_path/02_perception_core_steps_01_08.md).

If the sidecar is down, `scene_type` stays empty and later captions lose the hint.

## 4. Florence captions

Florence-2 produces a short description per keyframe. Captions are the first
human-readable layer and feed scene segments (token overlap between adjacent
captions). Runbook: [florence-2.md](../runbooks/florence-2.md).

Focus: read ten captions against the pixels; find one clearly wrong caption;
check whether segment boundaries match real scene changes. Failure: OOM (batch
size falls back to 1), generic captions on dark or blurry frames.

## 5. Whisper ASR

Extract audio, transcribe, align segments to frames. Speech often carries
callouts and coordinates that never appear visually. Runbook:
[whisper-asr.md](../runbooks/whisper-asr.md).

Focus: timestamp alignment with `t_sec`. Failure: VAD clipping short utterances;
no audio track.

## 6. OCR

Read visible text (HUD, signs, labels) into `frame_facts_json`. Runbook:
[ocr.md](../runbooks/ocr.md). Scene query can filter on these tokens.

## 7. Depth

Monocular depth gives geometry hints (near/far, occlusion), not metric scale
unless calibrated. Runbook: [depth.md](../runbooks/depth.md).

## 8. Detection

HF detectors and optional YOLO+SAM write boxes, tracks, and (when enabled) a
semantic environment graph. Runbooks: [detection-hf.md](../runbooks/detection-hf.md),
[yolo-sam.md](../runbooks/yolo-sam.md).

Optional later stages (Gemma-directed tracking, Qwen VLM, UniDriveVLA) are
ss-fusion models invoked from the worker when extras are on. Their study docs
stay in ss-fusion ([siblings.md](siblings.md)).

These stages currently enrich sampled frames or produce a mission-level semantic graph; they are not
yet a verified temporal 3D scene graph. The planned
[4D scene-analysis capability](../design/spec.md#four-dimensional-video-scene-analysis) adds
persistent mask tracks, uncertainty-aware geometry, graph deltas, strict verification, and
evidence-derived Video-QA without changing this current behavior until its plan tasks ship.

## Degradation rule

A missing extra skips that enricher and continues. Confirm in job logs which
stages actually ran before blaming search quality.

Next: [realtime and maps](04_realtime_and_maps.md).
