# Configuration

Defaults live in `src/selfsuvis/env/dev.env`, `test.env`, and `prod.env` (ss-perception package data). Set `APP_ENV=dev|test|prod` to pick one, then override them with a project-root `.env` or exported environment variables. Generate a resource-aware root `.env` with `ssv-env --env dev`. The authoritative sources are `selfsuvis.pipeline.core.config` (shared video/perception knobs) and
`selfsuvis.fusion_rt.config` (fusion). Both subclass `ss_kit.settings.KitSettings`.

## Core storage and paths

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `./.data` | Root for videos, frames, tiles, reports, maps, checkpoints, and edge assets |
| `VIDEOS_DIR` | `./.data/videos` | Stored indexed video copies |
| `FRAMES_DIR` | `./.data/frames` | Extracted frames |
| `TILES_DIR` | `./.data/tiles` | Extracted tiles |
| `REPORTS_DIR` | `./.data/reports` | Mission report output |
| `MAPS_DIR` | `./.data/maps` | SfM, splat, and semantic graph outputs |
| `DATABASE_URL` | env-specific | Video PostgreSQL DSN (database `selfsuvis`) |
| `FUSION_DATABASE_URL` | derived `.../selfsuvis_fusion` | Fusion PostgreSQL DSN; derived from `DATABASE_URL` when unset |
| `FUSION_RT_URL` | `http://fusion-rt:8000` (compose UI) | Streamlit UI base URL for `/api/v1/*`; host bind is `http://localhost:8001` |
| `CORRELATOR_POLL_INTERVAL_S` | `5` | fusion-rt correlator poll period; test overlay sets `1` |
| `QDRANT_HOST` / `QDRANT_PORT` | `qdrant` / `6333` | Qdrant connection (server image `qdrant/qdrant:v1.19.1`, client `qdrant-client>=1.19.1,<2`) |
| `QDRANT_COLLECTION` | `video_semantic` | Main vector collection |

The video API applies the video schema on startup. fusion-rt applies the fusion schema.
The worker applies the video schema.
To apply from the CLI:

```bash
python -m selfsuvis.scripts.migrate_postgres
python -m selfsuvis.scripts.migrate_postgres --owner video
python -m selfsuvis.scripts.migrate_postgres --owner fusion
```

## Retrieval and indexing

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `openclip` | `openclip`, `dinov2`, `dinov3`, or `gemma` |
| `OPENCLIP_MODEL` | `ViT-B-16` | Text/image retrieval backbone |
| `OPENCLIP_PRETRAINED` | `openai` | OpenCLIP weights tag |
| `DEVICE` | `auto` | Local inference device |
| `USE_FP16` | `true` | Half precision on CUDA |
| `SAMPLE_FPS_BASE` / `MIN` / `MAX` | `2 / 0.5 / 5` | Adaptive keyframe sampling |
| `HIST_THRESH` | `0.25` | Histogram/SSIM keep threshold |
| `EMBED_DRIFT_THRESH` | `0.15` | Embedding drift keep threshold |
| `MAX_GAP_SEC` | `10` | Force-keep interval |
| `TILE_SIZE` / `STRIDE` | `384 / 256` | Tile extraction geometry |
| `MAX_TILES_PER_SEGMENT` | `200` | Hard tile cap |
| `DEDUP_COS_SIM_THRESH` | `0.95` | Tile dedup cosine threshold |
| `PHASH_HAMMING_MAX` | `6` | Tile dedup perceptual-hash threshold |

## Captioning and multimodal enrichments

| Variable | Default | Purpose |
|---|---|---|
| `FLORENCE_BATCH_SIZE` | `16` | Local Florence batch size |
| `FLORENCE_API_URL` | empty | Use remote Florence endpoint instead of local model |
| `FLORENCE_MODEL` | `microsoft/Florence-2-large` | Remote Florence model ID |
| `QWEN_API_URL` | empty | Enables structured scene extraction sidecar |
| `QWEN_BACKEND` | `vllm` | `vllm` or `ollama` |
| `QWEN_MODEL` | backend-dependent | Qwen sidecar model |
| `QWEN_SIDECAR_CONCURRENCY` | env-specific | Parallel Qwen sidecar requests per local run |
| `QWEN_IMAGE_MAX_SIDE` | env-specific | Resize bound for Qwen frame uploads |
| `QWEN_MAX_FRAMES` | env-specific | Max frames selected for detailed Qwen captioning |
| `GEMMA_MODEL_ID` | `google/gemma-3-4b-it` | Local Gemma embedder |
| `GEMMA_API_URL` | empty | Gemma sidecar endpoint |
| `GEMMA_API_MODEL` | backend-dependent | Gemma sidecar model |
| `UNIDRIVE_ENABLED` | `false` | Enables UniDriveVLA expert analysis |
| `UNIDRIVE_API_URL` | empty | UniDriveVLA OpenAI-compatible bridge endpoint |
| `UNIDRIVE_MODEL` | `owl10/UniDriveVLA_Nusc_Base_Stage3` | UniDriveVLA model ID for bridge / sidecar |
| `UNIDRIVE_TIMEOUT_SEC` | `60` | UniDrive request timeout |
| `UNIDRIVE_MAX_FRAMES` | `24` | Max sampled frames analysed per video/job |
| `REASONING_API_URL` | Gemma default | Final local-run reasoning endpoint |
| `REASONING_MAX_TOKENS_SIMPLE` | `700` | Token cap for the simple agentic-audit attempt |
| `REASONING_MAX_TOKENS_COMPACT` | `900` | Token cap for the compact fallback agentic-audit attempt |
| `REASONING_MAX_TOKENS_FULL` | `1300` | Token cap for the full reasoning path |
| `ASR_ENABLED` / `ASR_MODEL` | `false` / `auto` | Whisper transcription |
| `OCR_ENABLED` / `OCR_MODEL` | `false` / `auto` | OCR extraction |
| `OCR_API_URL` | empty | OCR sidecar endpoint override |
| `OCR_SIDECAR_CONCURRENCY` | env-specific | Parallel OCR sidecar requests per local run |
| `OCR_IMAGE_MAX_SIDE` | env-specific | Resize bound for OCR frame uploads |
| `OCR_MIN_CAPTION_CONFIDENCE` | `0.55` | OCR prescreen threshold from Florence caption confidence |
| `DEPTH_ENABLED` / `DEPTH_MODEL` | `false` / `auto` | Depth estimation |
| `DEPTH_AUTO_PROFILE` | `fast` | Auto depth model policy: prefer fast or quality profile |
| `DEPTH_BATCH_SIZE` | `8` | Local depth outer batch size |
| `DEPTH_IMAGE_MAX_SIDE` | `768` | Resize bound before local depth inference |
| `DETECTION_ENABLED` / `DETECTION_MODEL` | `false` / `auto` | HF detection stage |
| `YOLO_ENABLED` / `YOLO_MODEL` | `true` / `yolo11l` | YOLO detection path |
| `YOLO_SSG_ENABLED` | `true` | Build YOLO semantic scene graphs from 3D frame anchors |
| `YOLO_SSG_MIN_OBSERVATIONS` | `1` | Minimum observations before a semantic node is kept |
| `YOLO_SSG_CLUSTER_RADIUS_METERS` | `12` | Merge radius for production ENU anchors |
| `YOLO_SSG_NEAR_EDGE_RADIUS_METERS` | `20` | `near` edge radius for production ENU anchors |
| `YOLO_SSG_CLUSTER_RADIUS_PCA` | `0.85` | Merge radius for local PCA/SfM anchors |
| `YOLO_SSG_NEAR_EDGE_RADIUS_PCA` | `1.5` | `near` edge radius for local PCA/SfM anchors |
| `SAM_ENABLED` / `SAM_MODEL` | `true` / `auto` | SAM mask refinement |
| `RFDETR_ENABLED` / `RFDETR_MODEL` | `true` / `base` | Gemma-directed RF-DETR tracking stage |
| `RFDETR_CONFIDENCE` | `0.35` | RF-DETR detection threshold before tracking |
| `WORLD_MODEL_ENABLED` / `WORLD_MODEL` | `false` / `nvidia/Cosmos-1.0-Autoregressive-4B` | Clip-level world-model embeddings |
| `DREAMER_ENABLED` | `true` | DreamerV3-inspired RSSM temporal surprise scoring for AL |
| `DREAMER_HIDDEN_DIM` | `256` | RSSM GRU hidden state dimension |
| `DREAMER_LATENT_DIM` | `32` | RSSM stochastic latent z_k dimension |
| `DREAMER_TRAIN_STEPS` | `20` | Online gradient steps per mission |
| `DREAMER_STORE_TEMPORAL` | `false` | Store recurrent state h_k in frame_facts_json |
| `STATE_FUSION_ENABLED` | `true` | Master switch for all probabilistic fusion layers |
| `STATE_FUSION_GPS_POS_STD_M` | `5.0` | GPS position noise (σ, metres) |
| `STATE_FUSION_BARO_ALT_STD_M` | `2.5` | Barometer altitude noise (σ, metres) |
| `STATE_FUSION_IMU_ACCEL_STD_MPS2` | `1.5` | IMU acceleration noise (σ, m/s²) |
| `STATE_FUSION_PROCESS_POS_STD_M` | `0.75` | CV process noise for position — multiplied by semantic prior scale at runtime |
| `STATE_FUSION_PROCESS_VEL_STD_MPS` | `1.5` | CV process noise for velocity — multiplied by semantic prior scale at runtime |
| `STATE_FUSION_INIT_VEL_STD_MPS` | `3.0` | Initial velocity uncertainty at filter bootstrap |
| `STATE_FUSION_CONTEXT_GAP_SEC` | `1.0` | Window for tagging which measurement kinds contributed to a posterior frame |
| `STATE_FUSION_SFM_POS_STD_M` | `2.0` | Baseline SfM position noise after Umeyama alignment (metres); actual std = this + alignment RMSE |
| `STATE_FUSION_SFM_MIN_FRAMES` | `6` | Minimum co-located SfM+GPS frames needed for Umeyama Sim(3) alignment |
| `OBJECT_FUSION_ENABLED` | `true` | Enable probabilistic per-object tracking (Mahalanobis + Hungarian + RTS) |
| `OBJECT_FUSION_OBS_NOISE` | `0.005` | Object bbox observation noise (σ, normalised image coords) |
| `OBJECT_FUSION_CONFIRM_HITS` | `3` | Frames needed before a track is promoted from tentative to confirmed |
| `OBJECT_FUSION_MAX_MISS` | `5` | Consecutive missed frames before a track is deleted |
| `MAP_FUSION_SMOOTH` | `true` | Run RTS backward smoother over the platform trajectory after the forward pass |

## Mapping and spatial queries

| Variable | Default | Purpose |
|---|---|---|
| `SFM_FPS` | `2` | Dense frame extraction for SfM |
| `PYCOLMAP_CAMERA_MODEL` | `SIMPLE_RADIAL` | Camera model for pycolmap |
| `NERFSTUDIO_API_URL` | `http://nerfstudio:8000` | Nerfstudio wrapper |
| `MAPPER_API_URL` | `http://mapper:8000` | ICP/mapper service |
| `GPS_SIDECAR_PATH` | empty | Override GPS sidecar autodetection |
| `GPS_FILTER_2D` | `false` | Use 2D Qdrant GPS filter instead of 1D+Python fallback |
| `STATIC_SERVER_URL` | `http://localhost:8080` | Static map hosting |
| `SUPERSPLAT_SERVER_URL` | `http://localhost:8090` | Viewer base URL |

## Active learning and training

| Variable | Default | Purpose |
|---|---|---|
| `AL_TAG_K` | `50` | Top-K frames marked `needs_annotation` |
| `KMEANS_BATCH_THRESHOLD` | `25000` | Switch to MiniBatchKMeans above this size |
| `CHANGE_DETECTION_THRESHOLD_CLIP` | `0.35` | CLIP change threshold |
| `CHANGE_DETECTION_THRESHOLD_DINO` | `0.25` | DINO change threshold |
| `DINO_CHECKPOINT` | empty | Override pretrained DINO weights |
| `SSL_CHECKPOINT_DIR` | `./.data/checkpoints` | Self-supervised checkpoint output |
| `SUP_CHECKPOINT_DIR` | `./.data/checkpoints/supervised` | Supervised checkpoint output |
| `SUP_AUTO_TRIGGER` | `true` | Allow CVAT webhook to enqueue finetune jobs |
| `MIN_ANNOTATED_FRAMES` | `50` | Minimum annotations before training starts |
| `MIN_NEW_ANNOTATED_SINCE_RETRAIN` | `100` | New-label delta required for retraining |
| `MODEL_VERSION_ID` | `base` | Provenance tag stored with frame payloads |

## Security and request limits

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | empty | API auth key when set |
| `ALLOWED_INDEX_PATHS` | empty | Enables path-based indexing when non-empty |
| `MAX_UPLOAD_BYTES` | `2 GiB` | Upload limit |
| `MAX_DOWNLOAD_BYTES` | `2 GiB` | URL download limit |
| `PRECHECK_URL_TIMEOUT` | `20` | URL precheck timeout in seconds |
| `MAX_REDIRECTS` | `5` | URL download redirect cap |
| `ALLOW_PRIVATE_URLS` | `false` | Private-network URL access |
| `RATE_LIMIT_PER_MIN` | `120` | Request rate limit |
| `RATE_LIMIT_BURST` | `60` | Token-bucket burst |
| `MAX_DIR_FILES` | `5000` | Directory scan file limit |
| `MAX_DIR_BYTES` | `50 GiB` | Directory scan byte limit |
| `MAX_DIR_DEPTH` | `10` | Directory recursion limit |

## 4D keyframes and tracks

These overrides apply to `workflows/analysis4d_tracks.py`. They are not `KitSettings`.
The pinned providers and the measured gate are in the
[4D model runbook](../runbooks/four-d-models.md).

| Variable | Default | Purpose |
|---|---|---|
| `ANALYSIS4D_GROUNDING_PROVIDER` | `grounding_dino` | Grounding provider. `scripted` and `unavailable` are test stand-ins. |
| `ANALYSIS4D_MASK_PROVIDER` | `kinematic` | Mask propagator. The shipped path is kinematic box motion. |
| `ANALYSIS4D_COUNT_PROVIDER` | empty | Count expert. Empty or `off` leaves counting disabled. |
| `ANALYSIS4D_SEED` | `0` | Tie-break seed for keyframe selection and track ids. |
| `ANALYSIS4D_VRAM_BUDGET_BYTES` | `8589934592` | Provider pin rejects a probe above this peak allocation (8 GiB). |
| `ANALYSIS4D_KEYFRAME_LATENCY_SEC` | `1.0` | Provider pin rejects a probe slower than this per keyframe after warmup. |
| `ANALYSIS4D_MAX_GAP_SEC` | `10` | Insert a forced keyframe when the last selection is older than this. |
| `ANALYSIS4D_CHUNK_SEC` | `4` | Fast-pass chunk length. The deep pass budgets across the whole span. |
| `ANALYSIS4D_KEYFRAME_BUDGET` | `2` | MaxInfo selections per chunk. Forced triggers may exceed it. |
| `ANALYSIS4D_MIN_SPACING_SEC` | `0.25` | Drop a non-forced keyframe closer than this to the previous one. |

Histogram, SSIM, and embedding-drift thresholds stay at the indexer defaults (`0.25`,
`0.25`, `0.15`) inside `SelectorConfig`. They are not separate environment variables.

## 4D geometry

Depth and appearance pins, and the synthetic-camera tolerances, are in the
[4D geometry runbook](../runbooks/four-d-geometry.md). The same VRAM and latency
budgets above apply to those probes.

`HF_TOKEN` in `.env` is sent on Hugging Face downloads. See
[secrets management](secrets-management.md). A gated or invisible repository with
an empty or rejected token fails
`python -m selfsuvis.pipeline.analysis4d.geometry_benchmark`. The failure detail
names `HF_TOKEN`. The token value is not written to logs or to the report.

## 4D scene graph

These overrides apply to `workflows/analysis4d_graph.py`. The predicates,
intervals, and the pinned comparison are in the
[temporal scene-graph runbook](../runbooks/four-d-scene-graph.md).

| Variable | Default | Purpose |
|---|---|---|
| `ANALYSIS4D_VLM_PROVIDER` | `unavailable` | Proposal provider. `unavailable` keeps the deterministic graph and records `provider_unavailable`. `smolvlm` loads `HuggingFaceTB/SmolVLM-256M-Instruct`. |

SceneGraphVLM training is not configured here. A remote proposal model is not the default.

## Strict verifier and Video-QA

These overrides apply to `workflows/analysis4d_verify.py`. The gate, the
graph programs, and the pinned rates are in the
[strict verifier runbook](../runbooks/four-d-strict-verifier.md).

| Variable | Default | Purpose |
|---|---|---|
| `ANALYSIS4D_REVIEW_PROVIDER` | `unavailable` | Semantic reviewer. `unavailable` leaves unsettled claims `uncertain` and keeps deterministic events. `qwen` loads the local Qwen-VL adapter. `remote` calls the Responses API. |
| `ANALYSIS4D_QWEN_VL_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Local Qwen-VL weights used when the reviewer is `qwen`. |
| `ANALYSIS4D_REVIEW_REMOTE_MODEL` | `gpt-6-astra` | Remote model id used when the reviewer is `remote`. |
| `ANALYSIS4D_REVIEW_TIMEOUT_SEC` | `30` | Reviewer timeout. A timeout fails closed. |

Remote review is not the default. It sends selected images and a strict JSON
schema. It does not send video.

## Notes

- If `ALLOWED_INDEX_PATHS` is empty, path-based indexing endpoints are disabled by design.
- The UI forwards `API_KEY` automatically when it is set in the UI container environment.
- Qdrant IDs and dedup hashes are SHA-256 based; changing embedding strategy may require a wipe and re-index.
- Production indexing writes mission-scoped semantic graph JSON to `MAPS_DIR/<mission_id>/semantic_environment_graph.json`.
- Production indexing stores optional UniDrive expert outputs in `frame_facts_json["unidrive_vla"]` and returns an aggregate `unidrive_summary` from `VideoIndexer.index_video(...)`.
- Local runs write semantic graph artifacts under `3d_map/semantic_environment_graph.{json,md}`.
- Local runs write Gemma tracking artifacts under `gemma_tracking/`, `gemma_tracking_results.json`, and `gemma_tracking_summary.md` when `GEMMA_API_URL` is set and `RFDETR_ENABLED=true`.
- Local runs write `unidrive_analysis.md` and `multi_model_comparison.md` when UniDrive is enabled.
- Production indexing writes fused platform-state summaries into `frame_facts_json["state_fusion"]` when GPS is available and `STATE_FUSION_ENABLED=true`.
- Local runs write `state_fusion.json` (GPS-only baseline) and `full_state_fusion.json` (all four layers: platform + visual-pose + object-state + map-state with RTS smoothing).
- GPS sidecar: place `<videoname>.gps.jsonl` next to the video; IMU: `<videoname>.imu.jsonl`; baro: `<videoname>.baro.jsonl`.
- `full_state_fusion.json` is written after step 15 (SfM join) so it has access to SfM poses, RF-DETR tracking results, and Gemma/RSSM semantic analysis.
- Semantic priors: Gemma's `scene_type` drives process noise scale; RSSM mean surprise drives temporal noise scale; urban canyon objects drive GPS noise inflation. See [ss-fusion probabilistic fusion](https://github.com/volod/ss-fusion/blob/v0.2.0/docs/learning_path/12_probabilistic_fusion_deep_dive.md) for the full noise table.
- On local Ollama defaults, Qwen uses a smaller sampled-frame budget and OCR only runs on lower-confidence Florence-captioned frames.
- Local agentic-flow audit runs a simple prompt first and only retries with a compact fallback when the first answer is empty or structurally incomplete.
- Depth `auto` uses the fast profile by default for local runs; set `DEPTH_AUTO_PROFILE=quality` or an explicit `DEPTH_MODEL` to opt into heavier inference.
- For a full variable list, use `src/selfsuvis/pipeline/core/config/` as the source of truth.

---
[← Helpers](../guide/helpers.md) | [Architecture →](../design/architecture.md)
