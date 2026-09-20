# Performance

## Primary throughput controls

- Lower `SAMPLE_FPS_MAX` and `SAMPLE_FPS_BASE` to decode fewer frames
- Increase `HIST_THRESH` and `EMBED_DRIFT_THRESH` to keep fewer keyframes
- Lower `MAX_TILES_PER_SEGMENT` to reduce tile extraction and embedding load
- Increase `STRIDE` to reduce tile overlap

## Primary memory controls

- Keep `USE_FP16=true` on CUDA
- Lower `FLORENCE_BATCH_SIZE` if captioning hits OOM
- Disable optional stages you do not need: `ASR_ENABLED`, `OCR_ENABLED`, `DEPTH_ENABLED`, `DETECTION_ENABLED`, `WORLD_MODEL_ENABLED`
- Use smaller sidecar models for Qwen, Gemma, or reasoning when sharing a single GPU

## Retrieval-quality controls

- Lower `EMBED_DRIFT_THRESH` to keep more frames
- Lower `DEDUP_COS_SIM_THRESH` only if you need more near-duplicate tiles retained
- Switch `MODEL_NAME` to `dinov3` when you want DINO vectors and active-learning scoring

## Example profiles

### Lightweight indexing

```bash
SAMPLE_FPS_BASE=1
SAMPLE_FPS_MAX=2
MAX_GAP_SEC=15
MAX_TILES_PER_SEGMENT=80
STRIDE=320
```

### Balanced default

```bash
SAMPLE_FPS_BASE=2
SAMPLE_FPS_MAX=5
HIST_THRESH=0.25
EMBED_DRIFT_THRESH=0.15
MAX_TILES_PER_SEGMENT=200
```

### Higher recall

```bash
SAMPLE_FPS_BASE=3
SAMPLE_FPS_MAX=6
MAX_GAP_SEC=6
EMBED_DRIFT_THRESH=0.10
MAX_TILES_PER_SEGMENT=300
STRIDE=192
```

---
[← Data layout](data_layout.md) | [Troubleshooting →](../operations/troubleshooting.md)
