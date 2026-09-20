# nvblox Sidecar

## Use when

Choose `nvblox` when:

- the mapper host has a CUDA-capable GPU
- dense volumetric mapping matters
- downstream planning benefits from TSDF / ESDF products

## Strengths

- strongest dense occupancy option in the current OSS set
- good throughput on GPU
- suitable for planning-oriented map products

## Weaknesses

- requires GPU capacity
- unstable depth or pose will damage map quality quickly

## Select it

```bash
export REALTIME_OCCUPANCY_BACKEND=nvblox
export REALTIME_NVBLOX_API_URL=http://realtime-nvblox:8101
```

## Deploy it

```bash
export REALTIME_NVBLOX_IMAGE=registry.example/nvblox-sidecar:latest
docker compose -f docker/core/docker-compose.yml -f docker/realtime/docker-compose.realtime-engines.yml up -d realtime-nvblox
```

## Integration contract

The sidecar must expose:

- `POST /integrate_frame`
- `GET /map_tile/{tile_key}`
- `GET /stats`
- `GET /health`

Required inputs:

- pose
- depth

Recommended inputs:

- RGB frame metadata for semantic overlays

Expected output:

- `tile.map_type="occupancy"`
- storage reference or fetchable tile
- resolution and stats metadata

## Operational notes

- pair with `DEPTH_OUTPUT_MODE=dense` when using monocular depth
- monitor GPU headroom under bursty replay workloads
