# voxblox Sidecar

## Use when

Choose `voxblox` when:

- the mapper host is CPU-only
- dense mapping is still required
- lower throughput is acceptable

## Strengths

- no GPU dependency
- simpler fit for lightweight edge hosts

## Weaknesses

- lower throughput than `nvblox`
- resolution must be tuned carefully to stay real-time

## Select it

```bash
export REALTIME_OCCUPANCY_BACKEND=voxblox
export REALTIME_VOXBLOX_API_URL=http://realtime-voxblox:8101
```

## Deploy it

```bash
export REALTIME_VOXBLOX_IMAGE=registry.example/voxblox-sidecar:latest
docker compose -f docker/core/docker-compose.yml -f docker/realtime/docker-compose.realtime-engines.yml up -d realtime-voxblox
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

Expected output:

- occupancy tile metadata
- durable storage path or retrievable tile handle

## Operational notes

- reduce tile resolution before blaming CPU saturation
- use it as the default occupancy sidecar on non-GPU field laptops
