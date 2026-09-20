# MediaMTX Streaming

SelfSuvis uses MediaMTX as the production live-stream ingress layer. MediaMTX accepts RTSP / RTMP publishers, can proxy upstream RTSP / RTMP sources, and exposes a control API that the FastAPI service uses to create and manage live stream paths.

## Production topology

In the production compose stack:

- `mediamtx` runs as a dedicated container from `bluenviron/mediamtx:1`
- `config/mediamtx/mediamtx.yml` enables a publisher-friendly default path policy plus the MediaMTX control API
- the FastAPI container talks to MediaMTX over the internal compose network through `http://mediamtx:9997`
- the host publishes MediaMTX stream ports for publishers and viewers

Default host ports:

- `8554` — RTSP
- `1935` — RTMP
- `8888` — HLS / HTTP
- `8889` — WebRTC HTTP signaling
- `8890/udp`, `8189/udp` — WebRTC ICE / UDP

The MediaMTX control API port `9997` is intentionally kept internal to the compose network. SelfSuvis manages it through the API service; operators do not need to expose it publicly.

## Configuration files and env vars

Primary files:

- `docker/core/docker-compose.yml` — production service wiring
- `config/mediamtx/mediamtx.yml` — MediaMTX runtime configuration
- `src/selfsuvis/env/prod.env` — default production env values

Relevant env vars:

- `MEDIAMTX_API_URL` — internal control API base URL, default `http://mediamtx:9997`
- `MEDIAMTX_RTSP_BASE_URL` — internal RTSP base used by the analysis runtime, default `rtsp://mediamtx:8554`
- `MEDIAMTX_PUBLIC_RTSP_BASE_URL` — externally visible RTSP base returned to API callers, default `rtsp://localhost:8554`

If the host or reverse-proxy address differs from `localhost`, override `MEDIAMTX_PUBLIC_RTSP_BASE_URL` in `.env`.

## Live stream workflow

The live-stream path is:

1. Client calls `POST /realtime/streams`
2. SelfSuvis creates or reuses a MediaMTX path through the MediaMTX control API
3. SelfSuvis creates a realtime session in PostgreSQL
4. SelfSuvis starts a background RTSP caption runtime against `rtsp://mediamtx:8554/<path>`
5. A drone, camera, or ffmpeg publisher pushes media into MediaMTX
6. `RtspCaptioner` samples frames and writes results into `scene_timeline`

There are two supported stream sources:

- Direct publisher mode: SelfSuvis creates a MediaMTX path with `source: publisher`, then a drone or ffmpeg client publishes into `rtsp://<host>:8554/<path>`
- Proxy mode: SelfSuvis creates a MediaMTX path that pulls from an upstream `source_url`

## API endpoints

MediaMTX-backed stream control endpoints live under `/realtime`.

### `POST /realtime/streams`

Creates a managed live stream path and starts realtime analysis.

Example:

```bash
curl -X POST http://localhost:8000/realtime/streams \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "drone-1",
    "mission_id": "mission-live-drone-1",
    "path_name": "live/drone-1",
    "caption_fps": 1.0
  }'
```

Optional fields:

- `source_url` — upstream RTSP / RTMP source for MediaMTX to pull
- `source_on_demand` — request on-demand pull behavior for upstream sources
- `caption_fps` — frame analysis sampling rate for the realtime caption runtime

### `GET /realtime/streams`

Returns:

- active SelfSuvis live-analysis runtimes
- current MediaMTX path inventory from the control API

### `GET /realtime/streams/{session_id}`

Returns one live-analysis runtime entry.

### `POST /realtime/streams/{session_id}/stop`

Stops the SelfSuvis realtime analysis runtime for a session. With `{"delete_path": true}`, SelfSuvis also removes the corresponding MediaMTX path through the control API.

## Publishing a test stream

After creating a stream path:

```bash
ffmpeg -re -i /path/to/drone.mp4 -c copy -f rtsp rtsp://localhost:8554/live/drone-1
```

This is the simplest way to test the production streaming path without a physical drone.

## Operational notes

- `path_name` is validated by SelfSuvis and must contain only letters, digits, `/`, `_`, and `-`
- The FastAPI app does not expose the raw MediaMTX control API; it wraps the required operations instead
- Realtime analysis writes captions and facts to PostgreSQL, but it does not automatically enqueue full mission indexing
- For full post-flight indexing, stop the live stream and use the recorded output or a dedicated ingest path

## Troubleshooting

Common issues:

- If publishing to `rtsp://localhost:8554/...` fails, verify the `mediamtx` container is running and that port `8554` is published on the host
- If `POST /realtime/streams` returns `502`, SelfSuvis cannot reach MediaMTX at `MEDIAMTX_API_URL`
- If the stream starts but no analysis appears, verify the API container can read `MEDIAMTX_RTSP_BASE_URL`
- If external clients receive the wrong publish/read URL, set `MEDIAMTX_PUBLIC_RTSP_BASE_URL` to the externally reachable address

---
[← Setup](../guide/setup.md) | [API →](api.md)
