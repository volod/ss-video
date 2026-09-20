"""Realtime session and pose endpoints for autonomous-drone integration."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from selfsuvis.app.db import get_db_pool
from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.app.services.live_streams import (
    MediaMtxClient,
    RealtimeStreamManager,
    build_rtsp_stream_url,
    validate_stream_path,
)
from selfsuvis.app.services.realtime import (
    collect_realtime_stats,
    fetch_map_tiles,
    fetch_semantic_observations,
    finalize_realtime_session,
    ingest_realtime_packets,
    integrate_realtime_frame,
    list_realtime_backends,
    publish_map_tile,
    publish_semantic_observation,
    start_realtime_session,
)
from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.realtime import pose_freshness_ms
from selfsuvis.pipeline.storage.realtime import fetch_realtime_state, stop_robot_session

logger = get_logger(__name__)

router = APIRouter(
    prefix="/realtime",
    tags=["realtime"],
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)


class SessionStartRequest(BaseModel):
    robot_id: str = Field(min_length=1, default="robot_0")
    mission_id: str | None = None
    sensors: list[str] = Field(default_factory=lambda: ["camera", "imu", "gps"])


class SessionStartResponse(BaseModel):
    session_id: str
    robot_id: str
    mission_id: str | None
    sensor_profile: dict[str, Any]
    status: str


class SensorPacket(BaseModel):
    sensor_type: str
    t_device: float
    seq: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class PacketIngestRequest(BaseModel):
    packets: list[SensorPacket]


class PacketIngestResponse(BaseModel):
    session_id: str
    accepted_packets: int
    packet_summary: dict[str, int]
    pose_updated: bool


class PoseResponse(BaseModel):
    session_id: str
    source: str
    t_sec: float
    position_enu: dict[str, float]
    orientation_quat: dict[str, float] | None
    velocity_enu: dict[str, float] | None
    tracking_status: str
    global_map_id: int | None
    freshness_ms: int | None


class RealtimeStateResponse(BaseModel):
    session_id: str
    robot_id: str
    mission_id: str | None
    status: str
    packet_counts: dict[str, int]
    latest_pose: PoseResponse | None


class SessionStopResponse(BaseModel):
    session_id: str
    status: str
    mission_id: str | None = None
    job_id: str | None = None
    enqueued_index_job: bool = False


class LiveStreamStartRequest(BaseModel):
    robot_id: str = Field(min_length=1, default="robot_0")
    mission_id: str | None = None
    sensors: list[str] = Field(default_factory=lambda: ["camera"])
    path_name: str = Field(min_length=1)
    source_url: str | None = None
    source_on_demand: bool = False
    caption_fps: float | None = Field(default=None, gt=0.0, le=4.0)


class LiveStreamStatus(BaseModel):
    session_id: str
    mission_id: str
    robot_id: str
    path_name: str
    rtsp_url: str
    caption_fps: float
    started_at: str
    status: str
    error: str | None = None


class LivePathInfo(BaseModel):
    name: str | None = None
    ready: bool | None = None
    source: dict[str, Any] | None = None
    tracks: list[dict[str, Any]] | None = None
    bytes_received: int | None = None
    bytes_sent: int | None = None


class LiveStreamStartResponse(BaseModel):
    session_id: str
    mission_id: str
    robot_id: str
    path_name: str
    publish_url: str
    read_url: str
    mediamtx_path_created: bool
    analysis: LiveStreamStatus


class LiveStreamsResponse(BaseModel):
    streams: list[LiveStreamStatus]
    mediamtx_paths: list[LivePathInfo]


class LiveStreamStopRequest(BaseModel):
    delete_path: bool = False


class LiveStreamStopResponse(BaseModel):
    session_id: str
    path_name: str
    status: str
    deleted_path: bool = False


class MapTileIn(BaseModel):
    tile_key: str
    map_type: str = "occupancy"
    storage_path: str
    resolution_m: float = 0.2
    bounds: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    global_map_id: int | None = None


class MapTileOut(BaseModel):
    tile_key: str
    map_type: str
    storage_path: str
    resolution_m: float
    bounds: dict[str, Any]
    stats: dict[str, Any]
    global_map_id: int | None


class MapTilesResponse(BaseModel):
    session_id: str
    tiles: list[MapTileOut]


class SemanticObservationIn(BaseModel):
    frame_id: str | None = None
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    position_enu: dict[str, Any] | None = None
    bbox: dict[str, Any] | None = None
    mask_ref: str | None = None
    track_id: str | None = None
    facts: dict[str, Any] = Field(default_factory=dict)


class SemanticObservationOut(BaseModel):
    frame_id: str | None
    class_name: str
    confidence: float
    position_enu: dict[str, Any] | None
    bbox: dict[str, Any] | None
    mask_ref: str | None
    track_id: str | None
    facts: dict[str, Any]


class SemanticObservationsResponse(BaseModel):
    session_id: str
    observations: list[SemanticObservationOut]


class SessionFinalizeRequest(BaseModel):
    recording_path: str | None = None
    enqueue_index_job: bool = False


class FrameIntegrationRequest(BaseModel):
    frame_id: str | None = None
    t_sec: float
    image_path: str = Field(min_length=1)
    packets: list[SensorPacket] = Field(default_factory=list)
    semantic_observations: list[SemanticObservationIn] = Field(default_factory=list)
    pose: dict[str, Any] | None = None
    depth_path: str | None = None
    map_type: str = "occupancy"
    tile_key: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class FrameIntegrationResponse(BaseModel):
    session_id: str
    pose_updated: bool
    tile: MapTileOut
    semantic_count: int


class RealtimeStatsResponse(BaseModel):
    pose_backend: str
    occupancy_backend: str
    pose: dict[str, Any] | None = None
    occupancy: dict[str, Any] | None = None


class RealtimeBackendsResponse(BaseModel):
    selected: dict[str, str]
    pose_backends: dict[str, dict[str, Any]]
    occupancy_backends: dict[str, dict[str, Any]]


def _pose_response(session_id: str, row: dict[str, Any]) -> PoseResponse:
    return PoseResponse(
        session_id=session_id,
        source=row["source"],
        t_sec=float(row["t_sec"]),
        position_enu=dict(row["position_enu_json"]),
        orientation_quat=dict(row["orientation_quat_json"])
        if row.get("orientation_quat_json")
        else None,
        velocity_enu=dict(row["velocity_enu_json"]) if row.get("velocity_enu_json") else None,
        tracking_status=row["tracking_status"],
        global_map_id=row.get("global_map_id"),
        freshness_ms=pose_freshness_ms(row.get("created_at")),
    )


def _get_mediamtx_client(request: Request) -> MediaMtxClient:
    client = getattr(request.app.state, "mediamtx_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="MediaMTX client is not configured")
    return client


def _get_stream_manager(request: Request) -> RealtimeStreamManager:
    manager = getattr(request.app.state, "realtime_stream_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="realtime stream manager is not configured")
    return manager


def _live_stream_status(data: dict[str, Any]) -> LiveStreamStatus:
    return LiveStreamStatus(**data)


def _live_path_info(item: dict[str, Any]) -> LivePathInfo:
    return LivePathInfo(
        name=item.get("name"),
        ready=item.get("ready"),
        source=item.get("source") if isinstance(item.get("source"), dict) else None,
        tracks=item.get("tracks") if isinstance(item.get("tracks"), list) else None,
        bytes_received=item.get("bytesReceived"),
        bytes_sent=item.get("bytesSent"),
    )


@router.post("/session/start", response_model=SessionStartResponse)
async def start_session(body: SessionStartRequest, request: Request) -> SessionStartResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        started = await start_realtime_session(
            conn,
            robot_id=body.robot_id,
            mission_id=body.mission_id,
            sensors=body.sensors,
        )
    return SessionStartResponse(**started)


@router.post("/streams", response_model=LiveStreamStartResponse)
async def start_live_stream(
    body: LiveStreamStartRequest, request: Request
) -> LiveStreamStartResponse:
    db_pool = get_db_pool(request)
    manager = _get_stream_manager(request)
    mediamtx = _get_mediamtx_client(request)
    try:
        path_name = validate_stream_path(body.path_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    mission_id = body.mission_id or f"live-{path_name.replace('/', '-')}"
    try:
        mediamtx_created = await mediamtx.ensure_path(
            path_name,
            source_url=body.source_url,
            source_on_demand=body.source_on_demand,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    async with db_pool.acquire() as conn:
        started = await start_realtime_session(
            conn,
            robot_id=body.robot_id,
            mission_id=mission_id,
            sensors=body.sensors,
        )
    try:
        analysis = await manager.start(
            session_id=started["session_id"],
            mission_id=mission_id,
            robot_id=body.robot_id,
            path_name=path_name,
            caption_fps=body.caption_fps,
        )
    except RuntimeError as exc:
        async with db_pool.acquire() as conn:
            await stop_robot_session(conn, started["session_id"])
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    public_rtsp_url = build_rtsp_stream_url(path_name, public=True)
    return LiveStreamStartResponse(
        session_id=started["session_id"],
        mission_id=mission_id,
        robot_id=body.robot_id,
        path_name=path_name,
        publish_url=public_rtsp_url,
        read_url=public_rtsp_url,
        mediamtx_path_created=mediamtx_created,
        analysis=_live_stream_status(analysis),
    )


@router.get("/streams", response_model=LiveStreamsResponse)
async def list_live_streams(request: Request) -> LiveStreamsResponse:
    manager = _get_stream_manager(request)
    mediamtx = _get_mediamtx_client(request)
    try:
        mediamtx_paths = await mediamtx.list_paths()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    streams = await manager.list()
    return LiveStreamsResponse(
        streams=[_live_stream_status(item) for item in streams],
        mediamtx_paths=[_live_path_info(item) for item in mediamtx_paths],
    )


@router.get("/streams/{session_id}", response_model=LiveStreamStatus)
async def get_live_stream(session_id: str, request: Request) -> LiveStreamStatus:
    manager = _get_stream_manager(request)
    try:
        return _live_stream_status(await manager.get(session_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/streams/{session_id}/stop", response_model=LiveStreamStopResponse)
async def stop_live_stream(
    session_id: str,
    body: LiveStreamStopRequest,
    request: Request,
) -> LiveStreamStopResponse:
    db_pool = get_db_pool(request)
    manager = _get_stream_manager(request)
    mediamtx = _get_mediamtx_client(request)
    try:
        runtime = await manager.get(session_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        stopped = await manager.stop(session_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async with db_pool.acquire() as conn:
        state = await fetch_realtime_state(conn, session_id)
        if state is not None:
            await stop_robot_session(conn, session_id)

    deleted = False
    if body.delete_path:
        try:
            deleted = await mediamtx.delete_path(runtime["path_name"])
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return LiveStreamStopResponse(
        session_id=session_id,
        path_name=runtime["path_name"],
        status=stopped["status"],
        deleted_path=deleted,
    )


@router.post("/session/{session_id}/packet", response_model=PacketIngestResponse)
async def ingest_packets(
    session_id: str,
    body: PacketIngestRequest,
    request: Request,
) -> PacketIngestResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            result = await ingest_realtime_packets(
                conn,
                session_id=session_id,
                packets=[packet.model_dump() for packet in body.packets],
            )
        except ValueError as exc:
            detail = str(exc)
            status = 413 if detail.startswith("too many packets") else 422
            raise HTTPException(status_code=status, detail=detail) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PacketIngestResponse(**result)


@router.get("/session/{session_id}/state", response_model=RealtimeStateResponse)
async def get_state(session_id: str, request: Request) -> RealtimeStateResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")
    latest_pose = state["latest_pose"]
    return RealtimeStateResponse(
        session_id=session_id,
        robot_id=state["session"]["robot_id"],
        mission_id=state["session"].get("mission_id"),
        status=state["session"]["status"],
        packet_counts=state["packet_counts"],
        latest_pose=_pose_response(session_id, latest_pose) if latest_pose else None,
    )


@router.get("/session/{session_id}/pose/latest", response_model=PoseResponse)
async def get_latest_pose(session_id: str, request: Request) -> PoseResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")
    latest_pose = state["latest_pose"]
    if latest_pose is None:
        raise HTTPException(status_code=404, detail="pose not available")
    return _pose_response(session_id, latest_pose)


@router.post("/session/{session_id}/pose/estimate", response_model=PoseResponse)
async def estimate_pose_from_packets(
    session_id: str,
    body: PacketIngestRequest,
    request: Request,
) -> PoseResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        result = await ingest_realtime_packets(
            conn,
            session_id=session_id,
            packets=[packet.model_dump() for packet in body.packets],
        )
        if not result["pose_updated"]:
            raise HTTPException(status_code=404, detail="pose not available")
        state = await fetch_realtime_state(conn, session_id)
    latest_pose = state["latest_pose"] if state else None
    if latest_pose is None:
        raise HTTPException(status_code=404, detail="pose not available")
    return _pose_response(session_id, latest_pose)


@router.post("/session/{session_id}/map/tile", response_model=MapTilesResponse)
async def publish_tile(session_id: str, body: MapTileIn, request: Request) -> MapTilesResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            await publish_map_tile(conn, session_id=session_id, tile=body.model_dump())
            rows = await fetch_map_tiles(
                conn, session_id=session_id, map_type=body.map_type, limit=50
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MapTilesResponse(
        session_id=session_id,
        tiles=[
            MapTileOut(
                tile_key=row["tile_key"],
                map_type=row["map_type"],
                storage_path=row["storage_path"],
                resolution_m=float(row["resolution_m"]),
                bounds=dict(row.get("bounds_json") or {}),
                stats=dict(row.get("stats_json") or {}),
                global_map_id=row.get("global_map_id"),
            )
            for row in rows
        ],
    )


@router.post("/session/{session_id}/frame/integrate", response_model=FrameIntegrationResponse)
async def integrate_frame(
    session_id: str,
    body: FrameIntegrationRequest,
    request: Request,
) -> FrameIntegrationResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            result = await integrate_realtime_frame(
                conn,
                session_id=session_id,
                frame_id=body.frame_id,
                t_sec=body.t_sec,
                image_path=body.image_path,
                packets=[packet.model_dump() for packet in body.packets],
                semantic_observations=[obs.model_dump() for obs in body.semantic_observations],
                pose=body.pose,
                depth_path=body.depth_path,
                map_type=body.map_type,
                tile_key=body.tile_key,
                stats=body.stats,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    tile = result["tile"]
    return FrameIntegrationResponse(
        session_id=session_id,
        pose_updated=result["pose_updated"],
        tile=MapTileOut(
            tile_key=tile["tile_key"],
            map_type=tile["map_type"],
            storage_path=tile["storage_path"],
            resolution_m=float(tile["resolution_m"]),
            bounds=dict(tile.get("bounds") or {}),
            stats=dict(tile.get("stats") or {}),
            global_map_id=tile.get("global_map_id"),
        ),
        semantic_count=int(result["semantic_count"]),
    )


@router.get("/session/{session_id}/map/latest", response_model=MapTilesResponse)
async def get_latest_map(
    session_id: str,
    request: Request,
    map_type: str | None = None,
    limit: int = 20,
) -> MapTilesResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            rows = await fetch_map_tiles(
                conn, session_id=session_id, map_type=map_type, limit=limit
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MapTilesResponse(
        session_id=session_id,
        tiles=[
            MapTileOut(
                tile_key=row["tile_key"],
                map_type=row["map_type"],
                storage_path=row["storage_path"],
                resolution_m=float(row["resolution_m"]),
                bounds=dict(row.get("bounds_json") or {}),
                stats=dict(row.get("stats_json") or {}),
                global_map_id=row.get("global_map_id"),
            )
            for row in rows
        ],
    )


@router.post("/session/{session_id}/semantic", response_model=SemanticObservationsResponse)
async def publish_semantic(
    session_id: str,
    body: SemanticObservationIn,
    request: Request,
) -> SemanticObservationsResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            await publish_semantic_observation(
                conn, session_id=session_id, observation=body.model_dump()
            )
            rows = await fetch_semantic_observations(
                conn,
                session_id=session_id,
                class_name=body.class_name,
                limit=50,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SemanticObservationsResponse(
        session_id=session_id,
        observations=[
            SemanticObservationOut(
                frame_id=row.get("frame_id"),
                class_name=row["class_name"],
                confidence=float(row["confidence"]),
                position_enu=dict(row["position_enu_json"])
                if row.get("position_enu_json")
                else None,
                bbox=dict(row["bbox_json"]) if row.get("bbox_json") else None,
                mask_ref=row.get("mask_ref"),
                track_id=row.get("track_id"),
                facts=dict(row.get("facts_json") or {}),
            )
            for row in rows
        ],
    )


@router.get("/session/{session_id}/semantic-nearby", response_model=SemanticObservationsResponse)
async def get_semantic_nearby(
    session_id: str,
    request: Request,
    class_name: str | None = None,
    limit: int = 20,
) -> SemanticObservationsResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            rows = await fetch_semantic_observations(
                conn,
                session_id=session_id,
                class_name=class_name,
                limit=limit,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SemanticObservationsResponse(
        session_id=session_id,
        observations=[
            SemanticObservationOut(
                frame_id=row.get("frame_id"),
                class_name=row["class_name"],
                confidence=float(row["confidence"]),
                position_enu=dict(row["position_enu_json"])
                if row.get("position_enu_json")
                else None,
                bbox=dict(row["bbox_json"]) if row.get("bbox_json") else None,
                mask_ref=row.get("mask_ref"),
                track_id=row.get("track_id"),
                facts=dict(row.get("facts_json") or {}),
            )
            for row in rows
        ],
    )


@router.post("/session/{session_id}/stop", response_model=SessionStopResponse)
async def stop_session(session_id: str, request: Request) -> SessionStopResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        state = await fetch_realtime_state(conn, session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="session not found")
        await stop_robot_session(conn, session_id)
    logger.info("Realtime session stopped: %s", session_id)
    return SessionStopResponse(session_id=session_id, status="stopped")


@router.post("/session/{session_id}/finalize", response_model=SessionStopResponse)
async def finalize_session(
    session_id: str,
    body: SessionFinalizeRequest,
    request: Request,
) -> SessionStopResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            result = await finalize_realtime_session(
                conn,
                session_id=session_id,
                recording_path=body.recording_path,
                enqueue_index_job=body.enqueue_index_job,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    logger.info("Realtime session finalized: %s mission=%s", session_id, result["mission_id"])
    return SessionStopResponse(**result)


@router.get("/stats", response_model=RealtimeStatsResponse)
async def get_realtime_stats() -> RealtimeStatsResponse:
    return RealtimeStatsResponse(**(await collect_realtime_stats()))


@router.get("/backends", response_model=RealtimeBackendsResponse)
async def get_realtime_backends() -> RealtimeBackendsResponse:
    return RealtimeBackendsResponse(**list_realtime_backends())
