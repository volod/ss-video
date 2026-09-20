import os
import pathlib
import uuid
from collections.abc import Generator

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from selfsuvis.app.api_utils import ERROR_RESPONSES, error_response
from selfsuvis.app.db import get_db_pool
from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.app.services.upload_utils import hash_upload_limited, write_upload_to_path
from selfsuvis.app.state import logger
from selfsuvis.pipeline.core import ensure_dir, file_sha256, resolve_allowed_path, settings
from selfsuvis.pipeline.media import safe_request, validate_url
from selfsuvis.pipeline.storage import create_job
from selfsuvis.pipeline.storage.processed import aget_by_hash, aget_by_size, aget_by_url
from selfsuvis.pipeline.storage.processed import get_by_hash as _sync_get_by_hash

router = APIRouter(tags=["index"], dependencies=[Depends(require_api_key), Depends(rate_limit)])
get_by_hash = _sync_get_by_hash


async def _lookup_by_hash(file_hash: str):
    """Lookup hash while preserving the router-level sync mock hook used in tests."""
    if get_by_hash is not _sync_get_by_hash:
        result = get_by_hash(file_hash)
        if hasattr(result, "__await__"):
            result = await result
        return result
    return await aget_by_hash(file_hash)


async def _enqueue_job(payload: dict, request: Request) -> str:
    job_id = uuid.uuid4().hex
    pool = get_db_pool(request)
    acquired = pool.acquire()
    # Real asyncpg.Pool.acquire() is both awaitable and an async context manager.
    # Prefer the context manager so we do not consume the acquire as a coroutine.
    if not hasattr(acquired, "__aenter__"):
        acquired = await acquired
    async with acquired as conn:
        maybe_created = create_job(conn, job_id, payload, job_type="index")
        if hasattr(maybe_created, "__await__"):
            await maybe_created
    return job_id


def _check_dir_limits(root: str, base: str, file_count: int, total_bytes: int) -> str:
    rel = os.path.relpath(root, base)
    depth = 0 if rel == "." else rel.count(os.sep) + 1
    if settings.MAX_DIR_DEPTH and depth > settings.MAX_DIR_DEPTH:
        return "directory depth limit exceeded"
    if settings.MAX_DIR_FILES and file_count > settings.MAX_DIR_FILES:
        return "directory file count limit exceeded"
    if settings.MAX_DIR_BYTES and total_bytes > settings.MAX_DIR_BYTES:
        return "directory byte limit exceeded"
    return ""


class _DirLimitExceeded(Exception):
    """Raised when directory scan exceeds configured limits."""

    def __init__(self, msg: str) -> None:
        self.msg = msg
        super().__init__(msg)


def _iter_video_paths_in_allowed_dir(
    resolved: str,
) -> Generator[tuple[str, bool], None, None]:
    """Yield (video_path, stat_failed) for each video file under `resolved`.
    stat_failed is True when os.path.getsize failed. Raises _DirLimitExceeded on limit."""
    file_count = 0
    total_bytes = 0
    for root, _, files in os.walk(resolved):
        for name in files:
            if os.path.splitext(name)[1].lower() not in settings.VIDEO_EXTS:
                continue
            video_path = os.path.join(root, name)
            try:
                total_bytes += os.path.getsize(video_path)
            except OSError:
                yield video_path, True
                continue
            file_count += 1
            limit_error = _check_dir_limits(root, resolved, file_count, total_bytes)
            if limit_error:
                raise _DirLimitExceeded(limit_error)
            yield video_path, False


@router.post(
    "/index/video",
    summary="Index a video from upload or allowed path",
    responses={400: ERROR_RESPONSES[400], 403: ERROR_RESPONSES[403], 413: ERROR_RESPONSES[413]},
)
async def index_video(
    request: Request,
    file: UploadFile | None = File(default=None),
    path: str | None = Form(default=None),
    enable_tiles: bool = Form(default=True),
):
    """Accept video file upload or a path (within ALLOWED_INDEX_PATHS). Returns video_id and job_id."""
    if file is None and path is None:
        return error_response("file or path required")
    ensure_dir(settings.VIDEOS_DIR)
    video_id = uuid.uuid4().hex
    if file is not None:
        upload_ext = pathlib.Path(file.filename or "").suffix.lower()
        if upload_ext not in settings.VIDEO_EXTS:
            return error_response("unsupported video extension")
        video_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}{upload_ext}")
        try:
            await write_upload_to_path(file, video_path, settings.MAX_UPLOAD_BYTES)
        except ValueError as exc:
            logger.warning("Upload overflow path=%s err=%s", video_path, exc)
            return error_response(str(exc), status_code=413)
    else:
        if path is None:
            return error_response("path required")
        resolved = resolve_allowed_path(path, must_be_file=True)
        if resolved is None:
            return error_response("path not allowed or not a file", status_code=403)
        video_path = resolved

    job_id = await _enqueue_job(
        {"video_id": video_id, "video_path": video_path, "enable_tiles": enable_tiles},
        request,
    )
    logger.info("Enqueued video_id=%s job_id=%s", video_id, job_id)
    return {"video_id": video_id, "job_id": job_id}


@router.post("/index/url", summary="Index a video from URL", responses={400: ERROR_RESPONSES[400]})
async def index_url(
    request: Request,
    url: str | None = Form(default=None),
    stream_url: str | None = Form(default=None),
    enable_tiles: bool = Form(default=True),
):
    url = url or stream_url
    if not url:
        return error_response("url required", status_code=422)
    try:
        validate_url(url)
    except ValueError as exc:
        return error_response(str(exc))
    video_id = uuid.uuid4().hex
    job_id = await _enqueue_job(
        {"video_id": video_id, "video_url": url, "enable_tiles": enable_tiles},
        request,
    )
    logger.info("Enqueued url video_id=%s job_id=%s", video_id, job_id)
    return {"video_id": video_id, "job_id": job_id}


@router.post(
    "/index/dir",
    summary="Index all videos in an allowed directory",
    responses={403: ERROR_RESPONSES[403]},
)
async def index_dir(
    request: Request,
    path: str | None = Form(default=None),
    dir_path: str | None = Form(default=None),
    enable_tiles: bool = Form(default=True),
):
    path = path or dir_path
    if not path:
        return error_response("path required", status_code=422)
    resolved = resolve_allowed_path(path, must_be_dir=True)
    if resolved is None:
        return error_response("path not allowed or not a directory", status_code=403)
    if not os.path.isdir(resolved):
        return error_response("path is not a directory")
    jobs = []
    try:
        for video_path, stat_failed in _iter_video_paths_in_allowed_dir(resolved):
            if stat_failed:
                continue
            video_id = uuid.uuid4().hex
            job_id = await _enqueue_job(
                {"video_id": video_id, "video_path": video_path, "enable_tiles": enable_tiles},
                request,
            )
            jobs.append({"video_id": video_id, "job_id": job_id})
    except _DirLimitExceeded as e:
        return error_response(e.msg)
    logger.info("Enqueued directory jobs=%s path=%s", len(jobs), resolved)
    return {"jobs": jobs}


async def _precheck_file(file: UploadFile):
    """Precheck by file upload hash."""
    try:
        file_hash, _ = await hash_upload_limited(file, settings.MAX_UPLOAD_BYTES)
    except ValueError as exc:
        return error_response(str(exc), status_code=413)
    existing = await _lookup_by_hash(file_hash)
    if existing:
        return {"status": "duplicate", "reason": "hash", "existing": existing}
    return {"status": "new", "reason": "hash", "hash": file_hash}


async def _precheck_path(path: str):
    """Precheck by allowed path and file hash."""
    resolved = resolve_allowed_path(path, must_be_file=True)
    if resolved is None:
        return error_response("path not allowed or not a file", status_code=403)
    if not os.path.exists(resolved) or not os.path.isfile(resolved):
        return error_response("path not found")
    file_hash = file_sha256(resolved)
    existing = await _lookup_by_hash(file_hash)
    if existing:
        return {"status": "duplicate", "reason": "hash", "existing": existing}
    return {"status": "new", "reason": "hash", "hash": file_hash}


async def _precheck_url(url: str):
    """Precheck by URL (existing by URL or size match)."""
    try:
        validate_url(url)
    except ValueError as exc:
        return error_response(str(exc))
    existing = await aget_by_url(url)
    if existing:
        return {"status": "duplicate", "reason": "url", "existing": existing}
    size = None
    try:
        with safe_request("HEAD", url, timeout=settings.PRECHECK_URL_TIMEOUT) as head:
            if head.ok and head.headers.get("Content-Length"):
                size = int(head.headers["Content-Length"])
    except Exception:
        size = None
    if size is not None:
        by_size = await aget_by_size(size)
        if by_size:
            return {
                "status": "maybe",
                "reason": "size_match",
                "existing": by_size,
                "size_bytes": size,
            }
    return {"status": "unknown", "reason": "url_unmatched", "size_bytes": size}


@router.post(
    "/index/rtsp",
    summary="Index a live RTSP/RTMP stream",
    responses={400: ERROR_RESPONSES[400]},
)
async def index_rtsp(
    request: Request,
    stream_url: str = Form(...),
    mission_id: str | None = Form(default=None),
    duration_sec: int | None = Form(default=None),
    enable_tiles: bool = Form(default=True),
):
    """Record a live RTSP or RTMP stream and queue it for indexing.

    The worker records the stream to a local MP4 using ffmpeg, then indexes
    it through the standard pipeline.

    - stream_url:   rtsp:// or rtmp:// URL (credentials in URL not allowed)
    - mission_id:   Optional mission label; defaults to a generated video_id
    - duration_sec: Record at most this many seconds (capped at RTSP_MAX_DURATION_SEC)
    - enable_tiles: Whether to extract and index tiles (default true)
    """
    from selfsuvis.pipeline.media.rtsp_ingest import validate_rtsp_url

    try:
        validate_rtsp_url(stream_url)
    except ValueError as exc:
        return error_response(str(exc))

    video_id = uuid.uuid4().hex
    effective_mission_id = mission_id or video_id
    job_id = await _enqueue_job(
        {
            "video_id": video_id,
            "mission_id": effective_mission_id,
            "video_url": stream_url,
            "ingest_mode": "rtsp",
            "duration_sec": duration_sec,
            "enable_tiles": enable_tiles,
        },
        request,
    )
    logger.info(
        "Enqueued RTSP stream video_id=%s mission_id=%s job_id=%s url=%s",
        video_id,
        effective_mission_id,
        job_id,
        stream_url,
    )
    return {"video_id": video_id, "mission_id": effective_mission_id, "job_id": job_id}


@router.post("/index/precheck")
async def precheck(
    file: UploadFile | None = File(default=None),
    path: str | None = Form(default=None),
    file_path: str | None = Form(default=None),
    url: str | None = Form(default=None),
):
    path = path or file_path
    if not file and not path and not url:
        return error_response("file, path, or url required")
    if file is not None:
        return await _precheck_file(file)
    if path is not None:
        return await _precheck_path(path)
    return await _precheck_url(url)


@router.post("/index/precheck_dir")
async def precheck_dir(
    request: Request,
    path: str | None = Form(default=None),
    dir_path: str | None = Form(default=None),
    enqueue: bool = Form(default=False),
    enable_tiles: bool = Form(default=True),
):
    path = path or dir_path
    if not path:
        return error_response("path required", status_code=422)
    resolved = resolve_allowed_path(path, must_be_dir=True)
    if resolved is None:
        return error_response("path not allowed or not a directory", status_code=403)
    if not os.path.isdir(resolved):
        return error_response("path is not a directory")
    results = []
    jobs = []
    try:
        for video_path, stat_failed in _iter_video_paths_in_allowed_dir(resolved):
            if stat_failed:
                results.append(
                    {
                        "filename": os.path.basename(video_path),
                        "status": "error",
                        "reason": "stat_failed",
                    }
                )
                continue
            try:
                file_hash = file_sha256(video_path)
            except OSError as e:
                logger.debug("Hash failed for path=%s err=%s", video_path, e)
                results.append(
                    {
                        "filename": os.path.basename(video_path),
                        "status": "error",
                        "reason": "hash_failed",
                    }
                )
                continue
            existing = await _lookup_by_hash(file_hash)
            if existing:
                results.append(
                    {
                        "filename": os.path.basename(video_path),
                        "status": "duplicate",
                        "reason": "hash",
                        "existing": existing,
                    }
                )
            else:
                entry = {
                    "filename": os.path.basename(video_path),
                    "status": "new",
                    "reason": "hash",
                    "hash": file_hash,
                }
                if enqueue:
                    video_id = uuid.uuid4().hex
                    job_id = await _enqueue_job(
                        {
                            "video_id": video_id,
                            "video_path": video_path,
                            "enable_tiles": enable_tiles,
                        },
                        request,
                    )
                    entry["enqueued"] = True
                    entry["video_id"] = video_id
                    entry["job_id"] = job_id
                    jobs.append({"video_id": video_id, "job_id": job_id})
                results.append(entry)
    except _DirLimitExceeded as e:
        return error_response(e.msg)
    logger.info("Precheck dir path=%s results=%s enqueue=%s", resolved, len(results), enqueue)
    return {"results": results, "jobs": jobs if enqueue else []}
