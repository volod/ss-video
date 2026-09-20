import re

from fastapi import APIRouter, Depends, Request

from selfsuvis.app.api_utils import ERROR_RESPONSES, error_response
from selfsuvis.app.db import get_db_pool
from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.pipeline.storage import fetch_job

router = APIRouter(tags=["jobs"], dependencies=[Depends(require_api_key), Depends(rate_limit)])

# job_id is a UUID hex string (32 chars); allow up to 64 for flexibility
_JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{1,64}$", re.IGNORECASE)


def _validate_job_id(job_id: str) -> str | None:
    """Return error message if job_id is invalid, else None."""
    if not job_id or len(job_id) > 64:
        return "job_id must be 1-64 characters"
    if not _JOB_ID_PATTERN.match(job_id):
        return "job_id must contain only hex digits"
    return None


@router.get(
    "/jobs/{job_id}",
    summary="Get job status by id",
    responses={400: ERROR_RESPONSES[400], 404: ERROR_RESPONSES[404]},
)
async def job_status(job_id: str, request: Request):
    """Return status, progress, started_at, finished_at, and error for a job."""
    err = _validate_job_id(job_id)
    if err:
        return error_response(err, status_code=400)
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        job = await fetch_job(conn, job_id)
    if not job:
        return error_response("job not found", status_code=404)
    return {
        "status": job["status"],
        "type": job.get("type"),
        "progress": job["progress"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "error": job["error"],
    }
