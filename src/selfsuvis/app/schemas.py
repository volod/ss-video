from typing import Literal

from pydantic import BaseModel, Field


class JobResponse(BaseModel):
    video_id: str
    job_id: str


class JobStatus(BaseModel):
    status: str
    progress: dict
    started_at: float | None
    finished_at: float | None
    error: str | None


class TextQuery(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class Match(BaseModel):
    """Single search result (frame or tile)."""

    id: int | str
    score: float
    type: Literal["tile", "frame"]
    video_id: str
    segment_id: int
    t_sec: float
    thumbnail_path: str
    frame_path: str | None
    tile_path: str | None
    bbox: dict | None


class QueryResponse(BaseModel):
    results: list[Match]


# Query parameter constraints
SearchType = Literal["both", "frame", "tile"]
VectorSpace = Literal["clip", "dino"]

# Reusable query params for validation
TOP_K_MIN = 1
TOP_K_MAX = 100


class QueryParams(BaseModel):
    """Common query parameters for search endpoints."""

    top_k: int = Field(default=20, ge=TOP_K_MIN, le=TOP_K_MAX)
    search_type: SearchType = "both"
    enable_rerank: bool = True


class ImageQueryParams(QueryParams):
    """Query parameters for image search."""

    vector_space: VectorSpace = "clip"


class ErrorResponse(BaseModel):
    """Standard error response shape."""

    error: str
    detail: str | None = None
