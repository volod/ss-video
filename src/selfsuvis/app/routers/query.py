import io

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from selfsuvis.app.api_utils import ERROR_RESPONSES, error_response
from selfsuvis.app.db import get_db_pool_optional
from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.app.schemas import QueryResponse, TextQuery
from selfsuvis.app.services.search import search_vectors
from selfsuvis.app.services.upload_utils import read_upload_limited
from selfsuvis.app.state import clip_model, dino_model
from selfsuvis.pipeline.core import settings

router = APIRouter(tags=["query"], dependencies=[Depends(require_api_key), Depends(rate_limit)])

_VALID_SEARCH_TYPES = frozenset({"both", "frame", "tile"})
_VALID_VECTOR_SPACES = frozenset({"clip", "dino"})


def _validate_query_params(search_type: str, vector_space: str | None = None) -> str | None:
    """Return error message if params invalid, else None."""
    if search_type not in _VALID_SEARCH_TYPES:
        return "search_type must be both, frame, or tile"
    if vector_space is not None and vector_space not in _VALID_VECTOR_SPACES:
        return "vector_space must be clip or dino"
    return None


@router.post(
    "/query/image",
    response_model=QueryResponse,
    summary="Search by image",
    responses={400: ERROR_RESPONSES[400], 413: ERROR_RESPONSES[413]},
)
async def query_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Form(default=20, ge=1, le=100),
    search_type: str = Form(default="both"),
    vector_space: str = Form(default="clip"),
    enable_rerank: bool = Form(default=True),
):
    err = _validate_query_params(search_type, vector_space)
    if err:
        return error_response(err)
    try:
        content = await read_upload_limited(file, settings.MAX_UPLOAD_BYTES)
    except ValueError as exc:
        return error_response(str(exc), status_code=413)

    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return error_response("invalid or unsupported image")

    clip_vec = clip_model.encode_images([image], batch_size=1)[0]
    if vector_space == "dino" and dino_model:
        query_vec = dino_model.encode_images([image], batch_size=1)[0]
        vs = "dino"
    else:
        query_vec = clip_vec
        vs = "clip"

    db_pool = get_db_pool_optional(request)
    results = await search_vectors(
        vector_space=vs,
        query_vec=query_vec,
        search_type=search_type,
        top_k=top_k,
        enable_rerank=enable_rerank,
        image_query=image,
        db_pool=db_pool,
    )
    return QueryResponse(results=results)


@router.post(
    "/query/text",
    response_model=QueryResponse,
    summary="Search by text",
    responses={400: ERROR_RESPONSES[400]},
)
async def query_text(
    request: Request,
    payload: TextQuery,
    top_k: int = Query(default=20, ge=1, le=100),
    search_type: str = Query(default="both"),
    enable_rerank: bool = Query(default=True),
):
    err = _validate_query_params(search_type)
    if err:
        return error_response(err)
    text = payload.text
    clip_vec = clip_model.encode_texts([text], batch_size=1)[0]
    db_pool = get_db_pool_optional(request)
    results = await search_vectors(
        vector_space="clip",
        query_vec=clip_vec,
        search_type=search_type,
        top_k=top_k,
        enable_rerank=enable_rerank,
        image_query=None,
        db_pool=db_pool,
    )
    return QueryResponse(results=results)
