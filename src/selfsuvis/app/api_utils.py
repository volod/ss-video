from fastapi.responses import JSONResponse

from selfsuvis.app.schemas import ErrorResponse


def error_response(message: str, status_code: int = 400, detail: str | None = None) -> JSONResponse:
    """Return a consistent error response."""
    body = {"error": message}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status_code)


ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Bad request"},
    403: {"model": ErrorResponse, "description": "Forbidden"},
    404: {"model": ErrorResponse, "description": "Not found"},
    413: {"model": ErrorResponse, "description": "Payload too large"},
    429: {"model": ErrorResponse, "description": "Too many requests"},
    503: {"model": ErrorResponse, "description": "Service unavailable"},
}
