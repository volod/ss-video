from fastapi import Header, HTTPException, Request

from selfsuvis.pipeline.core import settings
from ss_kit.security import MAX_RATE_LIMIT_CLIENTS, RateLimiter
from ss_kit.web import check_api_key, client_key

_rate_limiter = RateLimiter(
    lambda: (float(settings.RATE_LIMIT_PER_MIN) / 60.0, float(settings.RATE_LIMIT_BURST))
)

# Backward-compatible exports used by tests and older call sites.
_MAX_LIMITERS = MAX_RATE_LIMIT_CLIENTS
_limiters = _rate_limiter.buckets
_RateLimiter = RateLimiter


def _get_client_key(request: Request) -> str:
    """Derive client key for rate limiting. When TRUST_PROXY_HEADERS is True,
    uses X-Forwarded-For. WARNING: Only enable TRUST_PROXY_HEADERS behind a
    trusted reverse proxy that strips/overwrites this header; otherwise clients
    can spoof it and bypass or dilute rate limiting."""
    return client_key(request, trust_proxy_headers=settings.TRUST_PROXY_HEADERS)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    check_api_key(x_api_key, settings.API_KEY, required=settings.API_AUTH_REQUIRED)


def _evict_oldest_limiter() -> None:
    _rate_limiter.evict_oldest()


def rate_limit(request: Request) -> None:
    if settings.RATE_LIMIT_PER_MIN <= 0:
        return
    if not _rate_limiter.check(_get_client_key(request)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
