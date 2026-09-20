"""Shared HTTP helpers for realtime sidecar clients."""

from selfsuvis.pipeline.core.sidecars import HttpSidecarClient


class RealtimeSidecarClient(HttpSidecarClient):
    """Compatibility alias for HTTP-backed realtime sidecars."""
