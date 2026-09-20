"""Video OpenAPI keeps cameras and omits fusion-rt routes."""

from fastapi import FastAPI

from selfsuvis.app.routers.site import router as site_router


def test_video_site_router_omits_fusion_routes() -> None:
    app = FastAPI()
    app.include_router(site_router)
    paths = set(app.openapi()["paths"])
    assert "/site/cameras" in paths
    assert "/site/state" not in paths
    assert "/site/threat" not in paths
    assert "/site/synthesis" not in paths
    assert "/api/v1/incidents" not in paths
