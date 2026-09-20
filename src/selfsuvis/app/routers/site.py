"""Live camera sessions owned by the video API.

GET /site/cameras. Fusion-rt serves /site/state, /site/threat, /site/synthesis,
and WS /site/stream. Sensor-only /site/sensors and /site/mesh are served by ss-sens.
"""

from typing import Any

from fastapi import APIRouter, Depends, Request

from selfsuvis.app.deps import require_api_key

router = APIRouter(tags=["site"])


@router.get("/site/cameras")
async def site_cameras(request: Request, _: None = Depends(require_api_key)) -> dict[str, Any]:
    streams = getattr(request.app.state, "camera_streams", None)
    cameras = []
    if streams is not None:
        for session in streams.active_cameras():
            cameras.append(
                {
                    "camera": session.get("camera"),
                    "last_seen": session.get("started_at"),
                    "recent_detections": [],
                    "active_labels": [],
                    "total_events": 0,
                    "session_id": session.get("session_id"),
                    "rtsp_url": session.get("rtsp_url"),
                }
            )
    return {"cameras": cameras, "camera_count": len(cameras)}
