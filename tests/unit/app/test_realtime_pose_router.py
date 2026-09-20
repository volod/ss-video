"""Focused pose-router tests."""

from unittest.mock import patch

import httpx
import pytest

from tests.unit.app.test_realtime_router import FakeConn, FakePool, _app


@pytest.mark.anyio
async def test_pose_estimate_endpoint_returns_latest_pose():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post("/realtime/session/start", json={"robot_id": "pose_drone"})
            session_id = start.json()["session_id"]

            resp = await client.post(
                f"/realtime/session/{session_id}/pose/estimate",
                json={
                    "packets": [
                        {
                            "sensor_type": "gps",
                            "t_device": 2.0,
                            "payload": {"east": 3.0, "north": 4.0},
                        },
                        {"sensor_type": "imu", "t_device": 2.01, "payload": {"yaw": 0.2}},
                    ]
                },
            )
            assert resp.status_code == 200
            assert resp.json()["source"] == "fused_gps_imu"
