"""Live path: ChirpStack fixture uplink -> ss-sens /site/sensors and API /site/*.

Requires a running ss-sens stack (`make ss-sens-up-min` in volod/ss-sens). Skips otherwise.
"""

import asyncio
import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport

from selfsuvis.fusion_rt.aggregator import RealtimeThreatAggregator
from selfsuvis.fusion_rt.mqtt_consumer import FusionContractConsumer
from selfsuvis.fusion_rt.routers.site import router as site_router
from selfsuvis.fusion_rt.sensor_ingest import SensorEventIngestor
from selfsuvis.fusion_rt.site_snapshot import CombinedSiteSnapshot
from ss_kit.mqtt import MqttSettings

REPO = Path(__file__).resolve().parents[2]
UPLINK = REPO / "tests" / "assets" / "chirpstack" / "uplink.json"
load_dotenv(REPO / ".data" / ".env")

SS_SENS_URL = f"http://127.0.0.1:{os.getenv('COOP_HTTP_PORT', '8081')}"
TOPIC = "application/a1b2c3d4-0000-4000-8000-000000000001/device/70b3d57ed0060001/event/up"


def _ss_sens_ready() -> bool:
    try:
        response = httpx.get(f"{SS_SENS_URL}/health", timeout=2.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _publish_uplink() -> None:
    user = os.getenv("CHIRPSTACK_MQTT_USERNAME", "chirpstack")
    password = os.getenv("CHIRPSTACK_MQTT_PASSWORD", "chirpstack")
    payload = json.loads(UPLINK.read_text(encoding="utf-8"))
    payload["time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    result = subprocess.run(
        [
            "docker",
            "exec",
            "ss-sens-mosquitto",
            "mosquitto_pub",
            "-h",
            "127.0.0.1",
            "-p",
            "1883",
            "-t",
            TOPIC,
            "-m",
            json.dumps(payload),
            "-u",
            user,
            "-P",
            password,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"mosquitto_pub failed: {result.stderr or result.stdout}")


async def _wait_json(
    url: str, predicate: Callable[[dict[str, Any]], bool], timeout_sec: float = 20.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last: dict[str, Any] | None = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    last = response.json()
                    if predicate(last):
                        return last
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.4)
    pytest.fail(f"timed out waiting for {url}; last={last}")


@pytest.mark.integration
def test_fixture_uplink_reaches_ss_sens_and_api_site_routes() -> None:
    if not _ss_sens_ready():
        pytest.skip("ss-sens is not up; run `make ss-sens-up-min` in volod/ss-sens")
    asyncio.run(_run())


async def _run() -> None:
    snapshot = CombinedSiteSnapshot()
    threat = RealtimeThreatAggregator()
    ingestor = SensorEventIngestor(threat)
    mqtt = MqttSettings.from_env("COOP_MQTT_")
    consumer = FusionContractConsumer(
        on_sensor_event=_fanout(snapshot.ingest_sensor_event, ingestor.on_sensor_event),
        on_sensor_state=snapshot.ingest_sensor_state,
        mqtt=mqtt,
    )
    task = asyncio.create_task(consumer.run())
    app = FastAPI()
    app.include_router(site_router)
    app.state.site_snapshot = snapshot
    app.state.site_threat_aggregator = threat
    try:
        await asyncio.sleep(1.0)
        _publish_uplink()
        sensors = await _wait_json(
            f"{SS_SENS_URL}/site/sensors",
            lambda body: any(
                row.get("dev_eui") == "70b3d57ed0060001" for row in body.get("sensors", [])
            ),
        )
        assert sensors["sensor_count"] >= 1
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            state = await snapshot.get_state()
            if any(row.dev_eui == "70b3d57ed0060001" for row in state.sensors):
                break
            await asyncio.sleep(0.4)
        else:
            pytest.fail("API snapshot never received the contract sensor event")
        with patch("selfsuvis.fusion_rt.deps.fusion_settings") as ms:
            ms.API_KEY = ""
            ms.API_AUTH_REQUIRED = False
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                state_resp = await client.get("/site/state")
                assert state_resp.status_code == 200
                state_body = state_resp.json()
                assert any(row["dev_eui"] == "70b3d57ed0060001" for row in state_body["sensors"])
                threat_resp = await client.get("/site/threat")
                assert threat_resp.status_code == 200
                threat_body = threat_resp.json()
                assert "last_update" in threat_body
                assert "global_threat_map" in threat_body
                json.dumps(threat_body)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _fanout(*cbs):
    async def _cb(event) -> None:
        for cb in cbs:
            await cb(event)

    return _cb
