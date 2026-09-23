import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from selfsuvis.app.db import close_db_pool, init_db_pool
from selfsuvis.app.routers.admin import router as admin_router
from selfsuvis.app.routers.analysis4d import router as analysis4d_router
from selfsuvis.app.routers.cvat import cvat_admin_router, webhook_router
from selfsuvis.app.routers.health import router as health_router
from selfsuvis.app.routers.index import router as index_router
from selfsuvis.app.routers.jobs import router as jobs_router
from selfsuvis.app.routers.query import router as query_router
from selfsuvis.app.routers.realtime import router as realtime_router
from selfsuvis.app.routers.robot import router as robot_router
from selfsuvis.app.routers.scene import router as scene_router
from selfsuvis.app.routers.site import router as site_router
from selfsuvis.app.services.camera_streams import CameraStreamService
from selfsuvis.app.services.form_templates import get_index_form_html
from selfsuvis.app.services.live_streams import MediaMtxClient, RealtimeStreamManager
from selfsuvis.pipeline.core import get_logger, log_preflight, run_production_preflight, settings
from selfsuvis.pipeline.storage.processed import ainit_db as init_processed_db
from ss_kit.web import SecurityHeadersMiddleware

logger = get_logger(__name__)


def _start_video_mqtt(app: FastAPI) -> list[asyncio.Task]:
    """Publish camera-event and scene-caption contracts; consume Frigate MQTT."""
    tasks: list[asyncio.Task] = []
    try:
        from selfsuvis.pipeline.realtime.contract_consumer import ContractEventConsumer
        from selfsuvis.pipeline.realtime.contract_publisher import VideoContractPublisher

        publisher = VideoContractPublisher()
        app.state.video_publisher = publisher
        tasks.append(asyncio.create_task(publisher.run(), name="video_mqtt_pub"))

        async def _on_camera(event) -> None:
            await publisher.publish_camera_event(event)

        consumer = ContractEventConsumer(
            on_camera_event=_on_camera,
            subscribe_frigate=True,
            subscribe_contracts=False,
        )
        tasks.append(asyncio.create_task(consumer.run(), name="video_frigate_mqtt"))
        logger.info("video MQTT publisher started (%s)", publisher.describe())
    except Exception as exc:
        logger.warning("video MQTT publisher not started: %s", exc)
        app.state.video_publisher = None
    return tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down shared video API resources."""
    report = run_production_preflight("api")
    log_preflight(report)
    if os.getenv("STARTUP_PREFLIGHT_STRICT", "false").lower() == "true":
        report.raise_for_errors()
    if settings.DATABASE_URL:
        from selfsuvis.pipeline.storage.migrate_video import migrate as migrate_video

        try:
            await migrate_video(settings.DATABASE_URL, verbose=False)
        except Exception as exc:
            logger.warning("video schema bootstrap failed: %s", exc)
    await init_processed_db()
    await init_db_pool(app)
    mqtt_tasks = _start_video_mqtt(app)
    publisher = getattr(app.state, "video_publisher", None)
    app.state.mediamtx_client = MediaMtxClient()
    app.state.realtime_stream_manager = RealtimeStreamManager(
        getattr(app.state, "db_pool", None),
        on_caption=publisher.publish_caption_row if publisher is not None else None,
    )

    camera_streams = CameraStreamService(
        mediamtx_client=app.state.mediamtx_client,
        stream_manager=app.state.realtime_stream_manager,
        db_pool=getattr(app.state, "db_pool", None),
        event_publisher=publisher,
    )
    app.state.camera_streams = camera_streams
    await camera_streams.start()

    try:
        yield
    finally:
        for task in mqtt_tasks:
            if task and not task.done():
                task.cancel()
        if mqtt_tasks:
            await asyncio.gather(*mqtt_tasks, return_exceptions=True)
        await camera_streams.shutdown()
        await app.state.realtime_stream_manager.shutdown()
        await close_db_pool(app)


app = FastAPI(title="ss-video", lifespan=lifespan)
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(site_router)
app.include_router(analysis4d_router)
app.include_router(admin_router)
app.include_router(cvat_admin_router)
app.include_router(health_router)
app.include_router(index_router)
app.include_router(jobs_router)
app.include_router(query_router)
app.include_router(realtime_router)
app.include_router(robot_router)
app.include_router(scene_router)
app.include_router(webhook_router)


@app.get("/index/form", response_class=HTMLResponse, include_in_schema=False)
async def index_form():
    """Simple HTML form to upload a local video or submit a URL for indexing."""
    return get_index_form_html()
