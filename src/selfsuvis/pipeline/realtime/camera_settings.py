"""Video-side camera / Frigate settings (``COOP_FRIGATE_*`` env names unchanged)."""

from ss_kit.env import env_int, env_str
from ss_kit.mqtt import MqttSettings


class CameraSettings:
    """Frigate and camera rolling-window settings used by the video API."""

    mqtt: MqttSettings = MqttSettings.from_env("COOP_MQTT_")
    frigate_topic_prefix: str = env_str("COOP_FRIGATE_TOPIC_PREFIX", "frigate")
    frigate_api_url: str = env_str("COOP_FRIGATE_API_URL", "http://localhost:8971")
    camera_event_window_sec: int = env_int("COOP_CAMERA_EVENT_WINDOW_SEC", 120)
    site_id: str = env_str("COOP_SITE_ID", "local")

    def validate(self) -> None:
        if self.camera_event_window_sec < 10:
            raise ValueError("COOP_CAMERA_EVENT_WINDOW_SEC must be >= 10 seconds")


camera_settings = CameraSettings()
