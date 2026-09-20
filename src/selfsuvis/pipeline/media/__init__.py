"""Media ingestion and extraction helpers."""

from importlib import import_module
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_EXPORTS = {
    "caption_rtsp_stream": (".rtsp_captioner", "caption_rtsp_stream"),
    "download_url": (".download", "download_url"),
    "extract_audio": (".audio", "extract_audio"),
    "extract_frames": (".ffmpeg", "extract_frames"),
    "extract_frames_adaptive": (".frames", "extract_frames_adaptive"),
    "extract_frames_fixed": (".frames", "extract_frames_fixed"),
    "extract_gps_for_frames": (".gps", "extract_gps_for_frames"),
    "extract_gps_track": (".gps", "extract_gps_track"),
    "map_subtitles_to_frames": (".audio", "map_subtitles_to_frames"),
    "record_rtsp": (".rtsp_ingest", "record_rtsp"),
    "safe_request": (".network", "safe_request"),
    "validate_rtsp_url": (".rtsp_ingest", "validate_rtsp_url"),
    "validate_url": (".network", "validate_url"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    return getattr(import_module(module_name, __name__), attr_name)
