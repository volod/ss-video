"""Storage, indexing, and persistence helpers."""

from importlib import import_module
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_EXPORTS = {
    "InMemoryStore": (".vector_store", "InMemoryStore"),
    "QdrantStore": (".qdrant", "QdrantStore"),
    "RecentEmbeddingIndex": (".recent_index", "RecentEmbeddingIndex"),
    "build_map_cache": (".map_cache", "build_map_cache"),
    "create_job": (".jobs", "create_job"),
    "fetch_and_claim_next_pending": (".jobs", "fetch_and_claim_next_pending"),
    "fetch_job": (".jobs", "fetch_job"),
    "list_global_maps": (".global_maps", "list_global_maps"),
    "create_robot_session": (".realtime", "create_robot_session"),
    "fetch_latest_realtime_pose": (".realtime", "fetch_latest_realtime_pose"),
    "fetch_mission": (".missions", "fetch_mission"),
    "fetch_realtime_state": (".realtime", "fetch_realtime_state"),
    "fetch_robot_session": (".realtime", "fetch_robot_session"),
    "insert_realtime_pose": (".realtime", "insert_realtime_pose"),
    "list_realtime_frames": (".realtime", "list_realtime_frames"),
    "list_mission_frames": (".missions", "list_mission_frames"),
    "insert_sensor_packets": (".realtime", "insert_sensor_packets"),
    "insert_semantic_observation": (".realtime", "insert_semantic_observation"),
    "list_map_tiles": (".realtime", "list_map_tiles"),
    "list_semantic_observations": (".realtime", "list_semantic_observations"),
    "stop_robot_session": (".realtime", "stop_robot_session"),
    "summarize_realtime_session": (".realtime", "summarize_realtime_session"),
    "upsert_realtime_frame": (".realtime", "upsert_realtime_frame"),
    "upsert_map_tile": (".realtime", "upsert_map_tile"),
    "update_job": (".jobs", "update_job"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    return getattr(import_module(module_name, __name__), attr_name)
