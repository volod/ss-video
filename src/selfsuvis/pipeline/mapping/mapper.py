"""nerfstudio splatfacto 3DGS mapper.

Calls the nerfstudio FastAPI wrapper (separate Docker container, GPU-only) to
train a splatfacto model from the SfM-registered frames for a mission.

Scene chunking: when pycolmap returns multiple disconnected connected components
(e.g. takeoff → transit → inspection), each component is trained as a separate
splatfacto scene and output to maps/{mission_id}/scene-{N}/splat.ply.
Single-component missions use maps/{mission_id}/splat.ply (legacy path).

This module is intentionally thin — it just drives the HTTP API of the
nerfstudio container and polls for completion.  The heavy lifting (ns-train
splatfacto) runs inside the nerfstudio container.

ICP fusion (Phase 2): after splatfacto completes, each new splat.ply is
optionally registered against an existing global-map splat via the mapper
service (docker/core/docker-compose.override.yml, http://mapper:8000).  Pass
target_splat_paths to run_mapper; results are returned in icp_results and
should be persisted to global_map_missions by the caller.

The mapper is called by pipeline/indexer.py AFTER pose_status=success is set
by pipeline/sfm.py.  Missions with fewer than MIN_FRAMES_FOR_3DGS registered
poses are skipped (map_status='skipped').

Environment variables:
    NERFSTUDIO_API_URL   Base URL of the nerfstudio wrapper (default: http://nerfstudio:8000)
    MAPPER_API_URL       Base URL of the ICP mapper service (default: http://mapper:8000)
    MAPS_DIR             Root for splat.ply output (default: .data/maps)
"""

import os
import time
from typing import Any

import requests

from selfsuvis.pipeline.core import ensure_dir, get_logger, settings
from selfsuvis.pipeline.mapping.splat_io import apply_transform_to_splat, merge_splats

logger = get_logger(__name__)

# Skip 3DGS for tiny missions — nerfstudio needs enough views to converge
MIN_FRAMES_FOR_3DGS = 30

# How long to wait for ns-train to finish (seconds)
_TRAIN_TIMEOUT_SEC = 3600  # 1 hour
_POLL_INTERVAL_SEC = 10


class MapperError(RuntimeError):
    """Raised when the nerfstudio wrapper returns an error."""


def _map_result(
    *,
    map_status: str,
    message: str,
    splat_path: str | None = None,
    splat_paths: list[str] | None = None,
    scene_count: int = 0,
    icp_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "map_status": map_status,
        "splat_path": splat_path,
        "splat_paths": list(splat_paths or []),
        "scene_count": int(scene_count),
        "message": message,
        "icp_results": list(icp_results or []),
    }


def _post(endpoint: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    url = f"{settings.NERFSTUDIO_API_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _get(endpoint: str, timeout: int = 30) -> dict[str, Any]:
    url = f"{settings.NERFSTUDIO_API_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _train_scene(
    mission_id: str,
    scene_label: str,
    frame_list: list[dict[str, Any]],
    output_dir: str,
) -> dict[str, Any]:
    """Submit and poll one splatfacto training job.

    Args:
        mission_id:   Mission identifier (for logging).
        scene_label:  Human-readable label, e.g. "scene-0" or "main".
        frame_list:   List of {frame_path, pose_json} dicts.
        output_dir:   Absolute path where nerfstudio should write splat.ply.

    Returns:
        {"map_status": "success"|"failed"|"skipped", "splat_path": str|None, "message": str}
    """
    splat_path = os.path.join(output_dir, "splat.ply")
    ensure_dir(output_dir)

    logger.info(
        "Mapper: starting splatfacto mission=%s %s frames=%d output=%s",
        mission_id,
        scene_label,
        len(frame_list),
        output_dir,
    )

    resp = _post(
        "/train",
        {
            "mission_id": f"{mission_id}/{scene_label}",
            "frames": frame_list,
            "output_dir": output_dir,
        },
        timeout=60,
    )
    train_id = resp.get("train_id")
    if not train_id:
        raise MapperError(f"nerfstudio did not return a train_id: {resp}")

    deadline = time.time() + _TRAIN_TIMEOUT_SEC
    while time.time() < deadline:
        status_resp = _get(f"/train/{train_id}/status")
        status = status_resp.get("status")

        if status == "done":
            if os.path.isfile(splat_path):
                logger.info(
                    "Mapper: 3DGS complete mission=%s %s splat=%s",
                    mission_id,
                    scene_label,
                    splat_path,
                )
                return _map_result(
                    map_status="success",
                    splat_path=splat_path,
                    message=f"splatfacto training complete ({scene_label})",
                )
            logger.error("Mapper: nerfstudio reported done but splat.ply missing at %s", splat_path)
            return _map_result(
                map_status="failed",
                message=f"nerfstudio done but splat.ply not found ({scene_label})",
            )

        if status == "error":
            err = status_resp.get("error", "unknown error")
            logger.error("Mapper: nerfstudio error mission=%s %s: %s", mission_id, scene_label, err)
            return _map_result(map_status="failed", message=err)

        logger.debug("Mapper: waiting mission=%s %s status=%s", mission_id, scene_label, status)
        time.sleep(_POLL_INTERVAL_SEC)

    msg = f"nerfstudio timed out after {_TRAIN_TIMEOUT_SEC}s ({scene_label})"
    logger.error("Mapper: %s mission=%s", msg, mission_id)
    return _map_result(map_status="failed", message=msg)


def _call_icp_fuse(
    source_path: str,
    target_path: str,
    source_meta: dict[str, Any] | None = None,
    target_meta: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Call POST /fuse on the ICP mapper service.

    Returns the parsed FuseResponse dict on success, or None if the mapper
    service is unavailable (ConnectionError) — caller treats as soft skip.

    Raises MapperError on unexpected HTTP errors.
    """
    url = f"{settings.MAPPER_API_URL.rstrip('/')}/fuse"
    payload: dict[str, Any] = {
        "source_path": source_path,
        "target_path": target_path,
    }
    if source_meta is not None:
        payload["source_meta"] = source_meta
    if target_meta is not None:
        payload["target_meta"] = target_meta

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.warning(
            "ICP mapper not reachable at %s — skipping ICP fusion for %s",
            settings.MAPPER_API_URL,
            source_path,
        )
        return None
    except requests.exceptions.HTTPError as exc:
        raise MapperError(f"ICP mapper returned HTTP error: {exc}") from exc


def _fuse_splat_files(
    source_path: str,
    target_path: str,
    transform_4x4: list[list[float]],
    mission_id: str,
    scene_label: str,
) -> str | None:
    """Transform source splat into target frame and merge with target.

    Writes the fused PLY alongside the source splat:
        <source_dir>/fused.ply

    Returns the fused path on success, or None if an error occurs
    (logged as a warning — the ICP result is still returned to the caller).
    """
    source_dir = os.path.dirname(source_path)
    aligned_path = os.path.join(source_dir, "_aligned_tmp.ply")
    fused_path = os.path.join(source_dir, "fused.ply")
    try:
        apply_transform_to_splat(source_path, transform_4x4, aligned_path)
        n_total = merge_splats([target_path, aligned_path], fused_path)
        logger.info(
            "Mapper: fused splat written mission=%s scene=%s gaussians=%d path=%s",
            mission_id,
            scene_label,
            n_total,
            fused_path,
        )
        return fused_path
    except Exception as exc:
        logger.warning(
            "Mapper: splat fusion failed mission=%s scene=%s: %s",
            mission_id,
            scene_label,
            exc,
        )
        return None
    finally:
        try:
            os.remove(aligned_path)
        except OSError:
            pass


def run_mapper(
    mission_id: str,
    sfm_results: list[dict[str, Any]],
    scene_count: int = 1,
    target_splat_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Train splatfacto 3DGS model(s) for a mission.

    Handles scene chunking: if scene_count > 1, trains one model per connected
    SfM component and writes to maps/{mission_id}/scene-{N}/splat.ply.
    Single-scene missions write to maps/{mission_id}/splat.ply.

    After each successful scene, optionally runs ICP registration against each
    path in target_splat_paths via the mapper service (POST /fuse).  If the
    mapper service is not reachable, ICP results are silently omitted.

    Args:
        mission_id:          Mission identifier.
        sfm_results:         Output of pipeline/sfm.run_sfm — list of frame
                             dicts with pose_json, pose_status, scene_index.
        scene_count:         Number of SfM connected components (from sfm).
        target_splat_paths:  Existing global-map splat.ply paths to register
                             the new scenes against (Phase 2 ICP fusion).
                             Typically from global_map_db.get_global_map_splats.

    Returns:
        {
            "map_status":  "success" | "failed" | "skipped",
            "splat_path":  str | None,   # primary splat.ply (scene-0 or single)
            "splat_paths": List[str],    # all splat.ply paths (empty on skip/fail)
            "scene_count": int,          # number of scenes attempted
            "message":     str,
            "icp_results": List[dict],   # one entry per (new_scene, target) pair;
                                         # empty when target_splat_paths is None
                                         # or mapper service is unreachable.
        }
    """
    registered = [r for r in sfm_results if r.get("pose_status") == "success"]

    if len(registered) < MIN_FRAMES_FOR_3DGS:
        msg = (
            f"Only {len(registered)} registered frames "
            f"(minimum {MIN_FRAMES_FOR_3DGS}); skipping 3DGS."
        )
        logger.info("Mapper: %s mission=%s", msg, mission_id)
        return _map_result(map_status="skipped", message=msg)

    # Group registered frames by scene_index
    scenes: dict[int, list[dict[str, Any]]] = {}
    for r in registered:
        idx = r.get("scene_index") or 0
        scenes.setdefault(idx, []).append(
            {"frame_path": r["frame_path"], "pose_json": r["pose_json"]}
        )

    # Filter scenes that have enough frames on their own
    valid_scenes = {
        idx: frames for idx, frames in scenes.items() if len(frames) >= MIN_FRAMES_FOR_3DGS
    }

    if not valid_scenes:
        # Registered frames exist but no single scene has enough for 3DGS
        msg = (
            f"{len(registered)} registered frames split across {len(scenes)} scene(s); "
            f"no scene meets the {MIN_FRAMES_FOR_3DGS}-frame minimum for 3DGS."
        )
        logger.info("Mapper: %s mission=%s", msg, mission_id)
        return _map_result(map_status="skipped", message=msg)

    mission_maps_dir = os.path.join(settings.MAPS_DIR, mission_id)
    use_chunking = len(valid_scenes) > 1

    try:
        splat_paths: list[str] = []
        failed_scenes: list[str] = []
        icp_results: list[dict[str, Any]] = []

        for scene_idx in sorted(valid_scenes.keys()):
            frame_list = valid_scenes[scene_idx]
            if use_chunking:
                scene_label = f"scene-{scene_idx}"
                output_dir = os.path.join(mission_maps_dir, scene_label)
            else:
                scene_label = "main"
                output_dir = mission_maps_dir

            result = _train_scene(mission_id, scene_label, frame_list, output_dir)
            if result["map_status"] == "success" and result["splat_path"]:
                new_splat = result["splat_path"]
                splat_paths.append(new_splat)

                # Phase 2: ICP registration against each provided target splat
                if target_splat_paths:
                    for target_path in target_splat_paths:
                        logger.info(
                            "Mapper: ICP fusion mission=%s scene=%s target=%s",
                            mission_id,
                            scene_label,
                            target_path,
                        )
                        fuse_resp = _call_icp_fuse(new_splat, target_path)
                        if fuse_resp is None:
                            continue

                        fused_path: str | None = None
                        if fuse_resp.get("converged") and fuse_resp.get("transform_4x4"):
                            fused_path = _fuse_splat_files(
                                new_splat,
                                target_path,
                                fuse_resp["transform_4x4"],
                                mission_id,
                                scene_label,
                            )

                        icp_results.append(
                            {
                                "source_splat": new_splat,
                                "target_splat": target_path,
                                "status": fuse_resp.get("status"),
                                "converged": fuse_resp.get("converged", False),
                                "transform_4x4": fuse_resp.get("transform_4x4"),
                                "rmse": fuse_resp.get("rmse"),
                                "fitness": fuse_resp.get("fitness"),
                                "message": fuse_resp.get("message", ""),
                                "fused_splat": fused_path,
                            }
                        )
            else:
                failed_scenes.append(scene_label)

        if not splat_paths:
            return _map_result(
                map_status="failed",
                message=f"All {len(failed_scenes)} scene(s) failed",
                scene_count=len(valid_scenes),
            )

        primary = splat_paths[0]
        status = "success" if not failed_scenes else "failed"
        msg = (
            f"{len(splat_paths)}/{len(valid_scenes)} scene(s) succeeded"
            if use_chunking
            else "splatfacto training complete"
        )
        return _map_result(
            map_status=status,
            splat_path=primary,
            splat_paths=splat_paths,
            scene_count=len(valid_scenes),
            message=msg,
            icp_results=icp_results,
        )

    except requests.exceptions.ConnectionError:
        msg = (
            f"Cannot connect to nerfstudio at {settings.NERFSTUDIO_API_URL}. "
            "Is the nerfstudio container running? (docker-compose.override.yml, GPU only)"
        )
        logger.warning("Mapper: %s — map_status=skipped", msg)
        return _map_result(map_status="skipped", message=msg)

    except Exception as exc:
        logger.error("Mapper: unexpected error mission=%s: %s", mission_id, exc)
        return _map_result(map_status="failed", message=str(exc))
