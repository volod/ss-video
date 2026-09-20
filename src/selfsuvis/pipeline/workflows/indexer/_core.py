"""Core VideoIndexer: initialisation, frame loop, index_video orchestration, tile indexing."""

import dataclasses
import itertools
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from selfsuvis.models.dino_model import DINOEmbedder
from selfsuvis.models.openclip_model import OpenCLIPEmbedder
from selfsuvis.models.rssm_model import RSSMEmbedder
from selfsuvis.pipeline.core import (
    RateTimer,
    ensure_dir,
    get_dino_model_name,
    get_logger,
    settings,
    stable_point_id,
)
from selfsuvis.pipeline.core.optional_deps import require_cv2, require_qdrant_models
from selfsuvis.pipeline.fusion import run_platform_state_fusion
from selfsuvis.pipeline.media import extract_frames
from selfsuvis.pipeline.media.dedup import PhashLRU, dhash
from selfsuvis.pipeline.media.heuristics import (
    downsample_gray,
    edge_density,
    frame_quality_ok,
    histogram_diff,
    mean_abs_diff,
    phase_corr_align,
    ssim_diff,
    tile_quality_ok,
)
from selfsuvis.pipeline.storage import QdrantStore, RecentEmbeddingIndex
from selfsuvis.pipeline.vision import (
    ASRModel,
    DepthModel,
    DetectionModel,
    FlorenceModel,
    OCRModel,
    QwenModel,
    RFSignalAnalyzer,
    SAMPredictor,
    UniDriveVLAModel,
    WorldModel,
    YOLODetector,
)

from ._caption import _CaptionMixin
from ._perception import _PerceptionMixin
from ._tracking import _TrackingMixin
from ._vision import _VisionMixin

if TYPE_CHECKING:
    from qdrant_client.http import models as qmodels  # pragma: no cover


@dataclass
class _IndexFrameState:
    """Mutable state carried across frame processing iterations."""

    prev_small: np.ndarray | None = None
    last_kept_embed: np.ndarray | None = None
    last_kept_time: float = -1.0
    last_kept_frame: np.ndarray | None = None
    last_kept_small: np.ndarray | None = None
    eff_fps: float = 0.0
    skip_step: int = 1
    segment_count: int = 0


class VideoIndexer(_CaptionMixin, _PerceptionMixin, _VisionMixin, _TrackingMixin):
    def __init__(self, enable_tiles: bool = True):
        self.logger = get_logger(__name__)
        self.enable_tiles = enable_tiles
        self.clip_model = OpenCLIPEmbedder()
        self.dino_model = None
        if settings.MODEL_NAME in {"dinov2", "dinov3"}:
            name = get_dino_model_name(settings.MODEL_NAME)
            if name is None:
                raise ValueError(f"Unsupported DINO model family: {settings.MODEL_NAME}")
            self.dino_model = DINOEmbedder(model_name=name)
        self._florence_model: FlorenceModel | None = None
        self.store = QdrantStore(clip_dim=self.clip_model.image_dim(), dino_dim=self._dino_dim())
        self.qwen_model = (
            QwenModel(clip_prescreen_fn=self._make_vehicle_prescreen())
            if settings.QWEN_API_URL
            else None
        )
        self.unidrive_model = (
            UniDriveVLAModel() if settings.UNIDRIVE_ENABLED and settings.UNIDRIVE_API_URL else None
        )
        self.asr_model = ASRModel() if settings.ASR_ENABLED else None
        self.ocr_model = OCRModel() if settings.OCR_ENABLED else None
        self.depth_model = DepthModel() if settings.DEPTH_ENABLED else None
        self.detection_model = DetectionModel() if settings.DETECTION_ENABLED else None
        self.world_model = WorldModel() if settings.WORLD_MODEL_ENABLED else None
        self.yolo_detector = YOLODetector() if settings.YOLO_ENABLED else None
        self.sam_predictor = SAMPredictor() if settings.SAM_ENABLED else None
        self.segmentation_predictor = (
            SAMPredictor(
                enabled=True,
                model_name=settings.SEGMENTATION_MODEL,
            )
            if settings.SEGMENTATION_ENABLED
            else None
        )
        self.rf_analyzer = RFSignalAnalyzer() if settings.RF_ENABLED else None
        # RF-DETR tracker is initialised lazily inside _run_gemma_directed_tracking_pass
        self.rfdetr_tracker = None
        self.rssm_embedder = (
            RSSMEmbedder(
                hidden_dim=settings.DREAMER_HIDDEN_DIM,
                latent_dim=settings.DREAMER_LATENT_DIM,
                train_steps=settings.DREAMER_TRAIN_STEPS,
            )
            if settings.DREAMER_ENABLED
            else None
        )
        self.phash_lru = PhashLRU(settings.PHASH_LRU_SIZE, settings.PHASH_HAMMING_MAX)
        self.recent_index = RecentEmbeddingIndex(
            dim=self.clip_model.image_dim(),
            max_size=settings.DEDUP_RECENT_TILES,
            ttl_sec=settings.DEDUP_TTL_SEC,
        )

    def _dino_dim(self) -> int | None:
        if self.dino_model is None:
            return None
        return self.dino_model.image_dim()

    @property
    def florence_model(self) -> FlorenceModel:
        if self._florence_model is None:
            self._florence_model = FlorenceModel()
        return self._florence_model

    # -- per-frame helpers -----------------------------------------------------

    def _stabilize(self, prev_small: np.ndarray, small: np.ndarray) -> np.ndarray:
        """Return `small` aligned to `prev_small` via phase correlation, or `small` unchanged."""
        if not settings.STAB_ENABLE:
            return small
        aligned, dx, dy, resp = phase_corr_align(
            prev_small.astype(np.float32), small.astype(np.float32)
        )
        if (
            resp >= settings.PHASECORR_MIN_RESPONSE
            and abs(dx) <= settings.STAB_MAX_SHIFT
            and abs(dy) <= settings.STAB_MAX_SHIFT
        ):
            return aligned.astype(np.uint8)
        return small

    def _update_adaptive_fps(self, eff_fps: float, motion: float) -> tuple[float, int]:
        """Adjust effective FPS based on motion level. Returns (new_eff_fps, skip_step)."""
        if motion < settings.MOTION_LOW:
            eff_fps = max(settings.SAMPLE_FPS_MIN, eff_fps * 0.5)
        elif motion > settings.MOTION_HIGH:
            eff_fps = min(settings.SAMPLE_FPS_MAX, eff_fps * 1.5)
        skip_step = max(1, int(round(settings.SAMPLE_FPS_MAX / eff_fps)))
        return eff_fps, skip_step

    def _visual_diffs(
        self, last_kept_small: np.ndarray | None, small_for_diff: np.ndarray
    ) -> tuple[float, float]:
        """Return (histogram_diff, ssim_diff) vs last kept frame, or (0, 0) if none."""
        if last_kept_small is None:
            return 0.0, 0.0
        hist_d = histogram_diff(last_kept_small, small_for_diff)
        try:
            ssim_d = ssim_diff(last_kept_small, small_for_diff)
        except (ValueError, TypeError):
            ssim_d = 0.0
        return hist_d, ssim_d

    def _should_keep(self, hist_d: float, ssim_d: float, drift: float, time_gap: float) -> bool:
        """Return True if the frame is sufficiently different from the last kept frame."""
        return (
            hist_d > settings.HIST_THRESH
            or ssim_d > settings.HIST_THRESH
            or drift > settings.EMBED_DRIFT_THRESH
            or time_gap > settings.MAX_GAP_SEC
        )

    def _build_frame_point(
        self,
        video_id: str,
        segment_id: int,
        t_sec: float,
        frame_path: str,
        frame_pil: Image.Image,
        clip_embed: np.ndarray,
        mission_id: str | None = None,
        gps: dict[str, float] | None = None,
        enu: dict[str, float] | None = None,
        robot_id: str | None = None,
        global_map_id: int | None = None,
    ) -> "qmodels.PointStruct":
        """Build a Qdrant point for one keyframe."""
        qmodels = require_qdrant_models()
        vectors: dict[str, Any] = {"clip": clip_embed.tolist()}
        if self.dino_model:
            vectors["dino"] = self.dino_model.encode_images([frame_pil], batch_size=1)[0].tolist()
        payload: dict[str, Any] = {
            "type": "frame",
            "video_id": video_id,
            "segment_id": segment_id,
            "t_sec": t_sec,
            "frame_path": frame_path,
        }
        if mission_id is not None:
            payload["mission_id"] = mission_id
        if robot_id is not None:
            payload["robot_id"] = robot_id
        if gps is not None:
            payload["gps"] = gps
        if enu is not None:
            payload["enu"] = enu
        if global_map_id is not None:
            payload["global_map_id"] = global_map_id
        # Model version provenance: lets queries be traced back to the model that
        # produced the embedding.  Stored as-is; "base" means the pretrained backbone.
        payload["model_version_id"] = settings.MODEL_VERSION_ID
        return qmodels.PointStruct(
            id=stable_point_id(video_id, segment_id, int(t_sec * 1000), "frame"),
            vector=vectors,
            payload=payload,
        )

    # -- main frame processing -------------------------------------------------

    def _process_frame(
        self,
        video_id: str,
        frame_path: str,
        t_sec: float,
        frame: np.ndarray,
        state: "_IndexFrameState",
        cell_state: dict[tuple[int, int], tuple[float, float]],
        mission_id: str | None = None,
        gps: dict[str, float] | None = None,
        enu: dict[str, float] | None = None,
        robot_id: str | None = None,
        global_map_id: int | None = None,
    ) -> tuple[
        tuple[dict[str, Any], dict[str, Any], list["qmodels.PointStruct"], int] | None,
        "_IndexFrameState",
    ]:
        """Process one frame. Returns (result, new_state); result is None if frame is skipped."""
        small = downsample_gray(frame, settings.STAB_SIZE)
        prev_small = state.prev_small if state.prev_small is not None else small
        small_for_diff = self._stabilize(prev_small, small)

        motion = mean_abs_diff(prev_small, small_for_diff)
        eff_fps, skip_step = self._update_adaptive_fps(state.eff_fps, motion)
        new_state = dataclasses.replace(
            state, prev_small=small, eff_fps=eff_fps, skip_step=skip_step
        )

        if not frame_quality_ok(frame):
            return None, new_state

        hist_d, ssim_d = self._visual_diffs(state.last_kept_small, small_for_diff)
        cv2 = require_cv2()
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        embed = self.clip_model.encode_images([frame_pil], batch_size=1)[0]
        drift = (
            float(1.0 - np.dot(state.last_kept_embed, embed))
            if state.last_kept_embed is not None
            else 1.0
        )
        time_gap = t_sec - state.last_kept_time

        if not self._should_keep(hist_d, ssim_d, drift, time_gap):
            return None, new_state

        segment_id = state.segment_count
        segment_dict = {
            "segment_id": segment_id,
            "start_t": t_sec,
            "end_t": t_sec,
            "rep_frame_t": t_sec,
            "rep_frame_path": frame_path,
        }
        qdrant_id = stable_point_id(video_id, segment_id, int(t_sec * 1000), "frame")
        frame_point = self._build_frame_point(
            video_id,
            segment_id,
            t_sec,
            frame_path,
            frame_pil,
            embed,
            mission_id=mission_id,
            gps=gps,
            enu=enu,
            robot_id=robot_id,
            global_map_id=global_map_id,
        )
        frame_record = {
            "id": f"{mission_id}:{segment_id}:{int(t_sec * 1000)}",
            "frame_path": frame_path,
            "t_sec": t_sec,
            "segment_id": segment_id,
            "caption": None,
            "caption_confidence": None,
            "caption_model": None,
            "subtitle_text": None,
            "ocr_text": None,
            "al_score": None,
            "al_tag": "none",
            "_clip_embed": embed,
            "cvat_label": None,
            "pose_status": "pending",
            "pose_json": None,
            "gps_json": gps,
            "global_pose_json": enu,
            "qdrant_id": qdrant_id,
        }

        tile_points: list[qmodels.PointStruct] = []
        tile_count = 0
        if self.enable_tiles or drift > settings.TILE_INDEX_IF_EMBED_DRIFT_GT:
            tile_points, tile_count = self._index_tiles(
                frame,
                frame_path,
                video_id,
                segment_id,
                t_sec,
                embed,
                cell_state,
                mission_id=mission_id,
                robot_id=robot_id,
                global_map_id=global_map_id,
            )

        new_state = dataclasses.replace(
            new_state,
            last_kept_embed=embed,
            last_kept_time=t_sec,
            last_kept_frame=frame,
            last_kept_small=downsample_gray(frame, settings.STAB_SIZE),
            segment_count=segment_id + 1,
        )
        return (segment_dict, frame_record, [frame_point] + tile_points, 1 + tile_count), new_state

    def index_video(
        self,
        video_path: str,
        video_id: str,
        mission_id: str | None = None,
        robot_id: str | None = None,
        site_enu_origin=None,
        global_map_id: int | None = None,
        progress_cb=None,
    ) -> dict[str, Any]:
        ensure_dir(settings.VIDEOS_DIR)
        self.logger.info("Indexing video_id=%s path=%s", video_id, video_path)
        dst_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}.mp4")
        if os.path.abspath(video_path) != os.path.abspath(dst_path):
            shutil.copy(video_path, dst_path)

        effective_mission_id = mission_id or video_id
        effective_robot_id = robot_id or settings.ROBOT_ID

        frames = extract_frames(dst_path, video_id)
        if not frames:
            return {"segments": 0, "tiles": 0, "frames": 0}

        gps_lookup: dict[float, dict[str, float] | None] = {}
        enu_lookup: dict[float, dict[str, float] | None] = {}
        try:
            from selfsuvis.pipeline.mapping.gps_registration import gps_to_enu
            from selfsuvis.pipeline.media.gps import extract_gps

            timestamps_ms = [t * 1000.0 for _, t in frames]
            gps_list = extract_gps(dst_path, timestamps_ms)
            enu_origin = site_enu_origin
            if enu_origin is None:
                for g in gps_list:
                    if g is not None:
                        enu_origin = (g["lat"], g["lon"], g["alt"])
                        break
            for (_, t_sec), g in zip(frames, gps_list):
                if g is not None:
                    gps_lookup[t_sec] = {"lat": g["lat"], "lon": g["lon"], "alt": g["alt"]}
                    if enu_origin is not None:
                        tx, ty, tz = gps_to_enu(g["lat"], g["lon"], g["alt"], *enu_origin)
                        enu_lookup[t_sec] = {"tx": float(tx), "ty": float(ty), "tz": float(tz)}
        except Exception:
            self.logger.debug("GPS extraction unavailable or failed for video_id=%s", video_id)

        eff_fps = settings.SAMPLE_FPS_BASE
        state = _IndexFrameState(
            eff_fps=eff_fps,
            skip_step=max(1, int(round(settings.SAMPLE_FPS_MAX / eff_fps))),
        )
        segments: list[dict[str, Any]] = []
        frame_records: list[dict[str, Any]] = []
        points: list[qmodels.PointStruct] = []
        cell_state: dict[tuple[int, int], tuple[float, float]] = {}
        tiles_indexed = 0
        frames_indexed = 0
        frame_timer = RateTimer()
        embed_timer = RateTimer()
        cv2 = require_cv2()

        idx = 0
        while idx < len(frames):
            frame_path, t_sec = frames[idx]
            idx += state.skip_step

            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frame_timer.tick()

            result, state = self._process_frame(
                video_id,
                frame_path,
                t_sec,
                frame,
                state,
                cell_state,
                mission_id=effective_mission_id,
                gps=gps_lookup.get(t_sec),
                enu=enu_lookup.get(t_sec),
                robot_id=effective_robot_id,
                global_map_id=global_map_id,
            )
            embed_timer.tick()

            if result is None:
                continue

            segment_dict, frame_record, frame_and_tile_points, count = result
            segments.append(segment_dict)
            frame_records.append(frame_record)
            points.extend(frame_and_tile_points)
            frames_indexed += 1
            tiles_indexed += count - 1

            if len(points) >= 128:
                self.store.upsert_points(points)
                points = []

            if progress_cb:
                progress_cb(
                    {
                        "frames_processed": frame_timer.count,
                        "segments_found": len(segments),
                        "frames_indexed": frames_indexed,
                        "tiles_indexed": tiles_indexed,
                        "frame_fps": frame_timer.rate(),
                        "embed_fps": embed_timer.rate(),
                    }
                )

        if points:
            self.store.upsert_points(points)

        if frame_records and self.asr_model and self.asr_model.is_enabled():
            self._run_asr_pass(dst_path, frame_records)

        if frame_records:
            if settings.GEMMA_API_URL:
                self._run_gemma_caption_pass(frame_records)
            else:
                self._run_florence_pass(frame_records)

        if frame_records and self.ocr_model and self.ocr_model.is_enabled():
            self._run_ocr_pass(frame_records)

        if frame_records and self.qwen_model:
            self._run_qwen_pass(frame_records)

        if frame_records and self.depth_model and self.depth_model.is_enabled():
            self._run_depth_pass(frame_records)

        if frame_records and self.detection_model and self.detection_model.is_enabled():
            self._run_detection_pass(frame_records)

        if (
            frame_records
            and self.segmentation_predictor
            and self.segmentation_predictor.is_available()
        ):
            self._run_segmentation_pass(frame_records)

        if frame_records and self.rf_analyzer and self.rf_analyzer.is_enabled():
            self._run_rf_analysis_pass(dst_path, frame_records)

        if frame_records and self.yolo_detector and self.yolo_detector.is_enabled():
            self._run_yolo_sam_pass(frame_records)

        semantic_graph_summary = None
        if frame_records and settings.YOLO_SSG_ENABLED:
            semantic_graph_summary = self._run_yolo_ssg_pass(
                video_id=video_id,
                mission_id=effective_mission_id,
                frame_records=frame_records,
            )

        if frame_records and settings.RFDETR_ENABLED and settings.GEMMA_API_URL:
            self._run_gemma_directed_tracking_pass(frame_records)

        if frame_records and self.world_model and self.world_model.is_enabled():
            self._run_world_model_pass(frame_records)

        if frame_records and self.unidrive_model and self.unidrive_model.is_enabled():
            self._run_unidrive_pass(frame_records)

        state_fusion_summary: dict[str, Any] = {
            "enabled": False,
            "status": "skipped",
            "reason": "no frame records",
        }
        if frame_records:
            state_fusion = run_platform_state_fusion(
                video_path=video_path,
                frame_times_sec=[record["t_sec"] for record in frame_records],
                gps_samples=[record.get("gps_json") for record in frame_records],
            )
            state_fusion_summary = state_fusion.summary()
            samples_by_t = {
                round(sample.t_sec, 3): sample for sample in state_fusion.posterior_samples
            }
            for record in frame_records:
                sample = samples_by_t.get(round(record["t_sec"], 3))
                if sample is None:
                    continue
                frame_facts = record.get("frame_facts_json") or {}
                frame_facts["state_fusion"] = {
                    "source": state_fusion.source,
                    "position_enu_m": dict(sample.position_enu_m),
                    "velocity_enu_mps": dict(sample.velocity_enu_mps),
                    "covariance_trace": sample.covariance_trace,
                    "quality": sample.quality,
                    "measurement_kinds": list(sample.measurement_kinds),
                }
                record["frame_facts_json"] = frame_facts

        if frame_records:
            self._run_al_rssm_pass(frame_records)

        self.logger.info(
            "Indexing complete video_id=%s segments=%s frames=%s tiles=%s",
            video_id,
            len(segments),
            frames_indexed,
            tiles_indexed,
        )
        duration_sec = max((record["t_sec"] for record in frame_records), default=0.0)
        gps_origin = next(
            (record["gps_json"] for record in frame_records if record.get("gps_json")), None
        )
        return {
            "segments": len(segments),
            "tiles": tiles_indexed,
            "frames": frames_indexed,
            "duration_sec": duration_sec,
            "gps_origin": gps_origin,
            "frame_records": frame_records,
            "semantic_graph": semantic_graph_summary,
            "state_fusion_summary": state_fusion_summary,
            "unidrive_summary": self._summarize_unidrive_records(frame_records),
        }

    def _make_vehicle_prescreen(self):
        """Build a vehicle pre-screen function reusing self.clip_model (avoids second CLIP load)."""
        from selfsuvis.pipeline.vision.qwen import _VEHICLE_LABELS

        threshold = settings.QWEN_CLIP_THRESHOLD
        labels = list(_VEHICLE_LABELS)
        prompts = [f"a photo of {label}" for label in labels]
        text_embeds = self.clip_model.encode_texts(prompts)

        def prescreen(image: Image.Image) -> bool:
            img_emb = self.clip_model.encode_images([image], batch_size=1)[0]
            sims = float(np.dot(text_embeds, img_emb).max())
            return sims >= threshold

        return prescreen

    def _index_tiles(
        self,
        frame: np.ndarray,
        frame_path: str,
        video_id: str,
        segment_id: int,
        t_sec: float,
        segment_embed: np.ndarray,
        cell_state: dict[tuple[int, int], tuple[float, float]],
        mission_id: str | None = None,
        robot_id: str | None = None,
        global_map_id: int | None = None,
    ) -> tuple[list["qmodels.PointStruct"], int]:
        qmodels = require_qdrant_models()
        cv2 = require_cv2()
        h, w, _ = frame.shape
        tile_points: list[qmodels.PointStruct] = []
        count = 0

        ys = range(0, h - settings.TILE_SIZE + 1, settings.STRIDE)
        xs = range(0, w - settings.TILE_SIZE + 1, settings.STRIDE)
        for y, x in itertools.product(ys, xs):
            if count >= settings.MAX_TILES_PER_SEGMENT:
                break

            tile = frame[y : y + settings.TILE_SIZE, x : x + settings.TILE_SIZE]
            if not tile_quality_ok(tile):
                continue

            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            ph = dhash(gray)
            if self.phash_lru.near_duplicate(ph):
                continue

            cell_key = (x // settings.CELL_SIZE, y // settings.CELL_SIZE)
            score = float(edge_density(gray) * 1000.0 + gray.std())
            last = cell_state.get(cell_key)
            if last and (t_sec - last[0]) <= settings.CELL_WINDOW_SEC and score <= last[1]:
                continue
            cell_state[cell_key] = (t_sec, score)

            tile_pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
            clip_vec = self.clip_model.encode_images([tile_pil], batch_size=1)[0]
            if self.recent_index.max_cosine(clip_vec) > settings.DEDUP_COS_SIM_THRESH:
                continue

            self.phash_lru.add(ph)
            self.recent_index.add(np.expand_dims(clip_vec, axis=0))

            tile_path = os.path.join(
                settings.TILES_DIR,
                video_id,
                str(segment_id),
                f"tile_{int(t_sec * 1000)}_{x}_{y}.jpg",
            )
            ensure_dir(os.path.dirname(tile_path))
            cv2.imwrite(tile_path, tile)

            vectors: dict[str, Any] = {"clip": clip_vec.tolist()}
            if self.dino_model:
                vectors["dino"] = self.dino_model.encode_images([tile_pil], batch_size=1)[
                    0
                ].tolist()

            tile_payload: dict[str, Any] = {
                "type": "tile",
                "video_id": video_id,
                "segment_id": segment_id,
                "t_sec": t_sec,
                "frame_path": frame_path,
                "tile_path": tile_path,
                "x": x,
                "y": y,
                "w": settings.TILE_SIZE,
                "h": settings.TILE_SIZE,
            }
            if mission_id is not None:
                tile_payload["mission_id"] = mission_id
            if robot_id is not None:
                tile_payload["robot_id"] = robot_id
            if global_map_id is not None:
                tile_payload["global_map_id"] = global_map_id
            tile_points.append(
                qmodels.PointStruct(
                    id=stable_point_id(video_id, segment_id, int(t_sec * 1000), "tile", x, y),
                    vector=vectors,
                    payload=tile_payload,
                )
            )
            count += 1

        return tile_points, count
