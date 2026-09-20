"""Vision pass mixin: Detection, Segmentation, YOLO+SAM, Semantic Scene Graph."""

import os
from collections import Counter
from typing import Any

from PIL import Image

from selfsuvis.pipeline.core import ensure_dir, settings
from selfsuvis.pipeline.mapping import build_semantic_environment_graph


def _detections_from_yolo_result(result: Any) -> list[dict[str, Any]]:
    """Extract detection dicts from one YOLO detect_batch item.

    ``YoloDetector.detect_batch`` returns one status/result dict per image
    (``{"detections": [...]}`` or ``{"detection_unavailable": True}``), not a
    bare list of boxes. A list is still accepted for older callers.
    """
    if isinstance(result, dict):
        raw = result.get("detections", [])
    elif isinstance(result, list):
        raw = result
    else:
        return []
    if not isinstance(raw, list):
        return []
    return [d for d in raw if isinstance(d, dict) and "bbox_norm" in d]


class _VisionMixin:
    def _run_detection_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Run object detection and store bounding boxes in frame_facts_json."""
        self.logger.info("Detection pass: %d frames", len(frame_records))
        for batch_start in range(0, len(frame_records), settings.DETECTION_BATCH_SIZE):
            batch = frame_records[batch_start : batch_start + settings.DETECTION_BATCH_SIZE]
            images = []
            for rec in batch:
                try:
                    images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))
            results = self.detection_model.detect_batch(images)
            for rec, res in zip(batch, results):
                fj = rec.get("frame_facts_json") or {}
                if isinstance(fj, dict):
                    fj.update(res)
                    rec["frame_facts_json"] = fj
        self.logger.info("Detection pass complete")

    @staticmethod
    def _bbox_iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def _segment_label_for_mask(
        self,
        mask_bbox_norm: list[float],
        frame_facts_json: dict[str, Any],
    ) -> str:
        detections = frame_facts_json.get("detections") or []
        best_label = "unlabeled"
        best_iou = 0.0
        for det in detections:
            bbox = det.get("bbox_norm")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            iou = self._bbox_iou(mask_bbox_norm, [float(v) for v in bbox])
            if iou > best_iou:
                best_iou = iou
                best_label = str(det.get("label") or "unlabeled")
        return best_label if best_iou >= 0.1 else "unlabeled"

    def _run_segmentation_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Run automatic SAM segmentation and store compact summaries.

        Writes ``frame_facts_json["segments"]`` with mask counts and class-label
        summaries only; raw masks are intentionally not persisted.
        """
        self.logger.info(
            "Segmentation pass: %d frames model=%s",
            len(frame_records),
            settings.SEGMENTATION_MODEL,
        )
        max_masks = max(1, int(settings.SEGMENTATION_MAX_MASKS))
        min_area_norm = float(settings.SEGMENTATION_MIN_AREA_NORM)
        points_per_side = max(2, int(settings.SEGMENTATION_POINTS_PER_SIDE))

        for rec in frame_records:
            try:
                image = Image.open(rec["frame_path"]).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224))

            masks = self.segmentation_predictor.generate_auto_masks(
                image,
                points_per_side=points_per_side,
            )
            masks = [m for m in masks if float(m.get("area_norm", 0.0) or 0.0) >= min_area_norm]
            masks.sort(
                key=lambda m: (
                    float(m.get("score", 0.0) or 0.0),
                    float(m.get("area_norm", 0.0) or 0.0),
                ),
                reverse=True,
            )
            masks = masks[:max_masks]

            fj = rec.get("frame_facts_json") or {}
            if not isinstance(fj, dict):
                fj = {}

            label_counts: Counter[str] = Counter()
            area_norms: list[float] = []
            for mask_info in masks:
                x, y, w_box, h_box = [float(v) for v in (mask_info.get("bbox") or [0, 0, 0, 0])]
                img_w, img_h = image.size
                bbox_norm = [
                    x / max(1.0, img_w),
                    y / max(1.0, img_h),
                    (x + w_box) / max(1.0, img_w),
                    (y + h_box) / max(1.0, img_h),
                ]
                label = self._segment_label_for_mask(bbox_norm, fj)
                label_counts[label] += 1
                area_norms.append(float(mask_info.get("area_norm", 0.0) or 0.0))

            fj["segments"] = {
                "count": len(masks),
                "labels": sorted(label_counts.keys()),
                "label_counts": dict(sorted(label_counts.items())),
                "mean_area_norm": round(sum(area_norms) / len(area_norms), 6)
                if area_norms
                else 0.0,
                "max_area_norm": round(max(area_norms), 6) if area_norms else 0.0,
                "model": settings.SEGMENTATION_MODEL,
                "points_per_side": points_per_side,
            }
            rec["frame_facts_json"] = fj

        self.logger.info("Segmentation pass complete")

    def _run_yolo_sam_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Run YOLO11 detection (+ optional SAM3/SAM2 masks) and store results in frame_facts_json.

        Detections are priority-sorted (human=1 → vehicle=2 → artificial=3 → other=4).
        Stored under ``frame_facts_json["yolo_detections"]`` as a list of dicts:
        ``{label, confidence, bbox_norm, priority, priority_label, mask_area_norm}``.

        When ``SAM_ENABLED`` is true and a SAM backend is available, each bounding
        box is refined with a segmentation mask and ``mask_area_norm`` is populated.
        """
        use_sam = self.sam_predictor is not None and self.sam_predictor.is_available()
        self.logger.info(
            "YOLO+SAM pass: %d frames  model=%s  sam=%s",
            len(frame_records),
            self.yolo_detector.model_id,
            "enabled" if use_sam else "disabled",
        )
        batch_size = getattr(settings, "DETECTION_BATCH_SIZE", 8)
        total_dets = 0
        for batch_start in range(0, len(frame_records), batch_size):
            batch = frame_records[batch_start : batch_start + batch_size]
            images: list[Image.Image] = []
            for rec in batch:
                try:
                    images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))

            yolo_results = self.yolo_detector.detect_batch(images)

            for rec, img, result in zip(batch, images, yolo_results):
                detections = _detections_from_yolo_result(result)
                if use_sam and detections:
                    bboxes = [d["bbox_norm"] for d in detections]
                    try:
                        sam_masks = self.sam_predictor.predict_boxes(img, bboxes)
                        for det, mask_info in zip(detections, sam_masks):
                            det["mask_area_norm"] = mask_info.get("area_norm")
                    except Exception as exc:
                        self.logger.debug(
                            "SAM prediction failed for frame %s: %s", rec.get("frame_id"), exc
                        )

                fj = rec.get("frame_facts_json") or {}
                if isinstance(fj, dict):
                    fj["yolo_detections"] = detections
                    rec["frame_facts_json"] = fj
                total_dets += len(detections)

        self.logger.info(
            "YOLO+SAM pass complete: %d detections across %d frames",
            total_dets,
            len(frame_records),
        )

    def _run_yolo_ssg_pass(
        self,
        *,
        video_id: str,
        mission_id: str,
        frame_records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Build a YOLO semantic scene graph and attach node ids back to frames."""
        graph_dir = os.path.join(settings.MAPS_DIR, mission_id)
        ensure_dir(graph_dir)
        graph_path = os.path.join(graph_dir, "semantic_environment_graph.json")
        graph = build_semantic_environment_graph(
            frame_records,
            graph_id=mission_id or video_id,
            output_path=graph_path,
        )
        assignments = graph.get("frame_assignments", {})
        for record in frame_records:
            frame_key = str(record.get("id") or record.get("frame_path") or "")
            node_ids = assignments.get(frame_key, [])
            if not node_ids:
                continue
            facts = record.get("frame_facts_json") or {}
            if isinstance(facts, dict):
                facts["semantic_graph_node_ids"] = node_ids
                record["frame_facts_json"] = facts

        summary = {
            **graph.get("summary", {}),
            "output_path": graph.get("output_path", graph_path),
            "anchor_source": graph.get("anchor_source", "unknown"),
            "coordinate_frame": graph.get("coordinate_frame", "unknown"),
        }
        self.logger.info(
            "YOLO SSG pass complete: %d nodes, %d edges → %s",
            summary.get("node_count", 0),
            summary.get("edge_count", 0),
            summary["output_path"],
        )
        return summary
