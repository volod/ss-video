"""Tracking and temporal pass mixin: Gemma directed tracking, World model, RSSM/AL."""

import time
from typing import Any

import numpy as np
from PIL import Image

from selfsuvis.pipeline.core import settings


class _TrackingMixin:
    def _run_gemma_directed_tracking_pass(
        self,
        frame_records: list[dict[str, Any]],
    ) -> None:
        """Gemma directed tracking pass: Gemma scene understanding → SAM segmentation
        → RF-DETR tracking. Stores results in ``frame_facts_json["gemma_tracking"]``.

        Tracking results per frame:
            {
                "scene_type":        str,
                "tracking_priority": List[str],
                "detections":        List[detection_dict],
                "sam_masks":         [{"category": str, "area_norm": float, "source": str}],
            }
        """
        from selfsuvis.pipeline.vision.rfdetr import RFDETRTracker
        from ssv_vdp.steps.perception.gemma_tracking import (
            _gemma_structured_scene_analysis,
            _sam_directed_by_gemma,
        )

        frame_list = [
            (r["frame_path"], float(r.get("t_sec", 0.0)))
            for r in frame_records
            if r.get("frame_path")
        ]
        if not frame_list:
            return

        gemma_scene = _gemma_structured_scene_analysis(
            frame_list,
            api_url=settings.GEMMA_API_URL,
            model=settings.GEMMA_API_MODEL,
            timeout=float(settings.GEMMA_API_TIMEOUT_SEC),
            clip_model=self.clip_model,
        )
        tracking_priority = gemma_scene.get("tracking_priority", [])
        gemma_objects = gemma_scene.get("dominant_objects", [])
        scene_type = gemma_scene.get("scene_type", "other")
        self.logger.info(
            "Gemma directed tracking: scene_type=%s priority=%s objects=%d",
            scene_type,
            tracking_priority,
            len(gemma_objects),
        )

        if self.rfdetr_tracker is None:
            self.rfdetr_tracker = RFDETRTracker()
        tracking_results = self.rfdetr_tracker.track_sequence(
            frame_list,
            target_labels=tracking_priority if tracking_priority else None,
        )
        path_to_dets = {r["frame_path"]: r.get("detections", []) for r in tracking_results}

        use_sam = (
            self.sam_predictor is not None
            and self.sam_predictor.is_available()
            and bool(gemma_objects)
        )

        for rec in frame_records:
            fp = rec.get("frame_path", "")
            fj = rec.get("frame_facts_json") or {}
            if not isinstance(fj, dict):
                fj = {}
            tracking_dets = path_to_dets.get(fp, [])
            sam_masks_summary: list[dict] = []
            if use_sam and fp:
                try:
                    img = Image.open(fp).convert("RGB")
                    masks = _sam_directed_by_gemma(
                        img,
                        gemma_objects,
                        self.sam_predictor,
                        self.clip_model,
                    )
                    w_img, h_img = img.size
                    sam_masks_summary = [
                        {
                            "category": m.get("category", "unknown"),
                            "area_norm": round(float(m["mask"].sum()) / (w_img * h_img), 6)
                            if m.get("mask") is not None
                            else 0.0,
                            "source": m.get("source", "unknown"),
                        }
                        for m in masks
                    ]
                except Exception as exc:
                    self.logger.debug(
                        "Gemma directed tracking SAM pass failed for %s: %s",
                        rec.get("frame_id", fp),
                        exc,
                    )
            fj["gemma_tracking"] = {
                "scene_type": scene_type,
                "tracking_priority": tracking_priority,
                "detections": tracking_dets,
                "sam_masks": sam_masks_summary,
            }
            rec["frame_facts_json"] = fj

        self.logger.info(
            "Gemma directed tracking pass complete: %d frames, scene=%s",
            len(frame_records),
            scene_type,
        )

    def _run_world_model_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Run world model on sliding windows of consecutive kept frames."""
        clip_size = settings.WORLD_MODEL_CLIP_FRAMES
        self.logger.info(
            "World model pass: %d frames, clip_size=%d",
            len(frame_records),
            clip_size,
        )
        for batch_start in range(0, len(frame_records), clip_size):
            batch = frame_records[batch_start : batch_start + clip_size]
            images = []
            for rec in batch:
                try:
                    images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))
            result = self.world_model.process_clip(images)
            mid = len(batch) // 2
            rec = batch[mid]
            fj = rec.get("frame_facts_json") or {}
            if isinstance(fj, dict):
                fj.update(result)
                rec["frame_facts_json"] = fj
        self.logger.info("World model pass complete")

    def _run_al_rssm_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Compute active learning scores and assign al_tags.

        Integrates DreamerV3-inspired RSSM temporal surprise scoring
        (Romero et al., ICRA 2026) with the existing DINOv3-dist + caption
        confidence signal.

        Step 1 — collect per-frame data:
            - CLIP embeddings (stored temporarily in frame_records["_clip_embed"])
            - caption confidences (from Florence/Gemma captioning pass)

        Step 2 — RSSM temporal surprise (when DREAMER_ENABLED=true):
            Train a lightweight RSSM online on the mission's CLIP sequence,
            then compute surprise_k = cosine_distance(predicted_z̃_k, actual_z_k).
            Stores rssm_surprise in frame_facts_json["rssm"].

        Step 3 — active learning scoring:
            With RSSM:    al_score = 0.35*dino + 0.25*(1-conf) + 0.40*surprise
            Without RSSM: al_score = 0.60*dino + 0.40*(1-conf)

        Step 4 — strip temporary _clip_embed fields before DB write.
        """
        from selfsuvis.pipeline.analysis.active_learning import (
            assign_al_tags,
            dino_distances_from_centroids,
            fit_kmeans,
        )

        n = len(frame_records)
        self.logger.info("AL+RSSM pass: %d frames", n)

        clip_embeds_list = []
        valid_indices = []
        for i, rec in enumerate(frame_records):
            emb = rec.pop("_clip_embed", None)
            if emb is not None:
                clip_embeds_list.append(emb.astype(np.float32))
                valid_indices.append(i)

        caption_confidences = [float(rec.get("caption_confidence") or 0.5) for rec in frame_records]

        dino_dists = [0.5] * n
        if clip_embeds_list:
            try:
                all_embeds = np.stack(clip_embeds_list)
                kmeans = fit_kmeans(all_embeds, n_clusters=min(20, len(clip_embeds_list)))
                centroid_dists = dino_distances_from_centroids(all_embeds, kmeans.cluster_centers_)
                for rank, idx in enumerate(valid_indices):
                    dino_dists[idx] = float(centroid_dists[rank])
            except Exception as exc:
                self.logger.debug("AL k-means failed (%s) — using uniform dino_dists", exc)

        rssm_surprises: list[float] | None = None
        if self.rssm_embedder is not None and clip_embeds_list:
            try:
                t0 = time.time()
                all_embeds = np.stack(clip_embeds_list)
                rssm_result = self.rssm_embedder.encode_sequence(all_embeds)
                surprises_arr = rssm_result["surprise_scores"]
                method = rssm_result.get("method", "unknown")
                elapsed = time.time() - t0
                self.logger.info(
                    "RSSM pass complete: method=%s hidden=%d latent=%d elapsed=%.2fs",
                    method,
                    rssm_result["hidden_dim"],
                    rssm_result["latent_dim"],
                    elapsed,
                )
                rssm_surprises = [0.5] * n
                for rank, idx in enumerate(valid_indices):
                    rssm_surprises[idx] = float(surprises_arr[rank])
                for rank, idx in enumerate(valid_indices):
                    rec = frame_records[idx]
                    fj = rec.get("frame_facts_json") or {}
                    if not isinstance(fj, dict):
                        fj = {}
                    rssm_entry: dict[str, Any] = {
                        "surprise_score": float(surprises_arr[rank]),
                        "method": method,
                        "model": rssm_result.get("model", "RSSMEmbedder"),
                    }
                    if settings.DREAMER_STORE_TEMPORAL and "recurrent_states" in rssm_result:
                        rssm_entry["recurrent_state"] = rssm_result["recurrent_states"][
                            rank
                        ].tolist()
                    fj["rssm"] = rssm_entry
                    rec["frame_facts_json"] = fj
            except Exception as exc:
                self.logger.warning("RSSM temporal surprise failed (%s) — skipping", exc)

        scores, tags = assign_al_tags(
            dino_dists,
            caption_confidences,
            rssm_surprises=rssm_surprises,
        )
        for rec, score, tag in zip(frame_records, scores, tags):
            rec["al_score"] = float(score)
            rec["al_tag"] = tag

        needs = tags.count("needs_annotation")
        novel = tags.count("novel")
        formula = "rssm+dino+caption" if rssm_surprises is not None else "dino+caption"
        self.logger.info(
            "AL tagging complete: needs_annotation=%d novel=%d none=%d formula=%s",
            needs,
            novel,
            n - needs - novel,
            formula,
        )
