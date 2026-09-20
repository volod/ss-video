"""Perception pass mixin: ASR, OCR, Qwen, UniDrive, Depth, RF analysis."""

import json
import os
from typing import Any

from PIL import Image

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.utils import ensure_dir
from selfsuvis.pipeline.media import extract_audio, map_subtitles_to_frames


class _PerceptionMixin:
    def _run_asr_pass(self, video_path: str, frame_records: list[dict[str, Any]]) -> None:
        """Transcribe video audio and map subtitle segments to frame timestamps."""
        self.logger.info("ASR pass: transcribing audio from %s", video_path)
        audio_dir = settings.ASR_AUDIO_DIR
        ensure_dir(audio_dir)
        wav_path = extract_audio(video_path, audio_dir)
        if not wav_path:
            self.logger.info("ASR pass: no audio track — skipping")
            return
        segments = self.asr_model.transcribe(wav_path)
        if not segments:
            self.logger.info("ASR pass: no transcript segments produced")
            return
        timestamps = [rec["t_sec"] for rec in frame_records]
        subtitle_map = map_subtitles_to_frames(
            segments, timestamps, window_sec=settings.ASR_SUBTITLE_WINDOW_SEC
        )
        for rec in frame_records:
            text = subtitle_map.get(rec["t_sec"])
            if text:
                rec["subtitle_text"] = text
        subtitled = sum(1 for r in frame_records if r.get("subtitle_text"))
        self.logger.info(
            "ASR pass complete: %d/%d frames have subtitle text", subtitled, len(frame_records)
        )

    def _run_ocr_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Run OCR on kept frames and store text in frame_facts_json + ocr_text."""
        self.logger.info("OCR pass: %d frames", len(frame_records))
        for batch_start in range(0, len(frame_records), settings.OCR_BATCH_SIZE):
            batch = frame_records[batch_start : batch_start + settings.OCR_BATCH_SIZE]
            images = []
            for rec in batch:
                try:
                    images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))
            results = self.ocr_model.extract_text_batch(images)
            for rec, res in zip(batch, results):
                text = res.get("ocr_text", "") or ""
                rec["ocr_text"] = text if text else None
                if text:
                    fj = rec.get("frame_facts_json") or {}
                    if isinstance(fj, dict):
                        fj["ocr_text"] = text
                        rec["frame_facts_json"] = fj
        ocr_found = sum(1 for r in frame_records if r.get("ocr_text"))
        self.logger.info(
            "OCR pass complete: %d/%d frames contain text", ocr_found, len(frame_records)
        )

    def _run_qwen_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Run Qwen2.5-VL structured extraction, enriched with subtitle+OCR context."""
        if not self.qwen_model or not self.qwen_model.is_enabled():
            return
        self.logger.info("Qwen2.5-VL Phase 2 pass: %d frames", len(frame_records))
        for rec in frame_records:
            try:
                img = Image.open(rec["frame_path"]).convert("RGB")
            except Exception:
                fj = rec.get("frame_facts_json") or {}
                fj["file_error"] = True
                rec["frame_facts_json"] = fj
                continue
            subtitle = rec.get("subtitle_text")
            ocr = rec.get("ocr_text")
            qwen_result = self.qwen_model.extract_frame_facts(
                img, subtitle_text=subtitle, ocr_text=ocr
            )
            existing = rec.get("frame_facts_json")
            if isinstance(existing, dict) and isinstance(qwen_result, dict):
                merged = {**existing, **qwen_result}
                rec["frame_facts_json"] = merged
            else:
                rec["frame_facts_json"] = qwen_result
        self.logger.info("Qwen2.5-VL pass complete")

    def _run_unidrive_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Run UniDriveVLA expert analysis on a sparse sample and store results."""
        if not self.unidrive_model or not self.unidrive_model.is_enabled():
            return
        max_frames = max(1, int(getattr(settings, "UNIDRIVE_MAX_FRAMES", 24) or 24))
        sample_step = max(1, len(frame_records) // max_frames)
        sampled = frame_records[::sample_step][:max_frames]
        self.logger.info("UniDriveVLA pass: %d sampled frames", len(sampled))
        for rec in sampled:
            try:
                img = Image.open(rec["frame_path"]).convert("RGB")
            except Exception:
                fj = rec.get("frame_facts_json") or {}
                if isinstance(fj, dict):
                    fj["unidrive_vla"] = {"file_error": True}
                    rec["frame_facts_json"] = fj
                continue
            existing = rec.get("frame_facts_json") or {}
            extra_context = ""
            if isinstance(existing, dict) and existing:
                try:
                    extra_context = json.dumps(existing, ensure_ascii=True, sort_keys=True)[:2000]
                except Exception:
                    extra_context = ""
            result = self.unidrive_model.analyze_frame(
                img,
                subtitle_text=rec.get("subtitle_text"),
                ocr_text=rec.get("ocr_text"),
                extra_context=extra_context,
            )
            fj = rec.get("frame_facts_json") or {}
            if isinstance(fj, dict):
                fj["unidrive_vla"] = result
                rec["frame_facts_json"] = fj
        self.logger.info("UniDriveVLA pass complete")

    def _summarize_unidrive_records(self, frame_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarise UniDriveVLA outputs for worker/job status reporting."""
        analysed = 0
        high_risk = 0
        agreement_counts: dict[str, int] = {}
        for rec in frame_records:
            facts = rec.get("frame_facts_json") or {}
            if not isinstance(facts, dict):
                continue
            uv = facts.get("unidrive_vla")
            if not isinstance(uv, dict) or uv.get("service_unavailable") or uv.get("parse_error"):
                continue
            analysed += 1
            risk = (uv.get("understanding") or {}).get("risk_level", "unknown")
            if risk == "high":
                high_risk += 1
            agreement = (uv.get("mixture_of_experts") or {}).get("expert_agreement", "unknown")
            agreement_counts[agreement] = agreement_counts.get(agreement, 0) + 1
        return {
            "analysed_frames": analysed,
            "high_risk_frames": high_risk,
            "expert_agreement": agreement_counts,
        }

    def _run_depth_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Estimate monocular depth and store percentiles in frame_facts_json."""
        self.logger.info("Depth estimation pass: %d frames", len(frame_records))
        for rec in frame_records:
            try:
                img = Image.open(rec["frame_path"]).convert("RGB")
            except Exception:
                continue
            depth_result = self.depth_model.estimate(img)
            fj = rec.get("frame_facts_json") or {}
            if isinstance(fj, dict):
                fj.update(depth_result)
                rec["frame_facts_json"] = fj
        self.logger.info("Depth pass complete")

    def _run_rf_analysis_pass(self, video_path: str, frame_records: list[dict[str, Any]]) -> None:
        """Analyze IQ sidecar (or audio proxy) and write RF metrics to frame_facts_json.

        Stores ``frame_facts_json["rf_signal"]`` with SNR, spectral flatness,
        occupied bandwidth, peak frequency ratio, and optionally modulation class.
        If no IQ sidecar is present the analyzer falls back to the audio track
        extracted by the ASR pass (reuses the WAV in ASR_AUDIO_DIR if present).
        """
        base = os.path.splitext(os.path.basename(video_path))[0]
        audio_dir = settings.ASR_AUDIO_DIR
        audio_wav = os.path.join(audio_dir, f"{base}.wav")
        audio_wav = audio_wav if os.path.isfile(audio_wav) else None

        timestamps = [rec["t_sec"] for rec in frame_records]
        results = self.rf_analyzer.analyze_video(video_path, timestamps, audio_wav_path=audio_wav)

        for rec, res in zip(frame_records, results):
            if not res:
                continue
            fj = rec.get("frame_facts_json") or {}
            if isinstance(fj, dict):
                fj.update(res)
                rec["frame_facts_json"] = fj
