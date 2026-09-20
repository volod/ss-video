"""Unit tests for VideoIndexer enrichment pass methods.

Covers the pass-level logic in pipeline/indexer.py:
- _run_asr_pass: subtitle merging into frame_records
- _run_florence_pass: caption/confidence written, OOM fallback, Qdrant set_payload
- _run_ocr_pass: ocr_text + frame_facts_json merge, batching
- _run_qwen_pass: Qwen merge with existing frame_facts_json keys
- _run_depth_pass: depth result merge into frame_facts_json
- _run_detection_pass: detection result merge, batching
- _run_segmentation_pass: SAM auto-mask summary + label assignment
- _run_world_model_pass: middle-frame assignment, window logic

All tests run without GPU, Docker, or real model weights.
VideoIndexer.__init__ is bypassed via object.__new__ + direct attribute injection.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_records(n: int, tmp_path, *, base_t: float = 0.0) -> list[dict[str, Any]]:
    """Return n minimal frame_record dicts with valid PNGs under tmp_path."""
    records = []
    for i in range(n):
        p = tmp_path / f"frame_{i}.png"
        Image.new("RGB", (64, 64), color=(i * 20, 50, 100)).save(str(p))
        records.append(
            {
                "id": f"m:{i}:{int((base_t + i) * 1000)}",
                "frame_path": str(p),
                "t_sec": base_t + float(i),
                "segment_id": i,
                "caption": None,
                "caption_confidence": None,
                "caption_model": None,
                "subtitle_text": None,
                "ocr_text": None,
                "al_score": None,
                "al_tag": "none",
                "qdrant_id": f"qid-{i}",
            }
        )
    return records


def _make_indexer() -> Any:
    """Build a VideoIndexer shell without calling __init__ (no GPU/model loads)."""
    import selfsuvis.pipeline.workflows.indexer as idx_module

    obj = object.__new__(idx_module.VideoIndexer)
    obj.logger = MagicMock()
    obj.store = MagicMock()
    obj.store.collection = "test_collection"
    obj.clip_model = MagicMock()
    obj.dino_model = None
    obj._florence_model = MagicMock()
    obj.qwen_model = None
    obj.asr_model = None
    obj.ocr_model = None
    obj.depth_model = None
    obj.detection_model = None
    obj.world_model = None
    obj.yolo_detector = None
    obj.sam_predictor = None
    obj.segmentation_predictor = None
    obj.enable_tiles = False
    obj.phash_lru = MagicMock()
    obj.recent_index = MagicMock()
    return obj


# ── ASR pass ──────────────────────────────────────────────────────────────────


class TestRunASRPass:
    def test_asr_subtitles_written_to_matching_frames(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "ASR_AUDIO_DIR", str(tmp_path))

        fake_asr = MagicMock()
        fake_asr.is_enabled.return_value = True
        fake_asr.transcribe.return_value = [
            {"text": "target spotted", "timestamp": (1.0, 3.0)},
        ]
        indexer.asr_model = fake_asr

        records = _make_records(3, tmp_path)  # t_sec = 0, 1, 2
        wav = str(tmp_path / "audio.wav")

        with (
            patch(
                "selfsuvis.pipeline.workflows.indexer._perception.extract_audio", return_value=wav
            ),
            patch(
                "selfsuvis.pipeline.workflows.indexer._perception.map_subtitles_to_frames",
                return_value={1.0: "target spotted", 2.0: "target spotted"},
            ),
            patch("selfsuvis.pipeline.workflows.indexer._perception.ensure_dir"),
        ):
            indexer._run_asr_pass("/fake/video.mp4", records)

        assert records[0]["subtitle_text"] is None
        assert records[1]["subtitle_text"] == "target spotted"
        assert records[2]["subtitle_text"] == "target spotted"

    def test_asr_skips_when_no_audio_track(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "ASR_AUDIO_DIR", str(tmp_path))

        fake_asr = MagicMock()
        fake_asr.is_enabled.return_value = True
        indexer.asr_model = fake_asr

        records = _make_records(2, tmp_path)

        with (
            patch(
                "selfsuvis.pipeline.workflows.indexer._perception.extract_audio", return_value=None
            ),
            patch("selfsuvis.pipeline.workflows.indexer._perception.ensure_dir"),
        ):
            indexer._run_asr_pass("/fake/video.mp4", records)

        fake_asr.transcribe.assert_not_called()
        assert all(r["subtitle_text"] is None for r in records)

    def test_asr_skips_when_transcribe_returns_empty(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "ASR_AUDIO_DIR", str(tmp_path))

        fake_asr = MagicMock()
        fake_asr.is_enabled.return_value = True
        fake_asr.transcribe.return_value = []
        indexer.asr_model = fake_asr

        records = _make_records(2, tmp_path)
        wav = str(tmp_path / "audio.wav")

        with (
            patch(
                "selfsuvis.pipeline.workflows.indexer._perception.extract_audio", return_value=wav
            ),
            patch("selfsuvis.pipeline.workflows.indexer._perception.ensure_dir"),
        ):
            indexer._run_asr_pass("/fake/video.mp4", records)

        assert all(r["subtitle_text"] is None for r in records)


# ── Florence pass ─────────────────────────────────────────────────────────────


class TestRunFlorencePass:
    def _make_florence_model(self, captions: list[tuple]) -> MagicMock:
        """Return a fake FlorenceModel with given (caption, confidence) pairs."""
        m = MagicMock()
        m.model_tag = "florence-2-large"
        m.caption_batch.return_value = captions
        return m

    def test_caption_and_confidence_written_to_records(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "FLORENCE_BATCH_SIZE", 16)
        indexer._florence_model = self._make_florence_model(
            [
                ("five trucks on a mountain road", 0.87),
                ("empty road, clear sky", 0.91),
            ]
        )

        records = _make_records(2, tmp_path)
        indexer._run_florence_pass(records)

        assert records[0]["caption"] == "five trucks on a mountain road"
        assert records[0]["caption_confidence"] == pytest.approx(0.87)
        assert records[0]["caption_model"] == "florence-2-large"
        assert records[1]["caption"] == "empty road, clear sky"

    def test_oom_batch_failure_falls_back_to_empty_caption(self, tmp_path, monkeypatch):
        """If caption_batch raises (OOM), every frame in that batch gets ("", 0.5)."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "FLORENCE_BATCH_SIZE", 16)
        indexer._florence_model = MagicMock()
        indexer._florence_model.model_tag = "florence-2-large"
        indexer._florence_model.caption_batch.side_effect = RuntimeError("CUDA OOM")

        records = _make_records(3, tmp_path)
        indexer._run_florence_pass(records)

        for rec in records:
            assert rec["caption"] == ""
            assert rec["caption_confidence"] == pytest.approx(0.5)

    def test_bad_frame_path_uses_blank_image(self, tmp_path, monkeypatch):
        """Unreadable frame_path triggers blank PIL Image; captioning still runs."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "FLORENCE_BATCH_SIZE", 16)

        received_sizes: list[list] = []

        def fake_caption_batch(imgs, batch_size=16):
            received_sizes.append([img.size for img in imgs])
            return [("", 0.5)] * len(imgs)

        indexer._florence_model = MagicMock()
        indexer._florence_model.model_tag = "florence-2-large"
        indexer._florence_model.caption_batch.side_effect = fake_caption_batch

        records = _make_records(2, tmp_path)
        records[0]["frame_path"] = "/nonexistent/img.png"
        indexer._run_florence_pass(records)

        # First image was the fallback blank (224×224)
        assert received_sizes[0][0] == (224, 224)

    def test_florence_respects_batch_size(self, tmp_path, monkeypatch):
        """With FLORENCE_BATCH_SIZE=2 and 5 frames, caption_batch called 3 times."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "FLORENCE_BATCH_SIZE", 2)

        call_sizes: list[int] = []

        def fake_caption_batch(imgs, batch_size=2):
            call_sizes.append(len(imgs))
            return [("cap", 0.8)] * len(imgs)

        indexer._florence_model = MagicMock()
        indexer._florence_model.model_tag = "florence-2-large"
        indexer._florence_model.caption_batch.side_effect = fake_caption_batch

        records = _make_records(5, tmp_path)
        indexer._run_florence_pass(records)

        assert call_sizes == [2, 2, 1]

    def test_set_payload_called_per_frame(self, tmp_path, monkeypatch):
        """_set_caption_payload must call store.client.set_payload once per frame."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "FLORENCE_BATCH_SIZE", 16)
        indexer._florence_model = self._make_florence_model(
            [("a caption", 0.9), ("another", 0.8), ("third", 0.7)]
        )

        records = _make_records(3, tmp_path)
        indexer._run_florence_pass(records)

        assert indexer.store.client.set_payload.call_count == 3

    def test_set_payload_failure_does_not_raise(self, tmp_path, monkeypatch):
        """set_payload exception must be caught and logged — never propagated."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "FLORENCE_BATCH_SIZE", 16)
        indexer._florence_model = self._make_florence_model([("cap", 0.9)])
        indexer.store.client.set_payload.side_effect = RuntimeError("Qdrant down")

        records = _make_records(1, tmp_path)
        indexer._run_florence_pass(records)  # must not raise

        # caption still written to the record even though Qdrant failed
        assert records[0]["caption"] == "cap"

    def test_set_payload_skips_record_with_no_qdrant_id(self, tmp_path, monkeypatch):
        """A record without qdrant_id must not trigger a set_payload call."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "FLORENCE_BATCH_SIZE", 16)
        indexer._florence_model = self._make_florence_model([("cap", 0.9)])

        records = _make_records(1, tmp_path)
        records[0]["qdrant_id"] = None

        indexer._run_florence_pass(records)

        indexer.store.client.set_payload.assert_not_called()


# ── OCR pass ──────────────────────────────────────────────────────────────────


class TestRunOCRPass:
    def test_ocr_text_written_to_records(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "OCR_BATCH_SIZE", 8)
        indexer.ocr_model = MagicMock()
        indexer.ocr_model.extract_text_batch.return_value = [
            {"ocr_text": "STOP"},
            {"ocr_text": "CONVOY"},
        ]

        records = _make_records(2, tmp_path)
        indexer._run_ocr_pass(records)

        assert records[0]["ocr_text"] == "STOP"
        assert records[1]["ocr_text"] == "CONVOY"

    def test_ocr_empty_text_stored_as_none(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "OCR_BATCH_SIZE", 8)
        indexer.ocr_model = MagicMock()
        indexer.ocr_model.extract_text_batch.return_value = [{"ocr_text": ""}]

        records = _make_records(1, tmp_path)
        indexer._run_ocr_pass(records)

        assert records[0]["ocr_text"] is None

    def test_ocr_merges_into_frame_facts_json(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "OCR_BATCH_SIZE", 8)
        indexer.ocr_model = MagicMock()
        indexer.ocr_model.extract_text_batch.return_value = [{"ocr_text": "GRID 44N"}]

        records = _make_records(1, tmp_path)
        records[0]["frame_facts_json"] = {"depth": {"p50": 3.2}}

        indexer._run_ocr_pass(records)

        fj = records[0]["frame_facts_json"]
        assert fj["ocr_text"] == "GRID 44N"
        assert fj["depth"]["p50"] == 3.2  # pre-existing key preserved

    def test_ocr_respects_batch_size(self, tmp_path, monkeypatch):
        """With batch_size=2 and 5 records, extract_text_batch must be called 3 times."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "OCR_BATCH_SIZE", 2)

        call_sizes: list[int] = []

        def fake_batch(imgs):
            call_sizes.append(len(imgs))
            return [{"ocr_text": "x"}] * len(imgs)

        indexer.ocr_model = MagicMock()
        indexer.ocr_model.extract_text_batch.side_effect = fake_batch

        records = _make_records(5, tmp_path)
        indexer._run_ocr_pass(records)

        assert call_sizes == [2, 2, 1]

    def test_ocr_bad_image_falls_back_to_blank(self, tmp_path, monkeypatch):
        """A frame with an unreadable path should not crash the pass."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "OCR_BATCH_SIZE", 8)
        indexer.ocr_model = MagicMock()
        indexer.ocr_model.extract_text_batch.return_value = [{"ocr_text": ""}]

        records = _make_records(1, tmp_path)
        records[0]["frame_path"] = "/nonexistent/frame.png"

        indexer._run_ocr_pass(records)  # must not raise

        assert records[0]["ocr_text"] is None


# ── Qwen pass ─────────────────────────────────────────────────────────────────


class TestRunQwenPass:
    def test_qwen_result_stored_in_frame_facts_json(self, tmp_path):
        indexer = _make_indexer()
        indexer.qwen_model = MagicMock()
        indexer.qwen_model.is_enabled.return_value = True
        indexer.qwen_model.extract_frame_facts.return_value = {"vehicle_type": "truck"}

        records = _make_records(1, tmp_path)
        indexer._run_qwen_pass(records)

        assert records[0]["frame_facts_json"]["vehicle_type"] == "truck"

    def test_qwen_merges_with_existing_ocr_key(self, tmp_path):
        """Qwen result must not overwrite pre-existing OCR key unless Qwen also sets it."""
        indexer = _make_indexer()
        indexer.qwen_model = MagicMock()
        indexer.qwen_model.is_enabled.return_value = True
        indexer.qwen_model.extract_frame_facts.return_value = {"vehicle_type": "apc"}

        records = _make_records(1, tmp_path)
        records[0]["frame_facts_json"] = {"ocr_text": "GRID 44N"}

        indexer._run_qwen_pass(records)

        fj = records[0]["frame_facts_json"]
        assert fj["ocr_text"] == "GRID 44N"
        assert fj["vehicle_type"] == "apc"

    def test_qwen_passes_subtitle_and_ocr_context(self, tmp_path):
        indexer = _make_indexer()
        indexer.qwen_model = MagicMock()
        indexer.qwen_model.is_enabled.return_value = True
        indexer.qwen_model.extract_frame_facts.return_value = {}

        records = _make_records(1, tmp_path)
        records[0]["subtitle_text"] = "enemy contact"
        records[0]["ocr_text"] = "BEARING 270"

        indexer._run_qwen_pass(records)

        _, kwargs = indexer.qwen_model.extract_frame_facts.call_args
        assert kwargs["subtitle_text"] == "enemy contact"
        assert kwargs["ocr_text"] == "BEARING 270"

    def test_qwen_sets_file_error_on_bad_path(self, tmp_path):
        indexer = _make_indexer()
        indexer.qwen_model = MagicMock()
        indexer.qwen_model.is_enabled.return_value = True

        records = _make_records(1, tmp_path)
        records[0]["frame_path"] = "/nonexistent/img.png"

        indexer._run_qwen_pass(records)

        indexer.qwen_model.extract_frame_facts.assert_not_called()
        assert records[0]["frame_facts_json"].get("file_error") is True

    def test_qwen_skips_when_disabled(self, tmp_path):
        indexer = _make_indexer()
        indexer.qwen_model = MagicMock()
        indexer.qwen_model.is_enabled.return_value = False

        records = _make_records(2, tmp_path)
        indexer._run_qwen_pass(records)

        indexer.qwen_model.extract_frame_facts.assert_not_called()

    def test_qwen_skips_when_model_is_none(self, tmp_path):
        """qwen_model=None means the pass is a no-op."""
        indexer = _make_indexer()
        indexer.qwen_model = None

        records = _make_records(2, tmp_path)
        indexer._run_qwen_pass(records)  # must not raise

        assert "frame_facts_json" not in records[0]


# ── Depth pass ────────────────────────────────────────────────────────────────


class TestRunDepthPass:
    def test_depth_result_merged_into_frame_facts_json(self, tmp_path):
        indexer = _make_indexer()
        indexer.depth_model = MagicMock()
        indexer.depth_model.estimate.return_value = {"depth": {"percentiles": [1, 2, 3, 4, 5]}}

        records = _make_records(1, tmp_path)
        indexer._run_depth_pass(records)

        assert records[0]["frame_facts_json"]["depth"]["percentiles"] == [1, 2, 3, 4, 5]

    def test_depth_preserves_existing_facts(self, tmp_path):
        indexer = _make_indexer()
        indexer.depth_model = MagicMock()
        indexer.depth_model.estimate.return_value = {"depth": {"percentiles": [1, 2, 3, 4, 5]}}

        records = _make_records(1, tmp_path)
        records[0]["frame_facts_json"] = {"vehicle_type": "truck"}

        indexer._run_depth_pass(records)

        fj = records[0]["frame_facts_json"]
        assert fj["vehicle_type"] == "truck"
        assert "depth" in fj

    def test_depth_skips_unreadable_frame(self, tmp_path):
        """Bad frame path should not crash the pass; that record is skipped."""
        indexer = _make_indexer()
        indexer.depth_model = MagicMock()
        indexer.depth_model.estimate.return_value = {"depth": {"percentiles": [0] * 5}}

        records = _make_records(2, tmp_path)
        records[0]["frame_path"] = "/nonexistent/img.png"

        indexer._run_depth_pass(records)

        # Record 0 skipped — no frame_facts_json set
        assert "frame_facts_json" not in records[0]
        # Record 1 processed normally
        assert "depth" in records[1]["frame_facts_json"]

    def test_depth_called_once_per_frame(self, tmp_path):
        indexer = _make_indexer()
        indexer.depth_model = MagicMock()
        indexer.depth_model.estimate.return_value = {}

        records = _make_records(4, tmp_path)
        indexer._run_depth_pass(records)

        assert indexer.depth_model.estimate.call_count == 4


# ── Detection pass ────────────────────────────────────────────────────────────


class TestRunDetectionPass:
    def test_detections_merged_into_frame_facts_json(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "DETECTION_BATCH_SIZE", 8)
        indexer.detection_model = MagicMock()
        indexer.detection_model.detect_batch.return_value = [
            {"detections": [{"label": "vehicle", "confidence": 0.9}]}
        ]

        records = _make_records(1, tmp_path)
        indexer._run_detection_pass(records)

        assert records[0]["frame_facts_json"]["detections"][0]["label"] == "vehicle"

    def test_detection_batch_size_from_settings(self, tmp_path, monkeypatch):
        """detect_batch should be called with at most DETECTION_BATCH_SIZE images."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "DETECTION_BATCH_SIZE", 4)

        call_sizes: list[int] = []

        def fake_detect(imgs):
            call_sizes.append(len(imgs))
            return [{"detections": []}] * len(imgs)

        indexer.detection_model = MagicMock()
        indexer.detection_model.detect_batch.side_effect = fake_detect

        records = _make_records(10, tmp_path)
        indexer._run_detection_pass(records)

        assert call_sizes == [4, 4, 2]

    def test_detection_preserves_existing_depth_key(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "DETECTION_BATCH_SIZE", 8)
        indexer.detection_model = MagicMock()
        indexer.detection_model.detect_batch.return_value = [{"detections": []}]

        records = _make_records(1, tmp_path)
        records[0]["frame_facts_json"] = {"depth": {"percentiles": [1, 2, 3, 4, 5]}}

        indexer._run_detection_pass(records)

        assert records[0]["frame_facts_json"]["depth"]["percentiles"] == [1, 2, 3, 4, 5]

    def test_detection_bad_frame_uses_blank_image(self, tmp_path, monkeypatch):
        """Unreadable frame_path must not crash; blank image is passed instead."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "DETECTION_BATCH_SIZE", 8)

        received_sizes: list[tuple] = []

        def fake_detect(imgs):
            received_sizes.append(tuple(img.size for img in imgs))
            return [{"detections": []}] * len(imgs)

        indexer.detection_model = MagicMock()
        indexer.detection_model.detect_batch.side_effect = fake_detect

        records = _make_records(1, tmp_path)
        records[0]["frame_path"] = "/nonexistent/img.png"

        indexer._run_detection_pass(records)  # must not raise

        assert received_sizes == [((224, 224),)]


# ── Segmentation pass ────────────────────────────────────────────────────────


class TestRunSegmentationPass:
    def test_segmentation_summary_written_with_detection_labels(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "SEGMENTATION_MODEL", "facebook/sam2-hiera-small")
        monkeypatch.setattr(idx_module.settings, "SEGMENTATION_MAX_MASKS", 16)
        monkeypatch.setattr(idx_module.settings, "SEGMENTATION_MIN_AREA_NORM", 0.001)
        monkeypatch.setattr(idx_module.settings, "SEGMENTATION_POINTS_PER_SIDE", 8)

        indexer.segmentation_predictor = MagicMock()
        indexer.segmentation_predictor.generate_auto_masks.return_value = [
            {
                "bbox": [0, 0, 32, 32],
                "area": 1024,
                "area_norm": 0.25,
                "score": 0.91,
                "mask": None,
            },
            {
                "bbox": [32, 32, 16, 16],
                "area": 256,
                "area_norm": 0.0625,
                "score": 0.77,
                "mask": None,
            },
        ]

        records = _make_records(1, tmp_path)
        records[0]["frame_facts_json"] = {
            "detections": [
                {"label": "vehicle", "bbox_norm": [0.0, 0.0, 0.5, 0.5]},
            ]
        }

        indexer._run_segmentation_pass(records)

        segments = records[0]["frame_facts_json"]["segments"]
        assert segments["count"] == 2
        assert segments["label_counts"]["vehicle"] == 1
        assert segments["label_counts"]["unlabeled"] == 1
        assert segments["model"] == "facebook/sam2-hiera-small"

    def test_segmentation_filters_small_masks_and_caps_count(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "SEGMENTATION_MODEL", "facebook/sam2-hiera-small")
        monkeypatch.setattr(idx_module.settings, "SEGMENTATION_MAX_MASKS", 1)
        monkeypatch.setattr(idx_module.settings, "SEGMENTATION_MIN_AREA_NORM", 0.01)
        monkeypatch.setattr(idx_module.settings, "SEGMENTATION_POINTS_PER_SIDE", 8)

        indexer.segmentation_predictor = MagicMock()
        indexer.segmentation_predictor.generate_auto_masks.return_value = [
            {"bbox": [0, 0, 16, 16], "area": 256, "area_norm": 0.0625, "score": 0.4, "mask": None},
            {"bbox": [16, 16, 2, 2], "area": 4, "area_norm": 0.0009, "score": 0.9, "mask": None},
            {"bbox": [8, 8, 24, 24], "area": 576, "area_norm": 0.14, "score": 0.8, "mask": None},
        ]

        records = _make_records(1, tmp_path)
        indexer._run_segmentation_pass(records)

        segments = records[0]["frame_facts_json"]["segments"]
        assert segments["count"] == 1
        assert segments["max_area_norm"] == pytest.approx(0.14)


# ── World model pass ──────────────────────────────────────────────────────────


class TestRunWorldModelPass:
    def _make_world_model(self, result: dict[str, Any]) -> MagicMock:
        wm = MagicMock()
        wm.is_enabled.return_value = True
        wm.process_clip.return_value = result
        return wm

    def test_world_model_result_assigned_to_middle_frame_odd(self, tmp_path, monkeypatch):
        """clip_size=5 → middle index is 2."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "WORLD_MODEL_CLIP_FRAMES", 5)
        indexer.world_model = self._make_world_model({"world_model": {"embedding_dim": 768}})

        records = _make_records(5, tmp_path)
        indexer._run_world_model_pass(records)

        assert "world_model" in records[2]["frame_facts_json"]
        for i in [0, 1, 3, 4]:
            assert records[i].get("frame_facts_json", {}).get("world_model") is None

    def test_world_model_result_assigned_to_middle_frame_even(self, tmp_path, monkeypatch):
        """clip_size=4 → middle index is 2 (len//2)."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "WORLD_MODEL_CLIP_FRAMES", 4)
        indexer.world_model = self._make_world_model({"world_model": {"embedding_dim": 768}})

        records = _make_records(4, tmp_path)
        indexer._run_world_model_pass(records)

        assert "world_model" in records[2]["frame_facts_json"]

    def test_world_model_windows_are_non_overlapping(self, tmp_path, monkeypatch):
        """With clip_size=3 and 9 frames → 3 windows; process_clip called 3 times."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "WORLD_MODEL_CLIP_FRAMES", 3)
        wm = self._make_world_model({"world_model": {}})
        indexer.world_model = wm

        records = _make_records(9, tmp_path)
        indexer._run_world_model_pass(records)

        assert wm.process_clip.call_count == 3

    def test_world_model_partial_last_window(self, tmp_path, monkeypatch):
        """With clip_size=4 and 6 frames → windows of size 4 then 2; both get a middle frame."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "WORLD_MODEL_CLIP_FRAMES", 4)
        wm = self._make_world_model({"world_model": {}})
        indexer.world_model = wm

        records = _make_records(6, tmp_path)
        indexer._run_world_model_pass(records)

        # Window 0: frames 0-3, mid=2
        assert "world_model" in records[2]["frame_facts_json"]
        # Window 1: frames 4-5, mid=1 → global index 5
        assert "world_model" in records[5]["frame_facts_json"]

    def test_world_model_preserves_existing_depth(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "WORLD_MODEL_CLIP_FRAMES", 3)
        indexer.world_model = self._make_world_model({"world_model": {}})

        records = _make_records(3, tmp_path)
        records[1]["frame_facts_json"] = {"depth": {"percentiles": [1, 2, 3, 4, 5]}}

        indexer._run_world_model_pass(records)

        fj = records[1]["frame_facts_json"]
        assert fj["depth"]["percentiles"] == [1, 2, 3, 4, 5]
        assert "world_model" in fj

    def test_world_model_bad_frame_uses_blank_image(self, tmp_path, monkeypatch):
        """Unreadable frame_path must not crash the world model pass."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "WORLD_MODEL_CLIP_FRAMES", 3)
        wm = self._make_world_model({"world_model": {}})
        indexer.world_model = wm

        records = _make_records(3, tmp_path)
        records[0]["frame_path"] = "/nonexistent/img.png"

        indexer._run_world_model_pass(records)  # must not raise

        wm.process_clip.assert_called_once()
        images_passed = wm.process_clip.call_args[0][0]
        assert images_passed[0].size == (224, 224)


# ── Cross-pass isolation: failure in one pass leaves others intact ────────────


class TestPassIsolation:
    def test_failing_depth_does_not_corrupt_ocr_text(self, tmp_path, monkeypatch):
        """When depth returns depth_error (absorbed internally), ocr_text is untouched.

        DepthModel.estimate() absorbs its own exceptions and returns {"depth_error": True}
        rather than raising. The merged frame_facts_json must contain both keys.
        """
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "OCR_BATCH_SIZE", 8)

        indexer.ocr_model = MagicMock()
        indexer.ocr_model.extract_text_batch.return_value = [{"ocr_text": "CALL SIGN BRAVO"}]

        indexer.depth_model = MagicMock()
        indexer.depth_model.estimate.return_value = {"depth_error": True}

        records = _make_records(1, tmp_path)

        indexer._run_ocr_pass(records)
        indexer._run_depth_pass(records)  # depth returns error dict, does not raise

        assert records[0]["ocr_text"] == "CALL SIGN BRAVO"
        assert records[0]["frame_facts_json"]["depth_error"] is True

    def test_ocr_pass_does_not_overwrite_prior_frame_facts_keys(self, tmp_path, monkeypatch):
        """OCR only sets frame_facts_json['ocr_text']; other keys are left alone."""
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "OCR_BATCH_SIZE", 8)
        indexer.ocr_model = MagicMock()
        indexer.ocr_model.extract_text_batch.return_value = [{"ocr_text": "TARGET"}]

        records = _make_records(1, tmp_path)
        records[0]["frame_facts_json"] = {
            "vehicle_type": "mrap",
            "detections": [{"label": "person"}],
        }

        indexer._run_ocr_pass(records)

        fj = records[0]["frame_facts_json"]
        assert fj["vehicle_type"] == "mrap"
        assert fj["detections"][0]["label"] == "person"
        assert fj["ocr_text"] == "TARGET"

    def test_qwen_merge_does_not_clobber_depth_written_first(self, tmp_path):
        """Depth key written before Qwen pass must survive the Qwen dict merge."""
        indexer = _make_indexer()
        indexer.depth_model = MagicMock()
        indexer.depth_model.estimate.return_value = {"depth": {"percentiles": [1, 2, 3, 4, 5]}}
        indexer.qwen_model = MagicMock()
        indexer.qwen_model.is_enabled.return_value = True
        indexer.qwen_model.extract_frame_facts.return_value = {"vehicle_type": "ifv"}

        records = _make_records(1, tmp_path)

        indexer._run_depth_pass(records)
        indexer._run_qwen_pass(records)

        fj = records[0]["frame_facts_json"]
        assert fj["depth"]["percentiles"] == [1, 2, 3, 4, 5]
        assert fj["vehicle_type"] == "ifv"

    def test_qwen_bad_path_preserves_prior_ocr_data(self, tmp_path, monkeypatch):
        """When Qwen's frame_path is unreadable, prior OCR data must not be erased.

        Regression: indexer.py:505 previously overwrote frame_facts_json wholesale
        with {"file_error": True}, silently destroying any data written by prior passes.
        """
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "OCR_BATCH_SIZE", 8)

        indexer.ocr_model = MagicMock()
        indexer.ocr_model.extract_text_batch.return_value = [{"ocr_text": "GRID 44N"}]
        indexer.qwen_model = MagicMock()
        indexer.qwen_model.is_enabled.return_value = True

        records = _make_records(1, tmp_path)

        # OCR runs first and writes data
        indexer._run_ocr_pass(records)
        assert records[0]["ocr_text"] == "GRID 44N"
        assert records[0]["frame_facts_json"]["ocr_text"] == "GRID 44N"

        # Now Qwen fails because the path is bad
        records[0]["frame_path"] = "/nonexistent/img.png"
        indexer._run_qwen_pass(records)

        # Prior OCR data must survive the Qwen failure
        fj = records[0]["frame_facts_json"]
        assert fj["ocr_text"] == "GRID 44N", "OCR data was erased by Qwen file_error"
        assert fj["file_error"] is True


# ── YOLO+SAM pass ─────────────────────────────────────────────────────────────


class TestRunYoloSamPass:
    def test_unavailable_status_dict_does_not_crash(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "DETECTION_BATCH_SIZE", 8)
        indexer.yolo_detector = MagicMock()
        indexer.yolo_detector.model_id = "yolo11n"
        indexer.yolo_detector.detect_batch.return_value = [{"detection_unavailable": True}]
        indexer.sam_predictor = MagicMock()
        indexer.sam_predictor.is_available.return_value = True

        records = _make_records(1, tmp_path)
        indexer._run_yolo_sam_pass(records)

        assert records[0]["frame_facts_json"]["yolo_detections"] == []
        indexer.sam_predictor.predict_boxes.assert_not_called()

    def test_unwraps_detections_key_and_calls_sam(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "DETECTION_BATCH_SIZE", 8)
        det = {
            "label": "person",
            "confidence": 0.9,
            "bbox_norm": [0.1, 0.1, 0.2, 0.2],
            "priority": 1,
            "priority_label": "human",
            "mask_area_norm": None,
        }
        indexer.yolo_detector = MagicMock()
        indexer.yolo_detector.model_id = "yolo11n"
        indexer.yolo_detector.detect_batch.return_value = [{"detections": [det]}]
        indexer.sam_predictor = MagicMock()
        indexer.sam_predictor.is_available.return_value = True
        indexer.sam_predictor.predict_boxes.return_value = [{"area_norm": 0.04}]

        records = _make_records(1, tmp_path)
        indexer._run_yolo_sam_pass(records)

        assert records[0]["frame_facts_json"]["yolo_detections"][0]["mask_area_norm"] == 0.04
        indexer.sam_predictor.predict_boxes.assert_called_once()


class TestRunYoloSsgPass:
    def test_ssg_builds_graph_and_attaches_node_ids(self, tmp_path, monkeypatch):
        import selfsuvis.pipeline.workflows.indexer as idx_module

        indexer = _make_indexer()
        monkeypatch.setattr(idx_module.settings, "MAPS_DIR", str(tmp_path))

        records = _make_records(2, tmp_path)
        records[0]["global_pose_json"] = {"tx": 0.0, "ty": 0.0, "tz": 0.0}
        records[1]["global_pose_json"] = {"tx": 1.0, "ty": 0.0, "tz": 0.0}
        records[0]["frame_facts_json"] = {
            "yolo_detections": [
                {
                    "label": "truck",
                    "confidence": 0.91,
                    "bbox_norm": [0.1, 0.1, 0.4, 0.5],
                    "priority": 2,
                    "priority_label": "vehicle",
                    "mask_area_norm": 0.08,
                }
            ]
        }
        records[1]["frame_facts_json"] = {
            "yolo_detections": [
                {
                    "label": "truck",
                    "confidence": 0.88,
                    "bbox_norm": [0.12, 0.1, 0.42, 0.48],
                    "priority": 2,
                    "priority_label": "vehicle",
                    "mask_area_norm": 0.07,
                }
            ]
        }

        summary = indexer._run_yolo_ssg_pass(
            video_id="video-a",
            mission_id="mission-a",
            frame_records=records,
        )

        assert summary["node_count"] == 1
        assert summary["edge_count"] == 0
        assert summary["anchor_source"] == "enu"
        assert records[0]["frame_facts_json"]["semantic_graph_node_ids"]
        assert (
            records[1]["frame_facts_json"]["semantic_graph_node_ids"]
            == records[0]["frame_facts_json"]["semantic_graph_node_ids"]
        )
        assert (tmp_path / "mission-a" / "semantic_environment_graph.json").exists()
