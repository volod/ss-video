"""Caption pass mixin: Florence-2 and Gemma API captioning."""

from typing import Any

from PIL import Image

from selfsuvis.pipeline.core import settings


class _CaptionMixin:
    def _run_florence_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Caption all kept frames with Florence-2 and update Qdrant payloads.

        Loads each image from disk, runs caption_batch() in FLORENCE_BATCH_SIZE
        chunks, updates frame_records in-place, then pushes captions to Qdrant
        via set_payload in 128-frame batches.
        """
        self.logger.info(
            "Florence captioning pass: %d frames (batch_size=%d)",
            len(frame_records),
            settings.FLORENCE_BATCH_SIZE,
        )

        caption_model_tag = self.florence_model.model_tag

        for batch_start in range(0, len(frame_records), settings.FLORENCE_BATCH_SIZE):
            batch = frame_records[batch_start : batch_start + settings.FLORENCE_BATCH_SIZE]

            pil_images: list = []
            for rec in batch:
                try:
                    pil_images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    self.logger.warning(
                        "Florence: could not open %s; using blank image", rec["frame_path"]
                    )
                    pil_images.append(Image.new("RGB", (224, 224)))

            try:
                captions_and_confs = self.florence_model.caption_batch(
                    pil_images, batch_size=settings.FLORENCE_BATCH_SIZE
                )
            except Exception:
                self.logger.warning(
                    "Florence batch failed for frames %d–%d; using empty captions",
                    batch_start,
                    batch_start + len(batch) - 1,
                    exc_info=True,
                )
                captions_and_confs = [("", 0.5)] * len(batch)

            for rec, (caption, confidence) in zip(batch, captions_and_confs):
                rec["caption"] = caption
                rec["caption_confidence"] = confidence
                rec["caption_model"] = caption_model_tag

        self._set_caption_payload(frame_records)

    def _run_gemma_caption_pass(self, frame_records: list[dict[str, Any]]) -> None:
        """Caption frames via the Gemma sidecar API with Florence fallback.

        Strategy:
        - Rank all frames by absolute histogram-diff score (higher = more
          informative / more diverse).  Take the top GEMMA_MAX_CAPTION_FRAMES
          for Gemma; caption the rest with Florence.
        - Gemma frames are sent in chunks of GEMMA_CAPTION_CHUNK_SIZE with a
          50-second timeout and GEMMA_CAPTION_RETRIES retry.  On second failure
          the chunk falls back to Florence.
        - Every frame record gets caption, caption_confidence, caption_model set.
        """
        import httpx as _httpx

        max_gemma = settings.GEMMA_MAX_CAPTION_FRAMES
        chunk_size = settings.GEMMA_CAPTION_CHUNK_SIZE
        retries = settings.GEMMA_CAPTION_RETRIES
        timeout = 50.0
        api_url = settings.GEMMA_API_URL.rstrip("/")
        model = settings.GEMMA_API_MODEL
        endpoint = f"{api_url}/chat/completions"
        florence_tag = self.florence_model.model_tag
        gemma_tag = f"gemma-api:{model}"

        total = len(frame_records)
        scored = sorted(
            enumerate(frame_records),
            key=lambda iv: float(iv[1].get("hist_diff", 0.0) or 0.0),
            reverse=True,
        )
        gemma_indices = (
            {idx for idx, _ in scored[:max_gemma]} if max_gemma > 0 else set(range(total))
        )
        gemma_recs = [r for i, r in enumerate(frame_records) if i in gemma_indices]
        florence_recs = [r for i, r in enumerate(frame_records) if i not in gemma_indices]

        self.logger.info(
            "Gemma captioning pass: %d frames via Gemma API, %d via Florence fallback",
            len(gemma_recs),
            len(florence_recs),
        )

        def _caption_image_gemma(frame_path: str) -> tuple:
            """Return (caption, confidence) for one frame via Gemma API, or raise."""
            import base64 as _b64

            try:
                with open(frame_path, "rb") as _f:
                    img_b64 = _b64.b64encode(_f.read()).decode()
            except OSError:
                raise

            prompt = (
                "Describe this image in one concise sentence suitable for outdoor "
                "robotics scene understanding. Focus on terrain, objects, and activities."
            )
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 128,
                "temperature": 0.2,
            }
            resp = _httpx.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return text, 0.85

        def _caption_chunk_with_retry(chunk: list[dict[str, Any]]) -> None:
            """Caption a chunk of records in-place, retrying once then falling back to Florence."""
            for attempt in range(retries + 1):
                try:
                    for rec in chunk:
                        caption, conf = _caption_image_gemma(rec["frame_path"])
                        rec["caption"] = caption
                        rec["caption_confidence"] = conf
                        rec["caption_model"] = gemma_tag
                    return
                except Exception as exc:
                    if attempt < retries:
                        self.logger.debug(
                            "Gemma caption chunk attempt %d failed (%s) — retrying",
                            attempt + 1,
                            exc,
                        )
                    else:
                        self.logger.warning(
                            "Gemma caption chunk failed after %d attempt(s) (%s) — falling back to Florence",
                            retries + 1,
                            exc,
                        )
                        self._caption_records_with_florence(chunk, florence_tag)

        for chunk_start in range(0, len(gemma_recs), chunk_size):
            chunk = gemma_recs[chunk_start : chunk_start + chunk_size]
            _caption_chunk_with_retry(chunk)

        if florence_recs:
            self._caption_records_with_florence(florence_recs, florence_tag)

        self._set_caption_payload(frame_records)

    def _caption_records_with_florence(self, records: list[dict[str, Any]], model_tag: str) -> None:
        """Caption the given records in-place using the Florence model."""
        for batch_start in range(0, len(records), settings.FLORENCE_BATCH_SIZE):
            batch = records[batch_start : batch_start + settings.FLORENCE_BATCH_SIZE]
            pil_images: list = []
            for rec in batch:
                try:
                    pil_images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    pil_images.append(Image.new("RGB", (224, 224)))
            try:
                captions_and_confs = self.florence_model.caption_batch(
                    pil_images, batch_size=settings.FLORENCE_BATCH_SIZE
                )
            except Exception:
                self.logger.warning(
                    "Florence fallback batch failed for %d frames; using empty captions",
                    len(batch),
                    exc_info=True,
                )
                captions_and_confs = [("", 0.5)] * len(batch)
            for rec, (caption, confidence) in zip(batch, captions_and_confs):
                rec["caption"] = caption
                rec["caption_confidence"] = confidence
                rec["caption_model"] = model_tag

    def _set_caption_payload(self, frame_records: list[dict[str, Any]]) -> None:
        """Write caption into Qdrant point payloads (display-only in Phase 1).

        Qdrant set_payload applies a single payload dict to all listed points,
        so each distinct caption requires its own call. We bound the outer loop
        at 128-frame chunks for consistent log granularity and error reporting.
        """
        for batch_start in range(0, len(frame_records), 128):
            batch = frame_records[batch_start : batch_start + 128]
            failed = 0
            for rec in batch:
                qdrant_id = rec.get("qdrant_id")
                if qdrant_id is None:
                    continue
                try:
                    self.store.client.set_payload(
                        collection_name=self.store.collection,
                        payload={"caption": rec.get("caption", "")},
                        points=[qdrant_id],
                    )
                except Exception:
                    failed += 1
            if failed:
                self.logger.warning(
                    "Florence: %d/%d Qdrant set_payload calls failed in batch "
                    "starting at frame %d; DB has captions, backfill will sync Qdrant.",
                    failed,
                    len(batch),
                    batch_start,
                )
