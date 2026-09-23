"""Provider-neutral review of semantic claims the geometry did not settle.

A local Qwen-VL adapter is the offline option. A remote adapter calls the
Responses API with a strict JSON schema and image inputs. Timeout, refusal,
and malformed output raise ``ReviewError`` so the verifier can fail closed.
The default provider is unavailable and does not call a model.
"""

import base64
import json
import os
from typing import Protocol

from pydantic import ValidationError

from selfsuvis.pipeline.analysis4d.schemas import ContractModel, VerificationStatus

DEFAULT_QWEN_VL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_REMOTE_MODEL = "gpt-6-astra"
_STATUSES = ("accepted", "rejected", "uncertain", "corrected")

REVIEW_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "proposal_id",
                    "status",
                    "reasons",
                    "corrected_text",
                    "corrected_predicate",
                ],
                "properties": {
                    "proposal_id": {"type": "string"},
                    "status": {"type": "string", "enum": list(_STATUSES)},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                    "corrected_text": {"type": ["string", "null"]},
                    "corrected_predicate": {"type": ["string", "null"]},
                },
            },
        }
    },
}


class ReviewError(ValueError):
    """The reviewer timed out, refused, or returned output that is not the schema."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


class ReviewPacket(ContractModel):
    """One unresolved semantic claim plus measurements and counter-evidence."""

    proposal_id: str
    claim_kind: str
    text: str
    subject_id: str | None = None
    predicate: str | None = None
    object_id: str | None = None
    measurements: list[str]
    counter_evidence: list[str]
    image_refs: list[str] = []


class ReviewDecision(ContractModel):
    """Schema-constrained judgment. Unknown fields fail closed."""

    proposal_id: str
    status: VerificationStatus
    reasons: list[str] = []
    corrected_text: str | None = None
    corrected_predicate: str | None = None


class ReviewProvider(Protocol):
    """One multimodal reviewer."""

    failed: bool

    def review(self, packets: list[ReviewPacket]) -> list[ReviewDecision]:
        """Judge ``packets``. Timeout, refusal, and malformed output raise."""


class UnavailableReview:
    """No reviewer is configured. Semantic claims stay uncertain."""

    failed = True
    detail = "multimodal review is not configured"

    def review(self, packets: list[ReviewPacket]) -> list[ReviewDecision]:
        del packets
        return []


class ScriptedReview:
    """Return fixed decisions, or raise ``ReviewError`` for fail-closed tests."""

    failed = False

    def __init__(
        self,
        decisions: list[ReviewDecision] | None = None,
        error: ReviewError | None = None,
    ):
        self._decisions = list(decisions or [])
        self._error = error

    def review(self, packets: list[ReviewPacket]) -> list[ReviewDecision]:
        del packets
        if self._error is not None:
            raise self._error
        return list(self._decisions)


class QwenVlReview:
    """Local Qwen-VL family adapter. A missing model leaves ``failed`` set."""

    def __init__(self, model_id: str | None = None, timeout_sec: float | None = None):
        self.model_id = model_id or os.environ.get("ANALYSIS4D_QWEN_VL_MODEL", DEFAULT_QWEN_VL)
        self.timeout_sec = timeout_sec if timeout_sec is not None else _timeout_sec()
        self.failed = False
        self.detail = ""
        self._model = None
        self._processor = None
        try:
            self._model, self._processor = _load_qwen_vl(self.model_id)
        except Exception as exc:
            self.failed = True
            self.detail = _safe_detail(exc)

    def review(self, packets: list[ReviewPacket]) -> list[ReviewDecision]:
        if self.failed or self._model is None or self._processor is None:
            raise ReviewError("malformed", self.detail or "Qwen-VL is unavailable")
        prompt = _prompt(packets)
        images = [ref for packet in packets for ref in packet.image_refs]
        try:
            text = _generate_qwen(self._model, self._processor, prompt, images, self.timeout_sec)
        except TimeoutError as exc:
            raise ReviewError("timeout", "Qwen-VL review timed out") from exc
        except ReviewError:
            raise
        except Exception as exc:
            raise ReviewError("malformed", _safe_detail(exc)) from exc
        return parse_review_payload(text)


class RemoteJsonReview:
    """Strict JSON review through the Responses API. Images are sent, not video."""

    def __init__(self, client=None, model: str | None = None, timeout_sec: float | None = None):
        self._client = client
        self.model = model or os.environ.get("ANALYSIS4D_REVIEW_REMOTE_MODEL", DEFAULT_REMOTE_MODEL)
        self.timeout_sec = timeout_sec if timeout_sec is not None else _timeout_sec()
        self.failed = False
        self.detail = ""

    def review(self, packets: list[ReviewPacket]) -> list[ReviewDecision]:
        client = self._client if self._client is not None else _openai_client(self.timeout_sec)
        try:
            response = client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": _remote_content(packets)}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "claim_review",
                        "strict": True,
                        "schema": REVIEW_JSON_SCHEMA,
                    }
                },
                timeout=self.timeout_sec,
            )
        except TimeoutError as exc:
            raise ReviewError("timeout", "remote review timed out") from exc
        except ReviewError:
            raise
        except Exception as exc:
            name = exc.__class__.__name__
            if "Timeout" in name:
                raise ReviewError("timeout", "remote review timed out") from exc
            raise ReviewError("malformed", _safe_detail(exc)) from exc
        if _refused(response):
            raise ReviewError("refusal", "remote review refused")
        text = getattr(response, "output_text", None) or ""
        if not text:
            raise ReviewError("malformed", "remote review returned no text")
        return parse_review_payload(text)


def load_review_provider(name: str | None = None) -> ReviewProvider:
    """Return the configured reviewer.

    Args:
        name: ``unavailable``, ``qwen``, or ``remote``. The default is
            ``ANALYSIS4D_REVIEW_PROVIDER`` or ``unavailable``.

    Returns:
        A reviewer. An unknown name is the unavailable reviewer.
    """
    chosen = (
        name if name is not None else os.environ.get("ANALYSIS4D_REVIEW_PROVIDER", "unavailable")
    )
    if chosen in {"", "unavailable", "off"}:
        return UnavailableReview()
    if chosen == "qwen":
        return QwenVlReview()
    if chosen == "remote":
        return RemoteJsonReview()
    return UnavailableReview()


def parse_review_payload(text: str) -> list[ReviewDecision]:
    """Parse a strict review object. Unknown fields fail closed.

    Args:
        text: Model text. A fenced JSON block or a top-level list is accepted.

    Returns:
        Decisions. An empty list is valid and leaves claims uncertain.

    Raises:
        ReviewError: The payload is not schema-valid.
    """
    payload = _extract_json(text)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ReviewError("malformed", "malformed review JSON") from exc
    if isinstance(value, list):
        rows = value
    elif (
        isinstance(value, dict)
        and set(value) <= {"decisions"}
        and isinstance(value.get("decisions"), list)
    ):
        rows = value["decisions"]
    else:
        raise ReviewError("malformed", "review output must be a decisions object")
    decisions: list[ReviewDecision] = []
    allowed = {
        "proposal_id",
        "status",
        "reasons",
        "corrected_text",
        "corrected_predicate",
    }
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ReviewError("malformed", f"review decision {index} is not an object")
        extra = set(item) - allowed
        if extra:
            raise ReviewError("malformed", f"unknown_field: {sorted(extra)}")
        try:
            decisions.append(ReviewDecision.model_validate(item))
        except ValidationError as exc:
            raise ReviewError("malformed", f"malformed review decision {index}") from exc
    return decisions


def _timeout_sec() -> float:
    raw = os.environ.get("ANALYSIS4D_REVIEW_TIMEOUT_SEC", "30")
    try:
        value = float(raw)
    except ValueError:
        return 30.0
    return value if value > 0 else 30.0


def _prompt(packets: list[ReviewPacket]) -> str:
    body = [packet.model_dump(mode="json") for packet in packets]
    return (
        "Judge only these unresolved semantic claims. Return a JSON object "
        '{"decisions":[{"proposal_id":"...","status":"accepted|rejected|uncertain|corrected",'
        '"reasons":["..."],"corrected_text":null,"corrected_predicate":null}]}. '
        "Do not add fields. Geometry measurements override visual impression. "
        "Use uncertain when the images and the measurements disagree.\n"
        + json.dumps(body, sort_keys=True)
    )


def _remote_content(packets: list[ReviewPacket]) -> list[dict]:
    content: list[dict] = [{"type": "input_text", "text": _prompt(packets)}]
    seen: set[str] = set()
    for packet in packets:
        for ref in packet.image_refs:
            if ref in seen:
                continue
            seen.add(ref)
            encoded = _data_url(ref)
            if encoded is None:
                continue
            content.append({"type": "input_image", "image_url": encoded})
    return content


def _data_url(path: str) -> str | None:
    from pathlib import Path

    file_path = Path(path)
    if not file_path.is_file():
        return None
    payload = file_path.read_bytes()
    if len(payload) > 4_000_000:
        return None
    encoded = base64.standard_b64encode(payload).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _extract_json(text: str) -> str:
    fence = "```"
    if fence in text:
        start = text.find(fence)
        rest = text[start + len(fence) :]
        if rest.startswith("json"):
            rest = rest[4:]
        end = rest.find(fence)
        if end >= 0:
            return rest[:end].strip()
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    start_list = text.find("[")
    end_list = text.rfind("]")
    if start_obj >= 0 and end_obj > start_obj and (start_list < 0 or start_obj < start_list):
        return text[start_obj : end_obj + 1]
    if start_list >= 0 and end_list > start_list:
        return text[start_list : end_list + 1]
    return text.strip()


def _load_qwen_vl(model_id: str):
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model_cls = None
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_cls = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration

        model_cls = Qwen2VLForConditionalGeneration
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = model_cls.from_pretrained(model_id, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, processor


def _generate_qwen(model, processor, prompt: str, images: list[str], timeout_sec: float) -> str:
    import concurrent.futures
    from pathlib import Path

    existing = [path for path in images if Path(path).is_file()]

    def _run() -> str:
        import torch

        content: list[dict] = []
        for path in existing:
            content.append({"type": "image", "path": path})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content or [{"type": "text", "text": prompt}]}]
        templated = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs = {"text": [templated], "return_tensors": "pt", "padding": True}
        if existing:
            kwargs["images"] = existing
        inputs = processor(**kwargs)
        if torch.cuda.is_available():
            inputs = {
                key: value.to("cuda") if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=256)
        decoded = processor.batch_decode(output, skip_special_tokens=True)
        return decoded[0] if decoded else ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError("Qwen-VL review timed out") from exc


def _openai_client(timeout_sec: float):
    from openai import OpenAI

    return OpenAI(timeout=timeout_sec)


def _refused(response) -> bool:
    output = getattr(response, "output", None) or []
    for item in output:
        if getattr(item, "type", None) == "refusal":
            return True
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return True
    return False


def _safe_detail(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    lowered = text.lower()
    if "hf_token" in lowered or "api key" in lowered or "bearer" in lowered:
        return "review model is not visible; check the provider credentials"
    return text[:240] or exc.__class__.__name__
