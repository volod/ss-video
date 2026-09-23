"""Schema-constrained compact VLM proposals.

Training stays in ss-fusion. This module only parses a JSON list of claims.
A provider that is missing, refuses, or returns malformed JSON does not
block the deterministic graph.
"""

import json
from typing import Protocol

from pydantic import ValidationError

from selfsuvis.pipeline.analysis4d.geometry import relation_holds
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_PROPOSAL,
    GeometrySample,
    Proposal,
    VerificationStatus,
)

PROPOSAL_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


class VlmClaim(Proposal):
    """One model claim before it is assigned a mission id.

    ``proposal_id`` may be omitted. The runner fills it. Extra fields fail
    closed because ``Proposal`` forbids them.
    """

    schema_version: str = SCHEMA_PROPOSAL
    mission_id: str = "pending"
    proposal_id: str = "pending"
    verification_status: VerificationStatus = "uncertain"


class ProposalProvider(Protocol):
    """One schema-constrained proposal source."""

    failed: bool

    def propose(self, context: dict) -> list[VlmClaim]:
        """Return claims for ``context``. Malformed output raises ``ValueError``."""


class UnavailableVlm:
    """No model is configured. Callers keep the deterministic graph."""

    failed = True
    detail = "compact VLM is not configured"

    def propose(self, context: dict) -> list[VlmClaim]:
        del context
        return []


class ScriptedVlm:
    """Return a fixed claim list. Tests use this to keep rejected claims."""

    failed = False

    def __init__(self, claims: list[VlmClaim]):
        self._claims = list(claims)

    def propose(self, context: dict) -> list[VlmClaim]:
        del context
        return list(self._claims)


class CompactVlm:
    """Local SmolVLM adapter. A load or parse failure leaves ``failed`` set."""

    def __init__(self, model_id: str = PROPOSAL_MODEL_ID):
        self.model_id = model_id
        self.failed = False
        self.detail = ""
        self._model = None
        self._processor = None
        try:
            self._model, self._processor = _load_smolvlm(model_id)
        except Exception as exc:
            self.failed = True
            self.detail = _safe_detail(exc)

    def propose(self, context: dict) -> list[VlmClaim]:
        if self.failed or self._model is None or self._processor is None:
            raise ValueError(self.detail or "compact VLM is unavailable")
        prompt = _prompt(context)
        text = _generate(self._model, self._processor, prompt)
        return parse_vlm_payload(text)


def load_proposal_provider(name: str | None = None) -> ProposalProvider:
    """Return the configured proposal provider.

    Args:
        name: ``unavailable``, ``scripted``, or ``smolvlm``. The default is
            ``ANALYSIS4D_VLM_PROVIDER`` or ``unavailable``.

    Returns:
        A provider. An unknown name is the unavailable provider.
    """
    import os

    chosen = name if name is not None else os.environ.get("ANALYSIS4D_VLM_PROVIDER", "unavailable")
    if chosen in {"", "unavailable", "off"}:
        return UnavailableVlm()
    if chosen == "smolvlm":
        return CompactVlm()
    return UnavailableVlm()


def parse_vlm_payload(text: str) -> list[VlmClaim]:
    """Parse a JSON list of claims. Unknown fields and non-lists fail closed.

    Args:
        text: Model text. A fenced ```json block is accepted.

    Returns:
        Validated claims.

    Raises:
        ValueError: The payload is not a JSON list of schema-valid claims.
    """
    payload = _extract_json(text)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("malformed VLM JSON") from exc
    if not isinstance(value, list):
        raise ValueError("VLM output must be a JSON list")
    claims: list[VlmClaim] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"VLM claim {index} is not an object")
        body = {
            "schema_version": SCHEMA_PROPOSAL,
            "mission_id": "pending",
            "proposal_id": str(item.get("proposal_id") or f"pending-{index}"),
            "claim_kind": item.get("claim_kind"),
            "text": item.get("text"),
            "verification_status": item.get("verification_status", "uncertain"),
        }
        for key in ("subject_id", "predicate", "object_id", "start_sec", "end_sec", "supersedes"):
            if key in item:
                body[key] = item[key]
        if "reasons" in item:
            body["reasons"] = item["reasons"]
        allowed = set(body) | {"model"}
        extra = set(item) - allowed - {"proposal_id", "claim_kind", "text", "verification_status"}
        if extra:
            raise ValueError(f"unknown_field: {sorted(extra)}")
        try:
            claims.append(VlmClaim.model_validate(body))
        except ValidationError as exc:
            raise ValueError(f"malformed VLM claim {index}") from exc
    return claims


def proposals_from_claims(
    mission_id: str,
    claims: list[VlmClaim],
    samples: list[GeometrySample],
) -> list[Proposal]:
    """Copy claims into the mission log and reject geometric contradictions.

    Args:
        mission_id: Mission id stored on each row.
        claims: Provider output, including corrected and rejected rows.
        samples: Geometry used to check deterministic predicates.

    Returns:
        Proposals. Open-vocabulary claims stay at the provider status.
        Deterministic predicates that contradict geometry are ``rejected``.
    """
    grouped: dict[str, list[GeometrySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.subject_id, []).append(sample)
    proposals: list[Proposal] = []
    for index, claim in enumerate(claims):
        status = claim.verification_status
        reasons = list(claim.reasons)
        if _contradicts(claim, grouped):
            status = "rejected"
            if "geometry_contradiction" not in reasons:
                reasons.append("geometry_contradiction")
        proposal_id = claim.proposal_id if claim.proposal_id not in {"", "pending"} else ""
        if proposal_id.startswith("pending"):
            proposal_id = ""
        proposals.append(
            Proposal(
                schema_version=SCHEMA_PROPOSAL,
                mission_id=mission_id,
                proposal_id=proposal_id or f"proposal-{index:04d}",
                claim_kind=claim.claim_kind,
                text=claim.text,
                subject_id=claim.subject_id,
                predicate=claim.predicate,
                object_id=claim.object_id,
                start_sec=claim.start_sec,
                end_sec=claim.end_sec,
                verification_status=status,
                supersedes=claim.supersedes,
                reasons=reasons,
            )
        )
    return proposals


def _contradicts(claim: VlmClaim, grouped: dict[str, list[GeometrySample]]) -> bool:
    from selfsuvis.pipeline.analysis4d.schemas import DETERMINISTIC_PREDICATES

    if claim.predicate not in DETERMINISTIC_PREDICATES:
        return False
    if not claim.subject_id or not claim.object_id:
        return False
    subject = _nearest(grouped.get(claim.subject_id, []), claim.start_sec)
    obj = _nearest(grouped.get(claim.object_id, []), claim.start_sec)
    if subject is None or obj is None:
        return False
    band = None
    return not relation_holds(
        claim.predicate,
        subject.center_m,
        subject.extent_m,
        obj.center_m,
        obj.extent_m,
        distance_band=band,
        subject_depth_m=subject.depth_m,
        object_depth_m=obj.depth_m,
    )


def _nearest(samples: list[GeometrySample], t_sec: float | None) -> GeometrySample | None:
    if not samples:
        return None
    if t_sec is None:
        return samples[-1]
    return min(samples, key=lambda sample: abs(sample.t_sec - t_sec))


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
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text.strip()


def _prompt(context: dict) -> str:
    body = json.dumps(context, sort_keys=True)
    return (
        "Return a JSON list of scene-graph claims. Each object has claim_kind "
        "(relation, action, attribute, or event), text, and optional subject_id, "
        "predicate, object_id, start_sec, end_sec. Do not add other fields. "
        "Use only ids from this context:\n" + body
    )


def _load_smolvlm(model_id: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, processor


def _generate(model, processor, prompt: str) -> str:
    import torch

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    templated = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=templated, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256)
    decoded = processor.batch_decode(output, skip_special_tokens=True)
    return decoded[0] if decoded else ""


def _safe_detail(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    if "HF_TOKEN" in text or "token" in text.lower():
        return "compact VLM weights are not visible; check HF_TOKEN"
    return text[:240] or exc.__class__.__name__
