"""Versioned contracts for 4D tracks, graph deltas, timelines, and Video-QA.

Unknown fields and a missing or unexpected ``schema_version`` fail closed.
Time intervals are half-open: ``end_sec`` must be greater than ``start_sec``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_TIMELINE = "ss-video.scene-timeline.v1"
SCHEMA_TRACK = "ss-video.track.v1"
SCHEMA_DELTA = "ss-video.graph-delta.v1"
SCHEMA_PROPOSAL = "ss-video.proposal.v1"
SCHEMA_QA = "ss-video.qa.v1"
SCHEMA_MANIFEST = "ss-video.analysis4d-manifest.v1"
SCHEMA_MASK = "ss-video.mask.v1"
SCHEMA_GEOMETRY = "ss-video.geometry-sample.v1"
SCHEMA_GAP = "ss-video.gap.v1"
SCHEMA_TRUTH = "ss-video.analysis4d-truth.v1"
SCHEMA_BENCHMARK = "ss-video.analysis4d-benchmark.v1"

MetricScale = Literal["metric", "relative", "unavailable"]
ProfileName = Literal["fast", "deep"]
VerificationStatus = Literal["accepted", "rejected", "uncertain", "corrected"]
TrackState = Literal["tentative", "confirmed", "occluded", "lost", "ended"]
DeltaOp = Literal["add", "end", "correct", "supersede"]
NodeKind = Literal["track", "agent", "region", "place"]
ClaimKind = Literal["relation", "action", "attribute", "event"]
AnswerKind = Literal["track_ref", "region_ref", "count", "interval", "text", "unavailable"]
EdgeSource = Literal["deterministic", "vlm"]
LicenseDecision = Literal["allowed", "rejected", "unset"]
DegradationCode = Literal[
    "models_missing",
    "calibration_missing",
    "pose_missing",
    "metric_scale_missing",
    "evidence_gap",
    "relative_depth_only",
    "empty_scene",
    "provider_unavailable",
]

DETERMINISTIC_PREDICATES = frozenset(
    {
        "contains",
        "intersects",
        "distance_band",
        "supports",
        "contacts",
        "left_of",
        "right_of",
        "above",
        "below",
        "visible",
        "occludes",
        "relative_motion",
    }
)

METRIC_NAMES = (
    "keyframe_event_coverage",
    "grounding_ap",
    "grounding_recall",
    "count_mae",
    "mask_j",
    "mask_f",
    "hota",
    "idf1",
    "identity_switches",
    "depth_error",
    "box_iou",
    "box_center_error",
    "relation_precision",
    "relation_recall",
    "event_f1",
    "event_temporal_iou",
    "verifier_false_accept_rate",
    "verifier_false_reject_rate",
    "qa_accuracy",
    "stage_latency_sec",
    "real_time_factor",
    "peak_vram_bytes",
    "queue_depth",
    "artifact_size_bytes",
)

# Fields the specification requires on a stage report and on a degradation record.
STAGE_FIELDS = (
    "stage",
    "queue_delay_sec",
    "inference_time_sec",
    "processed_frames",
    "skipped_frames",
    "trigger_reason",
    "model",
    "degradation_flags",
    "queue_depth",
)
DEGRADATION_FIELDS = ("code", "stage", "detail")

_ID = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
_SHA = r"^sha256:[0-9a-f]{64}$"
_UTC = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


class ContractModel(BaseModel):
    """Base model: no undeclared fields, and ``model_`` names are allowed."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


def _check_interval(start_sec: float, end_sec: float) -> None:
    if start_sec < 0 or end_sec < 0 or end_sec <= start_sec:
        raise ValueError("invalid_interval: end_sec must be greater than start_sec and both >= 0")


class ModelProvenance(ContractModel):
    """Identity of one model invocation. Required on every model artifact."""

    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    weights_digest: str = Field(pattern=_SHA)
    license_decision: LicenseDecision
    prompt_config_digest: str = Field(pattern=_SHA)
    preprocessing_version: str = Field(min_length=1)


class CoordinateFrame(ContractModel):
    """Mission frame. Metric scale requires a calibration id."""

    name: str = Field(min_length=1, max_length=64)
    metric_scale: MetricScale
    calibration_id: str | None = None

    @model_validator(mode="after")
    def _metric_needs_calibration(self) -> "CoordinateFrame":
        if self.metric_scale == "metric" and not self.calibration_id:
            raise ValueError("metric_scale metric requires calibration_id")
        return self


class Degradation(ContractModel):
    """Why a stage did not produce the full metric claim."""

    code: DegradationCode
    stage: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class StageReport(ContractModel):
    """Telemetry the specification requires from every analysis stage."""

    stage: str = Field(min_length=1)
    queue_delay_sec: float = Field(ge=0)
    inference_time_sec: float = Field(ge=0)
    processed_frames: int = Field(ge=0)
    skipped_frames: int = Field(ge=0)
    trigger_reason: str = Field(min_length=1)
    model: ModelProvenance | None = None
    degradation_flags: list[str] = Field(default_factory=list)
    queue_depth: int = Field(ge=0, default=0)


class Box2D(ContractModel):
    """Axis-aligned box in normalized image coordinates, ``[x, y, w, h]``."""

    xywh_norm: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _inside_image(self) -> "Box2D":
        x, y, width, height = self.xywh_norm
        if min(x, y, width, height) < 0 or width <= 0 or height <= 0:
            raise ValueError("box xywh_norm must be positive and inside the unit square")
        if x + width > 1.000001 or y + height > 1.000001:
            raise ValueError("box xywh_norm must lie inside the unit square")
        return self


class IdentityLink(ContractModel):
    """Auditable re-identification. Does not rewrite the source observation."""

    linked_track_id: str = Field(pattern=_ID)
    reason: str = Field(min_length=1)


class TrackRecord(ContractModel):
    """One append-only 2D track observation."""

    schema_version: Literal["ss-video.track.v1"]
    mission_id: str = Field(pattern=_ID)
    observation_id: str = Field(pattern=_ID)
    track_id: str = Field(pattern=_ID)
    state: TrackState
    t_sec: float = Field(ge=0)
    prompt_id: str = Field(pattern=_ID)
    label_raw: str = Field(min_length=1)
    label_normalized: str = Field(min_length=1)
    box: Box2D
    mask_ref: str | None = None
    confidence: float = Field(ge=0, le=1)
    negative_prompts: list[str] = Field(default_factory=list)
    exemplars: list[str] = Field(default_factory=list)
    model: ModelProvenance | None = None
    identity_link: IdentityLink | None = None
    supersedes: str | None = None


class GraphNode(ContractModel):
    """Persistent track, agent, region, or place."""

    node_id: str = Field(pattern=_ID)
    kind: NodeKind
    label: str = Field(min_length=1)
    track_id: str | None = None


class DistanceBand(ContractModel):
    """Inclusive distance window in meters, used by ``distance_band`` edges."""

    min_m: float = Field(ge=0)
    max_m: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "DistanceBand":
        if self.max_m <= self.min_m:
            raise ValueError("invalid_interval: distance band max_m must exceed min_m")
        return self


class GraphEdge(ContractModel):
    """Typed subject-predicate-object claim over a half-open interval."""

    edge_id: str = Field(pattern=_ID)
    subject_id: str = Field(pattern=_ID)
    predicate: str = Field(min_length=1)
    object_id: str = Field(pattern=_ID)
    start_sec: float
    end_sec: float
    coordinate_frame: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    source: EdgeSource
    evidence_ids: list[str] = Field(min_length=1)
    verification_status: VerificationStatus
    distance_band: DistanceBand | None = None
    supersedes: str | None = None

    @model_validator(mode="after")
    def _interval_and_acceptance(self) -> "GraphEdge":
        _check_interval(self.start_sec, self.end_sec)
        if self.verification_status == "accepted":
            if self.predicate not in DETERMINISTIC_PREDICATES:
                raise ValueError(
                    "accepted edges must use a deterministic predicate; "
                    "open-vocabulary claims stay uncertain or rejected"
                )
            if self.predicate == "distance_band" and self.distance_band is None:
                raise ValueError("distance_band edges require distance_band")
        return self


class GraphDelta(ContractModel):
    """One append-only scene-graph edit."""

    schema_version: Literal["ss-video.graph-delta.v1"]
    mission_id: str = Field(pattern=_ID)
    delta_id: str = Field(pattern=_ID)
    op: DeltaOp
    t_sec: float = Field(ge=0)
    node: GraphNode | None = None
    edge: GraphEdge | None = None
    supersedes: str | None = None

    @model_validator(mode="after")
    def _payload(self) -> "GraphDelta":
        if (self.node is None) == (self.edge is None):
            raise ValueError("a graph delta carries exactly one of node or edge")
        if self.op in {"correct", "supersede"} and not self.supersedes:
            raise ValueError("dangling_id: correct and supersede deltas require supersedes")
        return self


class Proposal(ContractModel):
    """VLM or operator claim kept for audit, including rejected claims."""

    schema_version: Literal["ss-video.proposal.v1"]
    mission_id: str = Field(pattern=_ID)
    proposal_id: str = Field(pattern=_ID)
    claim_kind: ClaimKind
    text: str = Field(min_length=1)
    subject_id: str | None = None
    predicate: str | None = None
    object_id: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    model: ModelProvenance | None = None
    verification_status: VerificationStatus
    supersedes: str | None = None
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _interval(self) -> "Proposal":
        if self.start_sec is not None and self.end_sec is not None:
            _check_interval(self.start_sec, self.end_sec)
        elif self.start_sec is not None or self.end_sec is not None:
            raise ValueError("invalid_interval: proposal start_sec and end_sec are paired")
        return self


class EvidenceRef(ContractModel):
    """Pointer from an event back to a frame, track, mask, and geometry sample."""

    frame_id: str = Field(pattern=_ID)
    t_sec: float = Field(ge=0)
    track_ids: list[str] = Field(min_length=1)
    mask_ref: str | None = None
    geometry_ref: str | None = None


class Location3D(ContractModel):
    """Event location in a named frame. Covariance is the diagonal in meters squared."""

    frame: str = Field(min_length=1)
    center_m: list[float] = Field(min_length=3, max_length=3)
    covariance_diag: list[float] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _covariance(self) -> "Location3D":
        if any(value < 0 for value in self.covariance_diag):
            raise ValueError("covariance_diag must be non-negative")
        return self


class Verification(ContractModel):
    """How a claim was resolved. Geometry rules are named, not implied."""

    status: VerificationStatus
    rules: list[str] = Field(default_factory=list)
    vlm_claim_ref: str | None = None
    reasons: list[str] = Field(default_factory=list)


class TimelineEvent(ContractModel):
    """One verified, rejected, or uncertain timeline interval."""

    event_id: str = Field(pattern=_ID)
    type: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    start_sec: float
    end_sec: float
    participants: list[str] = Field(min_length=1)
    state_delta_refs: list[str] = Field(default_factory=list)
    location: Location3D | None = None
    confidence: float = Field(ge=0, le=1)
    verification: Verification
    evidence: list[EvidenceRef] = Field(default_factory=list)
    supersedes: str | None = None

    @model_validator(mode="after")
    def _interval_and_evidence(self) -> "TimelineEvent":
        _check_interval(self.start_sec, self.end_sec)
        if self.verification.status == "accepted" and not self.evidence:
            raise ValueError("accepted events require evidence")
        return self


class QaAnswer(ContractModel):
    """Answer produced by a graph program. ``unit`` is null when it does not apply."""

    kind: AnswerKind
    value: str | int | float | None = None
    unit: str | None = None


class QaPair(ContractModel):
    """Spatial Video-QA row embedded in a timeline."""

    qa_id: str = Field(pattern=_ID)
    type: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: QaAnswer
    graph_program: str = Field(min_length=1)
    interval_sec: list[float] = Field(min_length=2, max_length=2)
    evidence_event_ids: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus

    @model_validator(mode="after")
    def _interval_and_evidence(self) -> "QaPair":
        _check_interval(self.interval_sec[0], self.interval_sec[1])
        if self.verification_status == "accepted" and not self.evidence_event_ids:
            raise ValueError("accepted QA requires evidence_event_ids")
        if self.type == "causal" and self.verification_status == "accepted":
            raise ValueError("causal QA is excluded unless a later causal relation admits it")
        return self


class QaRecord(QaPair):
    """Append-only QA log line. The timeline embeds ``QaPair`` without these ids."""

    schema_version: Literal["ss-video.qa.v1"]
    mission_id: str = Field(pattern=_ID)


class SceneTimeline(ContractModel):
    """Materialized timeline. Empty ``events`` and ``qa_pairs`` are valid."""

    schema_version: Literal["ss-video.scene-timeline.v1"]
    mission_id: str = Field(pattern=_ID)
    profile: ProfileName
    coordinate_frame: CoordinateFrame
    model_manifest_ref: str = Field(min_length=1)
    degradations: list[Degradation] = Field(default_factory=list)
    events: list[TimelineEvent] = Field(default_factory=list)
    qa_pairs: list[QaPair] = Field(default_factory=list)
    supersedes_sha256: str | None = None

    @model_validator(mode="after")
    def _one_frame_and_metric_gate(self) -> "SceneTimeline":
        frame = self.coordinate_frame.name
        for event in self.events:
            if event.location is not None and event.location.frame != frame:
                raise ValueError(
                    "mixed_coordinate_frame: location.frame differs from coordinate_frame.name"
                )
            if (
                event.location is not None
                and event.verification.status == "accepted"
                and self.coordinate_frame.metric_scale != "metric"
            ):
                raise ValueError(
                    "metric location on an accepted event requires metric_scale metric"
                )
        return self


class ArtifactEntry(ContractModel):
    """One file named by the manifest, hashed as stored."""

    path: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA)
    bytes: int = Field(ge=0)


class AnalysisManifest(ContractModel):
    """Append-only manifest. A later profile sets ``supersedes_sha256``."""

    schema_version: Literal["ss-video.analysis4d-manifest.v1"]
    mission_id: str = Field(pattern=_ID)
    profile: ProfileName
    coordinate_frame: CoordinateFrame
    created_at: str = Field(pattern=_UTC)
    artifacts: list[ArtifactEntry]
    models: list[ModelProvenance] = Field(default_factory=list)
    degradations: list[Degradation] = Field(default_factory=list)
    stages: list[StageReport] = Field(min_length=1)
    keyframes_sec: list[float] = Field(default_factory=list)
    supersedes_sha256: str | None = None

    @model_validator(mode="after")
    def _keyframes(self) -> "AnalysisManifest":
        if any(value < 0 for value in self.keyframes_sec):
            raise ValueError("keyframes_sec must be non-negative")
        return self


class MaskArtifact(ContractModel):
    """Binary mask. ``bits`` is row-major ``0`` and ``1`` of length ``width * height``."""

    schema_version: Literal["ss-video.mask.v1"]
    mission_id: str = Field(pattern=_ID)
    track_id: str = Field(pattern=_ID)
    t_sec: float = Field(ge=0)
    width: int = Field(ge=1, le=64)
    height: int = Field(ge=1, le=64)
    bits: str

    @model_validator(mode="after")
    def _bits(self) -> "MaskArtifact":
        if len(self.bits) != self.width * self.height or any(ch not in "01" for ch in self.bits):
            raise ValueError("bits must be 0/1 and width*height long")
        return self


class GeometrySample(ContractModel):
    """One oriented box and optional depth sample in the mission frame."""

    schema_version: Literal["ss-video.geometry-sample.v1"]
    mission_id: str = Field(pattern=_ID)
    sample_id: str = Field(pattern=_ID)
    subject_id: str = Field(pattern=_ID)
    t_sec: float = Field(ge=0)
    frame: str = Field(min_length=1)
    center_m: list[float] = Field(min_length=3, max_length=3)
    extent_m: list[float] = Field(min_length=3, max_length=3)
    quaternion_xyzw: list[float] = Field(min_length=4, max_length=4)
    depth_m: float | None = None
    pose_position_m: list[float] | None = None

    @model_validator(mode="after")
    def _shape(self) -> "GeometrySample":
        if any(value <= 0 for value in self.extent_m):
            raise ValueError("extent_m must be positive")
        if self.pose_position_m is not None and len(self.pose_position_m) != 3:
            raise ValueError("pose_position_m must have length 3")
        norm = sum(value * value for value in self.quaternion_xyzw)
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("quaternion_xyzw must be a unit quaternion")
        return self


class GapRecord(ContractModel):
    """Discarded evidence. A stage that skips frames must emit one of these."""

    schema_version: Literal["ss-video.gap.v1"]
    mission_id: str = Field(pattern=_ID)
    gap_id: str = Field(pattern=_ID)
    start_sec: float
    end_sec: float
    reason: str = Field(min_length=1)
    skipped_frames: int = Field(ge=0)

    @model_validator(mode="after")
    def _interval(self) -> "GapRecord":
        _check_interval(self.start_sec, self.end_sec)
        return self


class TruthDetection(ContractModel):
    """Ground-truth box at one timestamp."""

    t_sec: float = Field(ge=0)
    gt_id: str = Field(pattern=_ID)
    xywh_norm: list[float] = Field(min_length=4, max_length=4)
    label: str = Field(min_length=1)


class TruthCount(ContractModel):
    """Ground-truth instance count for one label at one timestamp."""

    t_sec: float = Field(ge=0)
    label: str = Field(min_length=1)
    count: int = Field(ge=0)


class TruthMask(ContractModel):
    """Ground-truth mask bits aligned with a predicted track id."""

    t_sec: float = Field(ge=0)
    track_id: str = Field(pattern=_ID)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    bits: str


class TruthDepth(ContractModel):
    """Ground-truth depth in meters, or a relative value when scale is relative."""

    t_sec: float = Field(ge=0)
    subject_id: str = Field(pattern=_ID)
    depth_m: float


class TruthBox(ContractModel):
    """Ground-truth axis-aligned 3D box."""

    t_sec: float = Field(ge=0)
    subject_id: str = Field(pattern=_ID)
    center_m: list[float] = Field(min_length=3, max_length=3)
    extent_m: list[float] = Field(min_length=3, max_length=3)


class TruthRelation(ContractModel):
    """Ground-truth relation interval."""

    subject_id: str = Field(pattern=_ID)
    predicate: str = Field(min_length=1)
    object_id: str = Field(pattern=_ID)
    start_sec: float
    end_sec: float

    @model_validator(mode="after")
    def _interval(self) -> "TruthRelation":
        _check_interval(self.start_sec, self.end_sec)
        return self


class TruthEvent(ContractModel):
    """Ground-truth event interval."""

    type: str = Field(min_length=1)
    start_sec: float
    end_sec: float
    participants: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _interval(self) -> "TruthEvent":
        _check_interval(self.start_sec, self.end_sec)
        return self


class TruthQa(ContractModel):
    """Expected answer value for one QA id."""

    qa_id: str = Field(pattern=_ID)
    answer_value: str | int | float | None = None


class TruthVerifierLabel(ContractModel):
    """Whether a proposal should have been accepted."""

    proposal_id: str = Field(pattern=_ID)
    should_accept: bool


class MissionTruth(ContractModel):
    """Pinned annotations for one benchmark mission. Not a runtime artifact."""

    schema_version: Literal["ss-video.analysis4d-truth.v1"]
    mission_id: str = Field(pattern=_ID)
    media_duration_sec: float = Field(gt=0)
    conditions: list[str] = Field(min_length=1)
    keyframe_boundaries_sec: list[float] = Field(default_factory=list)
    tracks: list[TruthDetection] = Field(default_factory=list)
    counts: list[TruthCount] = Field(default_factory=list)
    masks: list[TruthMask] = Field(default_factory=list)
    depths: list[TruthDepth] = Field(default_factory=list)
    boxes: list[TruthBox] = Field(default_factory=list)
    relations: list[TruthRelation] = Field(default_factory=list)
    events: list[TruthEvent] = Field(default_factory=list)
    qa: list[TruthQa] = Field(default_factory=list)
    verifier: list[TruthVerifierLabel] = Field(default_factory=list)


class MetricReport(ContractModel):
    """Every specification metric. Null means the case does not define that score."""

    keyframe_event_coverage: float | None = None
    grounding_ap: float | None = None
    grounding_recall: float | None = None
    count_mae: float | None = None
    mask_j: float | None = None
    mask_f: float | None = None
    hota: float | None = None
    idf1: float | None = None
    identity_switches: float | None = None
    depth_error: float | None = None
    box_iou: float | None = None
    box_center_error: float | None = None
    relation_precision: float | None = None
    relation_recall: float | None = None
    event_f1: float | None = None
    event_temporal_iou: float | None = None
    verifier_false_accept_rate: float | None = None
    verifier_false_reject_rate: float | None = None
    qa_accuracy: float | None = None
    stage_latency_sec: float | None = None
    real_time_factor: float | None = None
    peak_vram_bytes: float | None = None
    queue_depth: float | None = None
    artifact_size_bytes: float | None = None


class CaseResult(ContractModel):
    """One corpus mission after validation and scoring."""

    mission_id: str
    valid: bool
    empty_timeline: bool
    conditions: list[str]
    metrics: MetricReport
    failures: list[str] = Field(default_factory=list)


class BenchmarkReport(ContractModel):
    """Machine-readable benchmark report. ``passed`` is contract integrity, not a quality bar."""

    schema_version: Literal["ss-video.analysis4d-benchmark.v1"]
    corpus_id: str
    passed: bool
    empty_timeline_valid: bool
    metrics: MetricReport
    degradations: list[Degradation]
    stages: list[StageReport]
    cases: list[CaseResult]
    failures: list[str] = Field(default_factory=list)
