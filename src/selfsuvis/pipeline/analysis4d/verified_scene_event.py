"""Contract verified-scene-event 1.0.0, generated from contracts/odcs/verified-scene-event.odcs.yaml; do not edit."""

from typing import ClassVar

from pydantic import Field

from ss_contracts.base import ContractModel, ContractTimestamp


class VerifiedSceneEvent(ContractModel):
    """One accepted 4D scene-timeline event published for site correlation. Rejected and uncertain
    claims are not valid messages. A deep-profile revision names the fast event it supersedes.
    """

    CONTRACT_ID: ClassVar[str] = "verified-scene-event"
    CONTRACT_VERSION: ClassVar[str] = "1.0.0"
    SS_BINDING: ClassVar[dict[str, str]] = {
        "modality": "video_4d",
        "mqttTopic": "ss/v1/site/{site_id}/video/{mission_id}/verified-event",
        "timeBase": "utc",
    }

    event_id: str = Field(
        description="Timeline event id. Stable across a replay of the same profile.",
        min_length=1,
        max_length=128,
    )
    mission_id: str = Field(
        description="Mission that produced the event.",
        min_length=1,
        max_length=128,
    )
    profile: str = Field(
        description="Analysis profile that accepted the event.",
        pattern="^(fast|deep)$",
    )
    event_type: str = Field(
        description="Timeline event type, such as entered_region or count_changed.",
        min_length=1,
        max_length=64,
    )
    summary: str = Field(
        description="Short description copied from the accepted timeline row.",
        min_length=1,
        max_length=512,
    )
    start_sec: float = Field(
        description="Media start of the half-open event interval, in seconds.",
        ge=0,
        json_schema_extra={
            "x-ss-binding": {
                "timeBase": "media",
                "unit": "s",
            },
        },
    )
    end_sec: float = Field(
        description="Media end of the half-open event interval, in seconds.",
        ge=0,
        json_schema_extra={
            "x-ss-binding": {
                "timeBase": "media",
                "unit": "s",
            },
        },
    )
    participants: list[str] = Field(
        description="Track and region ids that participate in the event.",
        min_length=1,
    )
    confidence: float = Field(
        description="Acceptance confidence copied from the timeline row.",
        ge=0,
        le=1,
    )
    verification_status: str = Field(
        description="Always accepted. The publisher drops every other status.",
        pattern="^accepted$",
    )
    supersedes: str | None = Field(
        default=None,
        description="Event id revised by this message. Unset when the event is new.",
    )
    artifact_uri: str | None = Field(
        default=None,
        description="Relative path of the timeline artifact inside the mission directory.",
    )
    published_at: ContractTimestamp = Field(
        description="UTC time the event was handed to the site envelope.",
        json_schema_extra={
            "x-ss-binding": {
                "timeBase": "utc",
            },
        },
    )
