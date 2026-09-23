"""Frame signals, file analysis, and deep-event linking."""

from pathlib import Path

from PIL import Image
from tests.unit.pipeline.analysis4d.test_profile import (
    _PROMPT,
    _WHEN,
    _detection,
    _frames,
    _grounding,
)

from selfsuvis.pipeline.analysis4d.analyze import analyze_video
from selfsuvis.pipeline.analysis4d.providers import Prompt
from selfsuvis.pipeline.analysis4d.schemas import SceneTimeline, Verification
from selfsuvis.pipeline.analysis4d.signals import signal_drift, signals_from_paths
from selfsuvis.pipeline.analysis4d.tracks import Detection
from selfsuvis.pipeline.workflows.analysis4d_profile import (
    _superseded_event_id,
    frames_from_rows,
    run_fast_profile,
)


def _save(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (16, 16), color).save(path)


def test_image_rows_keep_a_path_and_a_distinct_embedding(tmp_path: Path) -> None:
    red = tmp_path / "red.jpg"
    blue = tmp_path / "blue.jpg"
    _save(red, (220, 10, 10))
    _save(blue, (10, 10, 220))
    frames = frames_from_rows(
        [
            {"t_sec": 1.0, "frame_path": str(blue)},
            {"t_sec": 0.0, "frame_path": str(red)},
            {"t_sec": 2.0},
        ]
    )
    assert [frame.t_sec for frame in frames] == [0.0, 1.0, 2.0]
    assert frames[0].image == str(red)
    assert frames[0].quality_ok
    assert frames[2].image is None
    assert signal_drift(frames[0].embedding, frames[1].embedding) > 0.15
    missing = frames_from_rows([{"t_sec": 0.0, "frame_path": str(tmp_path / "absent.jpg")}])
    assert missing[0].quality_ok is False


def test_signals_ignore_unreadable_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not-an-image")
    frames = signals_from_paths([(0.0, str(broken))])
    assert frames[0].quality_ok is False
    assert frames[0].image is None


def test_deep_link_uses_the_overlapping_event() -> None:
    timeline = SceneTimeline(
        schema_version="ss-video.scene-timeline.v1",
        mission_id="mission-profile",
        profile="fast",
        coordinate_frame={"name": "mission_enu", "metric_scale": "unavailable"},
        model_manifest_ref="manifest.json",
        events=[
            _event("evt-early", 0.0, 1.0, ["track-a"]),
            _event("evt-late", 8.0, 9.0, ["track-b"]),
        ],
    )
    deep = SceneTimeline(
        schema_version="ss-video.scene-timeline.v1",
        mission_id="mission-profile",
        profile="deep",
        coordinate_frame={"name": "mission_enu", "metric_scale": "unavailable"},
        model_manifest_ref="manifest.json",
        events=[_event("evt-deep", 8.2, 9.2, ["track-deep"])],
    )
    assert _superseded_event_id(deep.events[0], list(timeline.events)) == "evt-late"


def test_unfinished_directory_can_be_rerun(tmp_path: Path) -> None:
    kwargs = {
        "grounding": _grounding(),
        "fixture_geometry": True,
        "capacity": 3,
        "slots": 0,
        "published_at": _WHEN,
        "duration_sec": 12.0,
    }
    run_fast_profile("mission-profile", _frames(), [_PROMPT], dest=tmp_path, **kwargs)
    (tmp_path / "orchestration-state.json").unlink()
    again = run_fast_profile("mission-profile", _frames(), [_PROMPT], dest=tmp_path, **kwargs)
    assert again.replayed is False
    assert (tmp_path / "orchestration-state.json").is_file()


def test_analyze_video_writes_a_summary(tmp_path: Path) -> None:
    video = tmp_path / "yard clip.mp4"
    video.write_bytes(b"placeholder")
    frame = tmp_path / "frame.jpg"
    _save(frame, (30, 30, 30))
    hit = _detection()

    def _decode(source: Path, dest: Path, fps: float) -> list[tuple[float, str]]:
        del source, dest, fps
        return [(0.0, str(frame))]

    summary = analyze_video(
        video,
        profile="fast",
        prompts=[Prompt(prompt_id="prompt-crate", text="crate", normalized="crate")],
        dest=tmp_path / "out" / "4d",
        decoder=_decode,
        grounding=_scripted(hit),
        fixture_geometry=True,
        slots=0,
        fps=1.0,
    )
    assert summary["mission_id"] == "mission-yard-clip"
    assert summary["frame_count"] == 1
    assert summary["track_count"] >= 1
    assert Path(summary["summary_path"]).is_file()
    assert Path(summary["tracks_path"]).is_file()


def _event(event_id: str, start: float, end: float, participants: list[str]) -> dict:
    return {
        "event_id": event_id,
        "type": "count_changed",
        "summary": event_id,
        "start_sec": start,
        "end_sec": end,
        "participants": participants,
        "confidence": 0.9,
        "verification": Verification(status="accepted", rules=["count"], reasons=[]),
        "evidence": [
            {
                "frame_id": f"mission-profile:{event_id}:0000000",
                "t_sec": start,
                "track_ids": participants,
            }
        ],
    }


def _scripted(hit: Detection):
    from selfsuvis.pipeline.analysis4d.providers import ScriptedGrounding

    return ScriptedGrounding({0.0: [hit]})
