"""Mission summary HTML report generator.

Writes reports/{mission_id}/summary.html containing:
- Mission metadata (id, frame count, duration)
- AL tag distribution (needs_annotation / novel / none counts)
- Frame gallery sorted by al_score descending
"""

import html as _html
import os
from typing import Any

from selfsuvis.pipeline.core import ensure_dir, get_logger, settings

logger = get_logger(__name__)

_AL_TAG_COLORS = {
    "needs_annotation": "#e53935",
    "novel": "#f9a825",
    "none": "#43a047",
}
_AL_TAG_LABELS = {
    "needs_annotation": "ANNOTATE",
    "novel": "NOVEL",
    "none": "",
}


def _badge_html(al_tag: str) -> str:
    label = _AL_TAG_LABELS.get(al_tag, al_tag)
    if not label:
        return ""
    color = _AL_TAG_COLORS.get(al_tag, "#9e9e9e")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 6px;'
        f'border-radius:4px;font-size:11px;margin-right:4px">'
        f"{_html.escape(label)}</span>"
    )


def generate_summary_html(
    mission_id: str,
    frames: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Build and return an HTML summary string for a mission.

    Args:
        mission_id: Mission identifier.
        frames: List of frame dicts with keys:
            frame_path (str), caption (str), al_tag (str), al_score (float), t_sec (float).
        metadata: Optional mission-level metadata dict (not displayed but reserved for future use).

    Returns:
        Complete HTML document as a string.
    """
    frame_count = len(frames)
    duration = max((f.get("t_sec", 0.0) for f in frames), default=0.0)

    tag_counts: dict[str, int] = {"needs_annotation": 0, "novel": 0, "none": 0}
    for f in frames:
        tag = f.get("al_tag", "none")
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    sorted_frames = sorted(frames, key=lambda f: f.get("al_score", 0.0), reverse=True)

    cards_html = ""
    for f in sorted_frames:
        frame_path = _html.escape(f.get("frame_path", ""))
        caption = _html.escape((f.get("caption") or "")[:80])
        al_tag = f.get("al_tag", "none")
        al_score = f.get("al_score", 0.0)
        t_sec = f.get("t_sec", 0.0)
        badge = _badge_html(al_tag)
        cards_html += (
            f'<div style="border:1px solid #ddd;border-radius:6px;padding:8px;'
            f'width:200px;display:inline-block;margin:4px;vertical-align:top">'
            f'<img src="{frame_path}" style="width:100%;height:150px;object-fit:cover;'
            f'border-radius:4px" onerror="this.style.display=\'none\'"/>'
            f'<div style="font-size:11px;margin-top:4px;color:#333">{caption}</div>'
            f'<div style="margin-top:4px">{badge}'
            f'<span style="font-size:11px;color:#666">score={al_score:.2f}</span></div>'
            f'<div style="font-size:10px;color:#999">t={t_sec:.1f}s</div>'
            f"</div>\n"
        )

    na_count = tag_counts.get("needs_annotation", 0)
    novel_count = tag_counts.get("novel", 0)
    none_count = tag_counts.get("none", 0)
    na_color = _AL_TAG_COLORS["needs_annotation"]
    novel_color = _AL_TAG_COLORS["novel"]
    none_color = _AL_TAG_COLORS["none"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Mission {_html.escape(mission_id)} — Summary</title>
<style>
body {{ font-family: sans-serif; margin: 16px; color: #333; }}
h1, h2 {{ color: #1a1a1a; }}
</style>
</head>
<body>
<h1>Mission {_html.escape(mission_id)}</h1>
<p>Frames: {frame_count} &nbsp;|&nbsp; Duration: {duration:.1f}s</p>
<h2>AL Tag Distribution</h2>
<ul>
  <li><span style="color:{na_color}">&#9679;</span> Needs annotation: {na_count}</li>
  <li><span style="color:{novel_color}">&#9679;</span> Novel: {novel_count}</li>
  <li><span style="color:{none_color}">&#9679;</span> None: {none_count}</li>
</ul>
<h2>Frames (by uncertainty score)</h2>
{cards_html}
</body>
</html>"""


def write_mission_report(
    mission_id: str,
    frames: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write HTML summary to reports/{mission_id}/summary.html under DATA_DIR.

    Returns:
        Absolute path to the written report file.
    """
    report_dir = os.path.join(settings.DATA_DIR, "reports", mission_id)
    ensure_dir(report_dir)
    report_path = os.path.join(report_dir, "summary.html")

    content = generate_summary_html(mission_id, frames, metadata)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info("Report written: %s (%d frames)", report_path, len(frames))
    return report_path
