"""Admin tab: stats, metrics, and 3DGS scene viewer."""

import os

import requests
import streamlit as st
import streamlit.components.v1 as components

from selfsuvis.pipeline.core.env import env_str
from selfsuvis.ui.api import API_URL, HEADERS


def render_admin_tab() -> None:
    st.header("Admin")
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh", key="admin_refresh"):
            st.rerun()

    stats = _fetch_admin_stats()
    if stats:
        _render_admin_stats(stats)

    render_scene_viewer()


def _fetch_admin_stats() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/admin/stats", headers=HEADERS, timeout=5)
        if resp.ok:
            return resp.json()
        st.error(f"API error {resp.status_code}: {resp.text}")
    except requests.RequestException as exc:
        st.error(f"Could not reach API: {exc}")
    return None


def _render_admin_stats(stats: dict) -> None:
    jobs = stats.get("jobs", {})
    al_tags = stats.get("al_tags", {})
    worker_active = stats.get("worker_active", False)

    if worker_active:
        st.success("Worker: ACTIVE")
    else:
        st.info("Worker: idle")

    st.metric("Queue depth (pending)", jobs.get("pending", 0))

    st.subheader("Job Status")
    col_p, col_r, col_d, col_e = st.columns(4)
    col_p.metric("Pending", jobs.get("pending", 0))
    col_r.metric("Running", jobs.get("running", 0))
    col_d.metric("Done", jobs.get("done", 0))
    col_e.metric("Error", jobs.get("error", 0))

    st.subheader("AL Tag Distribution")
    na = al_tags.get("needs_annotation", 0)
    novel = al_tags.get("novel", 0)
    none_count = al_tags.get("none", 0)
    total = na + novel + none_count

    if total == 0:
        st.caption("No indexed frames yet.")
        return

    import pandas as pd

    chart_data = pd.DataFrame(
        {"count": [na, novel, none_count]},
        index=["needs_annotation", "novel", "none"],
    )
    st.bar_chart(chart_data)
    st.caption(
        f"Total frames: {total} — "
        f"ANNOTATE: {na} ({100 * na // total}%) · "
        f"NOVEL: {novel} ({100 * novel // total}%) · "
        f"none: {none_count} ({100 * none_count // total}%)"
    )


def render_scene_viewer() -> None:
    st.subheader("3DGS Scene Viewer")
    supersplat_url = env_str("SUPERSPLAT_SERVER_URL", "http://localhost:8090")
    static_url = env_str("STATIC_SERVER_URL", "http://localhost:8080")

    try:
        missions_resp = requests.get(f"{API_URL}/admin/missions", headers=HEADERS, timeout=5)
        missions_list = missions_resp.json() if missions_resp.ok else []
    except requests.RequestException:
        missions_list = []

    done_missions = [m for m in missions_list if m.get("splat_paths")]
    if not done_missions:
        st.caption("No missions with 3DGS maps yet.")
        return

    mission_options = {f"{m['id']} ({m.get('scene_count', 1)} scene(s))": m for m in done_missions}
    selected_label = st.selectbox("Mission", list(mission_options.keys()), key="viewer_mission")
    selected_mission = mission_options[selected_label]
    splat_paths = selected_mission.get("splat_paths", [])

    if len(splat_paths) > 1:
        scene_labels = [os.path.basename(os.path.dirname(p)) for p in splat_paths]
        chosen_label = st.selectbox("Scene", scene_labels, key="viewer_scene")
        scene_idx = scene_labels.index(chosen_label)
        chosen_splat = splat_paths[scene_idx]
    else:
        chosen_splat = splat_paths[0]

    rel_path = os.path.relpath(
        chosen_splat,
        env_str("MAPS_DIR", os.path.join(os.environ.get("DATA_DIR", "./.data"), "maps")),
    )
    splat_static_url = f"{static_url}/static/maps/{rel_path}"
    viewer_url = f"{supersplat_url}/?load={splat_static_url}"

    st.caption(f"splat.ply: `{chosen_splat}`")
    components.iframe(viewer_url, height=600, scrolling=False)
