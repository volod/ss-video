"""Shared search result grid."""

import os

import streamlit as st


def render_search_results(results: list) -> None:
    st.subheader("Results")
    cols = st.columns(4)
    videos_dir = os.path.join(os.environ.get("DATA_DIR", "./.data"), "videos")
    for i, row in enumerate(results):
        col = cols[i % 4]
        with col:
            thumb = row.get("thumbnail_path")
            if thumb and os.path.exists(thumb):
                st.image(thumb, use_container_width=True)
            st.write(f"Score: {row['score']:.4f}")
            st.write(f"Video: {row['video_id']}")
            st.write(f"t={row['t_sec']:.2f}s")
            if row.get("frame_path"):
                st.caption(row.get("frame_path"))
            if row.get("tile_path"):
                st.caption(row.get("tile_path"))
            if row.get("video_id"):
                st.code(f'mpv "{videos_dir}/{row["video_id"]}.mp4" --start={row["t_sec"]:.2f}')
