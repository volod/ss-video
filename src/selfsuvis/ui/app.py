"""Streamlit entrypoint for the Video Semantic Search UI."""

import streamlit as st

from selfsuvis.ui.pages import admin, image_query, index_video, site_monitor, text_query


def main() -> None:
    st.set_page_config(page_title="Video Semantic Search", layout="wide")
    st.title("Video Semantic Search (POC)")

    tab_index, tab_image, tab_text, tab_admin, tab_site = st.tabs(
        ["Index Video", "Image Query", "Text Query", "Admin", "Site Monitor"]
    )

    with tab_index:
        index_video.render_index_video_tab()
    with tab_image:
        image_query.render_image_query_tab()
    with tab_text:
        text_query.render_text_query_tab()
    with tab_admin:
        admin.render_admin_tab()
    with tab_site:
        site_monitor.render_site_monitor_tab()


main()
