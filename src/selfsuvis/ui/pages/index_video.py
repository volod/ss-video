"""Index Video tab."""

import requests
import streamlit as st

from selfsuvis.ui.api import API_URL, HEADERS


def render_index_video_tab() -> None:
    st.header("Index Video")
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv"])
    url_input = st.text_input("Or URL (HTTP/S)")
    path_input = st.text_input("Or directory path (inside container, for batch indexing)")
    enable_tiles = st.checkbox("Enable tile indexing", value=True)

    if st.button("Start Indexing"):
        resp = _submit_indexing(uploaded, url_input, path_input, enable_tiles)
        if resp is None:
            st.error("Upload a video or provide URL/path.")
        elif resp.ok:
            job = resp.json()
            if "job_id" in job:
                st.session_state["job_id"] = job["job_id"]
                st.success(f"Job created: {job['job_id']}")
            else:
                st.json(job)
        else:
            st.error(resp.text)

    job_id = st.text_input("Job ID", value=st.session_state.get("job_id", ""))
    if st.button("Refresh Status") and job_id:
        resp = requests.get(f"{API_URL}/jobs/{job_id}", headers=HEADERS)
        if resp.ok:
            st.json(resp.json())
        else:
            st.error(resp.text)


def _submit_indexing(uploaded, url_input: str, path_input: str, enable_tiles: bool):
    data = {"enable_tiles": str(enable_tiles).lower()}
    if uploaded:
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        return requests.post(f"{API_URL}/index/video", files=files, data=data, headers=HEADERS)
    if url_input:
        data["url"] = url_input
        return requests.post(f"{API_URL}/index/url", data=data, headers=HEADERS)
    if path_input:
        data["path"] = path_input
        return requests.post(f"{API_URL}/index/dir", data=data, headers=HEADERS)
    return None
