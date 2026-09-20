"""Image Query tab."""

import requests
import streamlit as st

from selfsuvis.ui.api import API_URL, HEADERS
from selfsuvis.ui.components.results import render_search_results


def render_image_query_tab() -> None:
    st.header("Image Query")
    img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img")
    search_type = st.selectbox("Search type", ["both", "frame", "tile"], index=0)
    vector_space = st.selectbox("Vector space", ["clip", "dino"], index=0)
    enable_rerank = st.checkbox("Enable rerank", value=True, key="img_rerank")
    top_k = st.slider("Top-K", min_value=5, max_value=50, value=20)

    if st.button("Search", key="img_search"):
        if not img:
            st.error("Please upload an image.")
            return
        files = {"file": (img.name, img.getvalue(), img.type)}
        data = {
            "top_k": str(top_k),
            "search_type": search_type,
            "vector_space": vector_space,
            "enable_rerank": str(enable_rerank).lower(),
        }
        resp = requests.post(f"{API_URL}/query/image", files=files, data=data, headers=HEADERS)
        if resp.ok:
            render_search_results(resp.json().get("results", []))
        else:
            st.error(resp.text)
