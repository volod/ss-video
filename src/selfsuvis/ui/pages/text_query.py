"""Text Query tab."""

import requests
import streamlit as st

from selfsuvis.ui.api import API_URL, HEADERS
from selfsuvis.ui.components.results import render_search_results


def render_text_query_tab() -> None:
    st.header("Text Query")
    text = st.text_input("Query", value="green field", max_chars=1000)
    search_type = st.selectbox("Search type", ["both", "frame", "tile"], index=0, key="text_type")
    enable_rerank = st.checkbox("Enable rerank", value=True, key="text_rerank")
    top_k = st.slider("Top-K", min_value=5, max_value=50, value=20, key="text_topk")

    if st.button("Search", key="text_search"):
        payload = {"text": text}
        params = {
            "top_k": top_k,
            "search_type": search_type,
            "enable_rerank": enable_rerank,
        }
        resp = requests.post(f"{API_URL}/query/text", json=payload, params=params, headers=HEADERS)
        if resp.ok:
            render_search_results(resp.json().get("results", []))
        else:
            st.error(resp.text)
