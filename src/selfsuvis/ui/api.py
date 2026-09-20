"""HTTP client settings and /api/v1 helpers for the Streamlit UI."""

import requests

from selfsuvis.pipeline.core.env import env_str

API_URL = env_str("API_URL", "http://api:8000")
FUSION_RT_URL = env_str("FUSION_RT_URL", "http://fusion-rt:8000")

_API_KEY = env_str("API_KEY", "")
HEADERS = {"X-API-Key": _API_KEY} if _API_KEY else {}
V1_HEADERS = {"X-Api-Key": _API_KEY} if _API_KEY else {}


def v1_get(path: str, params: dict | None = None):
    try:
        resp = requests.get(
            f"{FUSION_RT_URL}/api/v1{path}",
            headers=V1_HEADERS,
            params=params,
            timeout=5,
        )
        if resp.ok:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def v1_post(path: str, json_body: dict | None = None):
    try:
        resp = requests.post(
            f"{FUSION_RT_URL}/api/v1{path}",
            headers=V1_HEADERS,
            json=json_body or {},
            timeout=5,
        )
        return resp.ok, resp
    except requests.RequestException as exc:
        return False, str(exc)
