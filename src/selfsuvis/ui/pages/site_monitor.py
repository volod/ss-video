"""Site Monitor tab: zones, incidents, and search."""

import time

import streamlit as st

from selfsuvis.ui.api import v1_get, v1_post

_RISK_COLORS = {
    "low": "[low]",
    "medium": "[med]",
    "high": "[high]",
    "critical": "[critical]",
}


def render_site_monitor_tab() -> None:
    st.header("Site Monitor")

    col_refresh, col_auto, _ = st.columns([1, 2, 5])
    with col_refresh:
        if st.button("Refresh now", key="site_refresh"):
            st.rerun()
    with col_auto:
        auto_refresh = st.toggle("Auto-refresh (10s)", key="site_autorefresh", value=False)

    site_state = v1_get("/site/state")

    if site_state is None:
        st.error("Could not reach fusion-rt /api/v1/site/state — is fusion-rt running?")
    elif not site_state.get("zones"):
        st.info(
            "No zones configured. "
            "POST to /api/v1/zones or set COOP_FRIGATE_API_URL to auto-seed from Frigate."
        )
    else:
        _render_zone_overview(site_state["zones"])
        _render_zone_drilldown(site_state["zones"])

    _render_incident_search()

    if auto_refresh:
        time.sleep(10)
        st.rerun()


def _render_zone_overview(zones: list) -> None:
    st.subheader("Zone Risk Overview")
    zone_data = []
    for zone in zones:
        active = zone.get("active_incidents", [])
        last_inc = active[0]["ts"][:19] if active else "—"
        risk_level = zone.get("risk_level")
        risk_label = f"{_RISK_COLORS.get(risk_level, '')} {risk_level}" if risk_level else "— none"
        zone_data.append(
            {
                "Zone": zone["zone_id"],
                "Label": zone["label"],
                "Risk": risk_label,
                "Active Incidents": len(active),
                "Last Incident": last_inc,
            }
        )
    st.table(zone_data)


def _render_zone_drilldown(zones: list) -> None:
    zone_ids = [z["zone_id"] for z in zones]
    selected_zone = st.selectbox("Drill into zone", ["(none)"] + zone_ids, key="site_zone_sel")
    if selected_zone == "(none)":
        return

    status_filter = st.selectbox(
        "Incident status",
        ["active", "acknowledged", "dismissed", "all"],
        key="site_inc_status",
    )
    incidents_data = v1_get(
        "/incidents",
        params={"zone": selected_zone, "status": status_filter, "limit": 50},
    )
    incidents = incidents_data.get("incidents", []) if incidents_data else []

    if not incidents:
        st.info(f"No {status_filter} incidents in {selected_zone}.")
        return

    st.markdown(f"**{len(incidents)} incident(s)** in `{selected_zone}` ({status_filter})")
    for incident in incidents:
        _render_incident_expander(incident)


def _render_incident_expander(incident: dict) -> None:
    risk = incident["risk_level"]
    title = (
        f"{_RISK_COLORS.get(risk, '')} {risk.upper()} | "
        f"{incident['ts'][:19]} | {incident['incident_id'][:8]}…"
    )
    with st.expander(title):
        st.json(incident)

        col_ack, col_dis = st.columns(2)
        with col_ack:
            if st.button("Acknowledge", key=f"ack_{incident['incident_id']}"):
                ok, _ = v1_post(f"/incidents/{incident['incident_id']}/acknowledge")
                st.success("Acknowledged") if ok else st.error("Failed")
                st.rerun()
        with col_dis:
            reason = st.text_input("Dismiss reason", key=f"dis_reason_{incident['incident_id']}")
            if st.button("Dismiss", key=f"dis_{incident['incident_id']}"):
                ok, _ = v1_post(
                    f"/incidents/{incident['incident_id']}/dismiss",
                    {"reason": reason or None},
                )
                st.success("Dismissed") if ok else st.error("Failed")
                st.rerun()

        notes_data = v1_get(f"/incidents/{incident['incident_id']}/notes")
        notes = notes_data.get("notes", []) if notes_data else []
        if notes:
            st.markdown("**Notes:**")
            for note in notes:
                st.markdown(f"- `{note['created_at'][:19]}` {note['body']}")

        note_body = st.text_area("Add note", key=f"note_{incident['incident_id']}", height=60)
        if st.button("Save note", key=f"note_save_{incident['incident_id']}"):
            if note_body.strip():
                ok, _ = v1_post(
                    f"/incidents/{incident['incident_id']}/notes",
                    {"body": note_body},
                )
                st.success("Note saved") if ok else st.error("Failed")
                st.rerun()


def _render_incident_search() -> None:
    st.divider()
    st.subheader("Search Incidents")
    search_q = st.text_input("Search query", key="site_search_q")
    if not search_q:
        return

    results = v1_get("/incidents/search", params={"q": search_q, "limit": 20})
    hits = results.get("incidents", []) if results else []
    st.markdown(f"**{len(hits)} result(s)** for `{search_q}`")
    for incident in hits:
        st.markdown(
            f"- `{incident['risk_level']}` | `{incident['zone_id']}` | "
            f"`{incident['ts'][:19]}` — {incident.get('summary_text', '—')}"
        )
