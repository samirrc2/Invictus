"""
invictus.pages.dev_console
==========================
Developer analytics and state debugger.
"""
import streamlit as st

from invictus.design import render_section_header, render_metric_card
from invictus.analytics.tracker import get_summary_stats as _analytics_get_summary_stats


def render(sub):
    """Render the Dev Console page."""
    render_section_header("Developer Analytics")
    stats = _analytics_get_summary_stats()
    k1, k2, k3 = st.columns(3)
    with k1: render_metric_card("Sessions", str(stats.get("total_sessions", 0)))
    with k2: render_metric_card("Views",    str(stats.get("total_page_views", 0)))
    with k3: render_metric_card("Clicks",   str(stats.get("total_clicks", 0)))

    st.markdown("### State Debugger")
    for key in ["filing_intel", "earnings_intel", "flow_signals", "conviction_synthesis"]:
        val = st.session_state.get(key)
        with st.expander(f"Debug: {key}"):
            st.write(val)
