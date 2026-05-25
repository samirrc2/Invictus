"""
invictus.pages.dev_analytics.data_health
=========================================
Data Health — external data source reliability monitoring.
"""
import streamlit as st
import pandas as pd

from invictus.design import (
    render_section_header,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED,
)
from invictus.pages.dev_analytics._shared import subtitle, conviction_card, health_color


def render_data_health():
    from invictus.observability.analyzers.calibration import analyze_data_health

    dh = analyze_data_health()
    if dh.get("status") == "no_data":
        st.info("No data health metrics collected yet.")
        return

    # ── Verdict Summary ───────────────────────────────────────────
    sr = dh["success_rate"]
    source_stats = dh.get("source_stats", [])
    failing_sources = [s for s in source_stats if s.get("fetches", 0) > 0
                       and s.get("successes", 0) / max(s["fetches"], 1) < 0.8]

    flags = []
    if sr < 0.9:
        flags.append(f'Overall success rate <b style="color:{DANGER_RED};">{sr:.0%}</b> — data feeds degraded')
    for src in failing_sources:
        src_sr = src["successes"] / max(src["fetches"], 1)
        flags.append(f'<b>{src["source"]}</b> at {src_sr:.0%} success — API issues or rate limits')

    if flags:
        flag_html = "".join(f'<div style="margin:2px 0;">⚠ {f}</div>' for f in flags)
        st.markdown(
            f'<div style="background:#fef2f211;border-left:3px solid {DANGER_RED};padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:{DANGER_RED};">Red Flags</b>{flag_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:{SUCCESS_GREEN}08;border-left:3px solid {SUCCESS_GREEN};padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:{SUCCESS_GREEN};">All Clear</b> — '
            f'{dh["total_fetches"]} fetches at {sr:.0%} success rate. All data sources healthy.</div>',
            unsafe_allow_html=True,
        )

    render_section_header("Data Pipeline Health")
    subtitle(
        'External data source reliability. '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">High success rate</span> = stable feeds; '
        f'<span style="color:{DANGER_RED};font-weight:700;">failures</span> = API issues or rate limits.'
    )
    d1, d2 = st.columns(2)
    with d1: conviction_card("Total Fetches", str(dh["total_fetches"]))
    with d2:
        sr = dh["success_rate"]
        conviction_card("Success Rate", f"{sr:.0%}",
                        color=health_color(sr, 0.95, 0.8),
                        sub_label="HEALTHY" if sr > 0.95 else "DEGRADED")

    source_stats = dh.get("source_stats", [])
    if source_stats:
        src_df = pd.DataFrame(source_stats)
        src_df["success_rate"] = src_df["successes"] / src_df["fetches"]
        st.dataframe(
            src_df.style.format({"avg_latency": "{:.0f}ms", "avg_records": "{:.0f}", "success_rate": "{:.0%}"}),
            use_container_width=True, hide_index=True,
        )
