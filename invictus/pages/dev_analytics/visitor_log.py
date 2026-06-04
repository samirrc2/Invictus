"""
invictus.pages.dev_analytics.visitor_log
=========================================
Visitor Activity Log — IP geolocation, session tracking, access patterns.
Shows who accessed Invictus, from where, and what they did.
"""
import streamlit as st

from invictus.design import render_section_header, BRAND_BLUE, SUCCESS_GREEN, DANGER_RED
from invictus.pages.dev_analytics._shared import subtitle, conviction_card


def render_visitor_log():
    """Render the Visitor Log tab in Dev Console."""
    from invictus.observability.collectors.visitor_collector import get_visitor_analytics

    va = get_visitor_analytics()

    # ── Verdict Summary ──────────────────────────────────────────────
    total = va["total_sessions"]
    unique = va["unique_ips"]
    countries = va["unique_countries"]
    runs = va["total_pipeline_runs"]

    if total == 0:
        st.markdown(
            '<div style="background:#f0fdf4;border:1px solid #bbf7d0;'
            'border-radius:10px;padding:20px 24px;text-align:center;">'
            '<div style="font-size:15px;font-weight:800;color:#16a34a;'
            'margin-bottom:4px;">Awaiting Visitors</div>'
            '<div style="font-size:13px;color:#64748b;">'
            'No visitor sessions recorded yet. Visitor tracking is active — '
            'the first access will appear here with IP, location, and session details.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    # Summary banner
    verdict_color = SUCCESS_GREEN if unique > 1 else BRAND_BLUE
    st.markdown(
        f'<div style="background:{verdict_color}08;border-left:3px solid {verdict_color};'
        f'padding:10px 14px;border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
        f'<b style="color:{verdict_color};">{total} Session(s) Tracked</b> — '
        f'{unique} unique IP(s) from {countries} countr{"y" if countries == 1 else "ies"}, '
        f'{runs} pipeline run(s) triggered.</div>',
        unsafe_allow_html=True,
    )

    # ── Summary Cards ────────────────────────────────────────────────
    render_section_header("Visitor Overview")
    subtitle('Session counts, unique visitors, geographic reach, and pipeline engagement.')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        conviction_card("Total Sessions", str(total))
    with c2:
        conviction_card("Unique IPs", str(unique), color=BRAND_BLUE)
    with c3:
        conviction_card("Countries", str(countries), color=BRAND_BLUE)
    with c4:
        conviction_card("Pipeline Runs", str(runs), color=SUCCESS_GREEN if runs > 0 else "#94a3b8")

    # ── Geographic Breakdown ─────────────────────────────────────────
    top_countries = va["top_countries"]
    top_cities = va["top_cities"]

    if top_countries:
        render_section_header("Geographic Distribution")
        subtitle('Top access locations by country and city.')

        gc, cc = st.columns(2)
        with gc:
            st.markdown(
                '<div style="font-size:12px;font-weight:700;color:#0f172a;'
                'margin-bottom:6px;">By Country</div>',
                unsafe_allow_html=True,
            )
            for row in top_countries:
                pct = row["visits"] / max(total, 1) * 100
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:5px 12px;border-bottom:1px solid #f1f5f9;">'
                    f'<span style="font-size:12px;color:#334155;">{row["country"]}</span>'
                    f'<span style="font-size:12px;font-weight:700;color:{BRAND_BLUE};'
                    f'font-variant-numeric:tabular-nums;">{row["visits"]} '
                    f'<span style="color:#94a3b8;font-weight:400;">({pct:.0f}%)</span></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with cc:
            st.markdown(
                '<div style="font-size:12px;font-weight:700;color:#0f172a;'
                'margin-bottom:6px;">By City</div>',
                unsafe_allow_html=True,
            )
            for row in top_cities:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:5px 12px;border-bottom:1px solid #f1f5f9;">'
                    f'<span style="font-size:12px;color:#334155;">'
                    f'{row["city"]}, {row["country"]}</span>'
                    f'<span style="font-size:12px;font-weight:700;color:{BRAND_BLUE};'
                    f'font-variant-numeric:tabular-nums;">{row["visits"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Recent Visitor Table ─────────────────────────────────────────
    render_section_header("Recent Sessions")
    subtitle('Last 100 visitor sessions with IP, location, user agent, and activity.')

    visitors = va["recent_visitors"]
    if not visitors:
        st.caption("No sessions recorded.")
        return

    for i, v in enumerate(visitors):
        ip = v.get("ip_address", "unknown")
        city = v.get("city", "")
        country = v.get("country", "")
        region = v.get("region", "")
        isp = v.get("isp", "")
        ua = v.get("user_agent", "unknown")
        ref = v.get("referrer", "")
        tickers = v.get("tickers_loaded", "")
        runs = v.get("pipeline_runs", 0)
        is_demo = v.get("is_demo", 0)
        start = v.get("session_start", "")
        last = v.get("last_active", "")

        # Location string
        loc_parts = [p for p in [city, region, country] if p]
        location = ", ".join(loc_parts) if loc_parts else "Unknown"

        # Time display
        ts = start[:19].replace("T", " ") if start else ""

        # Mode badge
        mode_badge = (
            f'<span style="background:#fef2f2;color:#dc2626;font-size:10px;'
            f'font-weight:700;padding:2px 8px;border-radius:3px;">DEMO</span>'
            if is_demo else
            f'<span style="background:#f0fdf4;color:#16a34a;font-size:10px;'
            f'font-weight:700;padding:2px 8px;border-radius:3px;">LIVE</span>'
            if runs > 0 else ""
        )

        # Pipeline info
        pipeline_info = ""
        if runs > 0:
            pipeline_info = (
                f'<span style="font-size:11px;color:{BRAND_BLUE};font-weight:700;">'
                f'{runs} run{"s" if runs > 1 else ""}</span>'
            )
            if tickers:
                pipeline_info += (
                    f' <span style="font-size:11px;color:#94a3b8;">({tickers})</span>'
                )

        # Shorten user agent for display
        ua_short = ua[:80] + "..." if len(ua) > 80 else ua

        # Pre-build optional HTML fragments
        isp_html = f"<span>ISP: {isp}</span>" if isp else ""
        ref_short = ref[:60] if ref else ""
        ref_html = (
            f'<span>Ref: <span style="color:#64748b;">{ref_short}</span></span>'
            if ref else ""
        )

        st.markdown(
            f'<div style="border:1px solid #e2e8f0;border-radius:8px;'
            f'padding:12px 16px;margin-bottom:6px;background:#fafbfc;">'
            # Row 1: IP + Location + Timestamp + Mode badge
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:6px;">'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:13px;font-weight:800;color:#0f172a;'
            f'font-family:monospace;">{ip}</span>'
            f'<span style="font-size:12px;color:#64748b;">{location}</span>'
            f'{mode_badge}'
            f'</div>'
            f'<span style="font-size:11px;color:#94a3b8;font-family:monospace;">{ts}</span>'
            f'</div>'
            # Row 2: ISP + Pipeline + Referrer
            f'<div style="display:flex;gap:16px;font-size:11px;color:#94a3b8;">'
            f'{isp_html}{pipeline_info}{ref_html}'
            f'</div>'
            # Row 3: User Agent
            f'<div style="font-size:10px;color:#cbd5e1;margin-top:4px;'
            f'font-family:monospace;">{ua_short}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
