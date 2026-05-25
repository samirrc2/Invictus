"""
invictus.pages.dev_analytics.session
=====================================
Session Analytics — usage overview, feature adoption, pipeline breakdown.
"""
import streamlit as st

from invictus.design import render_section_header, BRAND_BLUE
from invictus.pages.dev_analytics._shared import subtitle, conviction_card, health_color


def render_session_analytics():
    # ── Pipeline Error Log (always visible if errors exist) ──────
    errors = st.session_state.get("pipeline_errors", [])
    if errors:
        render_section_header("Pipeline Error Log")
        subtitle(
            f'<span style="color:#ef4444;font-weight:700;">{len(errors)} error(s)</span> '
            f'captured from the last pipeline run. Full tracebacks shown below.'
        )
        for i, err in enumerate(errors):
            with st.expander(f"Error {i + 1}: {err[:80].split(chr(10))[0]}", expanded=(i == 0)):
                st.code(err, language="python")
        st.markdown('<div style="margin-bottom:16px;"></div>', unsafe_allow_html=True)

    from invictus.observability.analyzers.calibration import analyze_session_analytics

    sa = analyze_session_analytics()
    if sa.get("status") == "no_data":
        if not errors:
            st.info("No session data collected yet.")
        return

    # ── Verdict Summary ───────────────────────────────────────────
    pcr = sa["pipeline_completion_rate"]
    demo_pct = sa["demo_mode_pct"]

    flags = []
    if pcr < 0.8:
        flags.append(f'Pipeline completion <b style="color:#ef4444;">{pcr:.0%}</b> — pipelines failing before finishing')
    if demo_pct > 0.8:
        flags.append(f'{demo_pct:.0%} of runs are demo mode — limited real data validation')
    avg_time = (sa.get("pipeline_avg_time_ms") or 0) / 1000
    if avg_time > 120:
        flags.append(f'Avg pipeline time <b>{avg_time:.0f}s</b> — may need optimization')

    if flags:
        flag_html = "".join(f'<div style="margin:2px 0;">⚠ {f}</div>' for f in flags)
        st.markdown(
            f'<div style="background:#fef2f211;border-left:3px solid #ef4444;padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:#ef4444;">Red Flags</b>{flag_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:#22c55e08;border-left:3px solid #22c55e;padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:#22c55e;">All Clear</b> — '
            f'{sa["unique_sessions"]} sessions, {pcr:.0%} completion rate, '
            f'avg pipeline time {avg_time:.1f}s.</div>',
            unsafe_allow_html=True,
        )

    render_section_header("Usage Overview")
    subtitle(
        'Platform usage metrics and feature adoption. '
        '<span style="color:#94a3b8;">Pipeline completion rate measures end-to-end reliability.</span>'
    )
    u1, u2, u3, u4 = st.columns(4)
    with u1: conviction_card("Sessions", str(sa["unique_sessions"]))
    with u2: conviction_card("Events", str(sa["total_events"]))
    with u3: conviction_card("Pipeline Runs", str(sa["pipeline_starts"]))
    with u4:
        pcr = sa["pipeline_completion_rate"]
        conviction_card("Completion", f"{pcr:.0%}", color=health_color(pcr))

    features = sa.get("feature_adoption", [])
    if features:
        render_section_header("Feature Adoption")
        subtitle('Most-used platform features ranked by usage count.')
        for f in features[:10]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:6px 12px;'
                f'border-bottom:1px solid #f1f5f9;">'
                f'<span style="font-size:13px;color:#334155;">{f["detail"]}</span>'
                f'<span style="font-size:13px;font-weight:700;color:{BRAND_BLUE};'
                f'font-variant-numeric:tabular-nums;">{f["uses"]}</span></div>',
                unsafe_allow_html=True,
            )

    render_section_header("Pipeline Breakdown")
    subtitle('Execution mode and timing statistics.')
    p1, p2, p3 = st.columns(3)
    with p1: conviction_card("Demo Mode %", f"{sa['demo_mode_pct']:.0%}")
    with p2: conviction_card("Avg Time", f"{(sa['pipeline_avg_time_ms'] or 0) / 1000:.1f}s")
    with p3: conviction_card("Completion", f"{sa['pipeline_completion_rate']:.0%}")
