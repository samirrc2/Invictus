"""
invictus.pages.dev_analytics.conviction_analytics
===================================================
Conviction Analytics — signal synthesis stability, agreement patterns, driver frequency.
"""
import streamlit as st
import plotly.graph_objects as go

from invictus.design import (
    render_section_header, apply_invictus_layout,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED,
)
from invictus.pages.dev_analytics._shared import subtitle, conviction_card


def render_conviction_analytics():
    from invictus.observability.analyzers.drift import analyze_conviction_stability

    cs = analyze_conviction_stability()
    if cs.get("status") == "no_data":
        st.info("No conviction data collected yet.")
        return

    # ── Verdict Summary ───────────────────────────────────────────
    ci = cs.get("ci_width_stats", {})
    avg_ci = ci.get("avg_width", 1)
    agree_dist = cs.get("agreement_distribution", {})
    divergent = agree_dist.get("SIGNAL DIVERGENCE", 0) + agree_dist.get("MIXED SIGNALS", 0)
    convergent = agree_dist.get("STRONG CONVERGENCE", 0) + agree_dist.get("MODERATE AGREEMENT", 0)
    total_agree = max(divergent + convergent, 1)
    driver_dist = cs.get("dominant_driver_distribution", {})
    top_driver_pct = max(driver_dist.values()) / max(sum(driver_dist.values()), 1) if driver_dist else 0

    flags = []
    if avg_ci > 0.3:
        flags.append(f'Wide confidence intervals (avg {avg_ci:.3f}) — conviction engine is uncertain')
    if divergent / total_agree > 0.4:
        flags.append(f'{divergent}/{total_agree} observations show signal divergence or mixed signals')
    if top_driver_pct > 0.7 and driver_dist:
        top_driver = max(driver_dist, key=driver_dist.get)
        flags.append(f'<b>{top_driver}</b> dominates {top_driver_pct:.0%} of scores — over-reliance on single signal')

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
            f'{cs["total_scores"]} conviction scores, CI width {avg_ci:.3f}, '
            f'{convergent}/{total_agree} signals converge. Signal mix healthy.</div>',
            unsafe_allow_html=True,
        )

    render_section_header("Conviction Engine Health")
    subtitle(
        'Signal synthesis stability and agreement patterns. '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Convergent</span> signals strengthen conviction; '
        f'<span style="color:{DANGER_RED};font-weight:700;">divergent</span> signals require investigation.'
    )
    c1, c2, c3 = st.columns(3)
    with c1: conviction_card("Total Scores", str(cs["total_scores"]))
    with c2: conviction_card("Data Points", str(cs["signal_data_points"]))
    with c3:
        ci = cs.get("ci_width_stats", {})
        conviction_card("Avg CI Width", f"{ci.get('avg_width', 0):.3f}",
                        sub_label="INFORMATIVE" if ci.get("avg_width", 1) < 0.3 else "WIDE")

    agree_dist = cs.get("agreement_distribution", {})
    if agree_dist:
        render_section_header("Signal Agreement Distribution")
        subtitle(
            'How often the 4 conviction signals (filing, earnings, flows, ML) agree on direction.'
        )
        with st.container(border=True):
            colors_map = {"STRONG CONVERGENCE": SUCCESS_GREEN, "MODERATE AGREEMENT": BRAND_BLUE,
                          "MIXED SIGNALS": "#f59e0b", "SIGNAL DIVERGENCE": DANGER_RED}
            fig = go.Figure(go.Bar(
                x=list(agree_dist.keys()), y=list(agree_dist.values()),
                marker_color=[colors_map.get(l, BRAND_BLUE) for l in agree_dist],
                text=list(agree_dist.values()), textposition="outside",
            ))
            apply_invictus_layout(fig, height=280, title="Signal Agreement Frequency")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    driver_dist = cs.get("dominant_driver_distribution", {})
    if driver_dist:
        render_section_header("Dominant Driver Frequency")
        subtitle(
            'Which signal most often dominates the composite conviction score. '
            '<span style="color:#94a3b8;">Over-reliance on one signal = vulnerability to single-source failure.</span>'
        )
        with st.container(border=True):
            fig = go.Figure(go.Pie(
                labels=list(driver_dist.keys()), values=list(driver_dist.values()),
                hole=0.5, textinfo="label+percent",
            ))
            apply_invictus_layout(fig, height=280)
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
