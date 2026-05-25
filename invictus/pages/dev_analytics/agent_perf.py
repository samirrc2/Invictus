"""
invictus.pages.dev_analytics.agent_perf
========================================
Agent Performance — orchestration overview, latency breakdown, trend.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from invictus.design import (
    render_section_header, apply_invictus_layout,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED,
)
from invictus.pages.dev_analytics._shared import subtitle, conviction_card, health_color


def render_agent_performance():
    from invictus.observability.analyzers.calibration import analyze_agent_performance

    ap = analyze_agent_performance()
    if ap.get("status") == "no_data":
        st.info("No agent run data collected yet.")
        return

    # ── Verdict Summary ───────────────────────────────────────────
    sr = ap["success_rate"]
    bn = ap.get("bottleneck")
    err_types = ap.get("error_types", {})
    agent_stats = ap.get("agent_stats", [])
    failing = [a for a in agent_stats if a["runs"] > 0 and (a["successes"] / max(a["runs"], 1)) < 0.8]

    flags = []
    if sr < 0.9:
        flags.append(f'Pipeline success rate <b style="color:{DANGER_RED};">{sr:.0%}</b> — below 90% threshold')
    if bn and bn.get("avg_latency", 0) > 5000:
        flags.append(f'<b>{bn["agent_name"]}</b> averaging {bn["avg_latency"]:,.0f}ms — bottleneck candidate')
    if failing:
        names = ", ".join(a["agent_name"] for a in failing)
        flags.append(f'Agents with <80% success: <b style="color:{DANGER_RED};">{names}</b>')
    if err_types:
        top_err = max(err_types, key=err_types.get)
        flags.append(f'Most common error: <b>{top_err}</b> ({err_types[top_err]} occurrences)')

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
            f'{sr:.0%} success rate across {ap["total_runs"]} runs. '
            f'No agents below 80% threshold.</div>',
            unsafe_allow_html=True,
        )

    render_section_header("Agent Orchestration Overview")
    subtitle(
        'Pipeline execution health across all agents. '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">High success rate</span> = reliable pipeline; '
        f'<span style="color:{DANGER_RED};font-weight:700;">bottleneck agent</span> = optimization target.'
    )
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        conviction_card("Total Runs", str(ap["total_runs"]))
    with a2:
        sr = ap["success_rate"]
        conviction_card("Success Rate", f"{sr:.0%}", color=health_color(sr),
                        sub_label="HEALTHY" if sr > 0.9 else "DEGRADED")
    with a3:
        bn = ap.get("bottleneck")
        conviction_card("Bottleneck", bn["agent_name"] if bn else "N/A",
                        sub_label=f"{bn['avg_latency']:,.0f}ms avg" if bn else "")
    with a4:
        conviction_card("Error Types", str(len(ap.get("error_types", {}))))

    agent_stats = ap.get("agent_stats", [])
    if agent_stats:
        render_section_header("Agent Latency Breakdown")
        subtitle(
            'Per-agent execution time. '
            f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Green</span> = >95% success; '
            f'<span style="color:{DANGER_RED};font-weight:700;">red</span> = <80% success. '
            '<span style="color:#94a3b8;">Longest bars are optimization candidates.</span>'
        )
        with st.container(border=True):
            names = [r["agent_name"] for r in agent_stats]
            latencies = [r["avg_latency"] for r in agent_stats]
            sr_list = [r["successes"] / max(r["runs"], 1) for r in agent_stats]
            colors = [SUCCESS_GREEN if s > 0.95 else DANGER_RED if s < 0.8 else BRAND_BLUE for s in sr_list]
            fig = go.Figure(go.Bar(
                x=latencies, y=names, orientation="h", marker_color=colors,
                text=[f"{v:,.0f}ms" for v in latencies], textposition="outside",
            ))
            apply_invictus_layout(fig, height=max(300, len(names) * 30), title="Avg Latency (ms)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        df = pd.DataFrame(agent_stats)
        df["success_rate"] = df["successes"] / df["runs"]
        st.dataframe(
            df[["agent_name", "runs", "avg_latency", "max_latency", "successes", "errors", "fallbacks", "success_rate"]].style.format({
                "avg_latency": "{:.0f}ms", "max_latency": "{:.0f}ms", "success_rate": "{:.0%}",
            }),
            use_container_width=True, hide_index=True,
        )

    trend = ap.get("latency_trend", [])
    if trend:
        render_section_header("Pipeline Latency Trend")
        subtitle(
            'Total pipeline execution time over recent runs. '
            '<span style="color:#94a3b8;">Upward trend = potential resource contention or data growth.</span>'
        )
        with st.container(border=True):
            fig_t = go.Figure(go.Scatter(
                x=list(range(len(trend))), y=[r["total_latency"] for r in trend],
                mode="lines+markers", line=dict(color=BRAND_BLUE, width=2),
            ))
            apply_invictus_layout(fig_t, height=250, title="Total Pipeline Time (ms) — Last 20 Runs")
            st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})
