"""
invictus.pages.dev_analytics.cost
==================================
Cost Analytics — token consumption, per-agent cost, prompt caching opportunities.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from invictus.design import (
    render_section_header, apply_invictus_layout,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED,
)
from invictus.pages.dev_analytics._shared import subtitle, conviction_card


def render_cost_analytics():
    from invictus.observability.store import query, query_one

    token_data = query_one(
        "SELECT SUM(tokens_in) as tin, SUM(tokens_out) as tout, COUNT(*) as calls "
        "FROM llm_calls"
    ) or {"tin": 0, "tout": 0, "calls": 0}
    tin = token_data["tin"] or 0
    tout = token_data["tout"] or 0
    total_tokens = tin + tout

    cost_in = (tin / 1_000_000) * 2.50
    cost_out = (tout / 1_000_000) * 10.00
    total_cost = cost_in + cost_out
    calls = token_data["calls"] or 0

    # ── Verdict Summary ───────────────────────────────────────────
    flags = []
    if total_cost > 1.0:
        flags.append(f'Total cost <b style="color:{DANGER_RED};">${total_cost:.2f}</b> — review token-heavy agents')
    if calls > 0 and tout / max(calls, 1) > 2000:
        flags.append(f'Avg {tout / calls:.0f} output tokens/call — consider shorter prompts or summaries')
    if calls > 0 and cost_out > cost_in * 3:
        flags.append('Output cost 3x+ input cost — output-heavy; consider response length limits')

    if total_cost == 0:
        st.markdown(
            f'<div style="background:{BRAND_BLUE}08;border-left:3px solid {BRAND_BLUE};padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:{BRAND_BLUE};">No Data</b> — '
            f'No LLM token data recorded yet. Cost tracking begins after the first pipeline run.</div>',
            unsafe_allow_html=True,
        )
    elif flags:
        flag_html = "".join(f'<div style="margin:2px 0;">⚠ {f}</div>' for f in flags)
        st.markdown(
            f'<div style="background:#fef2f211;border-left:3px solid {DANGER_RED};padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:{DANGER_RED};">Cost Alerts</b>{flag_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        cost_per_call = total_cost / max(calls, 1)
        st.markdown(
            f'<div style="background:{SUCCESS_GREEN}08;border-left:3px solid {SUCCESS_GREEN};padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:{SUCCESS_GREEN};">Cost Healthy</b> — '
            f'${total_cost:.4f} total across {calls} calls (${cost_per_call:.4f}/call). '
            f'Output/input ratio reasonable.</div>',
            unsafe_allow_html=True,
        )

    render_section_header("API Cost Analytics")
    subtitle(
        'Token consumption and cost estimation using GPT-4o pricing '
        '($2.50/1M input, $10/1M output). '
        '<span style="color:#94a3b8;">Optimize by reducing output tokens and enabling prompt caching.</span>'
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: conviction_card("Total Tokens", f"{total_tokens:,}")
    with c2: conviction_card("Tokens In", f"{tin:,}")
    with c3: conviction_card("Tokens Out", f"{tout:,}")
    with c4: conviction_card("Est. Total Cost", f"${total_cost:.4f}",
                              color=DANGER_RED if total_cost > 1 else SUCCESS_GREEN)

    render_section_header("Cost Breakdown")
    subtitle('Input vs output cost split.')
    cb1, cb2, cb3 = st.columns(3)
    with cb1: conviction_card("Input Cost", f"${cost_in:.4f}")
    with cb2: conviction_card("Output Cost", f"${cost_out:.4f}")
    with cb3:
        calls = token_data["calls"] or 1
        conviction_card("Cost per Call", f"${total_cost / calls:.4f}")

    # Per-agent token usage
    agent_tokens = query(
        "SELECT agent_name, SUM(tokens_in) as tin, SUM(tokens_out) as tout, "
        "COUNT(*) as calls, AVG(latency_ms) as avg_lat "
        "FROM llm_calls GROUP BY agent_name ORDER BY (SUM(tokens_in) + SUM(tokens_out)) DESC"
    )
    if agent_tokens:
        render_section_header("Token Usage by Agent")
        subtitle('Per-agent token consumption and estimated cost.')
        with st.container(border=True):
            names = [r["agent_name"] for r in agent_tokens]
            totals = [(r["tin"] or 0) + (r["tout"] or 0) for r in agent_tokens]
            costs = [((r["tin"] or 0) / 1e6 * 2.50 + (r["tout"] or 0) / 1e6 * 10.0) for r in agent_tokens]
            fig = go.Figure(go.Bar(
                x=totals, y=names, orientation="h", marker_color=BRAND_BLUE,
                text=[f"{v:,} tok (${c:.3f})" for v, c in zip(totals, costs)],
                textposition="outside",
            ))
            apply_invictus_layout(fig, height=max(250, len(names) * 35), title="Token Consumption by Agent")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        at_df = pd.DataFrame(agent_tokens)
        at_df["total_tokens"] = (at_df["tin"].fillna(0) + at_df["tout"].fillna(0)).astype(int)
        at_df["est_cost"] = at_df["tin"].fillna(0) / 1e6 * 2.50 + at_df["tout"].fillna(0) / 1e6 * 10.0
        st.dataframe(
            at_df[["agent_name", "calls", "tin", "tout", "total_tokens", "est_cost", "avg_lat"]].style.format({
                "tin": "{:,.0f}", "tout": "{:,.0f}", "total_tokens": "{:,}",
                "est_cost": "${:.4f}", "avg_lat": "{:.0f}ms",
            }),
            use_container_width=True, hide_index=True,
        )

    # Cost trending
    run_costs = query(
        "SELECT run_id, SUM(tokens_in) as tin, SUM(tokens_out) as tout, "
        "MIN(created_at) as ts "
        "FROM llm_calls WHERE run_id IS NOT NULL "
        "GROUP BY run_id ORDER BY ts DESC LIMIT 20"
    )
    if run_costs:
        render_section_header("Cost per Pipeline Run")
        subtitle('Estimated cost trend over recent pipeline executions.')
        with st.container(border=True):
            run_totals = [((r["tin"] or 0) / 1e6 * 2.50 + (r["tout"] or 0) / 1e6 * 10.0) for r in run_costs]
            fig_c = go.Figure(go.Scatter(
                x=list(range(len(run_costs))), y=run_totals,
                mode="lines+markers", line=dict(color=BRAND_BLUE, width=2),
                text=[f"${v:.4f}" for v in run_totals],
            ))
            apply_invictus_layout(fig_c, height=250, title="Estimated Cost per Run ($)")
            st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})

    if total_tokens == 0:
        st.info("No LLM token data recorded yet. Cost tracking begins after the first LLM-powered pipeline run.")
