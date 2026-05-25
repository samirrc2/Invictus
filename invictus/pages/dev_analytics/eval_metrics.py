"""
invictus.pages.dev_analytics.eval_metrics
==========================================
Eval Metrics — cost intelligence, consistency & reliability, determinism, cross-agent agreement.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from invictus.design import (
    render_section_header, apply_invictus_layout,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED,
)
from invictus.pages.dev_analytics._shared import subtitle, conviction_card, health_color


def render_eval_metrics():
    # ── Verdict Summary ───────────────────────────────────────────
    try:
        from invictus.evaluation.consistency_evaluator import (
            analyze_determinism, analyze_workflow_reliability,
        )
        reliability = analyze_workflow_reliability()
        det = analyze_determinism()

        rc = reliability.get("composite_reliability", 0)
        grade = det.get("grade", "?")
        fbr = reliability.get("fallback_rate", 0)

        flags = []
        if rc < 0.85:
            flags.append(f'Reliability score <b style="color:{DANGER_RED};">{rc:.0%}</b> — below production threshold')
        if grade in ("D", "F"):
            flags.append(f'Determinism grade <b style="color:{DANGER_RED};">{grade}</b> — outputs not reproducible')
        if fbr > 0.15:
            flags.append(f'Fallback rate <b style="color:{DANGER_RED};">{fbr:.0%}</b> — primary model failing too often')

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
                f'Reliability {rc:.0%}, determinism grade {grade}, fallback rate {fbr:.0%}. '
                f'Pipeline meets production quality standards.</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass  # Verdict unavailable — no data yet

    render_section_header("Evaluation Framework")
    subtitle(
        'Multi-dimensional evaluation of pipeline quality — '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">cost</span>, '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">consistency</span>, '
        f'<span style="color:#6366f1;font-weight:700;">reliability</span>.'
    )

    # ── 1. Cost Modeling ──────────────────────────────────────────
    _render_cost_intelligence()

    st.divider()

    # ── 2. Consistency & Reliability ──────────────────────────────
    _render_consistency_reliability()


# ── Cost Intelligence ──────────────────────────────────────────────────

def _render_cost_intelligence():
    render_section_header("Cost Intelligence")
    subtitle('Per-ticker and per-agent cost analysis with optimization recommendations.')

    try:
        from invictus.evaluation.cost_analyzer import (
            analyze_cost_per_ticker, analyze_agent_cost_breakdown,
            analyze_cost_per_run, analyze_latency_waterfall,
            analyze_prompt_caching_opportunity,
        )

        cpt = analyze_cost_per_ticker()
        if cpt.get("tickers"):
            e1, e2, e3, e4 = st.columns(4)
            with e1: conviction_card("Tickers Analyzed", str(cpt["ticker_count"]))
            with e2: conviction_card("Total Cost", f"${cpt['total_cost']:.4f}")
            with e3: conviction_card("Avg $/Ticker", f"${cpt['avg_cost_per_ticker']:.4f}")
            with e4: conviction_card("Most Expensive", cpt["most_expensive"],
                                      color=DANGER_RED, sub_label="HIGHEST COST")

            with st.container(border=True):
                tickers = [t["ticker"] for t in cpt["tickers"]]
                costs = [t["cost"] for t in cpt["tickers"]]
                fig = go.Figure(go.Bar(
                    x=costs, y=tickers, orientation="h", marker_color=BRAND_BLUE,
                    text=[f"${c:.4f}" for c in costs], textposition="outside",
                ))
                apply_invictus_layout(fig, height=max(200, len(tickers) * 30), title="Cost per Ticker ($)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        acb = analyze_agent_cost_breakdown()
        if acb.get("agents"):
            render_section_header("Agent Cost Breakdown")
            subtitle('Percentage of total cost consumed by each agent.')
            with st.container(border=True):
                agents = [a["agent"] for a in acb["agents"]]
                pcts = [a["pct_of_total"] for a in acb["agents"]]
                fig = go.Figure(go.Pie(
                    labels=agents, values=pcts, hole=0.5,
                    textinfo="label+percent",
                ))
                apply_invictus_layout(fig, height=280)
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        wf = analyze_latency_waterfall()
        if wf.get("waterfall"):
            render_section_header("Latency Waterfall (Latest Run)")
            subtitle(
                'Agent-by-agent execution time for the most recent pipeline run. '
                f'<span style="color:{DANGER_RED};font-weight:700;">Bottleneck</span> = '
                f'{wf["bottleneck"]} ({wf["bottleneck_pct"]:.0%} of total).'
            )
            wa1, wa2 = st.columns(2)
            with wa1: conviction_card("Total Latency", f"{wf['total_latency_ms']:,.0f}ms")
            with wa2: conviction_card("Bottleneck", wf["bottleneck"],
                                       color=DANGER_RED, sub_label=f"{wf['bottleneck_pct']:.0%} OF TOTAL")

            with st.container(border=True):
                names = [w["agent"] for w in wf["waterfall"]]
                lats = [w["latency_ms"] for w in wf["waterfall"]]
                colors = [DANGER_RED if w["status"] == "error" else BRAND_BLUE for w in wf["waterfall"]]
                fig = go.Figure(go.Waterfall(
                    y=names, x=lats, orientation="h",
                    connector={"line": {"color": "#e2e8f0"}},
                    textposition="outside",
                    text=[f"{v:,.0f}ms" for v in lats],
                ))
                apply_invictus_layout(fig, height=max(250, len(names) * 30), title="Agent Latency Waterfall (ms)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        cache = analyze_prompt_caching_opportunity()
        if cache.get("cacheable_calls", 0) > 0:
            render_section_header("Optimization: Prompt Caching")
            subtitle(
                f'<span style="color:{SUCCESS_GREEN};font-weight:700;">{cache["cacheable_calls"]}</span> '
                'duplicate LLM calls detected — caching could save '
                f'<span style="color:{SUCCESS_GREEN};font-weight:700;">{cache["savings_pct"]:.0%}</span> of cost.'
            )
            with st.container(border=True):
                o1, o2, o3 = st.columns(3)
                with o1: conviction_card("Before", f"${cache['current_total_cost']:.4f}", color=DANGER_RED)
                with o2: conviction_card("After (Cached)", f"${cache['optimized_cost']:.4f}", color=SUCCESS_GREEN)
                with o3: conviction_card("Savings", f"{cache['savings_pct']:.0%}", color=SUCCESS_GREEN,
                                          sub_label="OPTIMIZABLE")
    except Exception as e:
        st.warning(f"Cost analysis unavailable: {e}")


# ── Consistency & Reliability ──────────────────────────────────────────

def _render_consistency_reliability():
    render_section_header("Consistency & Reliability")
    subtitle('Pipeline determinism, answer stability, and cross-agent agreement.')

    try:
        from invictus.evaluation.consistency_evaluator import (
            analyze_answer_stability, analyze_cross_agent_consistency,
            analyze_determinism, analyze_workflow_reliability,
        )

        reliability = analyze_workflow_reliability()
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            rc = reliability["composite_reliability"]
            conviction_card("Reliability Score", f"{rc:.0%}",
                            color=health_color(rc),
                            sub_label="PRODUCTION-READY" if rc > 0.9 else "NEEDS WORK")
        with r2: conviction_card("Pipeline Completion", f"{reliability['pipeline_completion_rate']:.0%}")
        with r3: conviction_card("Agent Success", f"{reliability['agent_success_rate']:.0%}")
        with r4:
            fbr = reliability["fallback_rate"]
            conviction_card("Fallback Rate", f"{fbr:.0%}",
                            color=SUCCESS_GREEN if fbr < 0.1 else DANGER_RED)

        det = analyze_determinism()
        render_section_header("Determinism Scoring")
        subtitle(
            'How reproducible are outputs across identical inputs? '
            f'Grade: <span style="font-weight:700;">{det["grade"]}</span>.'
        )
        d1, d2, d3 = st.columns(3)
        with d1: conviction_card("LLM Determinism", f"{det['llm_determinism']:.0%}")
        with d2: conviction_card("Conviction Determinism", f"{det['conviction_determinism']:.0%}")
        with d3:
            cd = det["composite_determinism"]
            conviction_card(f"Composite — Grade {det['grade']}", f"{cd:.0%}",
                            color=health_color(cd))

        stab = analyze_answer_stability()
        if stab.get("tickers"):
            render_section_header("Answer Stability by Ticker")
            subtitle(
                'Conviction score variance per ticker across runs. '
                f'<span style="color:{SUCCESS_GREEN};font-weight:700;">HIGH</span> stability = consistent signals.'
            )
            stab_df = pd.DataFrame(stab["tickers"])
            st.dataframe(
                stab_df[["ticker", "runs", "mean_score", "std_score", "cv", "stability"]].style.format({
                    "mean_score": "{:.3f}", "std_score": "{:.3f}", "cv": "{:.3f}",
                }).applymap(
                    lambda v: f"color: {SUCCESS_GREEN}" if v == "HIGH"
                    else f"color: {DANGER_RED}" if v == "LOW" else "",
                    subset=["stability"]
                ),
                use_container_width=True, hide_index=True,
            )

        xac = analyze_cross_agent_consistency()
        if xac.get("total_checked", 0) > 0:
            render_section_header("Cross-Agent Consistency")
            subtitle(
                'Do different agents agree on the same ticker? '
                f'Grade: <span style="font-weight:700;">{xac["grade"]}</span>.'
            )
            x1, x2 = st.columns(2)
            with x1:
                conviction_card(f"Consistency — Grade {xac['grade']}", f"{xac['consistency_rate']:.0%}",
                                color=health_color(xac["consistency_rate"]))
            with x2: conviction_card("Inconsistencies", str(xac["inconsistency_count"]),
                                      color=DANGER_RED if xac["inconsistency_count"] > 0 else SUCCESS_GREEN)

            if xac.get("inconsistencies"):
                with st.expander(f"{xac['inconsistency_count']} Signal Disagreements Detected"):
                    for inc in xac["inconsistencies"][:5]:
                        st.markdown(
                            f"**{inc['ticker']}** — Composite: {inc['composite']:.2f} ({inc['composite_direction']}), "
                            f"Filing: {inc['filing']:.2f}, Earnings: {inc['earnings']:.2f}, "
                            f"Flow: {inc['flow']:.2f}, ML: {inc['ml']:.2f}"
                        )
    except Exception as e:
        st.warning(f"Consistency analysis unavailable: {e}")
