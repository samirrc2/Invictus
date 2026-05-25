"""
invictus.pages.dev_analytics.llm_quality
=========================================
LLM Quality — reliability metrics, sentiment analysis, token consumption.
"""
import streamlit as st

from invictus.design import (
    render_section_header,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED,
)
from invictus.pages.dev_analytics._shared import subtitle, conviction_card, health_color


def render_llm_quality():
    from invictus.observability.analyzers.hallucination import analyze_llm_quality

    lq = analyze_llm_quality()
    if lq.get("status") == "no_data":
        st.info("No LLM call data collected yet.")
        return

    # ── Verdict Summary ───────────────────────────────────────────
    fr = lq["fallback_rate"]
    det = lq["determinism_score"]
    bias = lq["sentiment_bias"]
    uniform = lq.get("sentiment_uniform", False)

    flags = []
    if fr > 0.2:
        flags.append(f'Fallback rate <b style="color:{DANGER_RED};">{fr:.0%}</b> — primary LLM failing frequently')
    if det < 0.8:
        flags.append(f'Determinism <b style="color:{DANGER_RED};">{det:.0%}</b> — outputs vary across identical inputs')
    if bias != "neutral":
        flags.append(f'Sentiment bias detected: <b style="color:{DANGER_RED};">{bias}</b> — model may be systematically skewed')
    if uniform:
        flags.append('Low sentiment variance — LLM may be giving templated/repetitive responses')

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
            f'{lq["total_calls"]} LLM calls, {fr:.0%} fallback rate, {det:.0%} determinism, '
            f'sentiment {bias}. No issues detected.</div>',
            unsafe_allow_html=True,
        )

    render_section_header("LLM Quality Metrics")
    subtitle(
        'Language model reliability and output quality. '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Low fallback</span> = primary model reliable; '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">high determinism</span> = consistent outputs.'
    )
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        conviction_card("Total Calls", str(lq["total_calls"]))
    with l2:
        fr = lq["fallback_rate"]
        conviction_card("Fallback Rate", f"{fr:.0%}",
                        color=SUCCESS_GREEN if fr < 0.1 else DANGER_RED,
                        sub_label="LOW" if fr < 0.1 else "HIGH" if fr > 0.5 else "MODERATE")
    with l3:
        det = lq["determinism_score"]
        conviction_card("Determinism", f"{det:.0%}",
                        color=health_color(det),
                        sub_label="CONSISTENT" if det > 0.9 else "VARIABLE")
    with l4:
        bias = lq["sentiment_bias"]
        conviction_card("Sentiment Bias", bias.upper(),
                        color=SUCCESS_GREEN if bias == "neutral" else DANGER_RED,
                        sub_label="UNBIASED" if bias == "neutral" else "SKEWED")

    render_section_header("Sentiment Analysis")
    subtitle(
        'LLM output sentiment distribution. '
        '<span style="color:#94a3b8;">Uniform variance suggests templated responses; high variance suggests inconsistency.</span>'
    )
    s1, s2 = st.columns(2)
    with s1:
        conviction_card("Mean Sentiment", f"{lq['sentiment_mean']:+.2f}")
        conviction_card("Std Dev", f"{lq['sentiment_std']:.3f}")
        if lq["sentiment_uniform"]:
            st.warning("Low variance — LLM may be giving uniform responses.")
    with s2:
        conviction_card("Avg Latency", f"{lq['latency_avg_ms']:,.0f}ms")
        conviction_card("Max Latency", f"{lq['latency_max_ms']:,.0f}ms")

    render_section_header("Token Consumption")
    subtitle(
        'Total token usage across all LLM calls. '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">Input</span> = prompts sent; '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Output</span> = responses generated.'
    )
    t1, t2, t3, t4 = st.columns(4)
    with t1: conviction_card("Total In", f"{lq['tokens_total_in']:,}")
    with t2: conviction_card("Total Out", f"{lq['tokens_total_out']:,}")
    with t3: conviction_card("Avg In/Call", f"{lq['tokens_avg_in']:,.0f}")
    with t4: conviction_card("Avg Out/Call", f"{lq['tokens_avg_out']:,.0f}")
