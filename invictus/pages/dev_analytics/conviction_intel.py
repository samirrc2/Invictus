"""
invictus.pages.dev_analytics.conviction_intel
===============================================
Conviction Intelligence — deep evaluation: calibration, signal quality, hit-rate analysis.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from invictus.design import (
    render_section_header, apply_invictus_layout,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT,
)
from invictus.pages.dev_analytics._shared import subtitle, conviction_card


def render_conviction_intelligence():
    """Deep conviction evaluation — calibration, signal quality, hit-rate analysis."""
    render_section_header("Conviction Intelligence")
    subtitle(
        'Deep evaluation of conviction engine quality. '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">Calibration</span> = are probabilities accurate? '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Signal quality</span> = which signals add value?'
    )

    try:
        from invictus.observability.store import query
    except Exception:
        st.info("Observability store not available.")
        return

    # ── 1. Conviction Score Distribution ─────────────────────────
    scores = query(
        "SELECT ticker, composite_score, outperformance_prob, signal_agreement, "
        "dominant_driver, ci_width, filing_score, earnings_score, flow_score, ml_score "
        "FROM conviction_scores ORDER BY created_at DESC"
    )

    if not scores:
        st.info("No conviction scores recorded yet. Run the predictive intelligence pipeline first.")
        return

    df = pd.DataFrame(scores)

    # Overview metrics
    n_scores = len(df)
    avg_score = df["composite_score"].mean() if "composite_score" in df.columns else 0
    avg_prob = df["outperformance_prob"].mean() if "outperformance_prob" in df.columns else 0
    unique_tickers = df["ticker"].nunique()

    # ── Verdict Summary ───────────────────────────────────────────
    flags = []
    if abs(avg_prob - 0.5) > 0.08:
        direction = "bullish" if avg_prob > 0.5 else "bearish"
        flags.append(f'Avg probability {avg_prob:.1%} — systematic <b>{direction}</b> bias detected')
    if "outperformance_prob" in df.columns:
        extreme = ((df["outperformance_prob"] > 0.75) | (df["outperformance_prob"] < 0.25)).sum()
        if extreme > n_scores * 0.2:
            flags.append(f'{extreme} scores ({extreme/n_scores:.0%}) are extreme (>75% or <25%) — possible overconfidence')
    signal_cols = ["filing_score", "earnings_score", "flow_score", "ml_score"]
    avail = [c for c in signal_cols if c in df.columns and df[c].notna().sum() > 2]
    if len(avail) >= 2:
        corr = df[avail].corr()
        high_corr = [(avail[i], avail[j], corr.iloc[i, j])
                     for i in range(len(avail)) for j in range(i+1, len(avail))
                     if abs(corr.iloc[i, j]) > 0.7]
        if high_corr:
            pairs = ", ".join(f'{a.replace("_score","")}/{b.replace("_score","")} ({c:.2f})'
                            for a, b, c in high_corr)
            flags.append(f'High signal correlation: <b>{pairs}</b> — signals may be redundant')

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
            f'{n_scores} scores across {unique_tickers} tickers, avg probability {avg_prob:.1%} '
            f'(near 50% = unbiased). No systematic issues detected.</div>',
            unsafe_allow_html=True,
        )

    o1, o2, o3, o4 = st.columns(4)
    with o1: conviction_card("Total Scores", str(n_scores))
    with o2: conviction_card("Unique Tickers", str(unique_tickers))
    with o3:
        conviction_card("Avg Composite", f"{avg_score:.3f}",
                        color=SUCCESS_GREEN if avg_score > 0.6 else DANGER_RED if avg_score < 0.4 else BRAND_BLUE)
    with o4:
        conviction_card("Avg Probability", f"{avg_prob:.1%}",
                        color=SUCCESS_GREEN if avg_prob > 0.55 else DANGER_RED if avg_prob < 0.45 else BRAND_BLUE)

    # ── 2. Score Distribution Histogram ──────────────────────────
    render_section_header("Score Distribution")
    subtitle(
        'Distribution of composite conviction scores. '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Right-skewed</span> = bullish bias; '
        f'<span style="color:{DANGER_RED};font-weight:700;">left-skewed</span> = bearish bias; '
        '<span style="color:#94a3b8;">centered = well-calibrated.</span>'
    )
    if "composite_score" in df.columns:
        with st.container(border=True):
            fig_hist = go.Figure(go.Histogram(
                x=df["composite_score"], nbinsx=20,
                marker_color=BRAND_BLUE, opacity=0.8,
            ))
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#94a3b8",
                              annotation_text="Neutral (0.5)")
            apply_invictus_layout(fig_hist, height=280, title="Composite Score Distribution")
            fig_hist.update_layout(xaxis_title="Composite Score", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # ── 3. Signal Correlation Heatmap ────────────────────────────
    signal_cols = ["filing_score", "earnings_score", "flow_score", "ml_score"]
    available_signals = [c for c in signal_cols if c in df.columns and df[c].notna().sum() > 2]

    if len(available_signals) >= 2:
        render_section_header("Signal Correlation Matrix")
        subtitle(
            'Cross-correlation between the 4 conviction signals. '
            f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Low correlation</span> = independent signals (good diversification); '
            f'<span style="color:{DANGER_RED};font-weight:700;">high correlation</span> = redundant information.'
        )
        corr = df[available_signals].corr()
        with st.container(border=True):
            labels = [c.replace("_score", "").title() for c in available_signals]
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=labels, y=labels,
                colorscale=[[0, DANGER_RED_ALT], [0.5, "#f8fafc"], [1, SUCCESS_GREEN]],
                text=corr.round(2).values, texttemplate="%{text}",
                zmin=-1, zmax=1,
            ))
            apply_invictus_layout(fig_corr, height=300, title="Signal Correlation")
            fig_corr.update_layout(margin=dict(t=30, b=10, l=80, r=10))
            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

    # ── 4. Per-Signal Quality ────────────────────────────────────
    if available_signals:
        render_section_header("Signal Quality Metrics")
        subtitle(
            'Per-signal statistics. '
            '<span style="color:#94a3b8;">Signals with high standard deviation are more informative; '
            'signals clustered near 0.5 add little discriminating power.</span>'
        )
        sig_cols = st.columns(len(available_signals))
        for i, col_name in enumerate(available_signals):
            with sig_cols[i]:
                vals = df[col_name].dropna()
                label = col_name.replace("_score", "").title()
                conviction_card(label, f"{vals.mean():.3f}",
                                color=BRAND_BLUE,
                                sub_label=f"σ={vals.std():.3f}")

    # ── 5. Confidence Bucket Analysis ────────────────────────────
    if "outperformance_prob" in df.columns:
        render_section_header("Confidence Bucket Analysis")
        subtitle(
            'Conviction scores grouped by confidence level. '
            f'<span style="color:{SUCCESS_GREEN};font-weight:700;">High confidence</span> (>65%) should have the best hit rate; '
            'if not, the model is overconfident.'
        )

        df["bucket"] = pd.cut(
            df["outperformance_prob"],
            bins=[0, 0.35, 0.45, 0.55, 0.65, 1.0],
            labels=["Strong Bearish", "Bearish", "Neutral", "Bullish", "Strong Bullish"],
        )
        bucket_counts = df["bucket"].value_counts().sort_index()

        with st.container(border=True):
            _bucket_colors = [DANGER_RED, DANGER_RED_ALT, "#94a3b8", BRAND_BLUE, SUCCESS_GREEN]
            fig_b = go.Figure(go.Bar(
                x=bucket_counts.index.tolist(),
                y=bucket_counts.values,
                marker_color=_bucket_colors[:len(bucket_counts)],
                text=bucket_counts.values, textposition="outside",
            ))
            apply_invictus_layout(fig_b, height=280, title="Conviction by Confidence Bucket")
            st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

    # ── 6. Per-Ticker Conviction Table ───────────────────────────
    render_section_header("Per-Ticker Conviction Detail")
    subtitle('Latest conviction scores per ticker with signal breakdown.')

    latest = df.drop_duplicates(subset=["ticker"], keep="first")
    display_cols = ["ticker", "composite_score", "outperformance_prob", "signal_agreement", "dominant_driver"]
    display_cols = [c for c in display_cols if c in latest.columns]
    for sc in available_signals:
        if sc in latest.columns:
            display_cols.append(sc)

    fmt = {}
    for c in display_cols:
        if "score" in c:
            fmt[c] = "{:.3f}"
        if "prob" in c:
            fmt[c] = "{:.1%}"

    st.dataframe(
        latest[display_cols].style.format(fmt, na_rep="—"),
        use_container_width=True, hide_index=True,
    )
