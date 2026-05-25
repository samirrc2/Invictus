"""
Management Outlook — Layer 2 of 3.

What management says about the future.
6 qualitative dimensions extracted from news, transcripts, press releases.
Each dimension scored [-1, +1], weighted composite = outlook_score.
"""
import streamlit as st
import plotly.graph_objects as go

from invictus.pages.conviction._shared import (
    render_section_header, render_metric_card, apply_invictus_layout,
    score_color, ticker_section_header, sub_section_header, source_line,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, SLATE_500,
)


DIM_DISPLAY = {
    "demand_environment": "Demand Environment",
    "competitive_positioning": "Competitive Position",
    "strategic_confidence": "Strategic Confidence",
    "macro_industry_outlook": "Macro / Industry",
    "headwinds_tailwinds": "Headwinds & Tailwinds",
    "investment_thesis_clarity": "Thesis Clarity",
}

DIM_ORDER = [
    "demand_environment", "competitive_positioning", "strategic_confidence",
    "macro_industry_outlook", "headwinds_tailwinds", "investment_thesis_clarity",
]


def _tone_label(score):
    if score > 0.4:  return "Strong Bullish", SUCCESS_GREEN
    if score > 0.15: return "Mildly Bullish", BRAND_BLUE
    if score < -0.4: return "Strong Bearish", DANGER_RED
    if score < -0.15: return "Mildly Bearish", "#f59e0b"
    return "Neutral", SLATE_500


def _explain(score):
    if score > 0.3:  return "Management painting a convincingly bullish picture"
    if score > 0.1:  return "Mildly positive management outlook"
    if score < -0.3: return "Bearish signals — caution warranted"
    if score < -0.1: return "Some negative management signals"
    return "No clear directional management signal"


def render(pi_tickers, pi_outlook):
    """Render the Management Outlook sub-tab — 6 qualitative dimensions."""
    render_section_header("Management Outlook")

    # Scale explainer
    st.markdown(
        f'<div style="font-size:12px;color:#64748b;margin:-4px 0 12px 0;line-height:1.5;">'
        f'Scores range from <span style="color:{DANGER_RED};font-weight:700;">−1</span> '
        f'(bearish management commentary) to '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">+1</span> '
        f'(bullish forward outlook). '
        f'<span style="color:#94a3b8;">Extracted from news, transcripts, and press releases.</span></div>',
        unsafe_allow_html=True,
    )

    # ── Overview Cards ───────────────────────────────────────────
    overview_cols = st.columns(len(pi_tickers))
    for idx, t in enumerate(pi_tickers):
        d = pi_outlook.get(t, {})
        os_raw = d.get("outlook_score_raw", 0)
        tone_lbl, tone_clr = _tone_label(os_raw)
        sources = d.get("data_sources", "N/A")

        with overview_cols[idx]:
            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                f'background:#fafbfc;">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">'
                f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{t}</span>'
                f'<span style="font-size:18px;font-weight:800;color:{tone_clr};">{os_raw:+.2f}</span></div>'
                f'<div style="font-size:12px;font-weight:700;color:{tone_clr};'
                f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">{tone_lbl}</div>'
                f'<div style="font-size:12px;color:#64748b;margin-bottom:6px;line-height:1.4;'
                f'font-style:italic;">{_explain(os_raw)}</div>'
                f'<div style="font-size:12px;color:#64748b;border-top:1px solid #e2e8f0;padding-top:6px;">'
                f'{sources}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)

    # ── Per-Ticker Detail ────────────────────────────────────────
    for t in pi_tickers:
        d = pi_outlook.get(t, {})
        if d.get("status") not in ("Success", "Partial"):
            st.caption(f"{t} — Management outlook data not available")
            continue

        os_raw = d.get("outlook_score_raw", 0)
        ticker_section_header(t, "Management Outlook", os_raw)

        with st.expander("Details", expanded=True):
            outlook_data = d.get("outlook", {})
            dims = outlook_data.get("dimensions", {})

            if not dims:
                st.caption("No dimension data available.")
                continue

            # ── Bar chart — 6 dimensions ─────────────────────
            dim_names = [DIM_DISPLAY.get(k, k) for k in DIM_ORDER]
            dim_scores = [dims.get(k, {}).get("score", 0) for k in DIM_ORDER]

            fig = go.Figure(go.Bar(
                x=dim_scores, y=dim_names, orientation="h",
                marker=dict(color=[
                    DANGER_RED if v < -0.1 else SUCCESS_GREEN if v > 0.1 else "#94a3b8"
                    for v in dim_scores
                ]),
                text=[f"{v:+.2f}" for v in dim_scores], textposition="outside",
                hovertemplate="<b>%{y}</b><br>Score: %{x:+.2f}<extra></extra>",
            ))
            apply_invictus_layout(fig, height=240)
            fig.update_layout(
                xaxis=dict(range=[-1.15, 1.15], tickformat=".1f", title="Dimension Score"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=140, r=50, t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # ── Dimension detail table ───────────────────────
            _thl = 'style="text-align:left;padding:4px 6px;color:#475569;font-weight:700;font-size:12px;"'
            _thc = 'style="text-align:center;padding:4px 6px;color:#475569;font-weight:700;font-size:12px;"'
            html = (
                '<table style="width:100%;border-collapse:collapse;font-size:12px;margin:4px 0 0 0;'
                'table-layout:fixed;">'
                '<colgroup>'
                '<col style="width:24%;">'
                '<col style="width:12%;">'
                '<col style="width:12%;">'
                '<col style="width:14%;">'
                '<col style="width:10%;">'
                '<col style="width:28%;">'
                '</colgroup>'
                f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                f'<th {_thl}>Dimension</th>'
                f'<th {_thc}>Weight</th>'
                f'<th {_thc}>Score</th>'
                f'<th {_thc}>Contribution</th>'
                f'<th {_thc}>Signal</th>'
                f'<th {_thl}>Evidence</th>'
                f'</tr></thead><tbody>'
            )
            for key in DIM_ORDER:
                dd = dims.get(key, {})
                score = dd.get("score", 0)
                weight = dd.get("weight", 0)
                contrib = dd.get("weighted_contribution", 0)
                signal = dd.get("signal", "Neutral")
                evidence = dd.get("evidence", [])
                ev_str = ", ".join(str(e)[:30] for e in evidence[:3]) if evidence else "—"
                sc = score_color(score)
                sig_color = SUCCESS_GREEN if "Positive" in signal or "Bullish" in signal else DANGER_RED if "Negative" in signal or "Bearish" in signal else "#94a3b8"
                _tdc = 'text-align:center;'
                html += (
                    f'<tr style="border-bottom:1px solid #f1f5f9;">'
                    f'<td style="padding:5px 6px;color:#1e293b;font-weight:500;">{DIM_DISPLAY.get(key, key)}</td>'
                    f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-variant-numeric:tabular-nums;">{weight:.0%}</td>'
                    f'<td style="padding:5px 6px;{_tdc}color:{sc};font-weight:700;font-variant-numeric:tabular-nums;">{score:+.2f}</td>'
                    f'<td style="padding:5px 6px;{_tdc}color:{sc};font-variant-numeric:tabular-nums;">{contrib:+.4f}</td>'
                    f'<td style="padding:5px 6px;{_tdc}color:{sig_color};font-weight:600;font-size:12px;">{signal}</td>'
                    f'<td style="padding:5px 6px;color:#64748b;font-size:12px;">{ev_str}</td>'
                    f'</tr>'
                )
            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)

            # Source + formula
            src = outlook_data.get("source", "N/A")
            source_line(src, f"outlook_score = Σ(weight × dim_score) = {os_raw:+.4f}")

            # ── Earnings context (quantitative anchor) ───────
