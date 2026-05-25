"""
Conviction Engine — the synthesis page.

3 inputs → 1 output.
Follows the same visual pattern as Outlook/Flows/Transcript:
  1. Overview cards (one per ticker)
  2. Per-ticker detail: bar chart + table + formula
"""
import streamlit as st
import plotly.graph_objects as go

from invictus.pages.conviction._shared import (
    render_section_header, render_metric_card, apply_invictus_layout,
    score_color, conviction_color, ticker_section_header, source_line,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, SLATE_500,
)


# ── Display maps ─────────────────────────────────────────────────────

DIM_DISPLAY = {
    "demand_environment": "Demand",
    "competitive_positioning": "Competitive",
    "strategic_confidence": "Strategy",
    "macro_industry_outlook": "Macro",
    "headwinds_tailwinds": "Headwinds",
    "investment_thesis_clarity": "Thesis",
}


def _prob_verdict(prob: float) -> tuple:
    if prob >= 0.78: return "Strong conviction — high confidence buy signal", SUCCESS_GREEN
    if prob >= 0.68: return "High conviction — most signals aligned bullish", "#10b981"
    if prob >= 0.53: return "Moderate positive — slight bullish tilt", BRAND_BLUE
    if prob >= 0.48: return "Neutral — no clear edge either way", "#94a3b8"
    if prob >= 0.35: return "Moderate negative — some signals flagging risk", "#f59e0b"
    return "Low / Risk — multiple layers signaling downside risk", DANGER_RED


def _flow_signal(score: float) -> str:
    if score > 0.3: return "Strong Accumulation"
    if score > 0.05: return "Mild Buying"
    if score < -0.3: return "Distribution"
    if score < -0.05: return "Mild Selling"
    return "Neutral"


def _outlook_signal(score: float) -> str:
    if score > 0.4: return "Strong Bullish"
    if score > 0.15: return "Mildly Bullish"
    if score < -0.4: return "Strong Bearish"
    if score < -0.15: return "Mildly Bearish"
    return "Neutral"


def _cred_signal(score: float) -> str:
    if score >= 0.9: return "High"
    if score >= 0.75: return "Moderate"
    return "Low"


# ── Insight Extractors ───────────────────────────────────────────────

def _flow_finding(fl: dict) -> str:
    insider = fl.get("insider_intelligence", {})
    fund = fl.get("fund_accumulation", {})
    parts = []
    if insider.get("summary"):
        parts.append(insider["summary"])
    if fund.get("summary"):
        parts.append(fund["summary"])
    return " ".join(parts[:2]) if parts else "No flow data available"


def _outlook_finding(ol: dict) -> str:
    dims = ol.get("outlook", {}).get("dimensions", {})
    if not dims:
        return "No outlook data available"
    # Top 2 dimensions by magnitude
    scored = []
    for key, dd in dims.items():
        s = dd.get("score", 0)
        name = DIM_DISPLAY.get(key, key)
        sig = dd.get("signal", "Neutral")
        scored.append((abs(s), s, name, sig))
    scored.sort(reverse=True)
    parts = []
    for _, s, name, sig in scored[:2]:
        parts.append(f"{name}: {sig} ({s:+.2f})")
    return ". ".join(parts)


def _cred_finding(ol: dict) -> str:
    cred = ol.get("credibility", {})
    sub_dims = cred.get("sub_dimensions", {})
    overall = cred.get("overall_credibility", "Moderate")
    if not sub_dims:
        return f"Assessment: {overall}"
    dim_names = {
        "hedging_density": "Hedging",
        "specificity": "Specificity",
        "forward_backward_ratio": "Forward-looking",
        "dodge_detection": "Evasion",
    }
    parts = [f"{overall} credibility"]
    for k, sd in sub_dims.items():
        s = sd.get("score", 0.5)
        if s < 0.4:
            parts.append(f"weak {dim_names.get(k, k).lower()} ({s:.2f})")
        elif s > 0.8:
            parts.append(f"strong {dim_names.get(k, k).lower()} ({s:.2f})")
    return ". ".join(parts[:3]).capitalize()


# ── Main Render ──────────────────────────────────────────────────────

def render(pi_tickers, pi_synth, pi_flow, pi_outlook):
    render_section_header("Conviction Engine")

    st.markdown(
        f'<div style="font-size:13px;color:#64748b;margin:-4px 0 12px 0;line-height:1.5;">'
        f'Outperformance probability: '
        f'<span style="color:{DANGER_RED};font-weight:700;">&lt;35% Low / Risk</span> · '
        f'<span style="color:#f59e0b;font-weight:700;">35-48% Moderate Negative</span> · '
        f'<span style="color:#94a3b8;font-weight:700;">48-53% Neutral</span> · '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">53-68% Moderate Positive</span> · '
        f'<span style="color:#10b981;font-weight:700;">68-78% High</span> · '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">&gt;78% Strong Conviction</span>. '
        f'<span style="color:#94a3b8;">Synthesized from capital flows, management outlook, '
        f'and transcript credibility.</span></div>',
        unsafe_allow_html=True,
    )

    # ── Overview Cards ───────────────────────────────────────────
    overview_cols = st.columns(len(pi_tickers))
    for idx, t in enumerate(pi_tickers):
        syn = pi_synth.get(t, {})
        fl = pi_flow.get(t, {})
        ol = pi_outlook.get(t, {})

        prob = syn.get("outperformance_probability", 0.5)
        level = syn.get("conviction_level", "N/A")
        lc = conviction_color(level)
        verdict, vc = _prob_verdict(prob)

        flow_score = fl.get("flow_composite", 0)
        outlook_raw = ol.get("outlook_score_raw", 0)
        cred_mult = ol.get("credibility_multiplier", 0.75)
        mgmt_signal = ol.get("management_signal", 0)

        with overview_cols[idx]:
            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                f'background:#fafbfc;">'
                # Row 1: Ticker + Probability
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                f'margin-bottom:4px;">'
                f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{t}</span>'
                f'<span style="font-size:20px;font-weight:800;color:{vc};'
                f'font-variant-numeric:tabular-nums;">{prob:.0%}</span>'
                f'</div>'
                # Row 2: Conviction level
                f'<div style="font-size:12px;font-weight:700;color:{lc};'
                f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">{level}</div>'
                # Row 3: Verdict
                f'<div style="font-size:12px;color:#64748b;margin-bottom:8px;line-height:1.4;'
                f'font-style:italic;">{verdict}</div>'
                # Row 4: 3 layer scores
                f'<div style="border-top:1px solid #e2e8f0;padding-top:8px;'
                f'display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;text-align:center;">'
                f'<div><div style="font-size:12px;color:#64748b;text-transform:uppercase;'
                f'letter-spacing:0.05em;">Capital Flows</div>'
                f'<div style="font-size:13px;font-weight:700;color:{score_color(flow_score)};'
                f'font-variant-numeric:tabular-nums;">{flow_score:+.2f}</div>'
                f'<div style="font-size:11px;color:#94a3b8;margin-top:1px;">What money is doing</div></div>'
                f'<div style="border-left:1px solid #f1f5f9;border-right:1px solid #f1f5f9;">'
                f'<div style="font-size:12px;color:#64748b;text-transform:uppercase;'
                f'letter-spacing:0.05em;">Management Outlook</div>'
                f'<div style="font-size:13px;font-weight:700;color:{score_color(outlook_raw)};'
                f'font-variant-numeric:tabular-nums;">{outlook_raw:+.2f}</div>'
                f'<div style="font-size:11px;color:#94a3b8;margin-top:1px;">What management says</div></div>'
                f'<div><div style="font-size:12px;color:#64748b;text-transform:uppercase;'
                f'letter-spacing:0.05em;">Credibility</div>'
                f'<div style="font-size:13px;font-weight:700;color:'
                f'{SUCCESS_GREEN if cred_mult >= 0.85 else "#f59e0b" if cred_mult >= 0.7 else DANGER_RED};'
                f'font-variant-numeric:tabular-nums;">{cred_mult:.2f}</div>'
                f'<div style="font-size:11px;color:#94a3b8;margin-top:1px;">How credibly they say it</div></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)

    # ── Per-Ticker Detail ────────────────────────────────────────
    for t in pi_tickers:
        syn = pi_synth.get(t, {})
        fl = pi_flow.get(t, {})
        ol = pi_outlook.get(t, {})

        prob = syn.get("outperformance_probability", 0.5)
        level = syn.get("conviction_level", "N/A")
        _, vc = _prob_verdict(prob)

        flow_score = fl.get("flow_composite", 0)
        outlook_raw = ol.get("outlook_score_raw", 0)
        cred_mult = ol.get("credibility_multiplier", 0.75)
        mgmt_signal = ol.get("management_signal", 0)

        ticker_section_header(t, "Conviction Analysis")

        with st.expander("Details", expanded=True):

            # ── Metrics row ──────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card(
                    "Outperformance", f"{prob:.0%}",
                    delta_val=prob - 0.5 if abs(prob - 0.5) > 0.05 else 0,
                )
            with m2:
                render_metric_card(
                    "Management Signal", f"{mgmt_signal:+.2f}",
                    delta_val=mgmt_signal if abs(mgmt_signal) > 0.05 else 0,
                )
            with m3:
                render_metric_card(
                    "Flow Signal", f"{flow_score:+.2f}",
                    delta_val=flow_score if abs(flow_score) > 0.05 else 0,
                )
            with m4:
                lc = conviction_color(level)
                # Delta reflects conviction level, not raw prob offset
                _conv_delta = (
                    prob - 0.5 if level in ("STRONG CONVICTION", "HIGH", "MODERATE POSITIVE")
                    else -(0.5 - prob) if level in ("MODERATE NEGATIVE", "LOW / RISK")
                    else 0  # NEUTRAL → no directional indicator
                )
                render_metric_card(
                    "Conviction", level,
                    delta_val=_conv_delta,
                )

            # ── Layer bar chart ──────────────────────────────
            layer_names = ["Capital Flows", "Management Outlook", "Credibility Gate"]
            layer_scores = [flow_score, outlook_raw, cred_mult * 2 - 1]  # normalize cred to [-1,1] for chart
            layer_display = [flow_score, outlook_raw, cred_mult]  # actual values for labels
            layer_colors = [
                DANGER_RED if v < -0.1 else SUCCESS_GREEN if v > 0.1 else "#94a3b8"
                for v in layer_scores
            ]

            fig = go.Figure(go.Bar(
                x=[flow_score, outlook_raw, mgmt_signal],
                y=["Capital Flows", "Management Outlook", "Management Signal (gated)"],
                orientation="h",
                marker=dict(color=[
                    DANGER_RED if flow_score < -0.1 else SUCCESS_GREEN if flow_score > 0.1 else "#94a3b8",
                    DANGER_RED if outlook_raw < -0.1 else SUCCESS_GREEN if outlook_raw > 0.1 else "#94a3b8",
                    DANGER_RED if mgmt_signal < -0.1 else SUCCESS_GREEN if mgmt_signal > 0.1 else "#94a3b8",
                ]),
                text=[f"{flow_score:+.2f}", f"{outlook_raw:+.2f}", f"{mgmt_signal:+.2f}"],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Score: %{x:+.2f}<extra></extra>",
            ))

            # Credibility annotation
            fig.add_annotation(
                x=mgmt_signal, y="Management Signal (gated)",
                text=f"(outlook {outlook_raw:+.2f} x cred {cred_mult:.2f})",
                showarrow=False, xanchor="left", xshift=50,
                font=dict(size=10, color="#94a3b8"),
            )

            apply_invictus_layout(fig, height=180)
            fig.update_layout(
                xaxis=dict(range=[-1.15, 1.15], tickformat=".1f", title="Signal Score"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=160, r=120, t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # ── Detail table ─────────────────────────────────
            _thl = 'style="text-align:left;padding:4px 6px;color:#475569;font-weight:700;font-size:12px;"'
            _thc = 'style="text-align:center;padding:4px 6px;color:#475569;font-weight:700;font-size:12px;"'

            rows_data = [
                ("Capital Flows", "What institutional money is doing",
                 flow_score, _flow_signal(flow_score), _flow_finding(fl)),
                ("Management Outlook", "What management says about the future",
                 outlook_raw, _outlook_signal(outlook_raw), _outlook_finding(ol)),
                ("Credibility Gate", "How credibly management communicates",
                 cred_mult, _cred_signal(cred_mult), _cred_finding(ol)),
            ]

            html = (
                '<table style="width:100%;border-collapse:collapse;font-size:12px;margin:4px 0 0 0;'
                'table-layout:fixed;">'
                '<colgroup>'
                '<col style="width:20%;">'
                '<col style="width:10%;">'
                '<col style="width:12%;">'
                '<col style="width:58%;">'
                '</colgroup>'
                f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                f'<th {_thl}>Signal Layer</th>'
                f'<th {_thc}>Score</th>'
                f'<th {_thc}>Signal</th>'
                f'<th {_thl}>Key Findings</th>'
                f'</tr></thead><tbody>'
            )

            for label, subtitle, score, signal, finding in rows_data:
                sc = score_color(score if label != "Credibility Gate" else score - 0.7)
                sig_color = (SUCCESS_GREEN if "Bullish" in signal or "Accumulation" in signal or signal == "High"
                             else DANGER_RED if "Bearish" in signal or "Distribution" in signal or signal == "Low"
                             else "#94a3b8")
                score_fmt = f"{score:+.2f}" if label != "Credibility Gate" else f"{score:.2f}"
                html += (
                    f'<tr style="border-bottom:1px solid #f1f5f9;">'
                    f'<td style="padding:5px 6px;color:#1e293b;font-weight:500;">{label}'
                    f'<div style="font-size:11px;color:#64748b;margin-top:1px;">{subtitle}</div></td>'
                    f'<td style="padding:5px 6px;text-align:center;color:{sc};font-weight:700;'
                    f'font-variant-numeric:tabular-nums;">{score_fmt}</td>'
                    f'<td style="padding:5px 6px;text-align:center;color:{sig_color};'
                    f'font-weight:600;font-size:12px;">{signal}</td>'
                    f'<td style="padding:5px 6px;color:#64748b;font-size:12px;line-height:1.4;">'
                    f'{finding}</td>'
                    f'</tr>'
                )

            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)

            # Source + formula
            source_line(
                "Synthesis Engine",
                f"Layer 1: Flows({flow_score:+.2f}) + "
                f"Layer 2: Outlook({outlook_raw:+.2f}) × Layer 3: Credibility({cred_mult:.2f}) "
                f"→ P(outperform) = {prob:.0%}",
            )
