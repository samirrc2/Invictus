"""
Transcript Analysis — Layer 3 of 3.

How credibly management communicates.
4 linguistic dimensions, each scored [0, 1].
Credibility multiplier = 0.5 + 0.5 × raw_credibility → [0.5, 1.0].
This layer GATES the Management Outlook — it does not produce an
independent signal; it attenuates Layer 2.
"""
import streamlit as st

from invictus.pages.conviction._shared import (
    render_section_header, render_metric_card,
    score_color, ticker_section_header, sub_section_header,
    source_line, formula_box, flags_two_column, analyst_ratings_banner,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, SLATE_500,
)


CRED_DISPLAY = {
    "hedging_density": "Hedging Density",
    "specificity": "Specificity",
    "forward_backward_ratio": "Forward / Backward Ratio",
    "dodge_detection": "Dodge Detection",
}

CRED_EXPLAIN = {
    "hedging_density": "Less hedging = higher credibility",
    "specificity": "Specific numbers/dates vs vague platitudes",
    "forward_backward_ratio": "More forward-looking = higher confidence",
    "dodge_detection": "Direct answers vs evasion",
}

CRED_ORDER = ["hedging_density", "specificity", "forward_backward_ratio", "dodge_detection"]


def _cred_label(cm):
    if cm >= 0.9:  return "High Credibility", SUCCESS_GREEN
    if cm >= 0.75: return "Moderate", BRAND_BLUE
    return "Low Credibility", "#f59e0b"


def _gate_explain(cm, os_raw, ms):
    """Plain-English explanation of the gating effect."""
    if cm >= 0.9:
        return "High credibility — outlook signal passes through mostly unattenuated"
    if cm >= 0.75:
        pct_loss = (1 - cm) * 100
        return f"Moderate credibility — outlook attenuated ~{pct_loss:.0f}%"
    pct_loss = (1 - cm) * 100
    return f"Low credibility — outlook attenuated ~{pct_loss:.0f}%, management claims discounted"


def render(pi_tickers, pi_outlook):
    """Render the Transcript Analysis sub-tab — credibility gate."""
    render_section_header("Transcript Analysis")

    # Scale explainer
    st.markdown(
        f'<div style="font-size:12px;color:#64748b;margin:-4px 0 12px 0;line-height:1.5;">'
        f'Credibility multiplier ranges from '
        f'<span style="color:#f59e0b;font-weight:700;">0.50</span> '
        f'(heavily discounted — vague, evasive) to '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">1.00</span> '
        f'(fully credible — specific, forward-looking). '
        f'<span style="color:#94a3b8;">Gates the Management Outlook signal: '
        f'mgmt_signal = outlook × credibility.</span></div>',
        unsafe_allow_html=True,
    )

    # ── Overview Cards ───────────────────────────────────────────
    overview_cols = st.columns(len(pi_tickers))
    for idx, t in enumerate(pi_tickers):
        d = pi_outlook.get(t, {})
        cm = d.get("credibility_multiplier", 0.75)
        os_raw = d.get("outlook_score_raw", 0)
        ms = d.get("management_signal", 0)
        cred_lbl, cred_clr = _cred_label(cm)

        with overview_cols[idx]:
            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                f'background:#fafbfc;">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">'
                f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{t}</span>'
                f'<span style="font-size:18px;font-weight:800;color:{cred_clr};">{cm:.2f}</span></div>'
                f'<div style="font-size:12px;font-weight:700;color:{cred_clr};'
                f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">{cred_lbl}</div>'
                f'<div style="font-size:12px;color:#64748b;margin-bottom:6px;line-height:1.4;'
                f'font-style:italic;">{_gate_explain(cm, os_raw, ms)}</div>'
                # Gating formula
                f'<div style="font-size:12px;color:#64748b;border-top:1px solid #e2e8f0;padding-top:6px;'
                f'font-family:monospace;">'
                f'{os_raw:+.2f} × {cm:.2f} = {ms:+.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)

    # ── Per-Ticker Detail ────────────────────────────────────────
    for t in pi_tickers:
        d = pi_outlook.get(t, {})
        cred_data = d.get("credibility", {})
        cm = cred_data.get("credibility_multiplier", 0.75)

        if cred_data.get("status") not in ("Success", None):
            if cred_data.get("status") == "Error":
                st.caption(f"{t} — Transcript analysis not available")
                continue

        cred_clr = SUCCESS_GREEN if cm >= 0.85 else "#f59e0b" if cm >= 0.7 else DANGER_RED
        ticker_section_header(t, "Transcript Analysis", cm)

        # ── Analyst Ratings Banner ───────────────────────────
        grades = d.get("earnings_context", {}).get("grades", {})
        if grades:
            analyst_ratings_banner(grades)

        with st.expander("Details", expanded=True):
            raw_cred = cred_data.get("raw_credibility", 0.5)
            cred_overall = cred_data.get("overall_credibility", "Moderate")
            sub_dims = cred_data.get("sub_dimensions", {})

            if not sub_dims:
                st.caption("No credibility data available.")
                continue

            # ── Metrics row ──────────────────────────────────
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                render_metric_card(
                    "Raw Credibility", f"{raw_cred:.2f}",
                    delta_val=raw_cred - 0.5 if abs(raw_cred - 0.5) > 0.05 else 0,
                )
            with cc2:
                render_metric_card(
                    "Multiplier", f"{cm:.2f}",
                    delta_val=cm - 0.75 if abs(cm - 0.75) > 0.05 else 0,
                )
            with cc3:
                render_metric_card(
                    "Assessment", cred_overall,
                    delta_val=cm - 0.75 if abs(cm - 0.75) > 0.05 else 0,
                )

            # ── Dimension table ──────────────────────────────
            _thl = 'style="text-align:left;padding:4px 6px;color:#475569;font-weight:700;font-size:12px;"'
            _thc = 'style="text-align:center;padding:4px 6px;color:#475569;font-weight:700;font-size:12px;"'
            html = (
                '<table style="width:100%;border-collapse:collapse;font-size:12px;margin:4px 0 0 0;'
                'table-layout:fixed;">'
                '<colgroup>'
                '<col style="width:22%;">'
                '<col style="width:12%;">'
                '<col style="width:12%;">'
                '<col style="width:14%;">'
                '<col style="width:40%;">'
                '</colgroup>'
                f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                f'<th {_thl}>Dimension</th>'
                f'<th {_thc}>Weight</th>'
                f'<th {_thc}>Score</th>'
                f'<th {_thc}>Contribution</th>'
                f'<th {_thl}>Reasoning</th>'
                f'</tr></thead><tbody>'
            )
            for key in CRED_ORDER:
                sd = sub_dims.get(key, {})
                score = sd.get("score", 0)
                weight = sd.get("weight", 0)
                contrib = sd.get("weighted_contribution", 0)
                reasoning = sd.get("reasoning", "—")
                sc = SUCCESS_GREEN if score >= 0.7 else "#f59e0b" if score >= 0.4 else DANGER_RED
                _tdc = 'text-align:center;'
                html += (
                    f'<tr style="border-bottom:1px solid #f1f5f9;">'
                    f'<td style="padding:5px 6px;color:#1e293b;font-weight:500;">{CRED_DISPLAY.get(key, key)}'
                    f'<div style="font-size:11px;color:#64748b;margin-top:1px;">{CRED_EXPLAIN.get(key, "")}</div></td>'
                    f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-variant-numeric:tabular-nums;">{weight:.0%}</td>'
                    f'<td style="padding:5px 6px;{_tdc}color:{sc};font-weight:700;font-variant-numeric:tabular-nums;">{score:.2f}</td>'
                    f'<td style="padding:5px 6px;{_tdc}color:{sc};font-variant-numeric:tabular-nums;">{contrib:.4f}</td>'
                    f'<td style="padding:5px 6px;color:#64748b;font-size:12px;">{reasoning}</td>'
                    f'</tr>'
                )
            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)

            # ── Credibility flags ────────────────────────────
            flags_two_column(
                cred_data.get("green_flags", []),
                cred_data.get("red_flags", []),
                green_label="Credibility Strengths",
                red_label="Credibility Concerns",
            )

            # Source + formula
            cred_src = cred_data.get("source", "N/A")
            source_line(cred_src, f"multiplier = 0.5 + 0.5 × {raw_cred:.4f} = {cm:.4f}")

            # ── Gating effect summary ────────────────────────
            os_raw = d.get("outlook_score_raw", 0)
            ms = d.get("management_signal", 0)
            _ms_color = score_color(ms)
            formula_box(
                f'<b>Gating Effect</b>: outlook ({os_raw:+.4f}) × credibility ({cm:.4f}) = '
                f'<span style="font-weight:800;color:{_ms_color};font-size:14px;">'
                f'{ms:+.4f}</span> management signal'
            )
