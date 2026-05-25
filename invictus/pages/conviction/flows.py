"""
Capital Flows — Layer 1 of 3.

What institutional money IS DOING (revealed preference).
Insider intelligence + fund accumulation trend.
"""
import streamlit as st

from invictus.pages.conviction._shared import (
    render_section_header, render_metric_card,
    score_color, ticker_section_header, sub_section_header,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, SLATE_500,
)


ACCUM_LABELS = {
    "strong_accumulation": "Strong Accumulation",
    "moderate_accumulation": "Moderate Accumulation",
    "moderate_distribution": "Moderate Distribution",
    "neutral": "Neutral",
    "distribution": "Distribution",
}
ACCUM_COLORS = {
    "strong_accumulation": SUCCESS_GREEN,
    "moderate_accumulation": BRAND_BLUE,
    "neutral": SLATE_500,
    "moderate_distribution": "#f59e0b",
    "distribution": DANGER_RED,
}


def _ins_explain(v):
    if v > 0.5: return "Insiders buying heavily — high conviction"
    if v > 0.1: return "More insider buying than selling"
    if v < -0.5: return "Heavy insider selling — caution"
    if v < -0.1: return "Insiders net selling their shares"
    return "Balanced — no clear insider signal"


def _fund_explain(v):
    if v > 0.3: return "Funds actively accumulating shares"
    if v > 0.05: return "Funds slightly increasing positions"
    if v < -0.3: return "Funds reducing exposure significantly"
    if v < -0.05: return "Funds trimming their positions"
    return "Institutional positions mostly unchanged"


def _comp_explain(v):
    if v > 0.4: return "Strong institutional demand — money flowing in"
    if v > 0.1: return "Mild positive signal — slight accumulation"
    if v < -0.4: return "Institutional selling pressure — money flowing out"
    if v < -0.1: return "Mild negative signal — slight distribution"
    return "No clear directional flow"


def render(pi_tickers, pi_flow):
    """Render the Capital Flows sub-tab."""
    render_section_header("Capital Flow Intelligence")

    # Scale explainer
    st.markdown(
        f'<div style="font-size:12px;color:#64748b;margin:-4px 0 12px 0;line-height:1.5;">'
        f'Scores range from <span style="color:{DANGER_RED};font-weight:700;">−1</span> '
        f'(heavy institutional selling / insider exits) to '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">+1</span> '
        f'(strong institutional accumulation / insider buying). '
        f'<span style="color:#94a3b8;">0 = no directional signal.</span></div>',
        unsafe_allow_html=True,
    )

    # ── Overview Cards ───────────────────────────────────────────
    overview_cols = st.columns(len(pi_tickers))
    for idx, t in enumerate(pi_tickers):
        d = pi_flow.get(t, {})
        comp = d.get("flow_composite", 0)
        accum = d.get("estimated_accumulation", "neutral")
        ins_s = d.get("insider_intelligence", {}).get("score", 0)
        fund_s = d.get("fund_accumulation", {}).get("score", 0)
        v_label = ACCUM_LABELS.get(accum, "Neutral")
        v_color = ACCUM_COLORS.get(accum, SLATE_500)

        with overview_cols[idx]:
            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                f'background:#fafbfc;">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">'
                f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{t}</span>'
                f'<span style="font-size:18px;font-weight:800;color:{v_color};">{comp:+.2f}</span></div>'
                f'<div style="font-size:12px;font-weight:700;color:{v_color};'
                f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">{v_label}</div>'
                f'<div style="font-size:12px;color:#64748b;margin-bottom:10px;line-height:1.4;'
                f'font-style:italic;">{_comp_explain(comp)}</div>'
                f'<div style="padding:5px 0;border-top:1px solid #e2e8f0;">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;">'
                f'<span style="color:#475569;font-weight:600;">Insider Intel</span>'
                f'<span style="color:{score_color(ins_s)};font-weight:700;">{ins_s:+.2f}</span></div>'
                f'<div style="font-size:12px;color:#64748b;margin-top:1px;">{_ins_explain(ins_s)}</div></div>'
                f'<div style="padding:5px 0;border-top:1px solid #f1f5f9;">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;">'
                f'<span style="color:#475569;font-weight:600;">Fund Trend</span>'
                f'<span style="color:{score_color(fund_s)};font-weight:700;">{fund_s:+.2f}</span></div>'
                f'<div style="font-size:12px;color:#64748b;margin-top:1px;">{_fund_explain(fund_s)}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)

    # ── Per-Ticker Detail ────────────────────────────────────────
    for t in pi_tickers:
        d = pi_flow.get(t, {})
        if d.get("status") != "Success":
            _err_detail = d.get("insider_intelligence", {}).get("summary", "")
            _err_msg = f"{t} — Flow data not available"
            if _err_detail and _err_detail != "No data":
                _err_msg += f" ({_err_detail})"
            st.caption(_err_msg)
            continue

        composite = d.get("flow_composite", 0)
        ticker_section_header(t, "Capital Flow Intelligence", composite)

        with st.expander("Details", expanded=True):
            insider_i = d.get("insider_intelligence", {})
            fund_a = d.get("fund_accumulation", {})

            # ── OWNERSHIP SNAPSHOT ───────────────────────────
            _own = d.get("ownership_breakdown", {})
            _inst_pct = _own.get("institutional_pct", 0)
            _ins_pct = _own.get("insider_pct", 0)
            _ret_pct = _own.get("retail_pct", 0)
            if _inst_pct > 0 or _ins_pct > 0:
                st.markdown(
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="font-size:12px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'text-transform:uppercase;margin-bottom:6px;">Shareholding Structure</div>'
                    f'<div style="display:flex;height:18px;border-radius:4px;overflow:hidden;'
                    f'margin-bottom:6px;border:1px solid #e2e8f0;">'
                    f'<div style="width:{_inst_pct*100:.1f}%;background:{BRAND_BLUE};"></div>'
                    f'<div style="width:{_ins_pct*100:.1f}%;background:{SUCCESS_GREEN};"></div>'
                    f'<div style="width:{_ret_pct*100:.1f}%;background:#e2e8f0;"></div>'
                    f'</div>'
                    f'<div style="display:flex;gap:16px;font-size:12px;">'
                    f'<span style="color:{BRAND_BLUE};font-weight:600;">■ Institutional {_inst_pct:.1%}</span>'
                    f'<span style="color:{SUCCESS_GREEN};font-weight:600;">■ Insiders {_ins_pct:.1%}</span>'
                    f'<span style="color:#94a3b8;font-weight:600;">■ Retail/Other {_ret_pct:.1%}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            # Top holders pills
            holder_rows_snap = fund_a.get("holders_detail", [])
            if holder_rows_snap:
                pills = []
                for h in holder_rows_snap[:5]:
                    nm = h.get("name", "")[:28]
                    stk = h.get("pct_held", 0)
                    tp = h.get("type", "")
                    _pill_color = "#6366f1" if "Smart" in tp else "#475569"
                    _pill_bg = "rgba(99,102,241,0.08)" if "Smart" in tp else "rgba(148,163,184,0.08)"
                    stk_str = f" ({stk:.1%})" if stk > 0 else ""
                    pills.append(
                        f'<span style="display:inline-block;font-size:12px;color:{_pill_color};'
                        f'background:{_pill_bg};padding:3px 10px;border-radius:12px;'
                        f'margin:2px 4px 2px 0;font-weight:500;">{nm}{stk_str}</span>'
                    )
                st.markdown(
                    f'<div style="margin-bottom:10px;">'
                    f'<span style="font-size:12px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                    f'text-transform:uppercase;margin-right:8px;">Largest Holders</span>'
                    + "".join(pills) + '</div>',
                    unsafe_allow_html=True,
                )

            # ── INSIDER INTELLIGENCE ─────────────────────────
            _ins_score = insider_i.get("score", 0)
            sub_section_header("Insider Intelligence", f"{_ins_score:+.2f}", score_color(_ins_score))
            _ins_verdict = (
                "Strong Buying" if _ins_score > 0.5 else
                "Net Buying" if _ins_score > 0.1 else
                "Net Selling" if _ins_score < -0.5 else
                "Selling Pressure" if _ins_score < -0.1 else
                "Neutral"
            )
            ic1, ic2, ic3 = st.columns(3)
            with ic1: render_metric_card("Buys", str(insider_i.get("buy_count", 0)), delta_val=insider_i.get("buy_count", 0))
            with ic2: render_metric_card("Sells", str(insider_i.get("sell_count", 0)), delta_val=-insider_i.get("sell_count", 0))
            _ins_delta = _ins_score if abs(_ins_score) > 0.1 else 0
            with ic3: render_metric_card("Insider Signal", _ins_verdict, delta_val=_ins_delta)

            # Insider transaction table
            notable = insider_i.get("notable_transactions", [])
            if notable:
                _thl = 'style="text-align:left;padding:4px 6px;color:#475569;font-weight:700;font-size:12px;"'
                _thc = 'style="text-align:center;padding:4px 6px;color:#475569;font-weight:700;font-size:12px;"'
                html = (
                    '<table style="width:100%;border-collapse:collapse;font-size:12px;margin:4px 0 0 0;'
                    'table-layout:fixed;">'
                    '<colgroup>'
                    '<col style="width:20%;"><col style="width:20%;"><col style="width:15%;">'
                    '<col style="width:15%;"><col style="width:14%;"><col style="width:16%;">'
                    '</colgroup>'
                    f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                    f'<th {_thl}>Insider</th><th {_thc}>Role</th>'
                    f'<th {_thc}>Stake in Company</th><th {_thc}>% Stake Sold</th>'
                    f'<th {_thc}>Direction</th><th {_thc}>Latest</th>'
                    f'</tr></thead><tbody>'
                )
                for tx in notable[:5]:
                    tx_type = tx.get("type", "SELL")
                    stake_pct = tx.get("stake_pct", 0)
                    pct_change = tx.get("pct_stake_change", 0)
                    if stake_pct >= 0.1:
                        stake_str = f"{stake_pct:.1f}%"
                    elif stake_pct >= 0.01:
                        stake_str = f"{stake_pct:.2f}%"
                    elif stake_pct > 0:
                        stake_str = "<0.01%"
                    else:
                        stake_str = '<span style="color:#94a3b8;font-style:italic;">Not available</span>'
                    if stake_pct <= 0:
                        pct_chg_str = '<span style="color:#94a3b8;font-style:italic;">N/A</span>'
                    elif pct_change > 0:
                        pct_chg_str = f"{pct_change:.1f}%"
                    else:
                        pct_chg_str = '<span style="color:#94a3b8;font-style:italic;">< 0.1%</span>'
                    pct_color = DANGER_RED if pct_change >= 15 else "#f59e0b" if pct_change >= 5 else "#64748b"
                    date_str = tx.get("date", "—") or "—"
                    if "BUY" in tx_type:
                        dir_color, dir_arrow = SUCCESS_GREEN, "▲ Buying"
                    elif "SELL" in tx_type:
                        dir_color, dir_arrow = DANGER_RED, "▼ Selling"
                    else:
                        dir_color, dir_arrow = "#94a3b8", "— Mixed"
                    _tdc = 'text-align:center;'
                    html += (
                        f'<tr style="border-bottom:1px solid #f1f5f9;">'
                        f'<td style="padding:5px 6px;color:#1e293b;font-weight:500;">{tx.get("name", "")}</td>'
                        f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-size:12px;">{tx.get("role", "")}</td>'
                        f'<td style="padding:5px 6px;{_tdc}color:#475569;font-weight:600;font-variant-numeric:tabular-nums;">{stake_str}</td>'
                        f'<td style="padding:5px 6px;{_tdc}font-weight:700;color:{pct_color};font-variant-numeric:tabular-nums;">{pct_chg_str}</td>'
                        f'<td style="padding:5px 6px;{_tdc}color:{dir_color};font-weight:600;font-size:12px;">{dir_arrow}</td>'
                        f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-size:12px;font-variant-numeric:tabular-nums;">{date_str}</td>'
                        f'</tr>'
                    )
                html += '</tbody></table>'
                st.markdown(html, unsafe_allow_html=True)

            # ── FUND ACCUMULATION TREND ──────────────────────
            _fa_score = fund_a.get("score", 0)
            sub_section_header("Fund Accumulation Trend", f"{_fa_score:+.2f}", score_color(_fa_score))
            _smt = fund_a.get('smart_money_trend', 0)
            _smt_label = (
                "Accumulating" if _smt > 0.3 else "Adding" if _smt > 0.05 else
                "Distributing" if _smt < -0.3 else "Reducing" if _smt < -0.05 else "Holding Steady"
            )
            fa1, fa2, fa3 = st.columns(3)
            with fa1: render_metric_card("Active Adding", str(fund_a.get("holders_increasing", 0)), delta_val=fund_a.get("holders_increasing", 0))
            with fa2: render_metric_card("Active Reducing", str(fund_a.get("holders_decreasing", 0)), delta_val=-fund_a.get("holders_decreasing", 0))
            _smt_delta = _smt if abs(_smt) > 0.05 else 0
            with fa3: render_metric_card("Smart Money", _smt_label, delta_val=_smt_delta)

            # Holder table
            holder_rows = fund_a.get("holders_detail", [])
            if holder_rows:
                _thl = 'style="text-align:left;padding:4px 6px;color:#475569;font-weight:700;font-size:12px;"'
                _thc = 'style="text-align:center;padding:4px 6px;color:#475569;font-weight:700;font-size:12px;"'
                h_html = (
                    '<table style="width:100%;border-collapse:collapse;font-size:12px;margin:4px 0 0 0;'
                    'table-layout:fixed;">'
                    '<colgroup>'
                    '<col style="width:28%;"><col style="width:16%;"><col style="width:14%;">'
                    '<col style="width:22%;"><col style="width:20%;">'
                    '</colgroup>'
                    f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                    f'<th {_thl}>Holder</th><th {_thc}>Stake</th>'
                    f'<th {_thc}>Direction</th><th {_thc}>Likely Reason</th><th {_thc}>Filed</th>'
                    f'</tr></thead><tbody>'
                )
                for h in holder_rows[:8]:
                    dir_val = h.get("direction", "Stable")
                    dir_color = SUCCESS_GREEN if dir_val == "Adding" else DANGER_RED if dir_val == "Reducing" else "#94a3b8"
                    dir_arrow = "▲ Adding" if dir_val == "Adding" else "▼ Reducing" if dir_val == "Reducing" else "— Stable"
                    holder_type = h.get("type", "Active")
                    if "Passive" in holder_type:
                        reason = "Index rebalance" if dir_val != "Stable" else "Index tracking"
                    elif "Smart" in holder_type:
                        reason = "Conviction buy" if dir_val == "Adding" else "Taking profits" if dir_val == "Reducing" else "Holding position"
                    else:
                        reason = "Increasing investment" if dir_val == "Adding" else "Trimming position" if dir_val == "Reducing" else "Maintaining position"
                    stake_str = h.get("stake_change", "")
                    if not stake_str:
                        pct_h = h.get("pct_held", 0)
                        stake_str = f"{pct_h:.2%}" if pct_h > 0 else "—"
                    date_filed = h.get("date_reported", "—") or "—"
                    _tdc = 'text-align:center;'
                    h_html += (
                        f'<tr style="border-bottom:1px solid #f1f5f9;">'
                        f'<td style="padding:5px 6px;color:#1e293b;font-weight:500;">{h.get("name", "")[:35]}</td>'
                        f'<td style="padding:5px 6px;{_tdc}color:#475569;font-weight:600;font-variant-numeric:tabular-nums;">{stake_str}</td>'
                        f'<td style="padding:5px 6px;{_tdc}color:{dir_color};font-weight:600;font-size:12px;">{dir_arrow}</td>'
                        f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-size:12px;font-style:italic;">{reason}</td>'
                        f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-size:12px;font-variant-numeric:tabular-nums;">{date_filed}</td>'
                        f'</tr>'
                    )
                h_html += '</tbody></table>'
                st.markdown(h_html, unsafe_allow_html=True)
