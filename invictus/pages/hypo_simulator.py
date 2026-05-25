"""
invictus.pages.hypo_simulator
=============================
Allocation Engine — Portfolio context overlay, simulation,
before/after comparison, AI portfolio fit analysis,
per-stock allocation rationale.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from invictus.design import (
    render_section_header, render_metric_card, apply_invictus_layout,
    fmt_currency,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT, SLATE_500, SLATE_100,
)


def _intelligence_callout(text: str) -> None:
    st.markdown(
        f'<div style="border-left:3px solid {BRAND_BLUE};background:rgba(29,78,216,0.04);'
        f'padding:12px 16px;border-radius:0 6px 6px 0;margin:12px 0;'
        f'font-size:13px;color:#1e293b;line-height:1.7;font-style:italic;">{text}</div>',
        unsafe_allow_html=True,
    )


def render(sub):
    """Render the Allocation Engine page."""

    if not st.session_state.portfolio_loaded:
        st.info("Load your portfolio first to use the Allocation Engine.")
        return
    if not st.session_state.pi_results or not st.session_state.pi_results.get("tickers"):
        st.info("Run Conviction Intelligence on candidate securities first — then simulate allocation here.")
        return

    pi = st.session_state.pi_results
    pi_tickers = pi["tickers"]
    ps = st.session_state.portfolio_state
    risk_s = st.session_state.risk_state
    summary = ps["summary"]

    # ══════════════════════════════════════════════════════════════
    # PORTFOLIO CONTEXT — current state before simulation
    # ══════════════════════════════════════════════════════════════
    render_section_header("Current Portfolio Context")

    st.markdown(
        f'<div style="font-size:13px;color:#64748b;margin:-4px 0 12px 0;line-height:1.5;">'
        f'Current portfolio snapshot before any allocation changes. '
        f'<span style="color:#94a3b8;">Risk metrics computed from historical returns.</span></div>',
        unsafe_allow_html=True,
    )

    # Key context metrics
    ctx1, ctx2, ctx3, ctx4, ctx5 = st.columns(5)
    rm = risk_s.get("risk_metrics") or {} if risk_s else {}

    with ctx1: render_metric_card("Portfolio Value", fmt_currency(ps["total_value"]))
    with ctx2: render_metric_card("Positions", str(len(summary)))
    with ctx3: render_metric_card("Volatility", f"{rm.get('annualized_volatility', 0):.1%}")
    with ctx4: render_metric_card("HHI Concentration", f"{rm.get('hhi_concentration', 0):.3f}")
    with ctx5:
        top_w = summary["Weight (%)"].max()
        conc = "LOW" if top_w < 20 else "MODERATE" if top_w < 35 else "HIGH"
        conc_c = SUCCESS_GREEN if conc == "LOW" else BRAND_BLUE if conc == "MODERATE" else DANGER_RED
        render_metric_card("Concentration", conc)

    # ══════════════════════════════════════════════════════════════
    # SIMULATION INPUTS
    # ══════════════════════════════════════════════════════════════
    render_section_header("Simulate Allocation")

    pi_synth = pi.get("synthesis", {})

    # Candidate conviction cards — before inputs
    card_cols = st.columns(len(pi_tickers))
    for idx, t in enumerate(pi_tickers):
        synth = pi_synth.get(t, {})
        prob = synth.get("outperformance_probability", 0)
        level = synth.get("conviction_level", "N/A")
        lc = SUCCESS_GREEN if prob > 0.6 else DANGER_RED if prob < 0.4 else BRAND_BLUE

        with card_cols[idx]:
            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;'
                f'background:#fafbfc;">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                f'margin-bottom:4px;">'
                f'<span style="font-size:15px;font-weight:800;color:#0f172a;">{t}</span>'
                f'<span style="font-size:16px;font-weight:800;color:{lc};'
                f'font-variant-numeric:tabular-nums;">{prob:.0%}</span></div>'
                f'<div style="font-size:12px;font-weight:700;color:{lc};'
                f'text-transform:uppercase;letter-spacing:0.05em;">{level}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Per-stock $ input + simulate button
    n = len(pi_tickers)
    input_cols = st.columns(n + 1)
    investment_amounts = {}
    for i, ticker in enumerate(pi_tickers):
        with input_cols[i]:
            val = st.number_input(
                f"{ticker} ($)", min_value=0.0, value=0.0,
                step=500.0, format="%.0f", key=f"hypo_amt_{ticker}",
            )
            investment_amounts[ticker] = val
    total_invest = sum(investment_amounts.values())
    with input_cols[n]:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        simulate_btn = st.button(
            f"Run Simulation (${total_invest:,.0f})" if total_invest > 0 else "Enter amounts",
            key="hypo_simulate_btn", type="primary",
            use_container_width=True, disabled=total_invest == 0,
        )

    # ══════════════════════════════════════════════════════════════
    # RUN SIMULATION
    # ══════════════════════════════════════════════════════════════
    if simulate_btn and total_invest > 0:
        from invictus.agents.hypo_agent import compute_before_after, generate_pros_cons
        from invictus.data.portfolio_loader import fetch_price_history

        with st.status("Running Allocation Simulation...", expanded=True) as sim_status:
            existing_tickers = ps["holdings"]["Ticker"].tolist()
            all_tickers = list(set(existing_tickers + [t for t in pi_tickers if investment_amounts.get(t, 0) > 0]))

            sim_status.update(label="Fetching price history...", state="running")
            st.write("▸ **Fetching prices**")
            prices = fetch_price_history(all_tickers)

            sim_status.update(label="Computing portfolio impact...", state="running")
            st.write("▸ **Computing risk metrics**")
            active_positions = {t: v for t, v in investment_amounts.items() if v > 0}
            comparison = compute_before_after(ps, risk_s, active_positions, prices)

            if "error" in comparison:
                sim_status.update(label=f"Error: {comparison['error']}", state="error")
            else:
                sim_status.update(label="Generating portfolio fit analysis...", state="running")
                st.write("▸ **Generating AI analysis**")
                commentary = generate_pros_cons(comparison)

                # Build per-stock rationale
                rationale = {}
                for t in active_positions:
                    synth = pi_synth.get(t, {})
                    prob = synth.get("outperformance_probability", 0)
                    level = synth.get("conviction_level", "N/A")
                    drivers = synth.get("drivers", [])
                    risks = synth.get("risks", [])
                    amt = active_positions[t]
                    pct_of_new = amt / comparison["new_total"] * 100

                    # Check if ticker already in portfolio
                    existing_weight = ps["weights"].get(t, 0) * 100
                    new_weight = comparison["new_weights"].get(t, 0) * 100

                    lines = []
                    lines.append(f"Conviction: {level} ({prob:.0%} outperformance probability).")
                    if existing_weight > 0:
                        lines.append(f"Existing position — weight moves from {existing_weight:.1f}% to {new_weight:.1f}%.")
                    else:
                        lines.append(f"New position — enters portfolio at {new_weight:.1f}% weight.")
                    if drivers:
                        lines.append(f"Supported by: {drivers[0]}.")
                    if risks:
                        lines.append(f"Risk factor: {risks[0]}.")

                    rationale[t] = " ".join(lines)

                st.session_state.hypo_results = {
                    "comparison": comparison,
                    "commentary": commentary,
                    "positions": active_positions,
                    "rationale": rationale,
                }
                sim_status.update(label="Simulation complete", state="complete", expanded=False)
                st.rerun()

    # ══════════════════════════════════════════════════════════════
    # RENDER RESULTS
    # ══════════════════════════════════════════════════════════════
    hypo = st.session_state.hypo_results
    if not hypo:
        return

    comp = hypo["comparison"]
    comm = hypo["commentary"]
    positions = hypo["positions"]
    rationale = hypo.get("rationale", {})
    before = comp["before"]
    after = comp["after"]
    deltas = comp["deltas"]

    # ── Verdict Banner ────────────────────────────────────────────
    verdict = comm["verdict"]
    verdict_colors = {
        "FAVORABLE": SUCCESS_GREEN, "UNFAVORABLE": DANGER_RED,
        "MIXED": "#f59e0b", "IMMATERIAL": SLATE_500, "NEUTRAL": SLATE_500,
    }
    vc = verdict_colors.get(verdict, BRAND_BLUE)
    pos_labels = [f"{t}: ${v:,.0f}" for t, v in positions.items()]
    st.markdown(
        f'<div style="background:{vc}15;border-left:4px solid {vc};padding:14px 20px;'
        f'border-radius:0 8px 8px 0;margin-bottom:16px;">'
        f'<span style="color:{vc};font-weight:800;font-size:15px;letter-spacing:0.1em;">'
        f'ALLOCATION VERDICT: {verdict}</span>'
        f'<div style="color:#1e293b;font-size:14px;margin-top:6px;line-height:1.6;">'
        f'{comm["verdict_detail"]}</div>'
        f'<div style="color:#475569;font-size:13px;margin-top:8px;font-weight:600;">'
        f'{" · ".join(pos_labels)} · Portfolio: ${before["total_value"]:,.0f} → ${comp["new_total"]:,.0f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _is_material = comm.get("is_material", True)

    # ── Metrics reference (used by Portfolio Fit Analysis below) ──
    _metrics_def = [
        ("sharpe_ratio", "Sharpe Ratio", "{:.2f}", False),
        ("annualized_volatility", "Ann. Volatility", "{:.1%}", True),
        ("max_drawdown", "Max Drawdown", "{:.1%}", False),
        ("var_95", "Value-at-Risk (95%)", "{:.2%}", False),
        ("hhi_concentration", "Concentration (HHI)", "{:.3f}", True),
        ("annualized_return", "Ann. Return", "{:.1%}", False),
    ]

    # ── Weight Changes (collapsed) ───────────────────────────────
    wc = comp.get("weight_changes", [])
    if wc:
        # Only show tickers that actually changed
        active_wc = [r for r in wc if abs(r.get("Change (pp)", 0)) > 0.05]
        if active_wc:
            with st.expander("Weight Changes", expanded=False):
                _thl = 'style="text-align:left;padding:6px 8px;color:#475569;font-weight:700;font-size:12px;"'
                _thr = 'style="text-align:right;padding:6px 8px;color:#475569;font-weight:700;font-size:12px;"'
                wc_html = (
                    '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
                    f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                    f'<th {_thl}>Ticker</th>'
                    f'<th {_thr}>Before</th>'
                    f'<th {_thr}>After</th>'
                    f'<th {_thr}>Change</th>'
                    f'</tr></thead><tbody>'
                )
                for r in sorted(active_wc, key=lambda x: abs(x.get("Change (pp)", 0)), reverse=True):
                    chg = r.get("Change (pp)", 0)
                    chg_c = SUCCESS_GREEN if chg > 0 else DANGER_RED
                    arrow = "▲" if chg > 0 else "▼"
                    wc_html += (
                        f'<tr style="border-bottom:1px solid #f1f5f9;">'
                        f'<td style="padding:6px 8px;font-weight:700;color:#1e293b;">{r.get("Ticker", "")}</td>'
                        f'<td style="padding:6px 8px;text-align:right;color:#64748b;'
                        f'font-variant-numeric:tabular-nums;">{r.get("Before (%)", 0):.1f}%</td>'
                        f'<td style="padding:6px 8px;text-align:right;color:{chg_c};font-weight:700;'
                        f'font-variant-numeric:tabular-nums;">{r.get("After (%)", 0):.1f}%</td>'
                        f'<td style="padding:6px 8px;text-align:right;color:{chg_c};font-weight:600;'
                        f'font-variant-numeric:tabular-nums;">{arrow} {chg:+.1f}pp</td>'
                        f'</tr>'
                    )
                wc_html += '</tbody></table>'
                st.markdown(wc_html, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # PER-STOCK ALLOCATION RATIONALE
    # ══════════════════════════════════════════════════════════════
    if rationale:
        render_section_header("Allocation Rationale")
        for t, text in rationale.items():
            amt = positions.get(t, 0)
            synth = pi_synth.get(t, {})
            prob = synth.get("outperformance_probability", 0)
            level = synth.get("conviction_level", "N/A")
            lc = SUCCESS_GREEN if prob > 0.6 else DANGER_RED if prob < 0.4 else BRAND_BLUE

            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-left:4px solid {lc};'
                f'border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                f'<div>'
                f'<span style="font-size:15px;font-weight:800;color:#1e293b;">{t}</span>'
                f'<span style="font-size:13px;font-weight:600;color:{lc};margin-left:10px;">{level}</span>'
                f'</div>'
                f'<span style="font-size:14px;font-weight:700;color:{lc};">${amt:,.0f}</span></div>'
                f'<div style="font-size:13px;color:#475569;line-height:1.7;">{text}</div></div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════
    # AI PORTFOLIO FIT ANALYSIS — Pros & Cons
    # ══════════════════════════════════════════════════════════════
    if not _is_material:
        _intelligence_callout(comm["verdict_detail"])
        return

    ranked = comm.get("ranked_metrics", [])
    _ai_lookup = comm.get("all_commentary", {})

    _pro_keys = {r["pro"]["key"] for r in comm.get("rows", []) if r.get("pro")}
    _con_keys = {r["con"]["key"] for r in comm.get("rows", []) if r.get("con")}

    positive_metrics = []
    negative_metrics = []
    for key, label, fmt, lower_better in _metrics_def:
        d_val = deltas.get(key, 0)
        if abs(d_val) < 1e-9:
            continue
        if lower_better:
            is_good = d_val <= 0
        elif key == "max_drawdown":
            is_good = d_val >= 0
        else:
            is_good = d_val >= 0
        entry = (key, label, fmt, lower_better, is_good, d_val)
        if is_good and key in _pro_keys:
            positive_metrics.append(entry)
        elif not is_good and key in _con_keys:
            negative_metrics.append(entry)

    def _render_metric_row(key, label, fmt, d_val, is_good, color):
        b_val = before.get(key, 0)
        a_val = after.get(key, 0)
        arrow = "▲" if d_val > 0 else "▼"
        card_col, ai_col = st.columns([1, 1.2])
        with card_col:
            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;'
                f'background:#fafbfc;margin-bottom:8px;">'
                f'<div style="font-size:12px;font-weight:700;color:#475569;'
                f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">{label}</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
                f'<span style="font-size:16px;font-weight:700;color:#1e293b;">{fmt.format(b_val)}</span>'
                f'<span style="font-size:17px;color:{color};font-weight:800;">{arrow}</span>'
                f'<span style="font-size:16px;font-weight:700;color:{color};">{fmt.format(a_val)}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with ai_col:
            ai = _ai_lookup.get(key)
            if ai:
                bg = "#f0fdf4" if is_good else "#fef2f2"
                bc = SUCCESS_GREEN if is_good else DANGER_RED
                st.markdown(
                    f'<div style="background:{bg};border-left:3px solid {bc};'
                    f'padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:8px;'
                    f'font-size:13px;color:#1e293b;line-height:1.7;">{ai["text"]}</div>',
                    unsafe_allow_html=True,
                )

    render_section_header("Portfolio Fit Analysis")

    if positive_metrics:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:14px 0 10px 0;">'
            f'<div style="width:4px;height:20px;background:{SUCCESS_GREEN};border-radius:2px;"></div>'
            f'<span style="font-weight:800;font-size:13px;letter-spacing:0.1em;'
            f'color:{SUCCESS_GREEN};text-transform:uppercase;">'
            f'Portfolio Improvements ({len(positive_metrics)})</span></div>',
            unsafe_allow_html=True,
        )
        for key, label, fmt, _, is_good, d_val in positive_metrics:
            _render_metric_row(key, label, fmt, d_val, True, SUCCESS_GREEN)

    if negative_metrics:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:14px 0 10px 0;">'
            f'<div style="width:4px;height:20px;background:{DANGER_RED};border-radius:2px;"></div>'
            f'<span style="font-weight:800;font-size:13px;letter-spacing:0.1em;'
            f'color:{DANGER_RED};text-transform:uppercase;">'
            f'Portfolio Risks ({len(negative_metrics)})</span></div>',
            unsafe_allow_html=True,
        )
        for key, label, fmt, _, is_good, d_val in negative_metrics:
            _render_metric_row(key, label, fmt, d_val, False, DANGER_RED)

    if not positive_metrics and not negative_metrics:
        st.info("No material metric changes detected at this allocation level.")
