"""
invictus.pages.dev_analytics.backtest
======================================
Backtest — walk-forward ex-ante 2024 backtest + conviction vs forward returns.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from invictus.design import (
    render_section_header, apply_invictus_layout,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT, SLATE_500,
)
from invictus.pages.dev_analytics._shared import subtitle, conviction_card, health_color


def render_backtest():
    # ══════════════════════════════════════════════════════════════
    # SECTION 1: Walk-Forward 2024 Backtest (the star feature)
    # ══════════════════════════════════════════════════════════════
    _render_walk_forward_backtest()

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: Live Conviction Backtest (from observability logs)
    # ══════════════════════════════════════════════════════════════
    _render_logged_backtest()


# ══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD EX-ANTE BACKTEST
# ══════════════════════════════════════════════════════════════════════════

def _render_walk_forward_backtest():
    render_section_header("Walk-Forward Ex-Ante Backtest — 2024")
    subtitle(
        'Replays the conviction pipeline across 2024 using <b>only data available at each evaluation date</b>. '
        'No look-ahead bias — every signal is computed from point-in-time financials and prices. '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">Fundamental</span> + '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Bayesian ML</span> signals evaluated monthly.'
    )

    # ── Config UI ────────────────────────────────────────────────
    with st.expander("Backtest Configuration", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            tickers_input = st.text_input(
                "Tickers (comma-separated)",
                value="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,V,JNJ",
                key="wf_tickers",
            )
        with c2:
            freq = st.selectbox("Frequency", ["monthly", "biweekly"],
                               key="wf_freq")
        with c3:
            year = st.selectbox("Year", [2024, 2023], key="wf_year")

        st.markdown("---")
        qm1, qm2 = st.columns([1, 2])
        with qm1:
            quality_mode = st.radio(
                "Signal Quality Mode",
                ["Full Signal Proxy", "Conservative (2-signal)"],
                key="wf_quality_mode",
                help="Controls how much conviction probabilities are shrunk toward 50%.",
            )
        with qm2:
            if quality_mode == "Conservative (2-signal)":
                st.caption(
                    "**Q = 0.60** — Heavy shrinkage. With only fundamental + technical signals "
                    "active (management & flows missing), probabilities are pulled strongly toward "
                    "50%. This is mathematically honest but produces few directional trades, "
                    "making hit-rate and P&L metrics unreliable due to small sample size."
                )
            else:
                st.caption(
                    "**Q = 0.85** — Light shrinkage. Treats missing signals (management & flows) "
                    "as approximately neutral rather than completely unknown. This widens the "
                    "probability spread, generating enough trades to measure hit rates, "
                    "calibration, and P&L meaningfully. The underlying signal ranking is identical."
                )

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    sig_quality = 0.85 if quality_mode == "Full Signal Proxy" else 0.6

    # ── Run Button ───────────────────────────────────────────────
    if st.button("Run Walk-Forward Backtest", key="wf_run_btn", type="primary"):
        _run_walk_forward(tickers, year, freq, sig_quality)

    # ── Render cached results ────────────────────────────────────
    wf = st.session_state.get("_wf_backtest_results")
    if wf and wf.get("status") == "ok":
        _render_wf_results(wf)
    elif wf and wf.get("status") == "error":
        st.error(wf.get("message", "Backtest failed"))
    elif not wf:
        st.info(
            "Click **Run Walk-Forward Backtest** to replay the conviction pipeline "
            "across 2024 with ex-ante data. This downloads ~18 months of price data "
            "and quarterly financials, then evaluates signals monthly."
        )


def _run_walk_forward(tickers, year, freq, signal_quality=0.85):
    """Execute the walk-forward backtest with progress tracking."""
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()

    steps = {"current": 0, "total": 4}

    def progress_cb(msg):
        steps["current"] += 1
        pct = min(steps["current"] / max(steps["total"] * 3, 1), 0.99)
        progress_bar.progress(pct, text=msg)
        status_text.caption(msg)

    try:
        from invictus.backtest import run_walk_forward

        results = run_walk_forward(
            tickers=tickers,
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            frequency=freq,
            signal_quality=signal_quality,
            progress_callback=progress_cb,
        )

        st.session_state["_wf_backtest_results"] = results
        progress_bar.progress(1.0, text="Complete!")
        status_text.empty()
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Backtest failed: {e}")
        st.session_state["_wf_backtest_results"] = {
            "status": "error", "message": str(e)
        }


def _render_wf_results(wf: dict):
    """Render walk-forward backtest results."""
    cfg = wf.get("config", {})
    summary = wf.get("summary", {})

    # ── Quality mode banner ────────────────────────────────────────
    q_mode = cfg.get("quality_mode", "conservative")
    q_val = cfg.get("signal_quality", 0.6)
    if q_mode == "full_proxy":
        st.markdown(
            f'<div style="background:linear-gradient(90deg,{BRAND_BLUE}11,{BRAND_BLUE}05);'
            f'border-left:3px solid {BRAND_BLUE};padding:10px 14px;border-radius:6px;'
            f'font-size:12px;color:#64748b;margin-bottom:12px;">'
            f'<b style="color:{BRAND_BLUE};">Full Signal Proxy Mode</b> — '
            f'Quality shrinkage Q={q_val:.2f}. Missing signals treated as neutral (not unknown). '
            f'Wider probability spread generates more directional trades for meaningful evaluation.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:linear-gradient(90deg,{SLATE_500}11,{SLATE_500}05);'
            f'border-left:3px solid {SLATE_500};padding:10px 14px;border-radius:6px;'
            f'font-size:12px;color:#64748b;margin-bottom:12px;">'
            f'<b style="color:{SLATE_500};">Conservative Mode</b> — '
            f'Quality shrinkage Q={q_val:.2f}. Heavy pull toward 50% (only 2/4 signals active). '
            f'Few trades will cross thresholds — ranking metrics (ρ, IC) are more informative than hit rates.</div>',
            unsafe_allow_html=True,
        )

    # ── Verdict Summary (cross-horizon) ─────────────────────────
    per_horizon = wf.get("per_horizon", {})
    if per_horizon:
        _render_verdict_summary(per_horizon, summary)

    # ── Overview Cards ────────────────────────────────────────────
    render_section_header("Backtest Overview")
    subtitle(
        f'{cfg.get("n_eval_dates", 0)} evaluation dates × '
        f'{summary.get("unique_tickers", 0)} tickers = '
        f'{summary.get("total_convictions", 0)} total conviction scores generated. '
        f'Each score uses only data available at its evaluation date — no look-ahead.'
    )

    o1, o2, o3, o4 = st.columns(4)
    with o1: conviction_card("Convictions", str(summary.get("total_convictions", 0)),
                              sub_label="SCORES GENERATED")
    with o2: conviction_card("Avg Probability", f"{summary.get('avg_probability', 0.5):.1%}",
                              sub_label="SHOULD BE ~50%")
    with o3: conviction_card("Bullish Calls", str(summary.get("bullish_count", 0)),
                              color=SUCCESS_GREEN, sub_label="P > 60%")
    with o4: conviction_card("Bearish Calls", str(summary.get("bearish_count", 0)),
                              color=DANGER_RED, sub_label="P < 40%")

    st.caption(
        "**How to read:** Convictions = total model evaluations (tickers × dates). "
        "Avg Probability near 50% means the model isn't systematically biased bullish or bearish. "
        "Bullish/Bearish counts show how many crossed the trading thresholds."
    )

    # ── Per-Horizon Results ──────────────────────────────────────
    per_horizon = wf.get("per_horizon", {})
    for h in sorted(per_horizon.keys()):
        hdata = per_horizon[h]
        _render_wf_horizon(h, hdata)

    # ── Conviction Heatmap (Timeline) ────────────────────────────
    timeline = wf.get("conviction_timeline", [])
    if timeline:
        _render_conviction_heatmap(timeline)

    # ── Quintile Analysis (best horizon) ─────────────────────────
    best_h = max(per_horizon.keys(), key=lambda k: per_horizon[k].get("overall_hit_rate", 0)) if per_horizon else None
    if best_h and per_horizon[best_h].get("quintile_returns"):
        _render_quintile_chart(per_horizon[best_h]["quintile_returns"], best_h)

    # ── Per-Ticker Breakdown ─────────────────────────────────────
    per_ticker = wf.get("per_ticker", [])
    if per_ticker:
        _render_wf_per_ticker(per_ticker, cfg.get("horizons", []))

    # ── Disclaimer ───────────────────────────────────────────────
    st.caption(wf.get("disclaimer", ""))


def _render_verdict_summary(per_horizon: dict, summary: dict):
    """
    Cross-horizon verdict — one scannable block that tells you
    whether the conviction signal works, at which horizons, and how strong.
    """
    _h_labels = {5: "1W", 10: "2W", 21: "1M", 63: "1Q"}

    render_section_header("Signal Verdict")
    subtitle(
        'Cross-horizon summary — does the conviction signal predict forward returns? '
        '<span style="color:#94a3b8;">Scans all horizons and reports what matters.</span>'
    )

    # ── Collect per-horizon stats ────────────────────────────────
    rows = []
    for h in sorted(per_horizon.keys()):
        hd = per_horizon[h]
        pnl = hd.get("hypothetical_pnl", {})
        quint = hd.get("quintile_returns", [])

        # Quintile monotonicity check
        if len(quint) >= 2:
            q_rets = [q["avg_return"] for q in quint]
            q1_q5 = (q_rets[-1] - q_rets[0]) * 100
            mono = all(q_rets[i] <= q_rets[i + 1] for i in range(len(q_rets) - 1))
        else:
            q1_q5 = 0.0
            mono = False

        rows.append({
            "horizon": h,
            "label": _h_labels.get(h, f"{h}d"),
            "hit_rate": hd.get("overall_hit_rate", 0.5),
            "rho": hd.get("spearman_rho", 0),
            "rho_p": hd.get("spearman_pval", 1),
            "ic": hd.get("information_coefficient", 0),
            "ic_p": hd.get("ic_pval", 1),
            "trades": pnl.get("trade_count", 0),
            "total_return": pnl.get("total_return", 0),
            "sharpe": pnl.get("sharpe", 0),
            "profit_factor": pnl.get("profit_factor", 0),
            "q1_q5": q1_q5,
            "monotonic": mono,
        })

    # ── Summary table ────────────────────────────────────────────
    with st.container(border=True):
        # Build HTML table
        header = (
            '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
            '<tr style="border-bottom:2px solid #e2e8f0;color:#64748b;font-weight:600;text-align:center;">'
            '<td style="text-align:left;padding:8px;">Horizon</td>'
            '<td style="padding:8px;">Hit Rate</td>'
            '<td style="padding:8px;">Spearman ρ</td>'
            '<td style="padding:8px;">IC</td>'
            '<td style="padding:8px;">Trades</td>'
            '<td style="padding:8px;">Return</td>'
            '<td style="padding:8px;">Sharpe</td>'
            '<td style="padding:8px;">Q5−Q1</td>'
            '<td style="padding:8px;">Verdict</td>'
            '</tr>'
        )

        body = ""
        for r in rows:
            # Color helpers
            def _c(val, good, bad):
                if val >= good: return SUCCESS_GREEN
                if val <= bad: return DANGER_RED
                return "#64748b"

            hr_c = _c(r["hit_rate"], 0.55, 0.48)
            rho_c = _c(r["rho"], 0.10, -0.05)
            ic_c = _c(r["ic"], 0.05, -0.03)
            ret_c = SUCCESS_GREEN if r["total_return"] > 0 else DANGER_RED if r["total_return"] < 0 else "#64748b"
            sh_c = _c(r["sharpe"], 0.5, 0)
            q_c = SUCCESS_GREEN if r["q1_q5"] > 0 else DANGER_RED if r["q1_q5"] < 0 else "#64748b"

            # Significance markers
            rho_sig = " ✓" if r["rho_p"] < 0.05 else " ~" if r["rho_p"] < 0.10 else ""
            ic_sig = " ✓" if r["ic_p"] < 0.05 else " ~" if r["ic_p"] < 0.10 else ""

            # Overall verdict for this horizon
            score = 0
            if r["rho"] > 0.10 and r["rho_p"] < 0.10: score += 2
            elif r["rho"] > 0: score += 1
            if r["ic"] > 0.05 and r["ic_p"] < 0.10: score += 2
            elif r["ic"] > 0: score += 1
            if r["hit_rate"] > 0.55 and r["trades"] > 5: score += 2
            elif r["hit_rate"] > 0.50 and r["trades"] > 0: score += 1
            if r["q1_q5"] > 1.0: score += 2
            elif r["q1_q5"] > 0: score += 1
            if r["monotonic"]: score += 1

            if score >= 6:
                verdict = f'<span style="color:{SUCCESS_GREEN};font-weight:700;">STRONG SIGNAL</span>'
            elif score >= 4:
                verdict = f'<span style="color:{SUCCESS_GREEN};font-weight:600;">SIGNAL PRESENT</span>'
            elif score >= 2:
                verdict = f'<span style="color:#f59e0b;font-weight:600;">WEAK SIGNAL</span>'
            else:
                verdict = f'<span style="color:{DANGER_RED};font-weight:600;">NO SIGNAL</span>'

            body += (
                f'<tr style="border-bottom:1px solid #f1f5f9;text-align:center;">'
                f'<td style="text-align:left;padding:8px;font-weight:700;">{r["label"]} ({r["horizon"]}d)</td>'
                f'<td style="padding:8px;color:{hr_c};font-weight:600;">{r["hit_rate"]:.0%}</td>'
                f'<td style="padding:8px;color:{rho_c};font-weight:600;">{r["rho"]:+.3f}{rho_sig}</td>'
                f'<td style="padding:8px;color:{ic_c};font-weight:600;">{r["ic"]:+.3f}{ic_sig}</td>'
                f'<td style="padding:8px;">{r["trades"]}</td>'
                f'<td style="padding:8px;color:{ret_c};font-weight:600;">'
                f'{r["total_return"]:+.2%}</td>' if r["trades"] > 0 else
                f'<td style="padding:8px;color:#94a3b8;">—</td>'
            )
            body += (
                f'<td style="padding:8px;color:{sh_c};font-weight:600;">'
                f'{r["sharpe"]:.2f}</td>' if r["trades"] > 0 else
                '<td style="padding:8px;color:#94a3b8;">—</td>'
            )
            body += (
                f'<td style="padding:8px;color:{q_c};font-weight:600;">'
                f'{r["q1_q5"]:+.1f}%{"  ↗" if r["monotonic"] else ""}</td>'
                f'<td style="padding:8px;">{verdict}</td>'
                f'</tr>'
            )

        st.markdown(header + body + '</table>', unsafe_allow_html=True)

    # ── Overall assessment ───────────────────────────────────────
    # Find best horizon
    best = max(rows, key=lambda r: (
        (1 if r["rho"] > 0.10 and r["rho_p"] < 0.10 else 0) * 3 +
        (1 if r["ic"] > 0.05 else 0) * 2 +
        (1 if r["q1_q5"] > 0 else 0) * 2 +
        (1 if r["hit_rate"] > 0.55 and r["trades"] > 5 else 0) * 2
    ))

    sig_horizons = [r for r in rows if r["rho"] > 0.05 and r["rho_p"] < 0.15]
    mono_horizons = [r for r in rows if r["monotonic"] and r["q1_q5"] > 0]
    profitable = [r for r in rows if r["total_return"] > 0 and r["trades"] > 5]

    # Compose assessment text
    parts = []

    if sig_horizons:
        labels = ", ".join(r["label"] for r in sig_horizons)
        best_rho = max(r["rho"] for r in sig_horizons)
        parts.append(
            f'Positive rank correlation at <b>{labels}</b> '
            f'(best ρ = {best_rho:+.3f}) — higher conviction scores are associated with higher returns.'
        )
    else:
        parts.append(
            'No statistically significant rank correlation found at any horizon. '
            'The conviction ranking does not reliably predict return ordering.'
        )

    if mono_horizons:
        labels = ", ".join(r["label"] for r in mono_horizons)
        parts.append(
            f'Quintile monotonicity confirmed at <b>{labels}</b> — '
            f'the most bullish quintile outperforms the most bearish with a clean staircase.'
        )

    if profitable:
        labels = ", ".join(r["label"] for r in profitable)
        best_pf = max(r["profit_factor"] for r in profitable)
        parts.append(
            f'Hypothetical long/short strategy profitable at <b>{labels}</b> '
            f'(best profit factor: {min(best_pf, 99):.2f}).'
        )
    elif any(r["trades"] > 0 for r in rows):
        parts.append(
            'Hypothetical long/short strategy was not consistently profitable. '
            'This may improve with all 4 signal sources active in live mode.'
        )

    total_trades = sum(r["trades"] for r in rows)
    if total_trades == 0:
        parts.append(
            '<b>No directional trades were generated</b> — all probabilities stayed within the '
            '40%-60% neutral band. Switch to Full Signal Proxy mode to widen the spread.'
        )

    # Signal sources caveat
    parts.append(
        '<span style="color:#94a3b8;">Backtest uses 2 of 4 signal sources '
        '(fundamental + technical). Live pipeline adds management outlook + institutional flows, '
        'which would widen conviction spread and improve signal strength.</span>'
    )

    assessment_color = SUCCESS_GREEN if len(sig_horizons) >= 2 else "#f59e0b" if sig_horizons else DANGER_RED
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{assessment_color}08,{assessment_color}03);'
        f'border-left:3px solid {assessment_color};padding:12px 16px;border-radius:6px;'
        f'font-size:12px;color:#475569;line-height:1.7;margin-top:8px;">'
        + "<br>".join(parts) +
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "**How to read the table:** ✓ = statistically significant (p<0.05), ~ = marginal (p<0.10). "
        "**Q5−Q1** = return spread between most bullish and most bearish quintiles; ↗ = perfectly monotonic. "
        "**Verdict** scores each horizon across 4 dimensions: rank correlation, IC, hit rate, and quintile separation."
    )


def _render_wf_horizon(h: int, hdata: dict):
    """Render results for a single forward-return horizon."""
    _h_labels = {5: "1 Week", 10: "2 Weeks", 21: "1 Month", 63: "1 Quarter"}
    h_label = _h_labels.get(h, f"{h}d")
    render_section_header(f"{h}-Day Forward Returns ({h_label})")
    subtitle(
        f'How well did conviction scores predict actual {h_label.lower()} stock returns? '
        f'<span style="color:#94a3b8;">{hdata["n_observations"]} observations evaluated.</span>'
    )

    # ── Row 1: Direction accuracy ────────────────────────────────
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        hr = hdata["overall_hit_rate"]
        conviction_card("Hit Rate", f"{hr:.0%}",
                        color=SUCCESS_GREEN if hr > 0.55 else DANGER_RED if hr < 0.50 else SLATE_500,
                        sub_label="EDGE" if hr > 0.55 else "COIN FLIP" if abs(hr - 0.5) < 0.02 else "NO EDGE")
    with h2:
        bhr = hdata['bullish_hit_rate']
        bt_count = hdata['bullish_count']
        conviction_card("Bullish Hits", f"{bhr:.0%}" if bt_count > 0 else "—",
                        color=SUCCESS_GREEN if bhr > 0.55 and bt_count > 0 else SLATE_500,
                        sub_label=f"{bt_count} TRADES")
    with h3:
        bear_hr = hdata['bearish_hit_rate']
        bear_count = hdata['bearish_count']
        conviction_card("Bearish Hits", f"{bear_hr:.0%}" if bear_count > 0 else "—",
                        color=SUCCESS_GREEN if bear_hr > 0.55 and bear_count > 0 else SLATE_500,
                        sub_label=f"{bear_count} TRADES")
    with h4:
        rho = hdata["spearman_rho"]
        p_val = hdata["spearman_pval"]
        sig_label = f"p={p_val:.3f}"
        if p_val < 0.05:
            sig_label += " ✓"
        conviction_card("Spearman ρ", f"{rho:+.3f}",
                        color=SUCCESS_GREEN if rho > 0.10 and p_val < 0.10 else DANGER_RED if rho < -0.05 else SLATE_500,
                        sub_label=sig_label)

    st.caption(
        f"**Hit Rate** — % of directional calls (P>60% or P<40%) where the stock moved in the predicted direction. "
        f"Above 55% = meaningful edge. "
        f"**Spearman ρ** — rank correlation between conviction and actual return. "
        f"Positive ρ means higher-conviction stocks tended to outperform. p<0.05 = statistically significant."
    )

    # ── Row 2: Signal quality & returns ──────────────────────────
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        ic = hdata.get("information_coefficient", 0)
        ic_p = hdata.get("ic_pval", 1.0)
        ic_label = f"p={ic_p:.3f}"
        if ic_p < 0.05:
            ic_label += " ✓"
        conviction_card("Info Coefficient", f"{ic:+.3f}",
                        color=SUCCESS_GREEN if ic > 0.05 and ic_p < 0.10 else SLATE_500,
                        sub_label=ic_label)
    with e2:
        conviction_card("Avg Fwd Return", f"{hdata['avg_fwd_return']:+.2%}",
                        sub_label="ALL OBSERVATIONS")
    with e3:
        bull_fwd = hdata.get('avg_fwd_bullish', 0)
        conviction_card("Avg Bullish Fwd", f"{bull_fwd:+.2%}" if hdata['bullish_count'] > 0 else "—",
                        color=SUCCESS_GREEN if bull_fwd > 0 and hdata['bullish_count'] > 0 else SLATE_500,
                        sub_label="LONG TRADES" if hdata['bullish_count'] > 0 else "NO TRADES")
    with e4:
        bear_fwd = hdata.get('avg_fwd_bearish', 0)
        conviction_card("Avg Bearish Fwd", f"{bear_fwd:+.2%}" if hdata['bearish_count'] > 0 else "—",
                        color=SUCCESS_GREEN if bear_fwd < 0 and hdata['bearish_count'] > 0 else SLATE_500,
                        sub_label="SHORT TRADES" if hdata['bearish_count'] > 0 else "NO TRADES")

    st.caption(
        f"**IC (Information Coefficient)** — Pearson correlation between conviction probability and forward return. "
        f"In institutional quant, IC of 0.05-0.10 is considered good for a single alpha signal. "
        f"**Avg Fwd** — mean {h_label.lower()} return; bullish avg should be positive, bearish should be negative."
    )

    # ── Calibration curve ────────────────────────────────────────
    cal = hdata.get("calibration", [])
    if cal:
        with st.container(border=True):
            pred = [c["predicted_prob"] for c in cal]
            actual = [c["actual_hit_rate"] for c in cal]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pred, y=actual, mode="markers+lines",
                name="Model", marker=dict(size=12, color=BRAND_BLUE),
                line=dict(color=BRAND_BLUE, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Perfect Calibration", line=dict(color=SLATE_500, dash="dash", width=1),
            ))
            apply_invictus_layout(fig, height=300, title=f"Calibration Curve — {h}d Horizon")
            fig.update_layout(xaxis_title="Predicted Probability", yaxis_title="Actual Hit Rate")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.caption(
            "**Calibration** — each dot is a group of predictions binned by predicted probability. "
            "The dashed line is perfect calibration (predicted = actual). "
            "Dots above the line = model is underconfident (it's right more often than it thinks). "
            "Dots below = overconfident. Tight clustering near 50% means quality shrinkage is compressing the spread."
        )

    # ── Hypothetical P&L ─────────────────────────────────────────
    pnl = hdata.get("hypothetical_pnl", {})
    if pnl.get("trade_count", 0) > 0:
        subtitle(
            f'<b>Hypothetical Long/Short P&L</b> — '
            f'Go long when P > 60%, short when P < 40%. Equal-weight 10% positions. '
            f'{pnl["trade_count"]} total trades ({pnl.get("long_count", 0)} long, {pnl.get("short_count", 0)} short).'
        )

        p1, p2, p3, p4 = st.columns(4)
        ret_c = SUCCESS_GREEN if pnl["total_return"] > 0 else DANGER_RED
        with p1: conviction_card("Total Return", f"{pnl['total_return']:+.2%}", color=ret_c,
                                  sub_label="CUMULATIVE")
        with p2: conviction_card("Win Rate", f"{pnl['win_rate']:.0%}",
                                  color=SUCCESS_GREEN if pnl['win_rate'] > 0.5 else DANGER_RED,
                                  sub_label="> 50% = PROFITABLE")
        with p3:
            pf = pnl.get('profit_factor', 0)
            conviction_card("Profit Factor", f"{min(pf, 99):.2f}",
                            color=SUCCESS_GREEN if pf > 1 else DANGER_RED,
                            sub_label="> 1.0 = NET GAIN")
        with p4: conviction_card("Sharpe (Ann.)", f"{pnl.get('sharpe', 0):.2f}",
                                  color=SUCCESS_GREEN if pnl.get('sharpe', 0) > 0.5 else SLATE_500,
                                  sub_label="> 0.5 = RESPECTABLE")

        st.caption(
            "**Win Rate** — % of trades that were profitable. "
            "**Profit Factor** — gross gains ÷ gross losses; above 1.0 means the strategy made money overall. "
            "**Sharpe** — risk-adjusted return (annualized); above 1.0 is strong, above 0.5 is respectable for a 2-signal backtest. "
            "**Max Drawdown** — worst peak-to-trough loss in the cumulative curve."
        )

        # Max drawdown card
        md = pnl.get("max_drawdown", 0)
        if md < 0:
            st.markdown(
                f'<div style="background:#fef2f2;border-left:3px solid {DANGER_RED};padding:8px 12px;'
                f'border-radius:6px;font-size:12px;color:#64748b;">'
                f'Max Drawdown: <b style="color:{DANGER_RED};">{md:+.2%}</b> — '
                f'worst consecutive losing streak before recovery.</div>',
                unsafe_allow_html=True,
            )

        cum = pnl.get("cumulative_pnl", [])
        if cum:
            with st.container(border=True):
                fig_pnl = go.Figure(go.Scatter(
                    x=list(range(1, len(cum) + 1)), y=[c * 100 for c in cum],
                    mode="lines+markers", fill="tozeroy",
                    line=dict(color=SUCCESS_GREEN if cum[-1] > 0 else DANGER_RED, width=2),
                    marker=dict(size=5),
                ))
                apply_invictus_layout(fig_pnl, height=250, title=f"Cumulative P&L — {h}d Horizon")
                fig_pnl.update_layout(yaxis_title="Return (%)", xaxis_title="Trade #")
                st.plotly_chart(fig_pnl, use_container_width=True, config={"displayModeBar": False})
    elif pnl.get("trade_count", 0) == 0:
        st.markdown(
            f'<div style="background:#f8fafc;border-left:3px solid {SLATE_500};padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#64748b;">'
            f'<b>No directional trades generated.</b> All conviction probabilities stayed between '
            f'40%-60%, so no long or short positions were triggered. '
            f'This typically happens in Conservative mode where quality shrinkage compresses the spread. '
            f'Try <b>Full Signal Proxy</b> mode to widen the probability range.</div>',
            unsafe_allow_html=True,
        )


def _render_conviction_heatmap(timeline: list):
    """Render monthly conviction probability heatmap by ticker."""
    render_section_header("Conviction Timeline")
    subtitle('Monthly conviction probability per ticker across the backtest period. '
             f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Green</span> = bullish, '
             f'<span style="color:{DANGER_RED};font-weight:700;">red</span> = bearish.')

    df = pd.DataFrame(timeline)
    if df.empty:
        return

    pivot = df.pivot_table(index="ticker", columns="eval_date", values="prob", aggfunc="mean")
    pivot = pivot.sort_index()

    # Shorten column labels to month
    col_labels = [c[:7] for c in pivot.columns]  # YYYY-MM

    with st.container(border=True):
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=col_labels,
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, DANGER_RED_ALT],
                [0.4, "#fef3c7"],
                [0.5, "#f8fafc"],
                [0.6, "#dcfce7"],
                [1.0, SUCCESS_GREEN],
            ],
            zmin=0.3, zmax=0.7,
            text=pivot.round(2).values,
            texttemplate="%{text:.0%}",
            textfont=dict(size=11),
        ))
        apply_invictus_layout(fig, height=max(250, len(pivot) * 35),
                             title="Monthly Conviction Probability")
        fig.update_layout(
            xaxis_title="Evaluation Month",
            yaxis_title="",
            margin=dict(l=80),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_quintile_chart(quintile_data: list, h: int):
    """Render quintile return monotonicity chart."""
    render_section_header("Return Monotonicity by Conviction Quintile")
    subtitle(
        f'All conviction scores sorted into 5 buckets from most bearish (Q1) to most bullish (Q5). '
        f'<span style="color:#94a3b8;">A clean staircase from left to right = the model ranks stocks correctly.</span>'
    )

    with st.container(border=True):
        labels = [q["quintile"] for q in quintile_data]
        returns = [q["avg_return"] * 100 for q in quintile_data]
        counts = [q["count"] for q in quintile_data]
        colors = [SUCCESS_GREEN if r > 0 else DANGER_RED for r in returns]

        fig = go.Figure(go.Bar(
            x=labels, y=returns, marker_color=colors,
            text=[f"{r:+.2f}%<br>n={c}" for r, c in zip(returns, counts)],
            textposition="outside",
        ))
        apply_invictus_layout(fig, height=320, title=f"Avg {h}d Return by Conviction Quintile")
        fig.update_layout(yaxis_title="Avg Forward Return (%)", xaxis_title="Conviction Quintile")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Monotonicity check
    rets = [q["avg_return"] for q in quintile_data]
    is_monotonic = all(rets[i] <= rets[i+1] for i in range(len(rets) - 1))
    q1_q5_spread = (rets[-1] - rets[0]) * 100 if len(rets) >= 2 else 0

    if is_monotonic and q1_q5_spread > 0:
        st.caption(
            f"✓ **Monotonic** — Q5 outperforms Q1 by {q1_q5_spread:+.2f}%. "
            f"The model successfully ranks stocks: higher conviction → higher returns."
        )
    elif q1_q5_spread > 0:
        st.caption(
            f"**Q5-Q1 spread: {q1_q5_spread:+.2f}%** — the extreme quintiles separate correctly, "
            f"but the middle quintiles aren't perfectly ordered. This is common with limited signal inputs."
        )
    else:
        st.caption(
            f"**Q5-Q1 spread: {q1_q5_spread:+.2f}%** — the highest-conviction quintile did not outperform "
            f"the lowest. This suggests the conviction signal lacks ranking power at this horizon."
        )


def _render_wf_per_ticker(per_ticker: list, horizons: list):
    """Render per-ticker breakdown table."""
    render_section_header("Per-Ticker Results")
    subtitle(
        'Which stocks did the model predict well vs. poorly? '
        'High hit rates on stable earners (AAPL, MSFT) vs. lower on volatile names (TSLA) '
        'tells you which market regimes the signal works best in.'
    )

    pt_df = pd.DataFrame(per_ticker)
    display_cols = ["ticker", "n_evaluations", "avg_conviction", "conviction_std",
                    "avg_fundamental", "avg_technical"]
    fmt = {
        "avg_conviction": "{:.1%}", "conviction_std": "{:.3f}",
        "avg_fundamental": "{:+.3f}", "avg_technical": "{:+.3f}",
    }

    for h in horizons:
        col_hr = f"hit_rate_{h}d"
        col_fwd = f"avg_fwd_{h}d"
        if col_hr in pt_df.columns:
            display_cols.extend([col_hr, col_fwd])
            fmt[col_hr] = "{:.0%}"
            fmt[col_fwd] = "{:+.2%}"

    available = [c for c in display_cols if c in pt_df.columns]
    st.dataframe(
        pt_df[available].style.format(
            {k: v for k, v in fmt.items() if k in available},
            na_rep="—"
        ),
        use_container_width=True, hide_index=True,
    )

    st.caption(
        "**avg_conviction** — mean outperformance probability (near 50% = unbiased). "
        "**conviction_std** — how much the model's opinion varied across dates (higher = more dynamic). "
        "**avg_fundamental / avg_technical** — average raw signal scores [-1, +1] per source. "
        "**hit_rate_Xd** — % of directional calls correct at that horizon. **—** = no directional trades for that ticker."
    )


# ══════════════════════════════════════════════════════════════════════════
# LOGGED CONVICTION BACKTEST (existing, from observability store)
# ══════════════════════════════════════════════════════════════════════════

def _render_logged_backtest():
    render_section_header("Live Conviction Backtest (From Pipeline Logs)")
    subtitle(
        'Compares conviction scores logged during actual pipeline runs against forward returns. '
        '<span style="color:#94a3b8;">Requires prior pipeline executions with conviction logging enabled.</span>'
    )

    try:
        from invictus.evaluation.backtest_tracker import run_backtest

        if st.button("Run Live Backtest", key="run_backtest_btn", type="secondary"):
            with st.spinner("Fetching forward returns from yfinance..."):
                bt = run_backtest(horizons=[5, 10, 30])
                st.session_state["_backtest_results"] = bt

        bt = st.session_state.get("_backtest_results")
        if bt:
            if bt.get("status") == "ok":
                _render_logged_results(bt)
            elif bt.get("status") in ("insufficient_history", "no_data"):
                st.info(bt["message"])
            else:
                st.info(bt.get("message", "No conviction data available."))
        else:
            st.info("Click **Run Live Backtest** to evaluate logged conviction accuracy.")
    except Exception as e:
        st.warning(f"Live backtest unavailable: {e}")


def _render_logged_results(bt: dict):
    """Render logged backtest results (existing functionality)."""
    b1, b2, b3 = st.columns(3)
    with b1: conviction_card("Convictions Evaluated", str(bt["evaluated"]))
    with b2: conviction_card("Total Logged", str(bt["total_convictions"]))
    with b3: conviction_card("Skipped", str(bt["skipped_too_recent"]),
                              sub_label="TOO RECENT")

    for h, hdata in bt.get("horizons", {}).items():
        if hdata.get("status") == "no_data":
            continue

        render_section_header(f"{h}-Day Forward Returns (Logged)")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            hr = hdata["overall_hit_rate"]
            conviction_card("Hit Rate", f"{hr:.0%}",
                            color=SUCCESS_GREEN if hr > 0.55 else DANGER_RED,
                            sub_label="EDGE" if hr > 0.55 else "NO EDGE")
        with h2: conviction_card("Bullish Hits", f"{hdata['bullish_hit_rate']:.0%}")
        with h3: conviction_card("Bearish Hits", f"{hdata['bearish_hit_rate']:.0%}")
        with h4:
            rho = hdata["spearman_rho"]
            conviction_card("Spearman ρ", f"{rho:+.3f}",
                            color=SUCCESS_GREEN if rho > 0.3 else DANGER_RED if rho < -0.1 else SLATE_500,
                            sub_label="STRONG" if rho > 0.3 else "WEAK")

        # Calibration curve
        cal = hdata.get("calibration", [])
        if cal:
            with st.container(border=True):
                pred = [c["predicted_prob"] for c in cal]
                actual = [c["actual_hit_rate"] for c in cal]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pred, y=actual, mode="markers+lines",
                    name="Model", marker=dict(size=10, color=BRAND_BLUE),
                    line=dict(color=BRAND_BLUE, width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    name="Perfect", line=dict(color=SLATE_500, dash="dash", width=1),
                ))
                apply_invictus_layout(fig, height=280, title=f"Calibration — {h}d")
                fig.update_layout(xaxis_title="Predicted Probability", yaxis_title="Actual Hit Rate")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    per_ticker = bt.get("per_ticker", [])
    if per_ticker:
        render_section_header("Per-Ticker Results (Logged)")
        pt_df = pd.DataFrame(per_ticker)
        display_cols = ["ticker", "conviction_count", "avg_conviction"]
        fmt = {"avg_conviction": "{:.3f}"}
        for hz in [5, 10, 30]:
            col_hr = f"hit_rate_{hz}d"
            col_fwd = f"avg_fwd_{hz}d"
            if col_hr in pt_df.columns:
                display_cols.extend([col_hr, col_fwd])
                fmt[col_hr] = "{:.0%}"
                fmt[col_fwd] = "{:+.2%}"
        available = [c for c in display_cols if c in pt_df.columns]
        st.dataframe(
            pt_df[available].style.format(
                {k: v for k, v in fmt.items() if k in available}, na_rep="—"
            ),
            use_container_width=True, hide_index=True,
        )

    st.caption(bt.get("disclaimer", ""))
