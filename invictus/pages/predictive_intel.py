"""
invictus.pages.predictive_intel
===============================
Conviction Intelligence — 6 sub-tabs:
  Summary, Conviction Engine, Filing Intelligence,
  Management Outlook, Transcript Analysis, Capital Flows.

Institutional research workflow. Conviction-building process.
"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from invictus.design import (
    render_section_header, render_metric_card, render_commentary_box,
    apply_invictus_layout, fmt_currency,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT, SLATE_500, SLATE_100,
)


def _intelligence_callout(text: str) -> None:
    st.markdown(
        f'<div style="border-left:3px solid {BRAND_BLUE};background:rgba(29,78,216,0.04);'
        f'padding:12px 16px;border-radius:0 6px 6px 0;margin:12px 0;'
        f'font-size:13px;color:#1e293b;line-height:1.7;font-style:italic;">{text}</div>',
        unsafe_allow_html=True,
    )


def _conviction_color(level: str) -> str:
    return {
        "STRONG CONVICTION": SUCCESS_GREEN, "HIGH": "#10b981",
        "MODERATE POSITIVE": BRAND_BLUE, "NEUTRAL": "#94a3b8",
        "MODERATE NEGATIVE": "#f59e0b", "LOW / RISK": DANGER_RED,
    }.get(level, BRAND_BLUE)


def _render_football_field(ticker, details, prob):
    """Conviction synthesis football-field chart."""
    signals = [
        {"Layer": "Fundamental Intelligence", "Score": float(details.get("fundamentals", {}).get("score", 0))},
        {"Layer": "Guidance Momentum",        "Score": float(details.get("guidance", {}).get("score", 0))},
        {"Layer": "Risk Environment",         "Score": -float(details.get("risk_env", {}).get("score", 0))},
        {"Layer": "Management Confidence",    "Score": float(details.get("management", {}).get("score", 0))},
        {"Layer": "Analyst Pressure",         "Score": -float(details.get("analyst", {}).get("score", 0))},
        {"Layer": "Capital Flow Intelligence", "Score": float(details.get("flows", {}).get("score", 0))},
        {"Layer": "ML / Technical",           "Score": float(details.get("technical", {}).get("score", 0))},
    ]
    df = pd.DataFrame(signals)
    fig = go.Figure(go.Bar(
        x=df["Score"], y=df["Layer"], orientation="h",
        marker=dict(color=[DANGER_RED if v < 0 else SUCCESS_GREEN for v in df["Score"]]),
        text=[f"{v:+.2f}" for v in df["Score"]], textposition="outside",
    ))
    target = (prob - 0.5) * 2
    fig.add_vline(x=target, line_width=3, line_dash="dash", line_color=BRAND_BLUE,
                  annotation_text=f"Composite: {prob:.0%}")
    apply_invictus_layout(fig, height=320)
    fig.update_layout(xaxis=dict(range=[-1.1, 1.1], tickformat=".1f", title="Signal Strength"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render(sub):
    """Render the Conviction Intelligence page for the given sub-tab."""

    # ── Stock Selection (always visible at top) ───────────────────
    portfolio_tickers = []
    if st.session_state.portfolio_loaded:
        portfolio_tickers = st.session_state.portfolio_state["holdings"]["Ticker"].tolist()

    pick_col, enter_col, btn_col = st.columns([1.2, 1.2, 0.6])
    with pick_col:
        picked = st.multiselect(
            "From portfolio", options=portfolio_tickers, default=[], max_selections=3,
            key="pi_portfolio_pick", placeholder="Select from portfolio...",
            disabled=len(portfolio_tickers) == 0,
        )
    with enter_col:
        new_input = st.text_input(
            "Enter tickers", placeholder="NVDA, AMZN, TSLA",
            key="pi_new_ticker_input", help="Comma-separated",
        )
    with btn_col:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        new_raw = [t.strip().upper() for t in new_input.replace(",", " ").split() if t.strip()] if new_input else []
        all_selected = list(dict.fromkeys(picked + new_raw))[:3]
        run_btn = st.button("Run Intel", key="pi_run_btn", type="primary",
                            use_container_width=True, disabled=len(all_selected) == 0)

    # ── Run Pipeline ──────────────────────────────────────────────
    if run_btn and all_selected:
        from invictus.agents.filing_agent import _extract_yfinance_fundamental_signals
        from invictus.agents.earnings_agent import _fetch_yfinance_sentiment_context, _analyze_sentiment_with_llm, _dictionary_sentiment
        from invictus.agents.flow_agent import _fetch_flow_data, _score_flows
        from invictus.agents.outlook_agent import analyze_management_outlook
        from invictus.agents.synthesis_agent import _calculate_stock_conviction

        pi_filing, pi_earnings, pi_flows, pi_outlook, pi_synthesis = {}, {}, {}, {}, {}

        # Reuse existing
        for t in all_selected:
            if st.session_state.filing_intel and t in st.session_state.filing_intel:
                pi_filing[t] = st.session_state.filing_intel[t]
            if st.session_state.earnings_intel and t in st.session_state.earnings_intel:
                pi_earnings[t] = st.session_state.earnings_intel[t]
            if st.session_state.flow_signals and t in st.session_state.flow_signals.get("intel", {}):
                pi_flows[t] = st.session_state.flow_signals["intel"][t]

        # Run fresh where needed
        tickers_to_run = [t for t in all_selected if t not in pi_filing]
        if tickers_to_run:
            progress = st.progress(0, text="Running Conviction Intelligence...")
            total = len(tickers_to_run) * 4
            step = 0
            for t in tickers_to_run:
                progress.progress((step + 1) / total, text=f"Filing Intelligence: {t}")
                pi_filing[t] = _extract_yfinance_fundamental_signals(t)
                step += 1
                progress.progress((step + 1) / total, text=f"Management Intelligence: {t}")
                ctx = _fetch_yfinance_sentiment_context(t)
                result = _analyze_sentiment_with_llm(t, ctx)
                pi_earnings[t] = result if (result and result.get("status") == "Success") else _dictionary_sentiment(ctx)
                step += 1
                progress.progress((step + 1) / total, text=f"Capital Flows: {t}")
                raw_flow = _fetch_flow_data(t)
                # Debug: surface fetch failures visibly
                _n_inst = len(raw_flow.get("institutional", []))
                _n_ins = len(raw_flow.get("insiders", []))
                _stat = raw_flow.get("status", "?")
                if _stat != "Success":
                    import logging as _lg
                    _lg.warning("Flow fetch %s: status=%s, inst=%d, ins=%d", t, _stat, _n_inst, _n_ins)
                try:
                    pi_flows[t] = _score_flows(t, raw_flow)
                except Exception as _flow_err:
                    import logging as _lg
                    _lg.error("Flow scoring CRASHED for %s: %s", t, _flow_err, exc_info=True)
                    pi_flows[t] = {"status": "Error", "flow_composite": 0,
                        "insider_intelligence": {"score": 0, "red_flags": [], "green_flags": [], "summary": str(_flow_err)},
                        "fund_accumulation": {"score": 0, "red_flags": [], "green_flags": [], "summary": str(_flow_err)},
                        "concentration": {"score": 0, "red_flags": [], "green_flags": [], "summary": str(_flow_err)},
                        "smart_money_pct": 0, "institutional_conviction": 0, "insider_alignment": 0,
                        "capital_participation": 0, "insider_buys": 0, "insider_sells": 0,
                        "estimated_accumulation": "neutral", "net_insider_value": 0, "notable_transactions": []}
                step += 1
                progress.progress((step + 1) / total, text=f"Management Outlook: {t}")
                try:
                    pi_outlook[t] = analyze_management_outlook(t)
                except Exception as _outlook_err:
                    import logging as _lg
                    _lg.error("Outlook analysis CRASHED for %s: %s", t, _outlook_err, exc_info=True)
                    pi_outlook[t] = {"status": "Error", "management_signal": 0,
                        "outlook_score_raw": 0, "credibility_multiplier": 0.75,
                        "outlook": {"dimensions": {}, "outlook_score": 0, "status": "Error"},
                        "credibility": {"sub_dimensions": {}, "raw_credibility": 0.5,
                            "credibility_multiplier": 0.75, "status": "Error"},
                        "data_sources": "Error", "ticker": t}
                step += 1
            progress.empty()

        # Synthesis
        vol_regime = st.session_state.vol_regime_state.get("current_regime") if st.session_state.vol_regime_state else None
        for t in all_selected:
            f = pi_filing.get(t, {"status": "N/A"})
            e = pi_earnings.get(t, {"status": "N/A"})
            fl = pi_flows.get(t, {"status": "N/A"})
            ml_pred = None
            if st.session_state.ml_state and "prediction_table" in (st.session_state.ml_state or {}):
                pt = st.session_state.ml_state["prediction_table"]
                if isinstance(pt, pd.DataFrame) and t in pt["Ticker"].values:
                    row = pt[pt["Ticker"] == t].iloc[0]
                    ml_pred = {"accumulation_prob": row.get("Accumulation Prob", 0.5)}
            pi_synthesis[t] = _calculate_stock_conviction(t, f, e, fl, ml_pred, vol_regime, "1 year")

        st.session_state.pi_results = {
            "filing": pi_filing, "earnings": pi_earnings,
            "flows": pi_flows, "outlook": pi_outlook,
            "synthesis": pi_synthesis, "tickers": all_selected,
        }
        st.session_state.pi_selected_tickers = all_selected
        st.rerun()

    # ── Gate: need results ────────────────────────────────────────
    pi = st.session_state.pi_results
    if not pi or not pi.get("tickers"):
        st.info("Select up to 3 securities and click Run Intel to begin conviction analysis.")
        return

    pi_tickers = pi["tickers"]
    pi_synth = pi["synthesis"]
    pi_filing = pi["filing"]
    pi_earn = pi["earnings"]
    pi_flow = pi["flows"]
    pi_outlook = pi.get("outlook", {})

    # ══════════════════════════════════════════════════════════════
    # 1. SUMMARY — Conviction Overview
    # ══════════════════════════════════════════════════════════════
    if sub == "Summary":

        # ── Conviction Scorecard Table ────────────────────────────
        render_section_header("Conviction Scorecard")
        score_rows = []
        for t in pi_tickers:
            res = pi_synth.get(t, {})
            score_rows.append({
                "Ticker": t,
                "Conviction": res.get("conviction_level", "N/A"),
                "Composite Score": res.get("composite_score", 0),
                "Outperformance Prob": res.get("outperformance_probability", 0),
                "Signal Confidence": res.get("signal_confidence", 0),
                "Dominant Driver": res.get("dominant_driver", "N/A").replace("_", " ").title(),
            })
        score_df = pd.DataFrame(score_rows)
        st.dataframe(
            score_df.style.format({
                "Composite Score": "{:+.3f}",
                "Outperformance Prob": "{:.0%}",
                "Signal Confidence": "{:.0%}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

        # ── Per-Security Deep Dive ────────────────────────────────
        for t in pi_tickers:
            res = pi_synth.get(t, {})
            prob = res.get("outperformance_probability", 0)
            level = res.get("conviction_level", "N/A")
            lc = _conviction_color(level)

            render_section_header(f"{t} — Conviction Analysis")

            # Metrics row
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">CONVICTION LEVEL</div>'
                    f'<div class="metric-value" style="color:{lc};font-size:16px;">{level}</div></div>',
                    unsafe_allow_html=True,
                )
            with k2: render_metric_card("Outperformance", f"{prob:.0%}")
            with k3: render_metric_card("Composite", f"{res.get('composite_score', 0):+.3f}", delta_val=res.get('composite_score', 0))
            with k4: render_metric_card("Confidence", f"{res.get('signal_confidence', 0):.0%}")

            # Football field + Monte Carlo side by side
            ff_col, ci_col = st.columns([1.2, 1])
            with ff_col:
                _render_football_field(t, res.get("signals_detail", {}), prob)
            with ci_col:
                ci = res.get("confidence_intervals", {})
                if ci:
                    ci_labels = ["5th", "25th", "Median", "75th", "95th"]
                    ci_vals = [ci.get("ci_5", 0), ci.get("ci_25", 0), ci.get("ci_50", 0), ci.get("ci_75", 0), ci.get("ci_95", 0)]
                    fig_ci = go.Figure(go.Bar(
                        x=ci_labels, y=ci_vals,
                        marker_color=[SLATE_500, BRAND_BLUE, SUCCESS_GREEN, BRAND_BLUE, SLATE_500],
                        text=[f"{v:.0%}" for v in ci_vals], textposition="outside",
                    ))
                    fig_ci.add_hline(y=0.5, line_dash="dot", line_color=SLATE_500, annotation_text="Neutral")
                    apply_invictus_layout(fig_ci, height=320, title="Monte Carlo Confidence Intervals")
                    fig_ci.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"))
                    st.plotly_chart(fig_ci, use_container_width=True, config={"displayModeBar": False})

            # Key Drivers & Risks
            drivers = res.get("drivers", [])
            risks = res.get("risks", [])
            if drivers or risks:
                dr_col, rk_col = st.columns(2)
                with dr_col:
                    if drivers:
                        st.markdown(
                            f'<div style="font-size:11px;font-weight:800;color:{SUCCESS_GREEN};'
                            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;">Key Drivers</div>',
                            unsafe_allow_html=True,
                        )
                        for d in drivers[:4]:
                            st.markdown(
                                f'<div style="border-left:2px solid {SUCCESS_GREEN};padding:4px 10px;'
                                f'margin-bottom:4px;font-size:12px;color:#334155;">{d}</div>',
                                unsafe_allow_html=True,
                            )
                with rk_col:
                    if risks:
                        st.markdown(
                            f'<div style="font-size:11px;font-weight:800;color:{DANGER_RED};'
                            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;">Key Risks</div>',
                            unsafe_allow_html=True,
                        )
                        for r in risks[:4]:
                            st.markdown(
                                f'<div style="border-left:2px solid {DANGER_RED};padding:4px 10px;'
                                f'margin-bottom:4px;font-size:12px;color:#334155;">{r}</div>',
                                unsafe_allow_html=True,
                            )

            # Signal Agreement
            agree = res.get("signal_agreement", {})
            if agree and agree.get("agreement_label") != "INSUFFICIENT DATA":
                a_label = agree.get("agreement_label", "N/A")
                a_colors = {
                    "STRONG CONVERGENCE": SUCCESS_GREEN, "MODERATE AGREEMENT": BRAND_BLUE,
                    "MIXED SIGNALS": "#f59e0b", "SIGNAL DIVERGENCE": DANGER_RED,
                }
                ac = a_colors.get(a_label, BRAND_BLUE)
                st.markdown(
                    f'<div style="display:inline-block;background:{ac}15;border:1px solid {ac};'
                    f'padding:4px 12px;border-radius:4px;font-size:11px;font-weight:700;color:{ac};'
                    f'letter-spacing:0.06em;margin-top:6px;">'
                    f'Signal Agreement: {a_label} — {agree.get("bullish_count", 0)} bullish / '
                    f'{agree.get("bearish_count", 0)} bearish</div>',
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════════
    # 2. CONVICTION ENGINE — ML Accumulation Classifier
    # ══════════════════════════════════════════════════════════════
    elif sub == "ML Model":
        render_section_header("Conviction Engine — Bayesian Accumulation Model")

        ml = st.session_state.ml_state
        if not ml:
            st.info("Load your portfolio first — the conviction engine requires portfolio-level price history for training.")
            return

        pred_table = ml.get("prediction_table")
        if pred_table is None or not isinstance(pred_table, pd.DataFrame) or pred_table.empty:
            st.info("ML predictions not available.")
            return

        # Filter to selected tickers
        filtered = pred_table[pred_table["Ticker"].isin(pi_tickers)]

        if filtered.empty:
            st.info("Selected tickers are not in the portfolio — ML requires portfolio-level training data.")
            return

        # ── Conviction Scores Table ───────────────────────────────
        render_section_header("Accumulation Probability")
        display_cols = ["Ticker", "Accumulation Prob", "Signal Strength"]
        extra_cols = ["LR Prob", "RF Prob", "RSI_14", "MACD_Hist", "Momentum_20d"]
        for ec in extra_cols:
            if ec in filtered.columns:
                display_cols.append(ec)
        display_df = filtered[[c for c in display_cols if c in filtered.columns]].copy()
        fmt_dict = {
            "Accumulation Prob": "{:.1%}", "LR Prob": "{:.1%}", "RF Prob": "{:.1%}",
            "RSI_14": "{:.1f}", "MACD_Hist": "{:.4f}", "Momentum_20d": "{:+.2%}",
        }
        st.dataframe(
            display_df.style.format({k: v for k, v in fmt_dict.items() if k in display_df.columns}, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

        # ── Accumulation Probability Bar Chart ────────────────────
        with st.container(border=True):
            fig_acc = go.Figure(go.Bar(
                x=filtered["Ticker"],
                y=filtered["Accumulation Prob"],
                marker_color=[SUCCESS_GREEN if v > 0.6 else BRAND_BLUE if v > 0.4 else DANGER_RED
                              for v in filtered["Accumulation Prob"]],
                text=[f"{v:.0%}" for v in filtered["Accumulation Prob"]], textposition="outside",
            ))
            fig_acc.add_hline(y=0.5, line_dash="dot", line_color=SLATE_500, annotation_text="Neutral (50%)")
            apply_invictus_layout(fig_acc, height=300, title="Accumulation Probability by Ticker")
            fig_acc.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"))
            st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})

        # ── Feature Group Decomposition ───────────────────────────
        sig_decomp = ml.get("signal_decomposition", {})
        for ticker in pi_tickers:
            groups = sig_decomp.get(ticker, {})
            if isinstance(groups, dict) and groups:
                with st.expander(f"{ticker} — Feature Group Breakdown"):
                    group_df = pd.DataFrame([
                        {"Group": g, "Score": float(s)} for g, s in groups.items()
                    ]).sort_values("Score", ascending=True)
                    fig_g = go.Figure(go.Bar(
                        x=group_df["Score"], y=group_df["Group"], orientation="h",
                        marker_color=[DANGER_RED if v < 0 else SUCCESS_GREEN for v in group_df["Score"]],
                        text=[f"{v:+.3f}" for v in group_df["Score"]], textposition="outside",
                    ))
                    apply_invictus_layout(fig_g, height=280, title="Signal Contribution by Feature Group")
                    st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

        # Model metadata
        _mc = ml.get('model_confidence', 0)
        _mc_str = f"{_mc:.1%}" if isinstance(_mc, (int, float)) else str(_mc)
        st.caption(
            f"Model: {ml.get('model_type', 'Ensemble')} | "
            f"Features: {ml.get('n_features', 'N/A')} | "
            f"Samples: {ml.get('n_samples', 'N/A')} | "
            f"CV Score: {_mc_str}"
        )

    # ══════════════════════════════════════════════════════════════
    # 3. FILING INTELLIGENCE — Fundamentals & Guidance
    # ══════════════════════════════════════════════════════════════
    elif sub == "10-K Filing":
        render_section_header("Filing Intelligence — Fundamental Analysis")

        for t in pi_tickers:
            d = pi_filing.get(t, {})
            status = d.get("status", "")
            status_ok = "Success" in str(status)

            with st.expander(f"{t}", expanded=status_ok):
                if not status_ok:
                    st.caption("Filing data not available for this security.")
                    continue

                # ── Signal Summary Cards ──────────────────────────
                s1, s2, s3 = st.columns(3)
                fc = d.get("fundamental_conviction", 0)
                gm = d.get("guidance_momentum", 0)
                rd = d.get("risk_deterioration", 0)

                with s1:
                    fc_color = SUCCESS_GREEN if fc > 0.3 else DANGER_RED if fc < -0.3 else BRAND_BLUE
                    fc_label = "Strong Positive" if fc > 0.5 else "Positive" if fc > 0.1 else "Neutral" if fc > -0.1 else "Negative"
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">FUNDAMENTAL CONVICTION</div>'
                        f'<div class="metric-value" style="color:{fc_color};font-size:16px;">{fc_label}</div>'
                        f'<div style="font-size:12px;color:#64748b;margin-top:2px;">Score: {fc:+.2f}</div></div>',
                        unsafe_allow_html=True,
                    )
                with s2:
                    gm_color = SUCCESS_GREEN if gm > 0.2 else DANGER_RED if gm < -0.2 else BRAND_BLUE
                    gm_label = "Improving" if gm > 0.2 else "Stable" if gm > -0.2 else "Deteriorating"
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">GUIDANCE MOMENTUM</div>'
                        f'<div class="metric-value" style="color:{gm_color};font-size:16px;">{gm_label}</div>'
                        f'<div style="font-size:12px;color:#64748b;margin-top:2px;">Score: {gm:+.2f}</div></div>',
                        unsafe_allow_html=True,
                    )
                with s3:
                    rd_color = SUCCESS_GREEN if rd < 0.3 else DANGER_RED if rd > 0.6 else BRAND_BLUE
                    rd_label = "Low" if rd < 0.3 else "Moderate" if rd < 0.6 else "Elevated"
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">STRUCTURAL RISK</div>'
                        f'<div class="metric-value" style="color:{rd_color};font-size:16px;">{rd_label}</div>'
                        f'<div style="font-size:12px;color:#64748b;margin-top:2px;">Score: {rd:.2f}</div></div>',
                        unsafe_allow_html=True,
                    )

                # ── Quantitative Metrics ──────────────────────────
                raw = d.get("raw_metrics", {})
                if raw:
                    render_section_header("Financial Metrics")
                    rm1, rm2, rm3, rm4 = st.columns(4)
                    with rm1: render_metric_card("Revenue Growth", f"{raw.get('revenue_growth', 0):+.1%}", delta_val=raw.get('revenue_growth', 0))
                    with rm2: render_metric_card("Net Income Growth", f"{raw.get('net_income_growth', 0):+.1%}", delta_val=raw.get('net_income_growth', 0))
                    with rm3: render_metric_card("Gross Margin", f"{raw.get('gross_margin', 0):.1%}")
                    with rm4: render_metric_card("Operating Margin", f"{raw.get('operating_margin', 0):.1%}")

                # ── Filing Evidence ───────────────────────────────
                reasoning = d.get("fundamental_reasoning", "")
                if reasoning:
                    render_commentary_box(reasoning)

                # ── Drivers & Risk Flags ──────────────────────────
                sup = d.get("supporting_drivers", [])
                rsk = d.get("risk_drivers", [])
                if sup or rsk:
                    dr1, dr2 = st.columns(2)
                    with dr1:
                        if sup:
                            st.markdown(f'<div style="font-size:11px;font-weight:800;color:{SUCCESS_GREEN};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">Growth Signals</div>', unsafe_allow_html=True)
                            for item in sup:
                                st.markdown(f'<div style="border-left:2px solid {SUCCESS_GREEN};padding:3px 10px;margin-bottom:3px;font-size:12px;color:#334155;">{item}</div>', unsafe_allow_html=True)
                    with dr2:
                        if rsk:
                            st.markdown(f'<div style="font-size:11px;font-weight:800;color:{DANGER_RED};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">Risk Changes</div>', unsafe_allow_html=True)
                            for item in rsk:
                                st.markdown(f'<div style="border-left:2px solid {DANGER_RED};padding:3px 10px;margin-bottom:3px;font-size:12px;color:#334155;">{item}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # 4. MANAGEMENT INTELLIGENCE — Outlook + Transcript Analysis
    # ══════════════════════════════════════════════════════════════
    elif sub == "Earnings":
        render_section_header("Management Intelligence")

        # Scale explainer — mirrors flows pattern
        st.markdown(
            f'<div style="font-size:12px;color:#64748b;margin:-4px 0 12px 0;line-height:1.5;">'
            f'Management Signal combines <b>Outlook</b> (what management says, '
            f'<span style="color:{DANGER_RED};font-weight:700;">−1</span> to '
            f'<span style="color:{SUCCESS_GREEN};font-weight:700;">+1</span>) '
            f'× <b>Credibility</b> (how credibly they say it, '
            f'<span style="color:#94a3b8;font-weight:700;">0.5</span> to '
            f'<span style="color:{SUCCESS_GREEN};font-weight:700;">1.0</span>). '
            f'<span style="color:#94a3b8;">Credibility gates outlook — vague/hedgy management '
            f'claims carry less weight.</span></div>',
            unsafe_allow_html=True,
        )

        # ── Overview Cards — per-ticker summary (mirrors flows pattern) ──
        def _outlook_tone_label(score):
            if score > 0.4: return "Strong Bullish", SUCCESS_GREEN
            if score > 0.15: return "Mildly Bullish", BRAND_BLUE
            if score < -0.4: return "Strong Bearish", DANGER_RED
            if score < -0.15: return "Mildly Bearish", "#f59e0b"
            return "Neutral", SLATE_500

        def _cred_label(cm):
            if cm >= 0.9: return "High Credibility", SUCCESS_GREEN
            if cm >= 0.75: return "Moderate", BRAND_BLUE
            return "Low Credibility", "#f59e0b"

        def _signal_explain(ms):
            if ms > 0.3: return "Management painting a convincingly bullish picture"
            if ms > 0.1: return "Mildly positive management outlook"
            if ms < -0.3: return "Bearish signals — caution warranted"
            if ms < -0.1: return "Some negative management signals"
            return "No clear directional management signal"

        def _sc(v):
            return SUCCESS_GREEN if v > 0.1 else DANGER_RED if v < -0.1 else "#94a3b8"

        overview_cols = st.columns(len(pi_tickers))
        for idx, t in enumerate(pi_tickers):
            d = pi_outlook.get(t, {})
            ms = d.get("management_signal", 0)
            os_raw = d.get("outlook_score_raw", 0)
            cm = d.get("credibility_multiplier", 0.75)
            tone_lbl, tone_clr = _outlook_tone_label(os_raw)
            cred_lbl, cred_clr = _cred_label(cm)
            sources = d.get("data_sources", "N/A")

            with overview_cols[idx]:
                st.markdown(
                    f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                    f'background:#fafbfc;">'
                    # Ticker + management signal
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">'
                    f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{t}</span>'
                    f'<span style="font-size:18px;font-weight:800;color:{tone_clr};">{ms:+.2f}</span></div>'
                    # Tone label
                    f'<div style="font-size:11px;font-weight:700;color:{tone_clr};'
                    f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">{tone_lbl}</div>'
                    # Explanation
                    f'<div style="font-size:11px;color:#64748b;margin-bottom:10px;line-height:1.4;'
                    f'font-style:italic;">{_signal_explain(ms)}</div>'
                    # Sub-scores
                    f'<div style="padding:5px 0;border-top:1px solid #e2e8f0;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:12px;">'
                    f'<span style="color:#475569;font-weight:600;">Outlook Score</span>'
                    f'<span style="color:{_sc(os_raw)};font-weight:700;">{os_raw:+.2f}</span></div>'
                    f'<div style="font-size:10px;color:#94a3b8;margin-top:1px;">What management says</div></div>'
                    f'<div style="padding:5px 0;border-top:1px solid #f1f5f9;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:12px;">'
                    f'<span style="color:#475569;font-weight:600;">Credibility</span>'
                    f'<span style="color:{cred_clr};font-weight:700;">{cm:.2f}</span></div>'
                    f'<div style="font-size:10px;color:#94a3b8;margin-top:1px;">How credibly they say it</div></div>'
                    # Formula
                    f'<div style="padding:6px 0 2px 0;border-top:1px solid #e2e8f0;">'
                    f'<div style="font-size:10px;color:#94a3b8;font-family:monospace;">'
                    f'{os_raw:+.2f} × {cm:.2f} = {ms:+.2f}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)

        # ── Per-Ticker Deep Dive ─────────────────────────────────
        for t in pi_tickers:
            d = pi_outlook.get(t, {})
            if d.get("status") not in ("Success", "Partial"):
                st.caption(f"{t} — Management intelligence not available")
                continue

            ms = d.get("management_signal", 0)
            _ms_color = SUCCESS_GREEN if ms > 0.1 else DANGER_RED if ms < -0.1 else SLATE_500

            # Per-ticker header — matches flows style
            st.markdown(
                f'<div style="font-size:13px;font-weight:800;color:{BRAND_BLUE};'
                f'background:#f8fafc;border-left:5px solid {BRAND_BLUE};'
                f'padding:10px 20px;margin:16px 0 8px 0;border-radius:0 8px 8px 0;'
                f'text-transform:uppercase;letter-spacing:1px;'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'{t} — Management Intelligence'
                f'<span style="color:{_ms_color};font-size:15px;">{ms:+.2f}</span></div>',
                unsafe_allow_html=True,
            )

            with st.expander("Details", expanded=True):
                outlook_data = d.get("outlook", {})
                cred_data = d.get("credibility", {})
                dims = outlook_data.get("dimensions", {})

                # ════════════════════════════════════════════
                # LAYER 2: MANAGEMENT OUTLOOK — 6 dimensions
                # ════════════════════════════════════════════
                os_raw = outlook_data.get("outlook_score", 0)
                _os_sc = SUCCESS_GREEN if os_raw > 0.1 else DANGER_RED if os_raw < -0.1 else SLATE_500
                st.markdown(
                    f'<div style="font-size:11px;font-weight:800;color:{BRAND_BLUE};letter-spacing:0.1em;'
                    f'text-transform:uppercase;margin:10px 0 8px 0;border-bottom:2px solid {BRAND_BLUE};'
                    f'padding-bottom:4px;">Management Outlook — 6 Dimensions '
                    f'<span style="color:{_os_sc};float:right;font-size:13px;">{os_raw:+.2f}</span></div>',
                    unsafe_allow_html=True,
                )

                # Dimension labels for display
                dim_display_names = {
                    "demand_environment": "Demand Environment",
                    "competitive_positioning": "Competitive Position",
                    "strategic_confidence": "Strategic Confidence",
                    "macro_industry_outlook": "Macro / Industry",
                    "headwinds_tailwinds": "Headwinds & Tailwinds",
                    "investment_thesis_clarity": "Thesis Clarity",
                }

                if dims:
                    # Horizontal bar chart — 6 dimensions
                    dim_names = []
                    dim_scores = []
                    dim_weights = []
                    dim_contribs = []
                    for key in ["demand_environment", "competitive_positioning",
                                "strategic_confidence", "macro_industry_outlook",
                                "headwinds_tailwinds", "investment_thesis_clarity"]:
                        dd = dims.get(key, {})
                        dim_names.append(dim_display_names.get(key, key))
                        dim_scores.append(dd.get("score", 0))
                        dim_weights.append(dd.get("weight", 0))
                        dim_contribs.append(dd.get("weighted_contribution", 0))

                    fig = go.Figure(go.Bar(
                        x=dim_scores, y=dim_names, orientation="h",
                        marker=dict(color=[DANGER_RED if v < -0.1 else SUCCESS_GREEN if v > 0.1 else "#94a3b8" for v in dim_scores]),
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

                    # Dimension detail table
                    _thl = 'style="text-align:left;padding:4px 6px;color:#475569;font-weight:700;font-size:11px;"'
                    _thc = 'style="text-align:center;padding:4px 6px;color:#475569;font-weight:700;font-size:11px;"'
                    dim_table = (
                        '<table style="width:100%;border-collapse:collapse;font-size:12px;margin:4px 0 0 0;'
                        'table-layout:fixed;">'
                        '<colgroup>'
                        '<col style="width:24%;">'   # Dimension
                        '<col style="width:12%;">'   # Weight
                        '<col style="width:12%;">'   # Score
                        '<col style="width:14%;">'   # Contribution
                        '<col style="width:10%;">'   # Signal
                        '<col style="width:28%;">'   # Evidence
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
                    for key in ["demand_environment", "competitive_positioning",
                                "strategic_confidence", "macro_industry_outlook",
                                "headwinds_tailwinds", "investment_thesis_clarity"]:
                        dd = dims.get(key, {})
                        score = dd.get("score", 0)
                        weight = dd.get("weight", 0)
                        contrib = dd.get("weighted_contribution", 0)
                        signal = dd.get("signal", "Neutral")
                        evidence = dd.get("evidence", [])
                        ev_str = ", ".join(str(e)[:30] for e in evidence[:3]) if evidence else "—"
                        sc = SUCCESS_GREEN if score > 0.1 else DANGER_RED if score < -0.1 else "#94a3b8"
                        sig_color = SUCCESS_GREEN if "Positive" in signal or "Bullish" in signal else DANGER_RED if "Negative" in signal or "Bearish" in signal else "#94a3b8"
                        _tdc = 'text-align:center;'
                        dim_table += (
                            f'<tr style="border-bottom:1px solid #f1f5f9;">'
                            f'<td style="padding:5px 6px;color:#1e293b;font-weight:500;">{dim_display_names.get(key, key)}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-variant-numeric:tabular-nums;">{weight:.0%}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:{sc};font-weight:700;font-variant-numeric:tabular-nums;">{score:+.2f}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:{sc};font-variant-numeric:tabular-nums;">{contrib:+.4f}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:{sig_color};font-weight:600;font-size:11px;">{signal}</td>'
                            f'<td style="padding:5px 6px;color:#64748b;font-size:11px;">{ev_str}</td>'
                            f'</tr>'
                        )
                    dim_table += '</tbody></table>'
                    st.markdown(dim_table, unsafe_allow_html=True)

                # Source + formula
                src = outlook_data.get("source", "N/A")
                st.markdown(
                    f'<div style="font-size:10px;color:#94a3b8;margin-top:6px;font-family:monospace;">'
                    f'Source: {src} | outlook_score = Σ(weight × dim_score) = {os_raw:+.4f}</div>',
                    unsafe_allow_html=True,
                )

                # ════════════════════════════════════════════
                # LAYER 3: TRANSCRIPT CREDIBILITY — 4 dimensions
                # ════════════════════════════════════════════
                raw_cred = cred_data.get("raw_credibility", 0.5)
                cred_mult = cred_data.get("credibility_multiplier", 0.75)
                cred_overall = cred_data.get("overall_credibility", "Moderate")
                _cred_color = SUCCESS_GREEN if cred_mult >= 0.85 else "#f59e0b" if cred_mult >= 0.7 else DANGER_RED

                st.markdown(
                    f'<div style="font-size:11px;font-weight:800;color:{BRAND_BLUE};letter-spacing:0.1em;'
                    f'text-transform:uppercase;margin:18px 0 8px 0;border-bottom:2px solid {BRAND_BLUE};'
                    f'padding-bottom:4px;">Transcript Credibility Analysis '
                    f'<span style="color:{_cred_color};float:right;font-size:13px;">{cred_mult:.2f}</span></div>',
                    unsafe_allow_html=True,
                )

                cred_display_names = {
                    "hedging_density": "Hedging Density",
                    "specificity": "Specificity",
                    "forward_backward_ratio": "Forward/Backward Ratio",
                    "dodge_detection": "Dodge Detection",
                }
                cred_explanations = {
                    "hedging_density": "Less hedging = higher credibility",
                    "specificity": "Specific numbers/dates vs vague platitudes",
                    "forward_backward_ratio": "More forward-looking = higher confidence",
                    "dodge_detection": "Direct answers vs evasion",
                }

                sub_dims = cred_data.get("sub_dimensions", {})
                if sub_dims:
                    # Credibility metrics row
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        render_metric_card(
                            "Raw Credibility", f"{raw_cred:.2f}",
                            delta_val=raw_cred - 0.5 if abs(raw_cred - 0.5) > 0.05 else 0,
                        )
                    with cc2:
                        render_metric_card(
                            "Multiplier", f"{cred_mult:.2f}",
                            delta_val=cred_mult - 0.75 if abs(cred_mult - 0.75) > 0.05 else 0,
                        )
                    with cc3:
                        render_metric_card(
                            "Assessment", cred_overall,
                            delta_val=cred_mult - 0.75 if abs(cred_mult - 0.75) > 0.05 else 0,
                        )

                    # Credibility dimension table
                    _thl = 'style="text-align:left;padding:4px 6px;color:#475569;font-weight:700;font-size:11px;"'
                    _thc = 'style="text-align:center;padding:4px 6px;color:#475569;font-weight:700;font-size:11px;"'
                    cred_table = (
                        '<table style="width:100%;border-collapse:collapse;font-size:12px;margin:4px 0 0 0;'
                        'table-layout:fixed;">'
                        '<colgroup>'
                        '<col style="width:22%;">'   # Dimension
                        '<col style="width:12%;">'   # Weight
                        '<col style="width:12%;">'   # Score
                        '<col style="width:14%;">'   # Contribution
                        '<col style="width:40%;">'   # Reasoning
                        '</colgroup>'
                        f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                        f'<th {_thl}>Dimension</th>'
                        f'<th {_thc}>Weight</th>'
                        f'<th {_thc}>Score</th>'
                        f'<th {_thc}>Contribution</th>'
                        f'<th {_thl}>Reasoning</th>'
                        f'</tr></thead><tbody>'
                    )
                    for key in ["hedging_density", "specificity", "forward_backward_ratio", "dodge_detection"]:
                        sd = sub_dims.get(key, {})
                        score = sd.get("score", 0)
                        weight = sd.get("weight", 0)
                        contrib = sd.get("weighted_contribution", 0)
                        reasoning = sd.get("reasoning", "—")
                        sc = SUCCESS_GREEN if score >= 0.7 else "#f59e0b" if score >= 0.4 else DANGER_RED
                        _tdc = 'text-align:center;'
                        cred_table += (
                            f'<tr style="border-bottom:1px solid #f1f5f9;">'
                            f'<td style="padding:5px 6px;color:#1e293b;font-weight:500;">{cred_display_names.get(key, key)}'
                            f'<div style="font-size:9px;color:#94a3b8;margin-top:1px;">{cred_explanations.get(key, "")}</div></td>'
                            f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-variant-numeric:tabular-nums;">{weight:.0%}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:{sc};font-weight:700;font-variant-numeric:tabular-nums;">{score:.2f}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:{sc};font-variant-numeric:tabular-nums;">{contrib:.4f}</td>'
                            f'<td style="padding:5px 6px;color:#64748b;font-size:11px;">{reasoning}</td>'
                            f'</tr>'
                        )
                    cred_table += '</tbody></table>'
                    st.markdown(cred_table, unsafe_allow_html=True)

                # Credibility flags
                red_flags = cred_data.get("red_flags", [])
                green_flags = cred_data.get("green_flags", [])
                if red_flags or green_flags:
                    fl1, fl2 = st.columns(2)
                    with fl1:
                        if green_flags:
                            st.markdown(
                                f'<div style="font-size:11px;font-weight:800;color:{SUCCESS_GREEN};'
                                f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">'
                                f'Credibility Strengths</div>', unsafe_allow_html=True,
                            )
                            for item in green_flags:
                                st.markdown(
                                    f'<div style="border-left:2px solid {SUCCESS_GREEN};padding:3px 10px;'
                                    f'margin-bottom:3px;font-size:12px;color:#334155;">{item}</div>',
                                    unsafe_allow_html=True,
                                )
                    with fl2:
                        if red_flags:
                            st.markdown(
                                f'<div style="font-size:11px;font-weight:800;color:{DANGER_RED};'
                                f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">'
                                f'Credibility Concerns</div>', unsafe_allow_html=True,
                            )
                            for item in red_flags:
                                st.markdown(
                                    f'<div style="border-left:2px solid {DANGER_RED};padding:3px 10px;'
                                    f'margin-bottom:3px;font-size:12px;color:#334155;">{item}</div>',
                                    unsafe_allow_html=True,
                                )

                # Formula + source
                cred_src = cred_data.get("source", "N/A")
                st.markdown(
                    f'<div style="font-size:10px;color:#94a3b8;margin-top:6px;font-family:monospace;">'
                    f'Source: {cred_src} | multiplier = 0.5 + 0.5 × {raw_cred:.4f} = {cred_mult:.4f}</div>',
                    unsafe_allow_html=True,
                )

                # ════════════════════════════════════════════
                # FINAL FORMULA
                # ════════════════════════════════════════════
                st.markdown(
                    f'<div style="margin-top:12px;padding:10px 16px;background:rgba(29,78,216,0.04);'
                    f'border-left:3px solid {BRAND_BLUE};border-radius:0 6px 6px 0;'
                    f'font-family:monospace;font-size:12px;color:#1e293b;">'
                    f'<b>Management Signal</b> = outlook × credibility = '
                    f'{d.get("outlook_score_raw", 0):+.4f} × {d.get("credibility_multiplier", 0.75):.4f} = '
                    f'<span style="font-weight:800;color:{_ms_color};font-size:14px;">'
                    f'{d.get("management_signal", 0):+.4f}</span>'
                    f'<span style="color:#94a3b8;margin-left:12px;">Data: {d.get("data_sources", "N/A")}</span></div>',
                    unsafe_allow_html=True,
                )

                # ════════════════════════════════════════════
                # EARNINGS CONTEXT (quantitative anchor)
                # ════════════════════════════════════════════
                ec = d.get("earnings_context", {})
                grades = ec.get("grades", {})
                earnings_hist = ec.get("earnings_history", [])
                if grades or earnings_hist:
                    st.markdown(
                        f'<div style="font-size:11px;font-weight:800;color:{BRAND_BLUE};letter-spacing:0.1em;'
                        f'text-transform:uppercase;margin:18px 0 8px 0;border-bottom:2px solid {BRAND_BLUE};'
                        f'padding-bottom:4px;">Earnings Context</div>',
                        unsafe_allow_html=True,
                    )
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        if grades:
                            consensus = grades.get("consensus", "N/A")
                            total_analysts = sum(grades.get(k, 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])
                            con_color = SUCCESS_GREEN if consensus in ("Buy", "Strong Buy") else DANGER_RED if consensus in ("Sell", "Strong Sell") else BRAND_BLUE
                            render_metric_card("Analyst Consensus", consensus, delta_val=1 if consensus in ("Buy", "Strong Buy") else -1 if consensus in ("Sell", "Strong Sell") else 0)
                            # Grade breakdown pills
                            pills = []
                            for label, key, color in [
                                ("Strong Buy", "strongBuy", SUCCESS_GREEN), ("Buy", "buy", "#10b981"),
                                ("Hold", "hold", "#94a3b8"), ("Sell", "sell", "#f59e0b"),
                                ("Strong Sell", "strongSell", DANGER_RED),
                            ]:
                                cnt = grades.get(key, 0)
                                if cnt > 0:
                                    pills.append(
                                        f'<span style="display:inline-block;font-size:10px;color:{color};'
                                        f'background:rgba(148,163,184,0.08);padding:2px 8px;border-radius:10px;'
                                        f'margin:2px 3px 2px 0;font-weight:600;">{label}: {cnt}</span>'
                                    )
                            if pills:
                                st.markdown("".join(pills), unsafe_allow_html=True)

                    with ec2:
                        if earnings_hist:
                            beats = sum(1 for e in earnings_hist if (e.get("epsActual") or 0) > (e.get("epsEstimated") or 0))
                            misses = sum(1 for e in earnings_hist if (e.get("epsActual") or 0) < (e.get("epsEstimated") or 0))
                            total = len(earnings_hist)
                            beat_rate = beats / total if total > 0 else 0
                            render_metric_card(
                                f"EPS Beat Rate ({total}Q)",
                                f"{beat_rate:.0%}",
                                delta_val=beat_rate - 0.5 if abs(beat_rate - 0.5) > 0.1 else 0,
                            )
                            # Most recent quarter
                            recent = earnings_hist[0] if earnings_hist else {}
                            eps_a = recent.get("epsActual")
                            eps_e = recent.get("epsEstimated")
                            if eps_a is not None and eps_e is not None:
                                surprise = ((eps_a - eps_e) / abs(eps_e) * 100) if eps_e else 0
                                st.markdown(
                                    f'<div style="font-size:11px;color:#64748b;margin-top:4px;">'
                                    f'Latest: ${eps_a:.2f} vs est ${eps_e:.2f} '
                                    f'(<span style="color:{SUCCESS_GREEN if surprise > 0 else DANGER_RED};">'
                                    f'{surprise:+.1f}%</span>)</div>',
                                    unsafe_allow_html=True,
                                )

    # ══════════════════════════════════════════════════════════════
    # 5. CAPITAL FLOWS — Institutional & Insider Activity
    # ══════════════════════════════════════════════════════════════
    elif sub == "Flows":
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

        # ── Flow Overview — per-ticker summary cards ─────────────
        accum_labels = {
            "strong_accumulation": "Strong Accumulation",
            "moderate_accumulation": "Moderate Accumulation",
            "moderate_distribution": "Moderate Distribution",
            "neutral": "Neutral",
            "distribution": "Distribution",
        }
        accum_color_lookup = {
            "strong_accumulation": SUCCESS_GREEN,
            "moderate_accumulation": BRAND_BLUE,
            "neutral": SLATE_500,
            "moderate_distribution": "#f59e0b",
            "distribution": DANGER_RED,
        }

        overview_cols = st.columns(len(pi_tickers))
        for idx, t in enumerate(pi_tickers):
            d = pi_flow.get(t, {})
            comp = d.get("flow_composite", 0)
            accum = d.get("estimated_accumulation", "neutral")
            ins_s = d.get("insider_intelligence", {}).get("score", 0)
            fund_s = d.get("fund_accumulation", {}).get("score", 0)
            v_label = accum_labels.get(accum, "Neutral")
            v_color = accum_color_lookup.get(accum, SLATE_500)

            def _sc(v):
                return SUCCESS_GREEN if v > 0.1 else DANGER_RED if v < -0.1 else "#94a3b8"

            # Human-readable explanations for each sub-score
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

            with overview_cols[idx]:
                st.markdown(
                    f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                    f'background:#fafbfc;">'
                    # Ticker + composite score
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">'
                    f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{t}</span>'
                    f'<span style="font-size:18px;font-weight:800;color:{v_color};">{comp:+.2f}</span></div>'
                    # Verdict label
                    f'<div style="font-size:11px;font-weight:700;color:{v_color};'
                    f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">{v_label}</div>'
                    # Plain-English explanation of composite
                    f'<div style="font-size:11px;color:#64748b;margin-bottom:10px;line-height:1.4;'
                    f'font-style:italic;">{_comp_explain(comp)}</div>'
                    # Sub-scores with explanations
                    f'<div style="padding:5px 0;border-top:1px solid #e2e8f0;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:12px;">'
                    f'<span style="color:#475569;font-weight:600;">Insider Intel</span>'
                    f'<span style="color:{_sc(ins_s)};font-weight:700;">{ins_s:+.2f}</span></div>'
                    f'<div style="font-size:10px;color:#94a3b8;margin-top:1px;">{_ins_explain(ins_s)}</div></div>'
                    f'<div style="padding:5px 0;border-top:1px solid #f1f5f9;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:12px;">'
                    f'<span style="color:#475569;font-weight:600;">Fund Trend</span>'
                    f'<span style="color:{_sc(fund_s)};font-weight:700;">{fund_s:+.2f}</span></div>'
                    f'<div style="font-size:10px;color:#94a3b8;margin-top:1px;">{_fund_explain(fund_s)}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Spacer between cards and detail ──────────────────────
        st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)

        # ── Per-Ticker Sub-Bucket Breakdown ───────────────────────
        for t in pi_tickers:
            d = pi_flow.get(t, {})
            if d.get("status") != "Success":
                _err_detail = d.get("insider_intelligence", {}).get("summary", "")
                _err_msg = f"{t} — Flow data not available"
                if _err_detail and _err_detail != "No data":
                    _err_msg += f" ({_err_detail})"
                st.caption(_err_msg)
                continue

            accum = d.get("estimated_accumulation", "neutral")
            accum_color_map = {
                "strong_accumulation": SUCCESS_GREEN,
                "moderate_accumulation": BRAND_BLUE,
                "neutral": SLATE_500,
                "moderate_distribution": DANGER_RED,
                "distribution": DANGER_RED,
            }
            accum_color = accum_color_map.get(accum, SLATE_500)
            composite = d.get("flow_composite", 0)
            _comp_color = SUCCESS_GREEN if composite > 0.1 else DANGER_RED if composite < -0.1 else SLATE_500

            # Per-ticker header — matches section-header style
            st.markdown(
                f'<div style="font-size:13px;font-weight:800;color:{BRAND_BLUE};'
                f'background:#f8fafc;border-left:5px solid {BRAND_BLUE};'
                f'padding:10px 20px;margin:16px 0 8px 0;border-radius:0 8px 8px 0;'
                f'text-transform:uppercase;letter-spacing:1px;'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'{t} — Capital Flow Intelligence'
                f'<span style="color:{_comp_color};font-size:15px;">{composite:+.2f}</span></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Details", expanded=True):
                insider_i = d.get("insider_intelligence", {})
                fund_a = d.get("fund_accumulation", {})

                # ════════════════════════════════════════════
                # OWNERSHIP SNAPSHOT
                # ════════════════════════════════════════════
                _own = d.get("ownership_breakdown", {})
                _inst_pct = _own.get("institutional_pct", 0)
                _ins_pct = _own.get("insider_pct", 0)
                _ret_pct = _own.get("retail_pct", 0)
                if _inst_pct > 0 or _ins_pct > 0:
                    # Ownership bar
                    st.markdown(
                        f'<div style="margin-bottom:10px;">'
                        f'<div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:0.08em;'
                        f'text-transform:uppercase;margin-bottom:6px;">Shareholding Structure</div>'
                        # Stacked bar
                        f'<div style="display:flex;height:18px;border-radius:4px;overflow:hidden;'
                        f'margin-bottom:6px;border:1px solid #e2e8f0;">'
                        f'<div style="width:{_inst_pct*100:.1f}%;background:{BRAND_BLUE};"></div>'
                        f'<div style="width:{_ins_pct*100:.1f}%;background:{SUCCESS_GREEN};"></div>'
                        f'<div style="width:{_ret_pct*100:.1f}%;background:#e2e8f0;"></div>'
                        f'</div>'
                        # Legend
                        f'<div style="display:flex;gap:16px;font-size:11px;">'
                        f'<span style="color:{BRAND_BLUE};font-weight:600;">■ Institutional {_inst_pct:.1%}</span>'
                        f'<span style="color:{SUCCESS_GREEN};font-weight:600;">■ Insiders {_ins_pct:.1%}</span>'
                        f'<span style="color:#94a3b8;font-weight:600;">■ Retail/Other {_ret_pct:.1%}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                # Top holders pills
                holder_rows_snap = fund_a.get("holders_detail", [])
                if holder_rows_snap:
                    top_names = [h.get("name", "")[:28] for h in holder_rows_snap[:5]]
                    top_stakes = [h.get("pct_held", 0) for h in holder_rows_snap[:5]]
                    top_types = [h.get("type", "") for h in holder_rows_snap[:5]]
                    pills = []
                    for nm, stk, tp in zip(top_names, top_stakes, top_types):
                        _pill_color = "#6366f1" if "Smart" in tp else "#475569"
                        _pill_bg = "rgba(99,102,241,0.08)" if "Smart" in tp else "rgba(148,163,184,0.08)"
                        stk_str = f" ({stk:.1%})" if stk > 0 else ""
                        pills.append(
                            f'<span style="display:inline-block;font-size:11px;color:{_pill_color};'
                            f'background:{_pill_bg};padding:3px 10px;border-radius:12px;'
                            f'margin:2px 4px 2px 0;font-weight:500;">{nm}{stk_str}</span>'
                        )
                    st.markdown(
                        f'<div style="margin-bottom:10px;">'
                        f'<span style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:0.08em;'
                        f'text-transform:uppercase;margin-right:8px;">Largest Holders</span>'
                        + "".join(pills) + '</div>',
                        unsafe_allow_html=True,
                    )

                # ════════════════════════════════════════════
                # BUCKET 1: INSIDER INTELLIGENCE
                # ════════════════════════════════════════════
                _ins_score = insider_i.get("score", 0)
                _ins_sc = SUCCESS_GREEN if _ins_score > 0.1 else DANGER_RED if _ins_score < -0.1 else SLATE_500
                st.markdown(
                    f'<div style="font-size:11px;font-weight:800;color:{BRAND_BLUE};letter-spacing:0.1em;'
                    f'text-transform:uppercase;margin:10px 0 8px 0;border-bottom:2px solid {BRAND_BLUE};'
                    f'padding-bottom:4px;">Insider Intelligence '
                    f'<span style="color:{_ins_sc};float:right;font-size:13px;">{_ins_score:+.2f}</span></div>',
                    unsafe_allow_html=True,
                )
                # Interpret the score into a human-readable verdict
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
                # Pass delta_val=0 when verdict is Neutral so the label stays grey
                _ins_delta = _ins_score if abs(_ins_score) > 0.1 else 0
                with ic3: render_metric_card("Insider Signal", _ins_verdict, delta_val=_ins_delta)

                # Insider Activity — aggregated per person (not per transaction)
                notable = insider_i.get("notable_transactions", [])
                if notable:
                    _thl = 'style="text-align:left;padding:4px 6px;color:#475569;font-weight:700;font-size:11px;"'
                    _thc = 'style="text-align:center;padding:4px 6px;color:#475569;font-weight:700;font-size:11px;"'
                    tx_rows_html = (
                        '<table style="width:100%;border-collapse:collapse;font-size:12px;margin:4px 0 0 0;'
                        'table-layout:fixed;">'
                        '<colgroup>'
                        '<col style="width:20%;">'   # Insider
                        '<col style="width:20%;">'   # Role
                        '<col style="width:15%;">'   # Stake in Company
                        '<col style="width:15%;">'   # % Stake Sold
                        '<col style="width:14%;">'   # Direction
                        '<col style="width:16%;">'   # Latest
                        '</colgroup>'
                        f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                        f'<th {_thl}>Insider</th>'
                        f'<th {_thc}>Role</th>'
                        f'<th {_thc}>Stake in Company</th>'
                        f'<th {_thc}>% Stake Sold</th>'
                        f'<th {_thc}>Direction</th>'
                        f'<th {_thc}>Latest</th>'
                        f'</tr></thead><tbody>'
                    )
                    for tx in notable[:5]:
                        tx_type = tx.get("type", "SELL")
                        stake_pct = tx.get("stake_pct", 0)
                        pct_change = tx.get("pct_stake_change", 0)
                        # Format stake in company
                        if stake_pct >= 0.1:
                            stake_str = f"{stake_pct:.1f}%"
                        elif stake_pct >= 0.01:
                            stake_str = f"{stake_pct:.2f}%"
                        elif stake_pct > 0:
                            stake_str = "<0.01%"
                        else:
                            stake_str = '<span style="color:#94a3b8;font-style:italic;">Not available</span>'
                        # Format % of stake sold/bought over the period
                        # If we don't know their stake in company, we can't show % sold
                        if stake_pct <= 0:
                            pct_chg_str = '<span style="color:#94a3b8;font-style:italic;">N/A</span>'
                        elif pct_change > 0:
                            pct_chg_str = f"{pct_change:.1f}%"
                        else:
                            pct_chg_str = '<span style="color:#94a3b8;font-style:italic;">< 0.1%</span>'
                        # Color by materiality
                        if pct_change >= 15:
                            pct_color = DANGER_RED
                        elif pct_change >= 5:
                            pct_color = "#f59e0b"
                        else:
                            pct_color = "#64748b"
                        date_str = tx.get("date", "—") or "—"
                        # Direction arrow
                        if "BUY" in tx_type:
                            dir_color = SUCCESS_GREEN
                            dir_arrow = "▲ Buying"
                        elif "SELL" in tx_type:
                            dir_color = DANGER_RED
                            dir_arrow = "▼ Selling"
                        else:
                            dir_color = "#94a3b8"
                            dir_arrow = "— Mixed"
                        _tdc = 'text-align:center;'
                        tx_rows_html += (
                            f'<tr style="border-bottom:1px solid #f1f5f9;">'
                            f'<td style="padding:5px 6px;color:#1e293b;font-weight:500;">{tx.get("name", "")}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-size:11px;">{tx.get("role", "")}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:#475569;font-weight:600;'
                            f'font-variant-numeric:tabular-nums;">{stake_str}</td>'
                            f'<td style="padding:5px 6px;{_tdc}font-weight:700;color:{pct_color};'
                            f'font-variant-numeric:tabular-nums;">{pct_chg_str}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:{dir_color};'
                            f'font-weight:600;font-size:11px;">{dir_arrow}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-size:11px;'
                            f'font-variant-numeric:tabular-nums;">{date_str}</td>'
                            f'</tr>'
                        )
                    tx_rows_html += '</tbody></table>'
                    st.markdown(tx_rows_html, unsafe_allow_html=True)

                # ════════════════════════════════════════════
                # BUCKET 2: FUND ACCUMULATION TREND
                # ════════════════════════════════════════════
                _fa_score = fund_a.get("score", 0)
                _fa_sc = SUCCESS_GREEN if _fa_score > 0.1 else DANGER_RED if _fa_score < -0.1 else SLATE_500
                st.markdown(
                    f'<div style="font-size:11px;font-weight:800;color:{BRAND_BLUE};letter-spacing:0.1em;'
                    f'text-transform:uppercase;margin:18px 0 8px 0;border-bottom:2px solid {BRAND_BLUE};'
                    f'padding-bottom:4px;">Fund Accumulation Trend '
                    f'<span style="color:{_fa_sc};float:right;font-size:13px;">{_fa_score:+.2f}</span></div>',
                    unsafe_allow_html=True,
                )
                _smt = fund_a.get('smart_money_trend', 0)
                _smt_label = (
                    "Accumulating" if _smt > 0.3 else
                    "Adding" if _smt > 0.05 else
                    "Distributing" if _smt < -0.3 else
                    "Reducing" if _smt < -0.05 else
                    "Holding Steady"
                )
                fa1, fa2, fa3 = st.columns(3)
                with fa1: render_metric_card("Active Adding", str(fund_a.get("holders_increasing", 0)), delta_val=fund_a.get("holders_increasing", 0))
                with fa2: render_metric_card("Active Reducing", str(fund_a.get("holders_decreasing", 0)), delta_val=-fund_a.get("holders_decreasing", 0))
                # Pass delta_val=0 when verdict is Holding Steady so label stays grey
                _smt_delta = _smt if abs(_smt) > 0.05 else 0
                with fa3: render_metric_card("Smart Money", _smt_label, delta_val=_smt_delta)

                # Holder table — proper styled table with from → to + date
                holder_rows = fund_a.get("holders_detail", [])
                if holder_rows:
                    _thl = 'style="text-align:left;padding:4px 6px;color:#475569;font-weight:700;font-size:11px;"'
                    _thc = 'style="text-align:center;padding:4px 6px;color:#475569;font-weight:700;font-size:11px;"'
                    h_html = (
                        '<table style="width:100%;border-collapse:collapse;font-size:12px;margin:4px 0 0 0;'
                        'table-layout:fixed;">'
                        '<colgroup>'
                        '<col style="width:28%;">'   # Holder
                        '<col style="width:16%;">'   # Stake
                        '<col style="width:14%;">'   # Direction
                        '<col style="width:22%;">'   # Likely Reason
                        '<col style="width:20%;">'   # Filed
                        '</colgroup>'
                        f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                        f'<th {_thl}>Holder</th>'
                        f'<th {_thc}>Stake</th>'
                        f'<th {_thc}>Direction</th>'
                        f'<th {_thc}>Likely Reason</th>'
                        f'<th {_thc}>Filed</th>'
                        f'</tr></thead><tbody>'
                    )
                    for h in holder_rows[:8]:
                        dir_val = h.get("direction", "Stable")
                        dir_color = SUCCESS_GREEN if dir_val == "Adding" else DANGER_RED if dir_val == "Reducing" else "#94a3b8"
                        dir_arrow = "▲ Adding" if dir_val == "Adding" else "▼ Reducing" if dir_val == "Reducing" else "— Stable"
                        holder_type = h.get("type", "Active")
                        # Interpret: what does this move mean for the user?
                        if "Passive" in holder_type:
                            if dir_val == "Adding":
                                reason = "Index rebalance"
                            elif dir_val == "Reducing":
                                reason = "Index rebalance"
                            else:
                                reason = "Index tracking"
                        elif "Smart" in holder_type:
                            if dir_val == "Adding":
                                reason = "Conviction buy"
                            elif dir_val == "Reducing":
                                reason = "Taking profits"
                            else:
                                reason = "Holding position"
                        else:
                            if dir_val == "Adding":
                                reason = "Increasing investment"
                            elif dir_val == "Reducing":
                                reason = "Trimming position"
                            else:
                                reason = "Maintaining position"
                        stake_str = h.get("stake_change", "")
                        if not stake_str:
                            pct_h = h.get("pct_held", 0)
                            stake_str = f"{pct_h:.2%}" if pct_h > 0 else "—"
                        date_filed = h.get("date_reported", "—") or "—"
                        _tdc = 'text-align:center;'
                        h_html += (
                            f'<tr style="border-bottom:1px solid #f1f5f9;">'
                            f'<td style="padding:5px 6px;color:#1e293b;font-weight:500;">{h.get("name", "")[:35]}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:#475569;font-weight:600;'
                            f'font-variant-numeric:tabular-nums;">{stake_str}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:{dir_color};'
                            f'font-weight:600;font-size:11px;">{dir_arrow}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:#64748b;font-size:11px;'
                            f'font-style:italic;">{reason}</td>'
                            f'<td style="padding:5px 6px;{_tdc}color:#64748b;'
                            f'font-size:11px;font-variant-numeric:tabular-nums;">{date_filed}</td>'
                            f'</tr>'
                        )
                    h_html += '</tbody></table>'
                    st.markdown(h_html, unsafe_allow_html=True)


