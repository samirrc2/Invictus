"""
invictus.pages.conviction
=========================
Conviction Intelligence — 4 sub-tabs:
  Conviction Engine, Capital Flows, Management Outlook, Transcript Analysis.

This package is a thin router:
  1. Stock selection UI (shared across all tabs)
  2. Pipeline execution (runs agents on selected tickers)
  3. Dispatches to the correct sub-tab module
"""
import pandas as pd
import streamlit as st

from invictus.pages.conviction import engine, flows, outlook, transcript


def render(sub: str):
    """Entry point called by app.py. `sub` is the internal route key."""

    # ── Stock Selection (always visible at top) ──────────────────
    portfolio_tickers = []
    if st.session_state.portfolio_loaded:
        portfolio_tickers = st.session_state.portfolio_state["holdings"]["Ticker"].tolist()

    # Default to demo tickers if results already exist from demo mode
    _demo_defaults = []
    if portfolio_tickers and st.session_state.pi_results and st.session_state.pi_results.get("tickers"):
        _demo_defaults = [t for t in st.session_state.pi_results["tickers"] if t in portfolio_tickers]

    pick_col, enter_col, btn_col = st.columns([1.2, 1.2, 0.6])
    with pick_col:
        picked = st.multiselect(
            "From portfolio", options=portfolio_tickers,
            default=_demo_defaults, max_selections=3,
            key="pi_portfolio_pick", placeholder="Select from portfolio...",
            disabled=len(portfolio_tickers) == 0,
        )
    with enter_col:
        new_input = st.text_input(
            "Enter tickers", placeholder="NVDA, AMZN, TSLA",
            key="pi_new_ticker_input",
        )
    with btn_col:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        new_raw = [t.strip().upper() for t in new_input.replace(",", " ").split() if t.strip()] if new_input else []
        all_selected = list(dict.fromkeys(picked + new_raw))[:3]
        run_btn = st.button("Run Intel", key="pi_run_btn", type="primary",
                            use_container_width=True, disabled=len(all_selected) == 0)

    _count = len(all_selected)
    st.markdown(
        f'<div style="font-size:11px;color:#94a3b8;margin:-8px 0 8px 0;">'
        f'Pick from portfolio or enter new — max 3 tickers combined '
        f'({_count}/3 selected)</div>',
        unsafe_allow_html=True,
    )

    # ── Run Pipeline ─────────────────────────────────────────────
    if run_btn and all_selected:
        _run_pipeline(all_selected)

    # ── Gate: need results ───────────────────────────────────────
    pi = st.session_state.pi_results
    if not pi or not pi.get("tickers"):
        st.info("Select up to 3 tickers from your portfolio or enter new ones, then click Run Intel.")
        return

    pi_tickers = pi["tickers"]
    pi_synth   = pi["synthesis"]
    pi_flow    = pi["flows"]
    pi_outlook = pi.get("outlook", {})

    # ── Route to sub-tab ─────────────────────────────────────────
    if sub == "Engine":
        engine.render(pi_tickers, pi_synth, pi_flow, pi_outlook)
    elif sub == "Flows":
        flows.render(pi_tickers, pi_flow)
    elif sub == "Outlook":
        outlook.render(pi_tickers, pi_outlook)
    elif sub == "Transcript":
        transcript.render(pi_tickers, pi_outlook)
    else:
        engine.render(pi_tickers, pi_synth, pi_flow, pi_outlook)


def _run_pipeline(all_selected):
    """Execute the conviction intelligence pipeline for selected tickers."""
    from invictus.agents.filing_agent import _extract_yfinance_fundamental_signals
    from invictus.agents.earnings_agent import (
        _fetch_sentiment_context, _analyze_sentiment_with_llm, _dictionary_sentiment,
    )
    from invictus.agents.flow_agent import _fetch_flow_data, _score_flows
    from invictus.agents.outlook_agent import analyze_management_outlook
    from invictus.agents.synthesis_agent import _calculate_stock_conviction

    pi_filing, pi_earnings, pi_flows, pi_outlook, pi_synthesis = {}, {}, {}, {}, {}

    # Reuse cached results
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
            # Filing
            progress.progress((step + 1) / total, text=f"Filing Intelligence: {t}")
            pi_filing[t] = _extract_yfinance_fundamental_signals(t)
            step += 1

            # Earnings (legacy — feeds synthesis)
            progress.progress((step + 1) / total, text=f"Earnings Intelligence: {t}")
            ctx = _fetch_sentiment_context(t)
            result = _analyze_sentiment_with_llm(t, ctx)
            pi_earnings[t] = result if (result and result.get("status") == "Success") else _dictionary_sentiment(ctx)
            step += 1

            # Capital Flows
            progress.progress((step + 1) / total, text=f"Capital Flows: {t}")
            raw_flow = _fetch_flow_data(t)
            try:
                pi_flows[t] = _score_flows(t, raw_flow)
            except Exception as _flow_err:
                import logging as _lg
                _lg.error("Flow scoring CRASHED for %s: %s", t, _flow_err, exc_info=True)
                pi_flows[t] = {
                    "status": "Error", "flow_composite": 0,
                    "insider_intelligence": {"score": 0, "red_flags": [], "green_flags": [], "summary": str(_flow_err)},
                    "fund_accumulation": {"score": 0, "red_flags": [], "green_flags": [], "summary": str(_flow_err)},
                    "concentration": {"score": 0, "red_flags": [], "green_flags": [], "summary": str(_flow_err)},
                    "smart_money_pct": 0, "institutional_conviction": 0, "insider_alignment": 0,
                    "capital_participation": 0, "insider_buys": 0, "insider_sells": 0,
                    "estimated_accumulation": "neutral", "net_insider_value": 0, "notable_transactions": [],
                }
            step += 1

            # Management Outlook
            progress.progress((step + 1) / total, text=f"Management Outlook: {t}")
            try:
                pi_outlook[t] = analyze_management_outlook(t)
            except Exception as _outlook_err:
                import logging as _lg
                _lg.error("Outlook CRASHED for %s: %s", t, _outlook_err, exc_info=True)
                pi_outlook[t] = {
                    "status": "Error", "management_signal": 0,
                    "outlook_score_raw": 0, "credibility_multiplier": 0.75,
                    "outlook": {"dimensions": {}, "outlook_score": 0, "status": "Error"},
                    "credibility": {"sub_dimensions": {}, "raw_credibility": 0.5,
                                    "credibility_multiplier": 0.75, "status": "Error"},
                    "data_sources": "Error", "ticker": t,
                }
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
