"""
Invictus Equity Portfolio Intelligence Platform
Main Streamlit Application — Thin routing shell.

Page content lives in invictus/pages/.
Design system lives in invictus/design/.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import plotly.io as pio

load_dotenv()  # Must run BEFORE any invictus.* imports (they read env vars at import time)

# ── Design System ─────────────────────────────────────────────────────
from invictus.design import (
    inject_styles, render_sidebar_brand, render_footer,
    render_metric_card, fmt_currency,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, SLATE_500,
)

# ── Analytics ─────────────────────────────────────────────────────────
from invictus.analytics.tracker import (
    create_session as _analytics_create_session,
)

# ── Page Modules ──────────────────────────────────────────────────────
from invictus.pages import portfolio as portfolio_analytics, hypo_simulator, dev_analytics
from invictus.pages import conviction
from invictus.pages import landing

pio.templates.default = "plotly"

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Invictus Portfolio Intelligence",
    page_icon="https://raw.githubusercontent.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# ── Session State ─────────────────────────────────────────────────────
for key, default in {
    "portfolio_loaded": False, "portfolio_state": None,
    "risk_state": None, "pca_state": None, "vol_regime_state": None,
    "stress_state": None, "greeks_state": None, "flow_signals": None,
    "ml_state": None, "rag_state": None, "pnl_state": None,
    "commentary_state": None, "eval_state": None, "filing_intel": None,
    "earnings_intel": None, "conviction_synthesis": None,
    "live_feed": True, "last_refresh_time": 0,
    "dev_authenticated": False, "_analytics_sid": None, "_last_tracked_page": None,
    "pi_results": None, "pi_selected_tickers": [],
    "hypo_results": None,
    "agent_status": {},
    "pipeline_running": False,
    "pipeline_errors": [],
    "nav_primary": "Portfolio Intelligence",
    "nav_sub": "Overview",
    "show_landing": True,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state._analytics_sid is None:
    st.session_state._analytics_sid = _analytics_create_session()

# ── Agent Display Mapping ─────────────────────────────────────────────
_AGENT_DISPLAY = {
    "load_portfolio": "Load Portfolio",
    "compute_risk": "Risk Analytics",
    "run_pca": "PCA Factor",
    "detect_vol_regime": "Vol Regime",
    "run_stress_tests": "Stress Test",
    "compute_greeks": "Greeks",
    "analyze_flows": "Inst. Flows",
    "retrieve_10k_context": "10-K RAG",
    "run_filing_intel": "Filing Intel",
    "run_earnings_intel": "Earnings Intel",
    "run_accumulation_model": "ML Engine",
    "run_conviction_synthesis": "Conviction",
    "attribute_pnl": "P&L Attribution",
    "generate_commentary": "AI Commentary",
    "evaluate_commentary": "Eval Harness",
}

# ── Routes ────────────────────────────────────────────────────────────
_ROUTES = {
    "Portfolio Intelligence": [
        "Overview", "Risk Analytics", "Factor Decomposition",
        "Volatility Regimes", "Stress Scenarios", "P&L Attribution",
    ],
    "Conviction Intelligence": [
        "Conviction Engine", "Capital Flows",
        "Management Outlook", "Transcript Analysis",
    ],
    "Allocation Engine": [
        "Run Allocation Simulation",
    ],
}
if st.query_params.get("dev") == "invictus":
    _ROUTES = {"Dev Analytics": [
        "Architecture", "Agent Performance", "LLM Quality", "ML Monitoring",
        "Conviction Analytics", "Conviction Intelligence", "Session Analytics",
        "Data Health", "Cost Analytics", "Eval Metrics", "Backtest",
    ]}

# ── Agent Status Helper ───────────────────────────────────────────────
_SIDEBAR_AGENTS = [
    "Risk Analytics", "PCA Factor", "Vol Regime", "Stress Test",
    "Greeks", "Inst. Flows", "10-K RAG", "Filing Intel", "Earnings Intel",
    "ML Engine", "Conviction", "P&L Attribution", "AI Commentary", "Eval Harness",
]
_RESULT_KEYS = {
    "Risk Analytics": "risk_state", "PCA Factor": "pca_state",
    "Vol Regime": "vol_regime_state", "Stress Test": "stress_state",
    "Greeks": "greeks_state", "Inst. Flows": "flow_signals",
    "10-K RAG": "rag_state", "Filing Intel": "filing_intel",
    "Earnings Intel": "earnings_intel",
    "ML Engine": "ml_state", "Conviction": "conviction_synthesis",
    "P&L Attribution": "pnl_state", "AI Commentary": "commentary_state",
    "Eval Harness": "eval_state",
}

def _agent_status(label):
    for node_name, display in _AGENT_DISPLAY.items():
        if display == label:
            s = st.session_state.agent_status.get(node_name)
            if s:
                return s
    sk = _RESULT_KEYS.get(label)
    if sk and st.session_state.get(sk) is not None:
        return "done"
    return "idle"

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — Brand, Navigation, Portfolio Upload, Snapshot
# ══════════════════════════════════════════════════════════════════════
logo_path = Path(__file__).parent / "invictus" / "static" / "logo.png"

with st.sidebar:
    render_sidebar_brand(logo_path)

    # ── Snapshot (always visible, -- when no portfolio) ────────
    if st.session_state.portfolio_loaded:
        ps = st.session_state.portfolio_state
        nw_val = f'${ps["total_value"]:,.2f}'
        day_chg = ps["total_daily_pnl"]
        day_pct = ps.get("daily_return_pct", 0)
        day_sign = "+" if day_chg >= 0 else ""
        day_val = f'{day_sign}${abs(day_chg):,.2f} ({day_sign}{day_pct:.2f}%)'
        day_dot = "#10b981" if day_chg >= 0 else "#ef4444"
        unr_pnl = ps["total_unrealized_pnl"]
        unr_pct = ps.get("unrealized_pnl_pct", 0)
        unr_sign = "+" if unr_pnl >= 0 else ""
        unr_val = f'{unr_sign}${abs(unr_pnl):,.2f} ({unr_sign}{unr_pct:.2f}%)'
        unr_dot = "#10b981" if unr_pnl >= 0 else "#ef4444"
    else:
        nw_val = "--"
        day_val = "--"
        day_dot = "#cbd5e1"
        unr_val = "--"
        unr_dot = "#cbd5e1"

    st.markdown(
        f'<div style="padding:14px 12px 12px 12px;border-bottom:1px solid #e2e8f0;margin-bottom:8px;">'
        f'<div style="font-size:10px;font-weight:800;color:#94a3b8;letter-spacing:0.1em;'
        f'text-transform:uppercase;margin-bottom:6px;">Portfolio Holdings</div>'
        f'<div style="font-size:22px;font-weight:800;color:#0f172a;'
        f'font-variant-numeric:tabular-nums;letter-spacing:-0.01em;">{nw_val}</div>'
        f'<div style="margin-top:10px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;padding:4px 0;">'
        f'<span style="font-size:12px;font-weight:600;color:#64748b;">Day Change</span>'
        f'<span style="font-size:13px;font-weight:700;color:{day_dot};'
        f'font-variant-numeric:tabular-nums;">{day_val}</span></div>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;padding:4px 0;">'
        f'<span style="font-size:12px;font-weight:600;color:#64748b;">Unrealized P&L</span>'
        f'<span style="font-size:13px;font-weight:700;color:{unr_dot};'
        f'font-variant-numeric:tabular-nums;">{unr_val}</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Navigation ────────────────────────────────────────────────
    st.markdown('<div class="snav-tree">', unsafe_allow_html=True)
    _landing_active = st.session_state.get("show_landing", False)
    for route, subs in _ROUTES.items():
        is_active = (not _landing_active) and st.session_state.nav_primary == route

        st.markdown(f'<div class="snav-heading-wrap{"  active" if is_active else ""}"></div>',
                    unsafe_allow_html=True)
        if st.button(route, key=f"snav_p__{route}", use_container_width=True):
            st.session_state.nav_primary = route
            st.session_state.nav_sub = (subs[0] if subs else None)
            st.session_state.show_landing = False
            st.rerun()

        if subs:
            for s in subs:
                is_sub = is_active and st.session_state.nav_sub == s
                st.markdown(f'<div class="snav-item-wrap{"  active" if is_sub else ""}"></div>',
                            unsafe_allow_html=True)
                if st.button(s, key=f"snav_s__{route}__{s}", use_container_width=True):
                    st.session_state.nav_primary = route
                    st.session_state.nav_sub = s
                    st.session_state.show_landing = False
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TOP BAR — Demo Mode + Portfolio Source + Load
# ══════════════════════════════════════════════════════════════════════
uploaded_file = None
manual_mapping = None
upload_method = "Default Portfolio"
load_btn = False
demo_mode = False

# ── Sticky Header: Page Title + Actions + Blue Line ──────────────────
primary = st.session_state.nav_primary
sub = st.session_state.nav_sub

_on_landing = st.session_state.get("show_landing", False)

if _on_landing:
    _page_title = (
        '<span style="color:#94a3b8;">Invictus</span>'
        '<span style="color:#cbd5e1;margin:0 8px;font-weight:400;">›</span>'
        '<span style="color:#0f172a;">How It Works</span>'
    )
elif sub:
    _page_title = (
        f'<span style="color:#94a3b8;">{primary}</span>'
        f'<span style="color:#cbd5e1;margin:0 8px;font-weight:400;">›</span>'
        f'<span style="color:#0f172a;">{sub}</span>'
    )
else:
    _page_title = f'<span style="color:#0f172a;">{primary}</span>'

_header = st.container()
with _header:
    st.markdown('<div class="inv-header-pin"></div>', unsafe_allow_html=True)
    _title_col, _hiw_col, _demo_col, _csv_col, _load_col = st.columns([3.4, 0.6, 0.6, 0.5, 0.6])

    with _title_col:
        st.markdown(
            f'<div style="font-size:18px;font-weight:700;color:#94a3b8;'
            f'padding:6px 0;letter-spacing:0.02em;">{_page_title}</div>',
            unsafe_allow_html=True,
        )

    with _hiw_col:
        if st.button("How It Works", key="top_hiw_btn", use_container_width=True, type="primary"):
            st.session_state.show_landing = True
            st.rerun()

    with _demo_col:
        demo_btn = st.button("Demo Mode", key="top_demo_btn", use_container_width=True, type="primary")

    with _csv_col:
        _csv_checked = st.checkbox("Upload CSV", key="top_csv_check", value=False)

    with _load_col:
        load_btn = st.button("Load Portfolio", key="top_load_btn", use_container_width=True, type="primary")

    st.markdown(
        '<div style="height:2px;background:linear-gradient(90deg,#1d4ed8 0%,#60a5fa 50%,transparent 100%);'
        'margin:4px 0 0 0;border-radius:1px;"></div>',
        unsafe_allow_html=True,
    )

# CSV upload panel (shown when checkbox is ticked)
if _csv_checked:
    uploaded_file = st.file_uploader(
        "Upload portfolio CSV (max 2MB) — AI will auto-detect your holdings",
        type=["csv"], key="top_csv_upload", label_visibility="collapsed",
    )
    if uploaded_file:
        if uploaded_file.size > 2 * 1024 * 1024:
            st.error("File too large. Maximum size is 2MB.")
            uploaded_file = None
        else:
            upload_method = "Upload CSV"
            st.caption("AI will automatically extract Ticker, Shares, and Cost Basis from your CSV — no manual mapping needed.")

# Demo Mode — direct trigger (no dialog on first click, confirm inline)
if demo_btn:
    load_btn = True
    demo_mode = True

# ══════════════════════════════════════════════════════════════════════
# PIPELINE — Load Portfolio + Run Agents
# ══════════════════════════════════════════════════════════════════════
if load_btn:
    from invictus.data.portfolio_loader import (
        load_portfolio_from_dict, fetch_price_history, compute_portfolio_state,
    )
    from invictus.data.smart_loader import smart_load_portfolio
    from invictus.agents.graph_state import PortfolioState as PState
    from invictus.agents.orchestrator import create_graph

    st.session_state.agent_status = {k: "idle" for k in _AGENT_DISPLAY}
    st.session_state.pipeline_running = True
    _completed_nodes = []

    # Observability
    from invictus.observability.store import generate_run_id
    from invictus.observability.collectors.agent_collector import log_agent_run
    from invictus.observability.collectors.session_collector import log_pipeline_start, log_pipeline_complete
    import time as _time
    _run_id = generate_run_id()
    _pipeline_start_time = _time.perf_counter()
    _mode = "demo" if demo_mode else "csv" if (upload_method == "Upload CSV") else "default"
    log_pipeline_start(session_id=st.session_state.get("_analytics_sid"), mode=_mode)

    with st.status("Executing Invictus Pipeline...", expanded=True) as status_ui:
        try:
            status_ui.update(label="Loading portfolio data...", state="running")
            if upload_method == "Upload CSV" and uploaded_file:
                holdings = smart_load_portfolio(uploaded_file.getvalue().decode("utf-8"),
                                                manual_mapping=manual_mapping)
            else:
                holdings = load_portfolio_from_dict()

            prices = fetch_price_history(holdings["Ticker"].tolist())
            state = compute_portfolio_state(holdings, prices)
            st.session_state.portfolio_state = state
            st.session_state.portfolio_loaded = True

            graph = create_graph()
            pstate = PState(
                holdings=state["holdings"], prices=state["prices"],
                returns=state["returns"], weights=state["weights"],
                total_value=state["total_value"],
            )

            _agent_timers = {}

            def _pipeline_progress(node_name, stage_idx, total_stages):
                display = _AGENT_DISPLAY.get(node_name, node_name)
                # Log completion of previous agent
                for prev in _completed_nodes:
                    st.session_state.agent_status[prev] = "done"
                    if prev in _agent_timers:
                        elapsed = (_time.perf_counter() - _agent_timers[prev]) * 1000
                        log_agent_run(prev, _run_id, "success", elapsed)
                # Start timing current agent
                st.session_state.agent_status[node_name] = "working"
                _agent_timers[node_name] = _time.perf_counter()
                _completed_nodes.append(node_name)
                stage_pct = int((stage_idx / total_stages) * 100)
                status_ui.update(label=f"Running {display}... ({stage_pct}%)", state="running")
                st.write(f"▸ **{display}**")

            pstate = graph.run(pstate, progress_callback=_pipeline_progress)

            # Log final agent + mark all done
            for node_name in _completed_nodes:
                st.session_state.agent_status[node_name] = "done"
                if node_name in _agent_timers:
                    elapsed = (_time.perf_counter() - _agent_timers[node_name]) * 1000
                    log_agent_run(node_name, _run_id, "success", elapsed)

            if pstate.errors:
                st.session_state.pipeline_errors = list(pstate.errors)
                for err_str in pstate.errors:
                    if err_str.startswith("[") and "]" in err_str:
                        err_node = err_str[1:err_str.index("]")]
                        if err_node in st.session_state.agent_status:
                            st.session_state.agent_status[err_node] = "error"
                            log_agent_run(err_node, _run_id, "error", 0,
                                          error_type="AgentError", error_message=err_str)

            st.session_state.risk_state = {
                "risk_metrics": pstate.risk_metrics,
                "correlation_matrix": pstate.correlation_matrix,
                "ticker_risk": pstate.ticker_risk,
            }
            st.session_state.pca_state            = pstate.pca_results
            st.session_state.vol_regime_state     = pstate.vol_regime
            st.session_state.stress_state         = pstate.stress_results
            st.session_state.greeks_state         = pstate.greeks_results
            st.session_state.flow_signals         = pstate.flow_signals
            st.session_state.ml_state             = pstate.ml_predictions
            st.session_state.filing_intel         = pstate.filing_intel
            st.session_state.earnings_intel       = pstate.earnings_intel
            st.session_state.conviction_synthesis = pstate.conviction_synthesis
            st.session_state.commentary_state     = pstate.commentary
            st.session_state.rag_state            = pstate.rag_insights
            st.session_state.pnl_state            = pstate.pnl_attribution

            st.session_state.pipeline_running = False

            # ── Demo Mode: also run PI + Hypo Simulator ──────────
            if demo_mode:
                status_ui.update(label="Demo: Running Predictive Intel...", state="running")
                st.write("▸ **Predictive Intel (demo)**")

                from invictus.agents.filing_agent import _extract_yfinance_fundamental_signals
                from invictus.agents.earnings_agent import _fetch_yfinance_sentiment_context, _analyze_sentiment_with_llm, _dictionary_sentiment
                from invictus.agents.flow_agent import _fetch_flow_data, _score_flows
                from invictus.agents.outlook_agent import analyze_management_outlook
                from invictus.agents.synthesis_agent import _calculate_stock_conviction

                demo_tickers = holdings["Ticker"].tolist()[:3]
                pi_filing, pi_earnings, pi_flows, pi_outlook, pi_synthesis = {}, {}, {}, {}, {}

                for t in demo_tickers:
                    if st.session_state.filing_intel and t in st.session_state.filing_intel:
                        pi_filing[t] = st.session_state.filing_intel[t]
                    else:
                        pi_filing[t] = _extract_yfinance_fundamental_signals(t)

                    if st.session_state.earnings_intel and t in st.session_state.earnings_intel:
                        pi_earnings[t] = st.session_state.earnings_intel[t]
                    else:
                        ctx = _fetch_yfinance_sentiment_context(t)
                        result = _analyze_sentiment_with_llm(t, ctx)
                        pi_earnings[t] = result if (result and result.get("status") == "Success") else _dictionary_sentiment(ctx)

                    if st.session_state.flow_signals and t in st.session_state.flow_signals.get("intel", {}):
                        pi_flows[t] = st.session_state.flow_signals["intel"][t]
                    else:
                        raw_flow = _fetch_flow_data(t)
                        pi_flows[t] = _score_flows(t, raw_flow)

                    try:
                        pi_outlook[t] = analyze_management_outlook(t)
                    except Exception:
                        pi_outlook[t] = {"status": "Error", "management_signal": 0,
                            "outlook_score_raw": 0, "credibility_multiplier": 0.75,
                            "outlook": {"dimensions": {}, "outlook_score": 0, "status": "Error"},
                            "credibility": {"sub_dimensions": {}, "raw_credibility": 0.5,
                                "credibility_multiplier": 0.75, "status": "Error"},
                            "data_sources": "Error", "ticker": t}

                vol_regime = st.session_state.vol_regime_state.get("current_regime") if st.session_state.vol_regime_state else None
                for t in demo_tickers:
                    f = pi_filing.get(t, {"status": "N/A"})
                    e = pi_earnings.get(t, {"status": "N/A"})
                    fl = pi_flows.get(t, {"status": "N/A"})
                    ml_pred = None
                    if st.session_state.ml_state and "prediction_table" in (st.session_state.ml_state or {}):
                        import pandas as _pd
                        pt = st.session_state.ml_state["prediction_table"]
                        if isinstance(pt, _pd.DataFrame) and t in pt["Ticker"].values:
                            row = pt[pt["Ticker"] == t].iloc[0]
                            ml_pred = {"accumulation_prob": row.get("Accumulation Prob", 0.5)}
                    pi_synthesis[t] = _calculate_stock_conviction(t, f, e, fl, ml_pred, vol_regime, "1 year")

                st.session_state.pi_results = {
                    "filing": pi_filing, "earnings": pi_earnings,
                    "flows": pi_flows, "outlook": pi_outlook,
                    "synthesis": pi_synthesis, "tickers": demo_tickers,
                }
                st.session_state.pi_selected_tickers = demo_tickers

                # Run Hypo Simulator with $5,000 per ticker
                status_ui.update(label="Demo: Running Hypo Simulator...", state="running")
                st.write("▸ **Hypo Simulator (demo)**")

                from invictus.agents.hypo_agent import compute_before_after, generate_pros_cons
                demo_positions = {t: 5000.0 for t in demo_tickers}
                comparison = compute_before_after(
                    st.session_state.portfolio_state,
                    st.session_state.risk_state,
                    demo_positions,
                    prices,
                )
                if "error" not in comparison:
                    commentary = generate_pros_cons(comparison)
                    st.session_state.hypo_results = {
                        "comparison": comparison,
                        "commentary": commentary,
                        "positions": demo_positions,
                    }

            # Log pipeline completion
            _pipeline_duration = (_time.perf_counter() - _pipeline_start_time) * 1000
            log_pipeline_complete(
                session_id=st.session_state.get("_analytics_sid"),
                duration_ms=_pipeline_duration, mode=_mode,
            )

            status_ui.update(label="Pipeline complete — all agents finished", state="complete", expanded=False)
            st.session_state.show_landing = False
            st.rerun()
        except Exception as e:
            import traceback as _tb
            st.session_state.pipeline_running = False
            st.session_state.pipeline_errors = st.session_state.get("pipeline_errors", []) + [
                f"[FATAL] {type(e).__name__}: {e}\n{_tb.format_exc()}"
            ]
            status_ui.update(label=f"Pipeline error: {e}", state="error")
            st.error(f"Execution Error: {e}")

# ══════════════════════════════════════════════════════════════════════
# LANDING PAGE — shows when show_landing is active
# ══════════════════════════════════════════════════════════════════════
if _on_landing:
    landing.render()

# ══════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ══════════════════════════════════════════════════════════════════════
elif primary == "Portfolio Intelligence":
    # Map new sub-tab names to old ones for the page module
    _pi_sub_map = {
        "Overview": "Dashboard", "Risk Analytics": "Analytics",
        "Factor Decomposition": "PCA", "Volatility Regimes": "Vol Regime",
        "Stress Scenarios": "Stress Test", "P&L Attribution": "Attribution",
    }
    portfolio_analytics.render(_pi_sub_map.get(sub, sub))
elif primary == "Conviction Intelligence":
    _ci_sub_map = {
        "Conviction Engine": "Engine", "Capital Flows": "Flows",
        "Management Outlook": "Outlook", "Transcript Analysis": "Transcript",
    }
    conviction.render(_ci_sub_map.get(sub, sub))
elif primary == "Allocation Engine":
    hypo_simulator.render(sub)
elif primary == "Agent Tracker":
    # Agent Tracker — show all agent statuses
    from invictus.design import render_section_header
    render_section_header("Agent Tracker")

    if not st.session_state.portfolio_loaded:
        st.info("Load a portfolio or run Demo Mode to activate agents.")
    else:
        done_agents = [a for a in _SIDEBAR_AGENTS if _agent_status(a) == "done"]
        working_agents = [a for a in _SIDEBAR_AGENTS if _agent_status(a) == "working"]
        error_agents = [a for a in _SIDEBAR_AGENTS if _agent_status(a) == "error"]
        idle_agents = [a for a in _SIDEBAR_AGENTS if _agent_status(a) == "idle"]

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            from invictus.design import render_metric_card as _rmc
            _rmc("Total Agents", str(len(_SIDEBAR_AGENTS)))
        with t2: _rmc("Completed", str(len(done_agents)), delta_val=len(done_agents))
        with t3: _rmc("Running", str(len(working_agents)))
        with t4: _rmc("Errors", str(len(error_agents)), delta_val=-len(error_agents) if error_agents else 0)

        # Agent list with status
        _status_colors = {"done": SUCCESS_GREEN, "working": "#f59e0b", "error": DANGER_RED, "idle": SLATE_500}
        _status_labels = {"done": "COMPLETE", "working": "RUNNING", "error": "ERROR", "idle": "IDLE"}
        agent_html = ""
        for agent in _SIDEBAR_AGENTS:
            status = _agent_status(agent)
            color = _status_colors[status]
            label = _status_labels[status]
            agent_html += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 12px;border-bottom:1px solid #f1f5f9;">'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:{color};'
                f'box-shadow:0 0 4px {color};"></div>'
                f'<span style="font-size:13px;font-weight:600;color:#1e293b;">{agent}</span></div>'
                f'<span style="font-size:10px;font-weight:700;color:{color};'
                f'letter-spacing:0.06em;">{label}</span></div>'
            )
        st.markdown(
            f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;'
            f'overflow:hidden;margin-top:12px;">{agent_html}</div>',
            unsafe_allow_html=True,
        )
elif primary == "Dev Analytics":
    dev_analytics.render(sub)

# ══════════════════════════════════════════════════════════════════════
# FOOTER — Disclaimer + Agent Status
# ══════════════════════════════════════════════════════════════════════
done_count = sum(1 for a in _SIDEBAR_AGENTS if _agent_status(a) == "done")
working_count = sum(1 for a in _SIDEBAR_AGENTS if _agent_status(a) == "working")
error_count = sum(1 for a in _SIDEBAR_AGENTS if _agent_status(a) == "error")
total = len(_SIDEBAR_AGENTS)

if working_count > 0:
    agent_indicator = f'<span style="color:#f59e0b;font-weight:800;">● RUNNING {done_count}/{total}</span>'
elif done_count == total and st.session_state.portfolio_loaded:
    agent_indicator = f'<span style="color:#10b981;font-weight:800;">● {done_count}/{total} COMPLETE</span>'
elif done_count > 0:
    agent_indicator = f'<span style="color:#10b981;">{done_count}/{total} done</span>'
    if error_count > 0:
        agent_indicator += f' <span style="color:#ef4444;">{error_count} err</span>'
else:
    agent_indicator = f'<span style="opacity:0.5;">{done_count}/{total}</span>'

# Footer — disclaimer centered, agent tracker pinned right
st.markdown(
    f'<div class="inv-footer">'
    f'<div style="flex:1;"></div>'
    f'<div style="display:flex;align-items:center;gap:12px;">'
    f'<span>© 2026 Invictus Equity Portfolio Intelligence</span>'
    f'<span class="sep">|</span>'
    f'<span>Not investment advice. Educational purposes only.</span>'
    f'<span class="sep">|</span>'
    f'<span>Data: Yahoo Finance, FMP API, SEC 13F Filings</span></div>'
    f'<div style="flex:1;text-align:right;">Agents {agent_indicator}</div>'
    f'</div>',
    unsafe_allow_html=True,
)
