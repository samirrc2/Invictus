"""
Invictus Equity Portfolio Intelligence Platform
Main Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import time
import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# ── Load Analytics ────────────────────────────────────────────────────
from invictus.analytics.tracker import (
    create_session as _analytics_create_session,
    track_click as _analytics_track_click,
    track_page_view as _analytics_track_page_view,
    get_summary_stats as _analytics_get_summary_stats,
    get_daily_traffic as _analytics_get_daily_traffic,
    get_page_popularity as _analytics_get_page_popularity,
)

# Load Environment Variables
load_dotenv()

# Force SVG renderer
pio.templates.default = "plotly"

# Page Config
st.set_page_config(
    page_title="Invictus Portfolio Intelligence",
    page_icon="https://raw.githubusercontent.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Institutional Design Tokens ── */
    :root {
        --brand-navy: #1e293b;
        --brand-blue: #1d4ed8;
        --brand-blue-hover: #2563eb;
        --brand-blue-light: #60a5fa;
        --brand-silver: #cbd5e1;
        --brand-silver-bright: #f8fafc;
        --capsule-bg: #eef2ff;
        --capsule-text: #1e3a8a;
        --capsule-muted: #64748b;
        --success-green: #10b981;
        --danger-red: #ef4444;
    }

    /* ── Global App Reset ── */
    html, body, .stApp { 
        background-color: #ffffff !important; 
        color: #0f172a !important; 
        font-family: 'Inter', sans-serif !important; 
    }

    /* ── Sidebar (Midnight Command Center) ── */
    [data-testid='stSidebar'], 
    [data-testid='stSidebarContent'],
    [data-testid='stSidebarUserContent'] {
        background-color: #020617 !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid='stSidebarUserContent'] { padding-top: 0 !important; margin-top: -45px !important; }

    /* Sidebar Brand Block */
    .sidebar-brand {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 0 0 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin: 0 0 12px 0;
    }
    .sidebar-brand img {
        height: 70px !important;
        width: auto;
        margin-bottom: 4px;
        filter: drop-shadow(0 0 10px rgba(29,78,216,0.3));
    }
    .sidebar-brand-name {
        color: var(--brand-silver) !important;
        font-size: 22px !important;
        font-weight: 900 !important;
        letter-spacing: 4px !important;
        text-transform: uppercase !important;
        line-height: 1.1;
    }
    .sidebar-brand-sub {
        color: #64748b !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    /* ── SURGICAL CAPSULE OVERRIDES ── */
    [data-testid='stSidebar'] [data-testid='stVerticalBlockBorderWrapper'],
    .agent-panel {
        background-color: var(--capsule-bg) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    .agent-panel-title {
        color: var(--capsule-muted) !important;
        font-weight: 700 !important;
        font-size: 10px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: 8px !important;
        border-bottom: 1px solid rgba(0,0,0,0.05) !important;
        padding-bottom: 4px !important;
        display: block !important;
    }

    [data-testid='stSidebar'] [data-testid='stVerticalBlockBorderWrapper'] label,
    [data-testid='stSidebar'] [data-testid='stVerticalBlockBorderWrapper'] p,
    [data-testid='stSidebar'] [data-testid='stVerticalBlockBorderWrapper'] span,
    .agent-panel label, .agent-panel p, .agent-panel span {
        color: var(--capsule-text) !important;
        font-size: 11px !important;
        font-weight: 600 !important;
    }

    .agent-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        padding: 3px 0;
        font-size: 11px !important;
    }
    
    .snapshot-value { color: var(--capsule-text) !important; font-size: 13px !important; font-weight: 800 !important; }

    /* ── BUTTON COMPACTION (32px) ── */
    [data-testid='stSidebar'] .stButton > button {
        background: var(--capsule-bg) !important;
        color: var(--capsule-text) !important;
        border: 1px solid rgba(29, 78, 216, 0.15) !important;
        font-weight: 700 !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        height: 32px !important;
        line-height: 32px !important;
        padding: 0 16px !important;
        margin-top: 8px !important;
        border-radius: 6px !important;
        width: 100% !important;
    }

    .agent-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; margin-right: 8px; flex-shrink: 0; }
    .agent-dot.active { background: #10b981; box-shadow: 0 0 6px #10b981; animation: heartbeat 2s ease-in-out infinite; }
    .agent-dot.inactive { background: #cbd5e1; }
    @keyframes heartbeat { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.3); opacity: 0.6; } }

    div[data-testid='stRadio'] label[data-baseweb='radio'] div:first-child { border-color: var(--brand-blue) !important; }
    div[data-testid='stRadio'] label[data-baseweb='radio'] [aria-checked='true'] + div { background-color: var(--brand-blue) !important; }
    .stTabs [aria-selected='true'] { color: var(--brand-blue) !important; border-bottom: 3px solid var(--brand-blue) !important; }

    /* Main Canvas Section Headers */
    .section-header {
        font-size: 13px !important;
        font-weight: 800 !important;
        color: var(--brand-blue) !important;
        background-color: var(--brand-silver-bright) !important;
        border-left: 5px solid var(--brand-blue) !important;
        padding: 12px 20px !important;
        margin: 24px 0 20px 0 !important;
        border-radius: 0 8px 8px 0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Anti-Dimming */
    [data-stale="true"] { opacity: 1 !important; filter: none !important; }

    /* Fixed Bottom Footer Bar */
    .inv-footer {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        z-index: 999999;
        height: 28px;
        background: var(--brand-silver-bright) !important;
        border-top: 1px solid var(--brand-blue) !important;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        font-size: 8px;
        color: var(--capsule-muted) !important;
        text-transform: uppercase;
        font-weight: 700 !important;
    }
    .inv-footer span { color: var(--capsule-muted) !important; }
    .inv-footer .sep { color: var(--brand-blue) !important; font-weight: 900; }
    
    /* ── Commentary Box ── */
    .commentary-box {
        border-left: 3px solid var(--brand-blue);
        padding: 16px 20px;
        background: var(--brand-silver-bright);
        border-radius: 6px;
        color: #0f172a;
        font-size: 14px;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    /* ── Main Metrics ── */
    .metric-card {
        background: #ffffff !important;
        border: 1px solid var(--brand-silver-bright) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        min-height: 120px !important;
    }
    .metric-label { color: #64748b !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase; }
    .metric-value { color: #0f172a !important; font-weight: 800 !important; font-size: 22px !important; }
    .pos { color: var(--success-green) !important; }
    .neg { color: var(--danger-red) !important; }
</style>
""", unsafe_allow_html=True)

# ── Ensure all state variables are present ────────────────────────────
for key, default in {
    "portfolio_loaded": False, "portfolio_state": None,
    "risk_state": None, "pca_state": None, "vol_regime_state": None,
    "stress_state": None, "greeks_state": None, "flow_signals": None,
    "ml_state": None, "rag_state": None, "pnl_state": None,
    "commentary_state": None, "eval_state": None, "filing_intel": None,
    "earnings_intel": None, "conviction_synthesis": None,
    "live_feed": True, "last_refresh_time": 0,
    "dev_authenticated": False, "_analytics_sid": None, "_last_tracked_page": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state._analytics_sid is None:
    st.session_state._analytics_sid = _analytics_create_session()

# ── UI Helpers ────────────────────────────────────────────────────────
def render_metric_card(label, value, delta_str=None, delta_val=None):
    status_class = "pos" if delta_val and delta_val > 0 else ("neg" if delta_val and delta_val < 0 else "")
    delta_html = f'<span class="metric-delta {status_class}" style="font-size:13px; font-weight:700; margin-left:8px;">{delta_str}</span>' if delta_str else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label.upper()}</div>'
        f'<div style="display:flex; align-items:baseline;">'
        f'<span class="metric-value">{value}</span>'
        f'{delta_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def apply_invictus_layout(fig, height=450, hovermode="closest", showlegend=False, title=None, margin=None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, sans-serif",
        font_color="#0f172a",
        hovermode=hovermode,
        height=height,
        title=title,
        showlegend=showlegend,
        margin=margin or dict(t=20, b=20, l=20, r=20),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(size=11, color="#64748b")),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(size=11, color="#64748b")),
        legend=dict(font=dict(size=11, color="#64748b"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1)
    )
    return fig

def fmt_currency(val, decimals=0):
    return f"${val:,.{decimals}f}" if val >= 0 else f"-${abs(val):,.{decimals}f}"

def render_football_field(ticker, details, prob):
    signals = [
        {"Signal": "Fundamentals", "Score": float(details.get("fundamentals", {}).get("score", 0))},
        {"Signal": "Technical Prob", "Score": (float(details.get("ml_accumulation", {}).get("score", 0)) - 0.5) * 2},
        {"Signal": "Inst. Context", "Score": float(details.get("institutional_flows", {}).get("score", 0))},
        {"Signal": "Risk Env.", "Score": -float(details.get("risk_environment", {}).get("score", 0))},
        {"Signal": "Analyst Pressure", "Score": -float(details.get("analyst_pressure", {}).get("score", 0))},
        {"Signal": "Mgmt Tone", "Score": float(details.get("management_tone", {}).get("score", 0))},
    ]
    df = pd.DataFrame(signals)
    fig = go.Figure(go.Bar(x=df["Score"], y=df["Signal"], orientation="h", marker=dict(color=["#ef4444" if v < 0 else "#10b981" for v in df["Score"]])))
    target = (prob - 0.5) * 2
    fig.add_vline(x=target, line_width=3, line_dash="dash", line_color="#1d4ed8")
    apply_invictus_layout(fig, height=350)
    fig.update_layout(xaxis=dict(range=[-1.1, 1.1], tickformat=".1f"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Sidebar ──
logo_path = Path(__file__).parent / "invictus" / "static" / "logo.png"
_logo_html = ""
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    _logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Invictus">'

with st.sidebar:
    st.markdown(f'<div class="sidebar-brand">{_logo_html}<div class="sidebar-brand-name">INVICTUS</div><div class="sidebar-brand-sub">Equity Portfolio Intelligence</div></div>', unsafe_allow_html=True)
    
    st.markdown("### Portfolio Input")
    upload_method = st.radio("Load portfolio from:", ["Default Portfolio", "Upload CSV"], index=0, label_visibility="collapsed")
    
    uploaded_file = None
    manual_mapping = None
    if upload_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            from invictus.data.smart_loader import analyze_csv_columns
            detected, confidence, all_cols = analyze_csv_columns(uploaded_file.getvalue().decode("utf-8"))
            with st.expander("Verify Data Mapping", expanded=(confidence < 0.9)):
                m_ticker = st.selectbox("Ticker", all_cols, index=all_cols.index(detected["Ticker"]) if detected["Ticker"] in all_cols else 0)
                m_shares = st.selectbox("Shares", all_cols, index=all_cols.index(detected["Shares"]) if detected["Shares"] in all_cols else 0)
                m_cost = st.selectbox("Cost Basis", ["None"] + all_cols, index=(all_cols.index(detected["CostBasis"])+1) if detected["CostBasis"] in all_cols else 0)
                manual_mapping = {"Ticker": m_ticker, "Shares": m_shares, "CostBasis": m_cost if m_cost != "None" else None}

    load_btn = st.button("Load Portfolio", use_container_width=True, type="primary")

    # Agents Panel
    agents = {
        "Risk Analytics": st.session_state.risk_state is not None,
        "PCA Factor": st.session_state.pca_state is not None,
        "Vol Regime": st.session_state.vol_regime_state is not None,
        "Stress Test": st.session_state.stress_state is not None,
        "Greeks": st.session_state.greeks_state is not None,
        "Inst. Flows": st.session_state.flow_signals is not None,
        "Filing Intel": st.session_state.filing_intel is not None,
        "Earnings Intel": st.session_state.earnings_intel is not None,
        "ML Engine": st.session_state.ml_state is not None,
        "Conviction": st.session_state.conviction_synthesis is not None,
    }
    active_count = sum(agents.values())
    agent_rows = "".join([f'<div class="agent-row"><span><div class="agent-dot {"active" if v else "inactive"}"></div>{k}</span><span style="font-size:9px;opacity:0.6;">{"LIVE" if v else "IDLE"}</span></div>' for k, v in agents.items()])
    st.markdown(f'<div class="agent-panel"><div class="agent-panel-title">Agents <span class="agent-panel-count">{active_count}/{len(agents)}</span></div>{agent_rows}</div>', unsafe_allow_html=True)

    if st.session_state.portfolio_loaded:
        ps = st.session_state.portfolio_state
        pnl_color = "#10b981" if ps["total_daily_pnl"] >= 0 else "#ef4444"
        st.markdown(f'<div class="agent-panel"><div class="agent-panel-title">Snapshot</div><div class="agent-row"><span>Net Worth</span><span class="snapshot-value">${ps["total_value"]:,.0f}</span></div><div class="agent-row"><span>Daily P&L</span><span style="color:{pnl_color};font-weight:700;">{ps["total_daily_pnl"]:,.0f}</span></div><div class="agent-row"><span>Positions</span><span style="font-weight:700;">{len(ps["summary"])}</span></div></div>', unsafe_allow_html=True)

# ── Tab Navigation ──
_tab_labels = ["Overview", "Portfolio Risk", "Predictive Intelligence", "Full Report"]
if st.query_params.get("dev") == "invictus": _tab_labels.append("Dev Console")
_tabs = st.tabs(_tab_labels)
_tab_contexts = {label: ctx for label, ctx in zip(_tab_labels, _tabs)}

# ── Pipeline ──
if load_btn:
    from invictus.data.portfolio_loader import load_portfolio_from_dict, fetch_price_history, compute_portfolio_state
    from invictus.data.smart_loader import smart_load_portfolio
    from invictus.agents.graph_state import PortfolioState as PState
    from invictus.agents.orchestrator import create_graph

    with st.spinner("Executing pipeline..."):
        try:
            if upload_method == "Upload CSV" and uploaded_file:
                holdings = smart_load_portfolio(uploaded_file.getvalue().decode("utf-8"), manual_mapping=manual_mapping)
            else:
                holdings = load_portfolio_from_dict()
            
            prices = fetch_price_history(holdings["Ticker"].tolist())
            state = compute_portfolio_state(holdings, prices)
            st.session_state.portfolio_state = state
            st.session_state.portfolio_loaded = True
            
            graph = create_graph()
            pstate = PState(holdings=state["holdings"], prices=state["prices"], returns=state["returns"], weights=state["weights"], total_value=state["total_value"])
            pstate = graph.run(pstate)
            
            # Map results back to session state
            st.session_state.risk_state = {"risk_metrics": pstate.risk_metrics, "correlation_matrix": pstate.correlation_matrix, "ticker_risk": pstate.ticker_risk}
            st.session_state.pca_state = pstate.pca_results
            st.session_state.vol_regime_state = pstate.vol_regime
            st.session_state.stress_state = pstate.stress_results
            st.session_state.greeks_state = pstate.greeks_results
            st.session_state.flow_signals = pstate.flow_signals
            st.session_state.ml_state = pstate.ml_predictions
            st.session_state.filing_intel = pstate.filing_intel
            st.session_state.earnings_intel = pstate.earnings_intel
            st.session_state.conviction_synthesis = pstate.conviction_synthesis
            st.session_state.commentary_state = pstate.commentary
            st.session_state.rag_state = pstate.rag_insights
            st.session_state.pnl_state = pstate.pnl_attribution
            st.rerun()
        except Exception as e: st.error(f"Execution Error: {e}")

# ── Overview ──
with _tab_contexts["Overview"]:
    if st.session_state.portfolio_loaded:
        s = st.session_state.portfolio_state
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: render_metric_card("Portfolio Value", fmt_currency(s["total_value"]))
        with m2: render_metric_card("Daily P&L", fmt_currency(s["total_daily_pnl"]), delta_val=s["total_daily_pnl"])
        with m3: render_metric_card("Unrealized P&L", fmt_currency(s["total_unrealized_pnl"]), delta_val=s["total_unrealized_pnl"])
        with m4: render_metric_card("Cost Basis", fmt_currency(s["total_cost"]))
        with m5: render_metric_card("Positions", str(len(s["summary"])))
        
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown('<div class="section-header">Allocation</div>', unsafe_allow_html=True)
            with st.container(border=True):
                df = s["summary"].sort_values("Market Value", ascending=False)
                fig = go.Figure(go.Pie(labels=df["Ticker"], values=df["Market Value"], hole=0.6, marker=dict(colors=px.colors.qualitative.Prism)))
                apply_invictus_layout(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="section-header">Relative Performance (Normalized to 100)</div>', unsafe_allow_html=True)
            with st.container(border=True):
                tickers = s["holdings"]["Ticker"].tolist()
                price_data = s["prices"][tickers]
                normalized = price_data / price_data.iloc[0] * 100
                fig_prices = go.Figure()
                colors_line = px.colors.qualitative.Plotly
                for i, ticker in enumerate(tickers):
                    fig_prices.add_trace(go.Scatter(
                        x=normalized.index, y=normalized[ticker], mode="lines", name=ticker,
                        line=dict(width=2, color=colors_line[i % len(colors_line)]),
                        hovertemplate=f"<b>{ticker}</b>: %{{y:.2f}}<extra></extra>"
                    ))
                apply_invictus_layout(fig_prices, height=500, showlegend=True)
                fig_prices.update_layout(yaxis=dict(side="right"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_prices, use_container_width=True, config={"displayModeBar": False})

        with c2:
            st.markdown('<div class="section-header">AI Commentary</div>', unsafe_allow_html=True)
            if st.session_state.commentary_state:
                cm = st.session_state.commentary_state
                ct1, ct2, ct3 = st.tabs(["Institutional PM", "Risk Manager", "Technicals"])
                with ct1: st.markdown(f'<div class="commentary-box">{cm.get("pm_summary", "")}</div>', unsafe_allow_html=True)
                with ct2: st.markdown(f'<div class="commentary-box">{cm.get("risk_manager", "")}</div>', unsafe_allow_html=True)
                with ct3: st.markdown(f'<div class="commentary-box">{cm.get("technical_summary", "")}</div>', unsafe_allow_html=True)
            else: st.info("Commentary results pending full pipeline run.")

# ── Risk ──
with _tab_contexts["Portfolio Risk"]:
    if st.session_state.risk_state:
        rt1, rt2, rt3, rt4, rt5, rt6 = st.tabs(["Analytics", "PCA", "Vol Regime", "Stress Test", "Greeks", "Attribution"])
        
        with rt1: # Analytics
            st.markdown('<div class="section-header">Core Risk Metrics</div>', unsafe_allow_html=True)
            rm = st.session_state.risk_state["risk_metrics"]
            c1, c2, c3, c4 = st.columns(4)
            with c1: render_metric_card("Ann. Volatility", f"{rm.get('annualized_volatility', 0):.1%}")
            with c2: render_metric_card("Sharpe Ratio", f"{rm.get('sharpe_ratio', 0):.2f}")
            with c3: render_metric_card("Max Drawdown", f"{rm.get('max_drawdown', 0):.1%}")
            with c4: render_metric_card("VAR (95%)", f"{rm.get('var_95_historical', 0):.2%}")
        
        with rt2:
            st.markdown('<div class="section-header">PCA Factor Decomposition</div>', unsafe_allow_html=True)
            if st.session_state.pca_state:
                pca = st.session_state.pca_state
                rc = {"HIGH": "#ff4b4b", "MODERATE": "#1d4ed8", "LOW": "#10b981"}[pca["concentration"]]
                st.markdown(f'<div style="background:#f8fafc; border-left:4px solid {rc}; padding:16px; border-radius:4px; margin-bottom:20px;"><span style="color:{rc}; font-weight:700; font-size:14px;">CONCENTRATION: {pca["concentration"]}</span><br/>{pca["assessment"]}</div>', unsafe_allow_html=True)
                c_scree, c_load = st.columns(2)
                with c_scree:
                    with st.container(border=True):
                        fig_s = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(len(pca["explained_variance"]))], y=pca["explained_variance"], marker_color="#1d4ed8"))
                        apply_invictus_layout(fig_s, height=400, title="Variance Explained")
                        st.plotly_chart(fig_s, use_container_width=True)
                with c_load:
                    with st.container(border=True):
                        loadings = pca["loadings"]
                        fig_l = go.Figure(go.Heatmap(z=loadings.values, x=loadings.columns.tolist(), y=loadings.index.tolist(), colorscale=[[0, "#ff4b4b"], [0.5, "#f8fafc"], [1, "#10b981"]]))
                        apply_invictus_layout(fig_l, height=400, title="Factor Loadings")
                        st.plotly_chart(fig_l, use_container_width=True)


        with rt3:
            st.markdown('<div class="section-header">Volatility Regime Detection</div>', unsafe_allow_html=True)
            if st.session_state.vol_regime_state:
                vr = st.session_state.vol_regime_state
                with st.container(border=True):
                    rc = {"Low": "#10b981", "Medium": "#1d4ed8", "High": "#ff4b4b"}[vr["current_regime"]]
                    st.markdown(f'<span style="color:{rc}; font-weight:700; font-size:20px;">CURRENT REGIME: {vr["current_regime"].upper()}</span><br/>Ann. Vol: {vr["current_vol"]:.1%} | Days in Regime: {vr["days_in_regime"]}', unsafe_allow_html=True)
                    
                    fig_v = go.Figure()
                    for n, c in {"Low": "#10b981", "Medium": "#1d4ed8", "High": "#ff4b4b"}.items():
                        mask = vr["regime_series"] == n
                        fig_v.add_trace(go.Scatter(x=vr["rolling_vol"].index, y=vr["rolling_vol"].where(mask), mode="lines", name=n, line=dict(color=c, width=2), connectgaps=False))
                    apply_invictus_layout(fig_v, height=400, showlegend=True, title="Rolling Volatility with Regime Overlay")
                    st.plotly_chart(fig_v, use_container_width=True)

        with rt4:
            st.markdown('<div class="section-header">Historical Shocks & Scenarios</div>', unsafe_allow_html=True)
            if st.session_state.stress_state:
                sr = st.session_state.stress_state
                sum_df = sr["summary"]
                with st.container(border=True):
                    fig_s = go.Figure(go.Bar(x=sum_df["Scenario"], y=sum_df["Portfolio Return"], marker_color=["#ff4b4b" if v < 0 else "#10b981" for v in sum_df["Portfolio Return"]]))
                    apply_invictus_layout(fig_s, height=400, title="Historical Scenario Impact")
                    st.plotly_chart(fig_s, use_container_width=True)
                    st.dataframe(sum_df.style.format({"Portfolio Return": "{:.2%}", "Portfolio P&L": "${:+,.0f}"}), use_container_width=True, hide_index=True)

        with rt5:
            st.markdown('<div class="section-header">Portfolio Greeks</div>', unsafe_allow_html=True)
            if st.session_state.greeks_state:
                gr = st.session_state.greeks_state
                c1, c2, c3, c4 = st.columns(4)
                with c1: render_metric_card("Delta", f"{gr.get('portfolio_greeks', {}).get('delta', 0):.2f}")
                with c2: render_metric_card("Gamma", f"{gr.get('portfolio_greeks', {}).get('gamma', 0):.4f}")
                with c3: render_metric_card("Vega", f"{gr.get('portfolio_greeks', {}).get('vega', 0):.2f}")
                with c4: render_metric_card("Theta", f"{gr.get('portfolio_greeks', {}).get('theta', 0):.2f}")

        with rt6:
            st.markdown('<div class="section-header">P&L Attribution (Alpha vs Beta)</div>', unsafe_allow_html=True)
            if st.session_state.pnl_state:
                pa = st.session_state.pnl_state
                c1, c2, c3 = st.columns(3)
                with c1: render_metric_card("Portfolio Return", f"{pa.get('portfolio_return', 0):+.3%}", delta_val=pa.get('portfolio_return', 0))
                with c2: render_metric_card("Single-Stock Alpha", f"{pa.get('single_stock_contribution', 0):+.3%}", delta_val=pa.get('single_stock_contribution', 0))
                with c3: render_metric_card("Macro/Factor Beta", f"{sum(pa.get('macro_contributions', {}).values()):+.3%}")

                with st.container(border=True):
                    tc = pa["ticker_contributions"].sort_values("Contribution")
                    fig_tc = go.Figure(go.Bar(x=tc["Contribution"], y=tc["Ticker"], orientation="h", marker_color=["#ff4b4b" if v < 0 else "#10b981" for v in tc["Contribution"]]))
                    apply_invictus_layout(fig_tc, height=450, title="Ticker-Level Attribution")
                    st.plotly_chart(fig_tc, use_container_width=True)

# ── Predictive ──
with _tab_contexts["Predictive Intelligence"]:
    if st.session_state.conviction_synthesis:
        sy = st.session_state.conviction_synthesis
        pt1, pt2, pt3, pt4 = st.tabs(["Summary", "10-K Sentiments", "Transcript Analysis", "Institutional Flows"])
        with pt1:
            st.markdown('<div class="section-header">Institutional Conviction Summary</div>', unsafe_allow_html=True)
            render_metric_card("Conviction Score", f"{sy.get('overall_portfolio_conviction', 0):.1%}")
            for t, res in sy["results"].items():
                with st.expander(f"{t} - {res['conviction_level']}"):
                    render_football_field(t, res["signals_detail"], res["outperformance_probability"])
        with pt2:
            if st.session_state.filing_intel:
                for t, d in st.session_state.filing_intel.items():
                    with st.expander(f"{t} - Filing Analysis"):
                        c1, c2, c3 = st.columns(3)
                        with c1: render_metric_card('Fundamentals', f"{d.get('fundamental_conviction', 0):+.2f}", delta_val=d.get('fundamental_conviction', 0))
                        with c2: render_metric_card('Guidance', f"{d.get('guidance_momentum', 0):+.2f}", delta_val=d.get('guidance_momentum', 0))
                        with c3: render_metric_card('Risk Deterioration', f"{d.get('risk_deterioration', 0):.2f}")
                        st.write(d.get("fundamental_reasoning", "N/A"))
        with pt3:
            if st.session_state.earnings_intel:
                for t, d in st.session_state.earnings_intel.items():
                    with st.expander(f"{t} - Management Tone"):
                        c1, c2 = st.columns(2)
                        with c1: render_metric_card('Confidence', f"{d.get('management_confidence', 0):+.2f}", delta_val=d.get('management_confidence', 0))
                        with c2: render_metric_card('Pressure', f"{d.get('analyst_pressure', 0):.2f}")
                        st.write(d.get("confidence_reasoning", "N/A"))
        with pt4:
            if st.session_state.flow_signals:
                st.markdown('<div class="section-header">Institutional Flow Aggregator</div>', unsafe_allow_html=True)
                rows = []
                for t, d in st.session_state.flow_signals.get("intel", {}).items():
                    rows.append({"Ticker": t, "Composite": d.get("flow_composite", 0), "Smart Money %": d.get("smart_money_pct", 0), "Insider Align": d.get("insider_alignment", 0)})
                st.dataframe(pd.DataFrame(rows).style.format({"Composite": "{:+.2f}", "Smart Money %": "{:.1%}", "Insider Align": "{:+.2f}"}), use_container_width=True, hide_index=True)


# ── Full Report ──
with _tab_contexts["Full Report"]:
    if st.session_state.conviction_synthesis:
        synth = st.session_state.conviction_synthesis
        st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
        render_metric_card("Portfolio Conviction", f"{synth.get('overall_portfolio_conviction', 0):.1%}")
        
        # Aggregated Commentary
        if st.session_state.commentary_state:
            cm = st.session_state.commentary_state
            st.markdown('<div class="section-header">Aggregated Commentary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="commentary-box">{cm.get("pm_summary", "Not available")}</div>', unsafe_allow_html=True)
        
        # Portfolio Inventory
        if st.session_state.portfolio_state:
            st.markdown('<div class="section-header">Portfolio Inventory</div>', unsafe_allow_html=True)
            state = st.session_state.portfolio_state
            fmt_cols = {"Cost Basis": "${:.2f}", "Current Price": "${:.2f}", "Market Value": "${:,.2f}", "Weight (%)": "{:.1f}%"}
            st.dataframe(state["summary"].style.format(fmt_cols, na_rep="—"), use_container_width=True, hide_index=True)
    else:
        st.info("Full institutional report is generated automatically on portfolio load.")


# ── Dev Console ──
if st.query_params.get("dev") == "invictus":
    with _tab_contexts["Dev Console"]:
        st.markdown('<div class="section-header">Developer Analytics</div>', unsafe_allow_html=True)
        stats = _analytics_get_summary_stats()
        k1, k2, k3 = st.columns(3)
        with k1: render_metric_card("Sessions", str(stats.get("total_sessions", 0)))
        with k2: render_metric_card("Views", str(stats.get("total_page_views", 0)))
        with k3: render_metric_card("Clicks", str(stats.get("total_clicks", 0)))

        st.markdown("### State Debugger")
        for key in ["filing_intel", "earnings_intel", "flow_signals", "conviction_synthesis"]:
            val = st.session_state.get(key)
            with st.expander(f"Debug: {key}"):
                st.write(val)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown('<div class="inv-footer"><span>© 2026 Invictus Portfolio Intelligence</span><span class="sep">|</span><span>Informational Only</span><span class="sep">|</span><span>Yahoo Finance & Finnhub</span></div>', unsafe_allow_html=True)

# ── Heartbeat ─────────────────────────────────────────────────────────
@st.fragment(run_every=60)
def heartbeat():
    if st.session_state.portfolio_loaded: st.rerun()
heartbeat()
