"""
Portfolio Dashboard — Overview tab.

Portfolio snapshot, exposure summary, relative performance, top movers.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from invictus.pages.portfolio._shared import (
    render_section_header, render_metric_card, apply_invictus_layout,
    fmt_currency, subtitle,
    BRAND_BLUE,
)


def render():
    """Render the Portfolio Dashboard sub-tab."""
    if not st.session_state.portfolio_loaded:
        st.info("Select a portfolio source and click Load Portfolio or try Demo Mode to begin.")
        return

    s = st.session_state.portfolio_state
    risk_s = st.session_state.risk_state
    summary = s["summary"]

    # ── Portfolio Snapshot ─────────────────────────────────────────
    render_section_header("Portfolio Snapshot")
    subtitle(
        'Real-time portfolio valuation, daily P&amp;L, and benchmark comparison. '
        '<span style="color:#94a3b8;">Updated on each portfolio load.</span>'
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: render_metric_card("Total Value", fmt_currency(s["total_value"]))
    with m2: render_metric_card("Daily P&L", fmt_currency(s["total_daily_pnl"]), delta_val=s["total_daily_pnl"])
    with m3: render_metric_card("Total Return", f"{s.get('unrealized_pnl_pct', 0):+.1f}%", delta_val=s.get("unrealized_pnl_pct", 0))
    with m4:
        bench_ret = 0
        if "prices" in s and "SPY" in s["prices"].columns:
            spy = s["prices"]["SPY"]
            bench_ret = (spy.iloc[-1] / spy.iloc[0] - 1) * 100
        render_metric_card("SPY (Benchmark)", f"{bench_ret:+.1f}%", delta_val=bench_ret)
    with m5: render_metric_card("Positions", str(len(summary)))

    # Health indicators row
    h1, h2, h3, h4, h5 = st.columns(5)
    if risk_s:
        rm = risk_s["risk_metrics"]
        vol = rm.get("annualized_volatility", 0)
        dd = abs(rm.get("max_drawdown", 0))
        risk_score = min(100, int(vol * 200 + dd * 100))
        with h1: render_metric_card("Risk Score", f"{risk_score}/100")

        hhi = rm.get("hhi_concentration", 0.5)
        div_score = max(0, min(100, int((1 - hhi) * 120)))
        with h2: render_metric_card("Diversification", f"{div_score}/100")

        top_weight = summary["Weight (%)"].max()
        conc_level = "LOW" if top_weight < 20 else "MODERATE" if top_weight < 35 else "HIGH"
        with h3: render_metric_card("Concentration", conc_level)

        sharpe = rm.get("sharpe_ratio", 0)
        with h4: render_metric_card("Sharpe Ratio", f"{sharpe:.2f}")
        with h5: render_metric_card("Ann. Volatility", f"{vol:.1%}")

    # ── Exposure Summary ──────────────────────────────────────────
    render_section_header("Exposure Summary")
    subtitle(
        'Complete position-level detail with cost basis, P&amp;L, risk metrics, and weight distribution.'
    )

    # Portfolio Allocation — stacked bar (same style as Shareholding Structure in Capital Flows)
    df_alloc = summary.sort_values("Weight (%)", ascending=False)
    _alloc_colors = px.colors.qualitative.Prism
    segments_html = ""
    legend_html = ""
    for i, (_, row) in enumerate(df_alloc.iterrows()):
        pct = row["Weight (%)"]
        color = _alloc_colors[i % len(_alloc_colors)]
        segments_html += f'<div style="width:{pct:.1f}%;background:{color};"></div>'
        legend_html += f'<span style="color:{color};font-weight:600;">■ {row["Ticker"]} {pct:.1f}%</span>'
    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<div style="font-size:12px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
        f'text-transform:uppercase;margin-bottom:6px;">Portfolio Allocation</div>'
        f'<div style="display:flex;height:18px;border-radius:4px;overflow:hidden;'
        f'margin-bottom:6px;border:1px solid #e2e8f0;">{segments_html}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px 16px;font-size:12px;">'
        f'{legend_html}</div></div>',
        unsafe_allow_html=True,
    )

    # Holdings table — full width with all columns
    fmt_cols = {
        "Cost Basis": "${:.2f}", "Current Price": "${:.2f}",
        "Market Value": "${:,.0f}", "Weight (%)": "{:.1f}%",
        "Daily P&L ($)": "${:+,.0f}",
        "Unrealized P&L ($)": "${:+,.0f}", "Unrealized P&L (%)": "{:+.1f}%",
        "Ann. Volatility": "{:.1%}", "Max Drawdown": "{:.1%}",
        "Beta (vs SPY)": "{:.2f}",
    }
    active_fmt = {k: v for k, v in fmt_cols.items() if k in summary.columns}
    st.dataframe(
        summary.style.format(active_fmt, na_rep="—"),
        use_container_width=True, hide_index=True,
        height=220,
    )

    # ── Relative Performance ──────────────────────────────────────
    render_section_header("Relative Performance (Normalized)")
    subtitle(
        'All holdings normalized to 100 at period start for direct comparison. '
        '<span style="color:#94a3b8;">Higher values indicate outperformance relative to entry.</span>'
    )
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
            ))
        # Add SPY benchmark
        if "SPY" in s["prices"].columns:
            spy_norm = s["prices"]["SPY"] / s["prices"]["SPY"].iloc[0] * 100
            fig_prices.add_trace(go.Scatter(
                x=spy_norm.index, y=spy_norm, mode="lines", name="SPY (Benchmark)",
                line=dict(width=2.5, color="#6b7280", dash="dot"),
            ))
        apply_invictus_layout(fig_prices, height=350, showlegend=True)
        fig_prices.update_layout(
            yaxis=dict(side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_prices, use_container_width=True, config={"displayModeBar": False})
