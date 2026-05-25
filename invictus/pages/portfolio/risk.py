"""
Risk Analytics — structural risk deep-dive.

Risk metrics, MCTR, drawdown, distribution, correlation.
"""
import streamlit as st
import plotly.graph_objects as go

from invictus.pages.portfolio._shared import (
    render_section_header, render_metric_card, apply_invictus_layout,
    intelligence_callout, subtitle,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT, SLATE_500,
)


def render():
    """Render the Risk Analytics sub-tab."""
    if not st.session_state.risk_state or not st.session_state.risk_state.get("risk_metrics"):
        st.info("Load portfolio to view risk analytics.")
        return

    rm = st.session_state.risk_state["risk_metrics"]

    # ── Risk Metrics ──────────────────────────────────────────────
    render_section_header("Risk Metrics")
    subtitle(
        f'Core risk statistics computed from historical returns. '
        f'<span style="color:{DANGER_RED};font-weight:700;">VaR/CVaR</span> = downside tail risk, '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Sharpe/Sortino</span> = risk-adjusted return quality. '
        f'<span style="color:#94a3b8;">Based on daily returns over the observation period.</span>'
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: render_metric_card("Ann. Volatility", f"{rm.get('annualized_volatility', 0):.1%}")
    with c2: render_metric_card("VaR 95%", f"{rm.get('var_95_historical', 0):.2%}")
    with c3: render_metric_card("CVaR 95%", f"{rm.get('cvar_95', 0):.2%}")
    with c4: render_metric_card("Max Drawdown", f"{rm.get('max_drawdown', 0):.1%}")
    with c5: render_metric_card("Sharpe Ratio", f"{rm.get('sharpe_ratio', 0):.2f}")

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1: render_metric_card("Sortino Ratio", f"{rm.get('sortino_ratio', 0):.2f}")
    with r2: render_metric_card("Calmar Ratio", f"{rm.get('calmar_ratio', 0):.2f}")
    with r3: render_metric_card("Omega Ratio", f"{rm.get('omega_ratio', 0):.2f}")
    with r4: render_metric_card("Ann. Return", f"{rm.get('annualized_return', 0):+.1%}", delta_val=rm.get('annualized_return', 0))
    with r5: render_metric_card("HHI Concentration", f"{rm.get('hhi_concentration', 0):.3f}")

    # ── Downside Behavior ─────────────────────────────────────────
    render_section_header("Downside Behavior")
    subtitle(
        'Drawdown history and return distribution shape. '
        '<span style="color:#94a3b8;">Non-normal distributions (fat tails, negative skew) '
        'indicate VaR may understate true tail risk.</span>'
    )
    ch1, ch2 = st.columns(2)
    with ch1:
        dd_series = rm.get("drawdown_series")
        if dd_series is not None and hasattr(dd_series, 'index'):
            with st.container(border=True):
                fig_dd = go.Figure(go.Scatter(
                    x=dd_series.index, y=dd_series.values,
                    fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
                    line=dict(color=DANGER_RED, width=1.5),
                    hovertemplate="%{x|%b %d, %Y}: %{y:.2%}<extra></extra>",
                ))
                apply_invictus_layout(fig_dd, height=320, title="Drawdown Series")
                fig_dd.update_layout(yaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        port_rets = rm.get("portfolio_returns")
        if port_rets is not None and hasattr(port_rets, 'values'):
            with st.container(border=True):
                fig_hist = go.Figure(go.Histogram(
                    x=port_rets.values, nbinsx=60,
                    marker_color=BRAND_BLUE, opacity=0.75,
                ))
                var_val = rm.get("var_95_historical", 0)
                fig_hist.add_vline(x=var_val, line_dash="dash", line_color=DANGER_RED,
                                   annotation_text=f"VaR 95%: {var_val:.2%}",
                                   annotation_position="top right")
                var_99 = rm.get("var_99_historical", 0)
                if not var_99 and hasattr(port_rets, 'quantile'):
                    var_99 = float(port_rets.quantile(0.01))
                fig_hist.add_vline(x=var_99, line_dash="dot", line_color=DANGER_RED_ALT,
                                   annotation_text=f"VaR 99%: {var_99:.2%}",
                                   annotation_position="top left")
                apply_invictus_layout(fig_hist, height=320, title="Return Distribution")
                fig_hist.update_layout(xaxis=dict(tickformat=".1%"))
                st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # Distribution stats below charts
    dist_stats = rm.get("distribution_stats", {})
    if dist_stats:
        jb_p = dist_stats.get("jarque_bera_pval", 0)
        st.caption(
            f"Jarque-Bera p={jb_p:.4f} — "
            f"{'Normal' if dist_stats.get('is_normal') else 'Non-normal (fat tails/skew)'} | "
            f"Skew: {dist_stats.get('skewness', 0):.2f} | Kurt: {dist_stats.get('kurtosis', 0):.2f}"
        )

    # ── Correlation Matrix ────────────────────────────────────────
    corr = st.session_state.risk_state.get("correlation_matrix")
    if corr is not None and not corr.empty:
        render_section_header("Correlation Structure")
        subtitle(
            'Pairwise return correlations between holdings. '
            '<span style="color:#94a3b8;">Pairs above 0.7 amplify drawdowns; '
            'negative correlations provide natural hedging.</span>'
        )
        with st.container(border=True):
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale=[[0, DANGER_RED_ALT], [0.5, "#ffffff"], [1, BRAND_BLUE]],
                zmin=-1, zmax=1,
                text=corr.round(2).values, texttemplate="%{text}",
            ))
            apply_invictus_layout(fig_corr, height=max(350, len(corr) * 45), title="Pairwise Correlation")
            fig_corr.update_layout(margin=dict(t=30, b=10, l=80, r=10))
            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

        high_corr = []
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    high_corr.append((corr.index[i], corr.columns[j], val))
        if high_corr:
            high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
            pairs = ", ".join(f"{a}-{b} ({c:.2f})" for a, b, c in high_corr[:3])
            intelligence_callout(f"Highly correlated pairs: {pairs}. These positions amplify each other during drawdowns.")
