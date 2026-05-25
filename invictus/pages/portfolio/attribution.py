"""
P&L Attribution — return decomposition.

Return decomposition cards, cumulative contribution chart, weight-vs-contribution
scatter, rolling window attribution, security-level bar chart.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from invictus.pages.portfolio._shared import (
    render_section_header, apply_invictus_layout, subtitle,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT,
)


def _return_color(val: float) -> str:
    """Color based on return sign/magnitude."""
    if val > 0.001:
        return SUCCESS_GREEN
    if val < -0.001:
        return DANGER_RED
    return "#94a3b8"


def render():
    """Render the P&L Attribution sub-tab."""
    if not st.session_state.pnl_state:
        st.info("Load portfolio to view P&L attribution.")
        return

    pa = st.session_state.pnl_state

    # ── Return Decomposition (conviction cards) ───────────────────
    render_section_header("Return Decomposition")
    subtitle(
        'Full-period return split into '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">alpha</span> (stock selection) and '
        f'<span style="color:#6366f1;font-weight:700;">beta</span> (systematic exposure). '
        '<span style="color:#94a3b8;">Alpha = idiosyncratic; Beta = market/sector/style.</span>'
    )
    _port_ret = pa.get('portfolio_return', 0)
    _alpha = pa.get('single_stock_contribution', 0)
    _beta = sum(pa.get('macro_contributions', {}).values())

    c1, c2, c3 = st.columns(3)
    _items = [
        ("Portfolio Return", _port_ret, _return_color(_port_ret)),
        ("Single-Stock Alpha", _alpha, _return_color(_alpha)),
        ("Macro/Factor Beta", _beta, _return_color(_beta)),
    ]
    for col, (label, val, color) in zip([c1, c2, c3], _items):
        with col:
            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                f'background:#fafbfc;">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                f'margin-bottom:4px;">'
                f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{label}</span>'
                f'<span style="font-size:20px;font-weight:800;color:{color};'
                f'font-variant-numeric:tabular-nums;">{val:+.3%}</span>'
                f'</div>'
                f'<div style="font-size:12px;font-weight:700;color:{color};'
                f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">'
                f'{"Gain" if val > 0.001 else "Loss" if val < -0.001 else "Flat"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Cumulative Contribution Chart ─────────────────────────────
    cum_contrib = pa.get("cumulative_contrib")
    if cum_contrib is not None and not cum_contrib.empty:
        render_section_header("Cumulative Contribution")
        subtitle(
            'How each position\'s contribution to portfolio return evolved over time. '
            f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Rising</span> = adding value; '
            f'<span style="color:{DANGER_RED};font-weight:700;">falling</span> = dragging performance. '
            '<span style="color:#94a3b8;">Reveals who drove returns and when.</span>'
        )
        with st.container(border=True):
            fig_cum = go.Figure()
            colors_line = px.colors.qualitative.Plotly
            # Sort by final contribution to show most impactful on top
            final_order = cum_contrib.iloc[-1].sort_values(ascending=False).index
            for i, ticker in enumerate(final_order):
                fig_cum.add_trace(go.Scatter(
                    x=cum_contrib.index, y=cum_contrib[ticker],
                    mode="lines", name=ticker,
                    line=dict(width=2, color=colors_line[i % len(colors_line)]),
                    hovertemplate=f"{ticker}: %{{y:+.3%}}<extra></extra>",
                ))
            # Portfolio total as bold dashed line
            port_cum = pa.get("portfolio_cumulative")
            if port_cum is not None:
                fig_cum.add_trace(go.Scatter(
                    x=port_cum.index, y=port_cum.values,
                    mode="lines", name="Portfolio",
                    line=dict(width=2.5, color="#0f172a", dash="dot"),
                    hovertemplate="Portfolio: %{y:+.3%}<extra></extra>",
                ))
            apply_invictus_layout(fig_cum, height=380, showlegend=True,
                                  title="Cumulative Attribution by Position")
            fig_cum.update_layout(
                yaxis=dict(tickformat=".1%", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=30, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": False})

    # ── Weight vs Contribution Scatter ────────────────────────────
    tc = pa["ticker_contributions"]
    if "Weight" in tc.columns and "Contribution" in tc.columns:
        render_section_header("Alpha Efficiency")
        subtitle(
            'Weight vs contribution — positions '
            f'<span style="color:{SUCCESS_GREEN};font-weight:700;">above</span> the diagonal '
            'outperform their weight allocation; '
            f'<span style="color:{DANGER_RED};font-weight:700;">below</span> = underperforming weight. '
            '<span style="color:#94a3b8;">Distance from diagonal = alpha magnitude.</span>'
        )
        with st.container(border=True):
            fig_sc = go.Figure()
            # Diagonal reference line (weight = contribution → no alpha)
            w_range = [0, max(tc["Weight"].max(), 0.4)]
            fig_sc.add_trace(go.Scatter(
                x=w_range, y=w_range, mode="lines", name="No Alpha Line",
                line=dict(color="#e2e8f0", width=1, dash="dash"),
                showlegend=False,
            ))
            # Color by alpha sign
            colors = [SUCCESS_GREEN if r > w else DANGER_RED
                      for r, w in zip(tc["Contribution"], tc["Weight"] * tc["Return"])]
            fig_sc.add_trace(go.Scatter(
                x=tc["Weight"], y=tc["Contribution"],
                mode="markers+text", name="Holdings",
                marker=dict(
                    size=14,
                    color=[_return_color(c) for c in tc["Contribution"]],
                    line=dict(width=1, color="#e2e8f0"),
                ),
                text=tc["Ticker"],
                textposition="top center",
                textfont=dict(size=11, color="#334155"),
                hovertemplate="%{text}<br>Weight: %{x:.1%}<br>Contribution: %{y:+.3%}<extra></extra>",
            ))
            apply_invictus_layout(fig_sc, height=380, title="Weight vs Contribution")
            fig_sc.update_layout(
                xaxis=dict(title="Portfolio Weight", tickformat=".0%"),
                yaxis=dict(title="Contribution to Return", tickformat=".2%", side="right"),
                margin=dict(t=30, b=40, l=20, r=20),
            )
            st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    # ── Rolling Window Attribution ────────────────────────────────
    rolling = pa.get("rolling_attribution", {})
    if rolling:
        render_section_header("Rolling Attribution")
        subtitle(
            'Attribution across different lookback windows — '
            f'<span style="color:{BRAND_BLUE};font-weight:700;">1W</span> = recent momentum, '
            f'<span style="color:#6366f1;font-weight:700;">1M</span> = medium-term, '
            f'<span style="color:#94a3b8;font-weight:700;">3M</span> = trend. '
            '<span style="color:#94a3b8;">Compare to see if alpha is recent or stale.</span>'
        )
        window_cols = st.columns(len(rolling))
        _window_colors = {"1W": BRAND_BLUE, "1M": "#6366f1", "3M": "#94a3b8"}
        for i, (window, data) in enumerate(rolling.items()):
            _wc = _window_colors.get(window, BRAND_BLUE)
            _wret = data.get("portfolio_return", 0)
            _wcolor = _return_color(_wret)
            with window_cols[i]:
                st.markdown(
                    f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                    f'background:#fafbfc;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                    f'margin-bottom:4px;">'
                    f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{window}</span>'
                    f'<span style="font-size:20px;font-weight:800;color:{_wcolor};'
                    f'font-variant-numeric:tabular-nums;">{_wret:+.3%}</span>'
                    f'</div>'
                    f'<div style="font-size:12px;font-weight:700;color:{_wcolor};'
                    f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">'
                    f'{"Gain" if _wret > 0.001 else "Loss" if _wret < -0.001 else "Flat"}</div>',
                    unsafe_allow_html=True,
                )
                # Top 3 contributors for this window
                tc_window = data.get("ticker_contrib", {})
                sorted_w = sorted(tc_window.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                drivers_html = ""
                for t, c in sorted_w:
                    _tc = _return_color(c)
                    drivers_html += (
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:3px 0;border-bottom:1px solid #f1f5f9;">'
                        f'<span style="font-size:12px;color:#334155;font-weight:600;">{t}</span>'
                        f'<span style="font-size:12px;font-weight:700;color:{_tc};'
                        f'font-variant-numeric:tabular-nums;">{c:+.3%}</span></div>'
                    )
                st.markdown(
                    f'{drivers_html}</div>',
                    unsafe_allow_html=True,
                )

