"""
Sensitivity Analysis — Greeks, IV, Beta exposure.

Portfolio-level greeks, per-ticker sensitivity table, IV chart, beta exposure.
"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from invictus.pages.portfolio._shared import (
    render_section_header, render_metric_card, apply_invictus_layout,
    intelligence_callout, subtitle,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, SLATE_500,
)


def render():
    """Render the Sensitivity Analysis sub-tab."""
    if not st.session_state.greeks_state:
        st.info("Load portfolio to view sensitivity analysis.")
        return

    gr = st.session_state.greeks_state
    pg = gr.get("portfolio_greeks", {})

    # ── Market Sensitivities ──────────────────────────────────────
    render_section_header("Market Sensitivities")
    subtitle(
        'Portfolio-level option greeks aggregated from ATM implied volatility surface. '
        '<span style="color:#94a3b8;">Delta = directional exposure, Gamma = convexity, '
        'Vega = volatility sensitivity, Theta = time decay.</span>'
    )
    g1, g2, g3, g4, g5 = st.columns(5)
    with g1: render_metric_card("Portfolio Delta", f"{pg.get('delta', 0):.2f}")
    with g2: render_metric_card("Gamma", f"{pg.get('gamma', 0):.4f}")
    with g3: render_metric_card("Vega", f"{pg.get('vega', 0):.2f}")
    with g4: render_metric_card("Theta", f"{pg.get('theta', 0):.2f}")
    with g5: render_metric_card("Coverage", str(gr.get("coverage", "N/A")))

    # ── Per-Ticker Sensitivity Table ──────────────────────────────
    greeks_summary = gr.get("summary")
    if greeks_summary is not None and isinstance(greeks_summary, pd.DataFrame) and not greeks_summary.empty:
        render_section_header("Per-Ticker Sensitivity & Implied Volatility")
        subtitle(
            'Individual position greeks and implied volatility from the options market. '
            '<span style="color:#94a3b8;">High IV positions are pricing significant near-term movement.</span>'
        )
        greeks_fmt = {
            "IV": "{:.1%}", "Delta": "{:.3f}", "Gamma": "{:.4f}",
            "Vega": "{:.3f}", "Theta": "{:.3f}", "Rho": "{:.3f}",
            "Volga": "{:.4f}", "Skew": "{:.3f}",
        }
        active_gfmt = {k: v for k, v in greeks_fmt.items() if k in greeks_summary.columns}
        st.dataframe(greeks_summary.style.format(active_gfmt, na_rep="—"), use_container_width=True, hide_index=True)

        # IV Chart
        ok_greeks = greeks_summary[greeks_summary["Status"] == "OK"].copy()
        if not ok_greeks.empty and "IV" in ok_greeks.columns:
            render_section_header("Implied Volatility Exposure")
            subtitle(
                'Market-implied forward volatility for each holding. '
                '<span style="color:#94a3b8;">Higher IV = market expects larger price swings.</span>'
            )
            with st.container(border=True):
                ok_greeks = ok_greeks.sort_values("IV", ascending=True)
                fig_iv = go.Figure(go.Bar(
                    x=ok_greeks["IV"], y=ok_greeks["Ticker"], orientation="h",
                    marker_color=BRAND_BLUE,
                    text=[f"{v:.1%}" for v in ok_greeks["IV"]], textposition="outside",
                ))
                apply_invictus_layout(fig_iv, height=max(280, len(ok_greeks) * 36))
                fig_iv.update_layout(xaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig_iv, use_container_width=True, config={"displayModeBar": False})

            # Intelligence
            highest_iv = ok_greeks.nlargest(1, "IV").iloc[0]
            intelligence_callout(
                f'{highest_iv["Ticker"]} has the highest implied volatility at {highest_iv["IV"]:.1%} — '
                f'market is pricing significant near-term movement for this position.'
            )

    # ── Beta Exposure ─────────────────────────────────────────────
    if st.session_state.portfolio_loaded:
        summary = st.session_state.portfolio_state["summary"]
        if "Beta (vs SPY)" in summary.columns:
            render_section_header("Beta Exposure")
            subtitle(
                'Market sensitivity per holding relative to SPY. '
                '<span style="color:#94a3b8;">Beta &gt; 1.0 amplifies market moves; '
                '&lt; 1.0 provides defensiveness.</span>'
            )
            beta_data = summary[["Ticker", "Beta (vs SPY)", "Weight (%)"]].dropna(subset=["Beta (vs SPY)"])
            if not beta_data.empty:
                weighted_beta = (beta_data["Beta (vs SPY)"] * beta_data["Weight (%)"] / 100).sum()
                b1, b2 = st.columns([1, 2])
                with b1:
                    render_metric_card("Portfolio Beta", f"{weighted_beta:.2f}")
                with b2:
                    with st.container(border=True):
                        fig_beta = go.Figure(go.Bar(
                            x=beta_data["Ticker"], y=beta_data["Beta (vs SPY)"],
                            marker_color=[DANGER_RED if b > 1.2 else SUCCESS_GREEN if b < 0.8 else BRAND_BLUE
                                          for b in beta_data["Beta (vs SPY)"]],
                            text=[f"{v:.2f}" for v in beta_data["Beta (vs SPY)"]], textposition="outside",
                        ))
                        fig_beta.add_hline(y=1.0, line_dash="dot", line_color=SLATE_500, annotation_text="Market (1.0)")
                        apply_invictus_layout(fig_beta, height=280)
                        st.plotly_chart(fig_beta, use_container_width=True, config={"displayModeBar": False})
