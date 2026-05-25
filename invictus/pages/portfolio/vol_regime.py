"""
Volatility Regimes — market environment classification.

Current regime summary, historical regime overlay, regime statistics.
"""
import streamlit as st
import plotly.graph_objects as go

from invictus.pages.portfolio._shared import (
    render_section_header, apply_invictus_layout, subtitle,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT,
)

_REGIME_COLORS = {"Low": SUCCESS_GREEN, "Medium": BRAND_BLUE, "High": DANGER_RED}
_REGIME_EXPLAIN = {
    "Low": "Stable conditions — positioning well-supported",
    "Medium": "Normal volatility — standard risk management applies",
    "High": "Elevated risk — consider reducing or hedging",
}


def render():
    """Render the Volatility Regimes sub-tab."""
    if not st.session_state.vol_regime_state:
        st.info("Load portfolio to view volatility regimes.")
        return

    vr = st.session_state.vol_regime_state

    # ── Current Regime (conviction card style) ────────────────────
    render_section_header("Current Regime")
    subtitle(
        'Hidden Markov Model classification of the volatility environment. '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Low</span> = calm, '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">Medium</span> = normal, '
        f'<span style="color:{DANGER_RED};font-weight:700;">High</span> = stressed. '
        '<span style="color:#94a3b8;">Regime shifts may warrant position sizing adjustments.</span>'
    )
    _rc = _REGIME_COLORS.get(vr["current_regime"], BRAND_BLUE)
    st.markdown(
        f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
        f'background:#fafbfc;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'margin-bottom:4px;">'
        f'<span style="font-size:16px;font-weight:800;color:#0f172a;">Volatility Regime</span>'
        f'<span style="font-size:20px;font-weight:800;color:{_rc};'
        f'font-variant-numeric:tabular-nums;">{vr["current_vol"]:.1%}</span>'
        f'</div>'
        f'<div style="font-size:12px;font-weight:700;color:{_rc};'
        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">'
        f'{vr["current_regime"]}</div>'
        f'<div style="font-size:12px;color:#64748b;margin-bottom:6px;'
        f'font-style:italic;line-height:1.4;">'
        f'{_REGIME_EXPLAIN.get(vr["current_regime"], "")}</div>'
        f'<div style="font-size:12px;color:#94a3b8;">'
        f'Days in regime: {vr["days_in_regime"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Regime Statistics (conviction card grid) ──────────────────
    regime_stats = vr.get("regime_stats")
    if regime_stats and isinstance(regime_stats, dict):
        render_section_header("Regime Statistics")
        subtitle(
            'Historical distribution across volatility regimes. '
            f'<span style="color:{DANGER_RED};font-weight:700;">Frequency</span> = time spent in regime; '
            f'<span style="color:{BRAND_BLUE};font-weight:700;">Avg Vol</span> = mean annualized volatility within that state. '
            '<span style="color:#94a3b8;">Based on full observation period.</span>'
        )
        rs_cols = st.columns(len(regime_stats))
        for i, (regime, stats) in enumerate(regime_stats.items()):
            rc = _REGIME_COLORS.get(regime, BRAND_BLUE)
            with rs_cols[i]:
                if isinstance(stats, dict):
                    _days = stats.get("days", "N/A")
                    _avg_vol = stats.get("mean_vol", 0)
                    _freq = stats.get("pct_time", 0)
                    st.markdown(
                        f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                        f'background:#fafbfc;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                        f'margin-bottom:4px;">'
                        f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{regime}</span>'
                        f'<span style="font-size:20px;font-weight:800;color:{rc};'
                        f'font-variant-numeric:tabular-nums;">{_avg_vol:.1%}</span>'
                        f'</div>'
                        f'<div style="font-size:12px;font-weight:700;color:{rc};'
                        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">'
                        f'Avg Volatility</div>'
                        f'<div style="font-size:12px;color:#64748b;margin-bottom:6px;'
                        f'font-style:italic;line-height:1.4;">'
                        f'{_freq:.0%} of observation period ({_days} days)</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Regime History Chart ──────────────────────────────────────
    render_section_header("Historical Regime Behavior")
    subtitle(
        'Rolling portfolio volatility color-coded by detected regime. '
        '<span style="color:#94a3b8;">Transitions between regimes often precede '
        'major drawdowns or recovery phases.</span>'
    )
    with st.container(border=True):
        fig_v = go.Figure()
        for n, c in {"Low": SUCCESS_GREEN, "Medium": BRAND_BLUE, "High": DANGER_RED_ALT}.items():
            mask = vr["regime_series"] == n
            fig_v.add_trace(go.Scatter(
                x=vr["rolling_vol"].index,
                y=vr["rolling_vol"].where(mask),
                mode="lines", name=n,
                line=dict(color=c, width=2), connectgaps=False,
            ))
        apply_invictus_layout(fig_v, height=350, showlegend=True,
                              title="Rolling Volatility with Regime Overlay")
        fig_v.update_layout(yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig_v, use_container_width=True, config={"displayModeBar": False})
