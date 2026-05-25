"""
Stress Scenarios — vulnerability analysis.

Scenario overview cards, impact chart, per-scenario drill-down with ticker detail.
"""
import pandas as pd
import streamlit as st

from invictus.pages.portfolio._shared import (
    render_section_header, subtitle, fmt_currency,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT,
)


def _severity_color(ret: float) -> str:
    """Color based on scenario severity."""
    if ret < -0.20:
        return DANGER_RED
    if ret < -0.10:
        return DANGER_RED_ALT
    if ret < 0:
        return "#f59e0b"  # amber
    return SUCCESS_GREEN


def render():
    """Render the Stress Scenarios sub-tab."""
    if not st.session_state.stress_state:
        st.info("Load portfolio to view stress scenarios.")
        return

    sr = st.session_state.stress_state
    sum_df = sr["summary"]

    # ── Scenario Overview (conviction cards — all scenarios) ──────
    render_section_header("Scenario Overview")
    subtitle(
        'Portfolio impact under historical crisis scenarios. '
        f'<span style="color:{DANGER_RED};font-weight:700;">Severe</span> = drawdown &gt;20%, '
        f'<span style="color:{DANGER_RED_ALT};font-weight:700;">Moderate</span> = 10-20%, '
        f'<span style="color:#f59e0b;font-weight:700;">Mild</span> = &lt;10%. '
        '<span style="color:#94a3b8;">Projected using actual historical factor returns on current holdings.</span>'
    )

    if not sum_df.empty:
        # Render all scenarios in rows of 3
        n_scenarios = len(sum_df)
        for row_start in range(0, n_scenarios, 3):
            row_end = min(row_start + 3, n_scenarios)
            cols = st.columns(3)
            for i, idx in enumerate(range(row_start, row_end)):
                row = sum_df.iloc[idx]
                _name = str(row.get("Scenario", f"Scenario {idx+1}"))
                _ret = row.get("Portfolio Return", 0)
                _pnl = row.get("Portfolio P&L", 0)
                _stressed = row.get("Stressed Value", 0)
                _color = _severity_color(_ret)
                with cols[i]:
                    st.markdown(
                        f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                        f'background:#fafbfc;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                        f'margin-bottom:4px;">'
                        f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{_name}</span>'
                        f'<span style="font-size:20px;font-weight:800;color:{_color};'
                        f'font-variant-numeric:tabular-nums;">{_ret:+.1%}</span>'
                        f'</div>'
                        f'<div style="font-size:12px;font-weight:700;color:{_color};'
                        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">'
                        f'{"Severe" if _ret < -0.20 else "Moderate" if _ret < -0.10 else "Mild" if _ret < 0 else "Positive"}'
                        f'</div>'
                        f'<div style="font-size:12px;color:#64748b;margin-bottom:6px;'
                        f'font-style:italic;line-height:1.4;">'
                        f'P&L: {fmt_currency(_pnl)}</div>'
                        f'<div style="font-size:12px;color:#94a3b8;">'
                        f'Stressed Value: {fmt_currency(_stressed)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


    # ── Per-Scenario Drill-Down ───────────────────────────────────
    scenarios = sr.get("scenarios", {})
    if scenarios:
        render_section_header("Scenario Drill-Down")
        subtitle(
            'Per-ticker impact within each scenario. '
            f'<span style="color:{DANGER_RED};font-weight:700;">Vulnerable</span> = largest dollar losses; '
            f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Least Impacted</span> = smallest losses or gains. '
            '<span style="color:#94a3b8;">Expand any scenario for full ticker breakdown.</span>'
        )

    for scenario_name, scenario_data in scenarios.items():
        if "error" in scenario_data:
            continue
        td = scenario_data.get("ticker_detail")
        worst_t = scenario_data.get("worst_tickers", [])
        best_t = scenario_data.get("best_tickers", [])
        if td is not None and isinstance(td, pd.DataFrame) and not td.empty:
            # Compute scenario return for expander label
            _sc_ret = scenario_data.get("portfolio_return", 0)
            _label_color = "🔴" if _sc_ret < -0.10 else "🟡" if _sc_ret < 0 else "🟢"
            with st.expander(f"{_label_color} {scenario_name} ({_sc_ret:+.1%})"):
                tw1, tw2 = st.columns(2)
                with tw1:
                    if worst_t:
                        st.markdown(
                            f'<div style="font-size:12px;font-weight:800;color:{DANGER_RED};'
                            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">'
                            f'Most Vulnerable</div>',
                            unsafe_allow_html=True,
                        )
                        for item in worst_t:
                            _sr = item.get("Scenario Return", 0)
                            _pnl = item.get("P&L ($)", 0)
                            st.markdown(
                                f'<div style="border-left:3px solid {DANGER_RED};padding:6px 12px;'
                                f'margin-bottom:6px;font-size:12px;color:#334155;'
                                f'background:rgba(239,68,68,0.04);border-radius:0 4px 4px 0;">'
                                f'<span style="font-weight:700;">{item["Ticker"]}</span> '
                                f'<span style="color:{DANGER_RED};font-weight:800;">{_sr:+.1%}</span>'
                                f'<span style="color:#94a3b8;margin-left:8px;">'
                                f'{fmt_currency(_pnl)}</span></div>',
                                unsafe_allow_html=True,
                            )
                with tw2:
                    if best_t:
                        st.markdown(
                            f'<div style="font-size:12px;font-weight:800;color:{SUCCESS_GREEN};'
                            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">'
                            f'Least Impacted</div>',
                            unsafe_allow_html=True,
                        )
                        for item in best_t:
                            _sr = item.get("Scenario Return", 0)
                            _pnl = item.get("P&L ($)", 0)
                            st.markdown(
                                f'<div style="border-left:3px solid {SUCCESS_GREEN};padding:6px 12px;'
                                f'margin-bottom:6px;font-size:12px;color:#334155;'
                                f'background:rgba(34,197,94,0.04);border-radius:0 4px 4px 0;">'
                                f'<span style="font-weight:700;">{item["Ticker"]}</span> '
                                f'<span style="color:{SUCCESS_GREEN};font-weight:800;">{_sr:+.1%}</span>'
                                f'<span style="color:#94a3b8;margin-left:8px;">'
                                f'{fmt_currency(_pnl)}</span></div>',
                                unsafe_allow_html=True,
                            )
                td_fmt = {"Current Value": "${:,.0f}", "Scenario Return": "{:.2%}",
                          "P&L ($)": "${:+,.0f}", "Stressed Value": "${:,.0f}"}
                active_td = {k: v for k, v in td_fmt.items() if k in td.columns}
                st.dataframe(td.style.format(active_td, na_rep="—"),
                             use_container_width=True, hide_index=True, height=220)
