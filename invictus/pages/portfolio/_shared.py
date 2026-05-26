"""
Shared helpers for Portfolio Intelligence sub-tabs.
Intelligence callout, common imports re-exported.
"""
import streamlit as st

from invictus.design import (
    render_section_header, render_metric_card, apply_invictus_layout,
    render_concentration_banner, fmt_currency,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT,
    SLATE_50, SLATE_100, SLATE_500, SLATE_700,
)


def intelligence_callout(text: str) -> None:
    """Render a key intelligence signal callout — blue-left-border box."""
    st.markdown(
        f'<div style="border-left:3px solid {BRAND_BLUE};background:rgba(29,78,216,0.04);'
        f'padding:12px 16px;border-radius:0 6px 6px 0;margin:12px 0;'
        f'font-size:13px;color:#1e293b;line-height:1.7;font-style:italic;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def subtitle(text: str) -> None:
    """Render a grey subtitle explainer below a section header."""
    st.markdown(
        f'<div style="font-size:12px;color:#64748b;margin:-4px 0 12px 0;line-height:1.5;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def data_disclosure() -> None:
    """
    Render a compact data-period disclosure strip at the top of risk pages.
    Industry standard (Bloomberg PORT, MSCI RiskMetrics, Aladdin) — always
    disclose the observation window, data frequency, and standard disclaimer.
    """
    from invictus.config import LOOKBACK_DAYS, TRADING_DAYS_PER_YEAR, VAR_CONFIDENCE
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%b %d, %Y")
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%b %d, %Y")

    st.markdown(
        f'<div style="font-size:11px;color:#94a3b8;border-left:2px solid #334155;'
        f'padding:6px 12px;margin:0 0 16px 0;line-height:1.5;'
        f'background:rgba(51,65,85,0.03);border-radius:0 4px 4px 0;">'
        f'<span style="color:#64748b;font-weight:600;">DATA DISCLOSURE</span>&ensp;|&ensp;'
        f'All risk metrics computed from <b>daily returns</b> over a '
        f'<b>{LOOKBACK_DAYS}-day</b> observation window '
        f'({start_date} – {end_date}, ~{TRADING_DAYS_PER_YEAR} trading days). '
        f'VaR/CVaR at {VAR_CONFIDENCE:.0%} confidence. '
        f'Past performance is not indicative of future results.'
        f'</div>',
        unsafe_allow_html=True,
    )
