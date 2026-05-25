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
