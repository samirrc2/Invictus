"""
Shared helpers for Landing page sub-modules.
"""
import streamlit as st

from invictus.design import (
    render_section_header, apply_invictus_layout,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT,
    SLATE_50, SLATE_100, SLATE_500, SLATE_700,
)


def subtitle(text: str) -> None:
    """Render a grey subtitle explainer below a section header."""
    st.markdown(
        f'<div style="font-size:13px;color:#64748b;margin:-10px 0 16px 0;'
        f'line-height:1.6;">{text}</div>',
        unsafe_allow_html=True,
    )


def section_pill(label: str, color: str = BRAND_BLUE) -> str:
    """Return HTML for a small colored pill badge."""
    return (
        f'<span style="display:inline-block;font-size:10px;font-weight:700;'
        f'color:{color};background:{color}12;border:1px solid {color}30;'
        f'border-radius:4px;padding:2px 8px;letter-spacing:0.04em;'
        f'text-transform:uppercase;">{label}</span>'
    )
