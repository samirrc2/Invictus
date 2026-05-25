"""
invictus.pages.dev_analytics._shared
=====================================
Reusable UI helpers for the Dev Analytics console.
"""
import streamlit as st

from invictus.design import (
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED,
)


# ── UI Helpers ──────────────────────────────────────────────────────────

def subtitle(text: str):
    """Render a colored subtitle under a section header."""
    st.markdown(
        f'<div style="font-size:13px;color:#64748b;margin:-10px 0 16px 0;'
        f'line-height:1.6;">{text}</div>',
        unsafe_allow_html=True,
    )


def conviction_card(label: str, value: str, color: str = "#0f172a",
                    sub_label: str = "", sub_color: str = "#94a3b8"):
    """Render a conviction-style metric card (16px/800 name, 20px/800 value, 12px/700 label)."""
    sub_html = ""
    if sub_label:
        sub_html = (
            f'<div style="font-size:12px;font-weight:700;color:{sub_color};'
            f'text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;">'
            f'{sub_label}</div>'
        )
    st.markdown(
        f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
        f'background:#fafbfc;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'margin-bottom:4px;">'
        f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{label}</span>'
        f'<span style="font-size:20px;font-weight:800;color:{color};'
        f'font-variant-numeric:tabular-nums;">{value}</span>'
        f'</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def health_color(val: float, good_thresh: float = 0.9, bad_thresh: float = 0.7) -> str:
    """Return color based on health thresholds."""
    if val >= good_thresh:
        return SUCCESS_GREEN
    if val < bad_thresh:
        return DANGER_RED
    return BRAND_BLUE
