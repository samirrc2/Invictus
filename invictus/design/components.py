"""
invictus.design.components
==========================
Reusable UI components — sidebar brand, section headers, metric cards,
commentary boxes, banners, footer.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import streamlit as st

from invictus.design.tokens import (
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT,
)


def render_sidebar_brand(logo_path: Optional[Path] = None) -> None:
    """Logo + INVICTUS wordmark + tagline. Pass the file path; we base64-encode it."""
    logo_html = ""
    if logo_path and Path(logo_path).exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Invictus">'
    st.markdown(
        f'<div class="sidebar-brand">'
        f'{logo_html}'
        f'<div class="sidebar-brand-name">INVICTUS</div>'
        f'<div class="sidebar-brand-sub">Equity Portfolio Intelligence</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_section_header(text: str) -> None:
    """The signature blue-rail / slate-50 section divider."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    delta_str: Optional[str] = None,
    delta_val: Optional[float] = None,
) -> None:
    """
    Labeled metric tile.

    Coloring rules:
      * `delta_val` set + `delta_str` set → the delta TEXT is colored.
      * `delta_val` set + no `delta_str`  → the VALUE itself is colored.
      * neither                           → value renders in default fg.
    """
    status_class = ""
    if delta_val is not None:
        if delta_val > 0:
            status_class = "pos"
        elif delta_val < 0:
            status_class = "neg"

    value_class = "metric-value"
    delta_html = ""
    if delta_str:
        delta_html = (
            f'<span class="metric-delta {status_class}" style="margin-left:8px;">'
            f'{delta_str}</span>'
        )
    elif status_class:
        value_class = f"metric-value {status_class}"

    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label.upper()}</div>'
        f'<div style="display:flex; align-items:baseline;">'
        f'<span class="{value_class}">{value}</span>'
        f'{delta_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_commentary_box(text: str) -> None:
    """3px-rail commentary block. Accepts plain text or pre-wrap content."""
    st.markdown(f'<div class="commentary-box">{text}</div>', unsafe_allow_html=True)


def render_concentration_banner(
    level: str,
    body: str,
    color: Optional[str] = None,
) -> None:
    """
    Severity banner used by PCA / regime / stress sections.
    `level` is one of HIGH / MODERATE / LOW.
    """
    palette = {"HIGH": DANGER_RED, "MODERATE": BRAND_BLUE, "LOW": SUCCESS_GREEN}
    c = color or palette.get(level.upper(), BRAND_BLUE)
    st.markdown(
        f'<div class="conc-banner" style="border-left:4px solid {c};">'
        f'<span class="lvl" style="color:{c};">CONCENTRATION: {level.upper()}</span>'
        f'<div class="body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_regime_banner(regime: str, body: str) -> None:
    """Volatility-regime banner (Low / Medium / High)."""
    palette = {"LOW": SUCCESS_GREEN, "MEDIUM": BRAND_BLUE, "HIGH": DANGER_RED_ALT}
    c = palette.get(regime.upper(), BRAND_BLUE)
    st.markdown(
        f'<span style="color:{c}; font-weight:700; font-size:20px;">'
        f'CURRENT REGIME: {regime.upper()}'
        f'</span><br/>{body}',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """
    Backwards-compatible no-op.

    The footer is now mounted via JS inside `inject_styles()` so that it
    persists across Streamlit reruns.
    """
    return
