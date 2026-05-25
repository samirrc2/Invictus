"""
How Invictus Works — interactive navigation + content display.

Pure rendering logic. All content lives in _content.py.
Images live in invictus/static/landing/{SubTabName}/*.png (folder per tab, 1+ images).

Layout:
    Row 1  — Heading pills (Portfolio Intelligence / Conviction Intelligence / Allocation Engine)
    Row 2  — Sub-tab pills for the active heading
    Row 3  — Left: explanation text  |  Right: screenshot image(s)
"""
import streamlit as st
from pathlib import Path

from invictus.design import BRAND_BLUE
from invictus.pages.landing._content import SECTIONS
from invictus.pages.landing import hero as _hero


# ── Image root ──────────────────────────────────────────────────────────
_IMG_DIR = Path(__file__).resolve().parents[2] / "static" / "landing"


# ── Helpers ─────────────────────────────────────────────────────────────

def _find_section(key: str):
    """Return section dict by key, or None."""
    return next((s for s in SECTIONS if s["key"] == key), None)


def _find_sub(section: dict, name: str):
    """Return sub-tab dict by name within a section, or first sub."""
    match = next((s for s in section["subs"] if s["name"] == name), None)
    return match or section["subs"][0]


# ── Main Entry ──────────────────────────────────────────────────────────

def render():
    """Render the full How It Works page."""

    # Defaults — System Architecture tile shown first on landing
    if "landing_active" not in st.session_state:
        st.session_state.landing_active = "arch"
    if "landing_active_sub" not in st.session_state:
        st.session_state.landing_active_sub = ""

    # ── Sticky CSS for nav container ─────────────────────────────
    st.markdown(
        '<style>'
        'div[data-testid="stVerticalBlockBorderWrapper"]:has(.landing-nav-pin) {'
        '  position: sticky !important;'
        '  top: 0;'
        '  z-index: 99;'
        '  background: #ffffff;'
        '  padding: 8px 0 12px 0;'
        '  border-bottom: 1px solid #e2e8f0;'
        '}'
        '</style>',
        unsafe_allow_html=True,
    )

    # ── Sticky nav container ─────────────────────────────────────
    nav = st.container()
    with nav:
        st.markdown('<div class="landing-nav-pin"></div>', unsafe_allow_html=True)
        _render_headings()
        section = _find_section(st.session_state.landing_active)

        # Sub-tabs only for sections that have them (not "arch")
        if section and section.get("subs"):
            _render_sub_tabs(section)

    # ── Scrollable content below ─────────────────────────────────
    if section:
        if st.session_state.landing_active == "arch":
            # System Architecture — render hero.py sections
            _hero._render_topology()
            _hero._render_numbers()
            _hero._render_methodology()
            _hero._render_code_arch()
            _hero._render_eval()
        elif section.get("subs"):
            _render_content(section)


# ── Row 1: Heading pills ───────────────────────────────────────────────

def _render_headings():
    cols = st.columns(len(SECTIONS))
    for i, section in enumerate(SECTIONS):
        with cols[i]:
            active = st.session_state.landing_active == section["key"]
            if active:
                st.markdown(
                    f'<div style="background:{BRAND_BLUE};border-radius:6px;'
                    f'padding:8px 0;text-align:center;">'
                    f'<span style="font-size:13px;font-weight:700;color:#fff;">'
                    f'{section["title"]}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(section["title"], key=f"hw_{section['key']}",
                             use_container_width=True):
                    st.session_state.landing_active = section["key"]
                    first_sub = section["subs"][0]["name"] if section["subs"] else ""
                    st.session_state.landing_active_sub = first_sub
                    st.rerun()


# ── Row 2: Sub-tab pills ───────────────────────────────────────────────

def _render_sub_tabs(section: dict):
    subs = section["subs"]
    sub_cols = st.columns(len(subs))
    for j, sub in enumerate(subs):
        with sub_cols[j]:
            is_active = st.session_state.landing_active_sub == sub["name"]
            if is_active:
                st.markdown(
                    f'<div style="border:1px solid {BRAND_BLUE};border-radius:6px;'
                    f'padding:8px 6px;background:rgba(29,78,216,0.06);text-align:center;'
                    f'border-bottom:2px solid {BRAND_BLUE};cursor:default;">'
                    f'<span style="font-size:12px;font-weight:700;color:{BRAND_BLUE};">'
                    f'{sub["name"]}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(sub["name"], key=f"hw_sub_{sub['name']}",
                             use_container_width=True):
                    st.session_state.landing_active_sub = sub["name"]
                    st.rerun()


# ── Row 3: Content — per-image sections, left text / right image ───────

def _render_content(section: dict):
    sub = _find_sub(section, st.session_state.landing_active_sub)
    st.session_state.landing_active_sub = sub["name"]  # sync if fallback

    # Gather images (sorted alphabetically → maps 1:1 to sections list)
    img_folder = _IMG_DIR / sub["name"]
    images = sorted(img_folder.glob("*.png")) if img_folder.is_dir() else []

    content_sections = sub.get("sections", [])

    for idx, sec in enumerate(content_sections):
        st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)

        left, divider, right = st.columns([4, 0.1, 6])

        with left:
            # Section title
            st.markdown(
                f'<div style="font-size:15px;font-weight:800;color:#0f172a;'
                f'margin-bottom:14px;">{sec["title"]}</div>',
                unsafe_allow_html=True,
            )
            # Feature bullets — fill the vertical space
            for label, desc in sec["bullets"]:
                st.markdown(
                    f'<div style="margin-bottom:12px;line-height:1.6;">'
                    f'<span style="font-size:12px;font-weight:700;color:{BRAND_BLUE};">'
                    f'{label}</span>'
                    f'<span style="font-size:12px;color:#64748b;"> — {desc}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with divider:
            st.markdown(
                f'<div style="width:2px;background:linear-gradient(180deg,'
                f'{BRAND_BLUE} 0%,#cbd5e1 50%,{BRAND_BLUE} 100%);'
                f'min-height:300px;margin:0 auto;border-radius:1px;"></div>',
                unsafe_allow_html=True,
            )

        with right:
            if idx < len(images):
                st.image(str(images[idx]), use_container_width=True)
            else:
                st.markdown(
                    f'<div style="border:2px dashed #cbd5e1;border-radius:8px;'
                    f'padding:60px 20px;text-align:center;background:#f8fafc;">'
                    f'<div style="font-size:13px;font-weight:600;color:#94a3b8;'
                    f'margin-bottom:4px;">Screenshot Pending</div>'
                    f'<div style="font-size:11px;color:#cbd5e1;">'
                    f'Add image #{idx + 1} to: static/landing/{sub["name"]}/</div></div>',
                    unsafe_allow_html=True,
                )

        # Separator between sections (not after the last one)
        if idx < len(content_sections) - 1:
            st.markdown(
                '<hr style="border:none;border-top:1px solid #e2e8f0;margin:24px 0 0 0;">',
                unsafe_allow_html=True,
            )
