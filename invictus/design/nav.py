"""
invictus.design.nav
===================
Navigation bar — labels outside tiles, pills/buttons inside.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st


def render_nav(
    routes: "dict[str, Optional[list[str]]]",
    open_label: str = "Open",
) -> "tuple[str, Optional[str]]":
    """
    Navigation — labels outside tiles, pills/buttons inside tiles.

    Structure:
      Row 1 (labels):  PORTFOLIO ANALYTICS   PREDICTIVE INTEL   FULL REPORT
      Row 2 (cards):   [Dashboard][Analytics] [Summary][ML..]   [Open]
    """
    primary_labels = list(routes.keys())

    # Init state
    if "nav_primary" not in st.session_state:
        st.session_state.nav_primary = primary_labels[0]
    if "nav_sub" not in st.session_state:
        first_subs = routes.get(primary_labels[0])
        st.session_state.nav_sub = first_subs[0] if first_subs else None

    # Heal state if route names changed
    if st.session_state.nav_primary not in routes:
        st.session_state.nav_primary = primary_labels[0]
        first_subs = routes.get(primary_labels[0])
        st.session_state.nav_sub = first_subs[0] if first_subs else None
    else:
        subs_now = routes[st.session_state.nav_primary]
        if subs_now and st.session_state.nav_sub not in subs_now:
            st.session_state.nav_sub = subs_now[0]
        elif not subs_now:
            st.session_state.nav_sub = None

    primary = st.session_state.nav_primary
    sub = st.session_state.nav_sub

    nav = st.container()
    with nav:
        st.markdown('<div class="inv-nav-c-marker"></div>', unsafe_allow_html=True)

        # Row 1 — Primary routes as segmented control
        route_cols = st.columns(len(primary_labels))
        for rc, route in zip(route_cols, primary_labels):
            is_active = route == primary
            with rc:
                if st.button(
                    route,
                    key=f"nav__primary__{route}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.nav_primary = route
                    subs = routes[route]
                    st.session_state.nav_sub = subs[0] if subs else None
                    st.rerun()

        # Row 2 — Sub-tabs for the active route only
        active_subs = routes.get(primary)
        if active_subs:
            st.markdown('<div class="inv-nav-sub-marker"></div>', unsafe_allow_html=True)
            sub_cols = st.columns(len(active_subs))
            for sc, s in zip(sub_cols, active_subs):
                is_pill_active = s == sub
                with sc:
                    if st.button(
                        s,
                        key=f"nav__sub__{primary}__{s}",
                        type="primary" if is_pill_active else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state.nav_sub = s
                        st.rerun()

    return st.session_state.nav_primary, st.session_state.nav_sub
