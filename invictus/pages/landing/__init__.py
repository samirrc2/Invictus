"""
invictus.pages.landing
======================
Landing page — unified "How It Works" with 4 tiles:
  Portfolio Intelligence | Conviction Intelligence | System Architecture | Allocation Engine

Hero branding renders at top, then the tile navigator + content below.
System Architecture tile renders the topology, methodology, code arch, and eval sections.
"""
import streamlit as st

from invictus.pages.landing import hero, how_it_works


def render():
    """Entry point called by app.py."""
    # Hero branding only for the System Architecture tile
    if st.session_state.get("landing_active", "arch") == "arch":
        hero._render_hero()

    # Unified How It Works navigator
    how_it_works.render()
