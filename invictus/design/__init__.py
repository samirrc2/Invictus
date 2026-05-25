"""
invictus.design
===============
Design system package — tokens, styles, components, charts, navigation.

All public symbols are re-exported here so existing code continues to work:
    from invictus.design import inject_styles, render_metric_card, BRAND_BLUE
"""

# ── Tokens ──
from invictus.design.tokens import (  # noqa: F401
    BRAND_NAVY_DEEP, BRAND_NAVY,
    BRAND_BLUE, BRAND_BLUE_HOVER, BRAND_BLUE_LIGHT, BRAND_BLUE_DEEP,
    BRAND_SILVER, BRAND_SILVER_BRIGHT,
    CAPSULE_BG, CAPSULE_TEXT, CAPSULE_MUTED,
    SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT, WARN_AMBER,
    SLATE_50, SLATE_100, SLATE_200, SLATE_300, SLATE_400,
    SLATE_500, SLATE_600, SLATE_700, SLATE_800, SLATE_900, SLATE_950,
    CHART_PALETTE,
)

# ── Styles ──
from invictus.design.styles import inject_styles  # noqa: F401

# ── Components ──
from invictus.design.components import (  # noqa: F401
    render_sidebar_brand,
    render_section_header,
    render_metric_card,
    render_commentary_box,
    render_concentration_banner,
    render_regime_banner,
    render_footer,
)

# ── Navigation (kept for backward compat, nav now in sidebar) ──
from invictus.design.nav import render_nav  # noqa: F401

# ── Charts ──
from invictus.design.charts import apply_invictus_layout  # noqa: F401

# ── Formatters ──
from invictus.design.formatters import (  # noqa: F401
    fmt_currency,
    fmt_pct,
    fmt_signed_currency,
)
