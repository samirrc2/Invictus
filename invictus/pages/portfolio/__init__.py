"""
invictus.pages.portfolio
========================
Portfolio Intelligence — 6 sub-tabs:
  Dashboard, Risk Analytics, Factor Decomposition, Volatility Regimes,
  Stress Scenarios, P&L Attribution.

Thin router dispatching to per-tab modules.
"""
from invictus.pages.portfolio import (
    dashboard, risk, pca, vol_regime, stress, attribution,
)

_ROUTES = {
    "Dashboard": dashboard,
    "Analytics": risk,
    "PCA": pca,
    "Vol Regime": vol_regime,
    "Stress Test": stress,
    "Attribution": attribution,
}


def render(sub: str):
    """Entry point called by app.py. `sub` is the internal route key."""
    module = _ROUTES.get(sub)
    if module:
        module.render()
    else:
        dashboard.render()
