"""
Invictus — P&L Attribution Agent
Decomposes portfolio returns into ticker-level, sector, and macro contributions.
Computes cumulative contribution time series and rolling window attribution.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

from invictus.agents.graph_state import PortfolioState
from invictus.config import BENCHMARK_TICKER, TRADING_DAYS_PER_YEAR


def attribute_pnl(state: PortfolioState) -> PortfolioState:
    """Decompose P&L into ticker, sector/factor, and macro contributions.

    Outputs cumulative contribution time series, rolling window attribution,
    and weight-vs-contribution data for scatter analysis.
    """
    returns = state.returns
    weights = state.weights
    holdings = state.holdings
    prices = state.prices

    if returns is None or weights is None or prices is None:
        raise ValueError("Returns, weights, and prices required.")

    tickers = [t for t in weights if t in returns.columns]

    if not tickers or returns[tickers].dropna(how="all").empty:
        raise ValueError("No valid return data for any ticker in portfolio.")

    # ── Full-period ticker-level contribution ─────────────────────
    # Weight-adjusted daily contributions for every day in the period
    daily_contrib = returns[tickers].copy()
    for t in tickers:
        daily_contrib[t] = daily_contrib[t] * weights.get(t, 0)

    # Cumulative contribution time series (sum of daily weighted returns)
    cumulative_contrib = daily_contrib.cumsum()

    # Portfolio cumulative return
    portfolio_cum = cumulative_contrib.sum(axis=1)

    # Latest daily returns (for backward compat)
    if len(returns[tickers]) == 0:
        raise ValueError("Returns DataFrame is empty — no data to attribute.")
    latest_returns = returns[tickers].iloc[-1]

    # Total-period contribution per ticker (sum of all daily contributions)
    total_contrib = daily_contrib.sum()
    portfolio_return_total = total_contrib.sum()

    # Latest single-day contribution
    ticker_contrib_daily = {}
    for t in tickers:
        w = weights.get(t, 0)
        r = latest_returns[t] if t in latest_returns.index else 0
        ticker_contrib_daily[t] = w * r
    portfolio_return_daily = sum(ticker_contrib_daily.values())

    # ── Rolling window attribution (1W, 1M, 3M) ──────────────────
    window_days = {"1W": 5, "1M": 21, "3M": 63}
    rolling_attribution = {}
    for label, days in window_days.items():
        if len(returns) >= days:
            window_rets = returns[tickers].iloc[-days:]
            window_contrib = {}
            for t in tickers:
                window_contrib[t] = float((window_rets[t] * weights.get(t, 0)).sum())
            rolling_attribution[label] = {
                "ticker_contrib": window_contrib,
                "portfolio_return": sum(window_contrib.values()),
            }

    # ── Macro proxy decomposition ─────────────────────────────────
    macro_tickers = {"QQQ": "Tech/Growth", "SPY": "Broad Market", "VTI": "Total Market",
                     "GLDM": "Gold/Safe Haven", "SCHD": "Dividend/Value", "VGK": "International"}
    macro_contrib = {}
    for t, label in macro_tickers.items():
        if t in total_contrib.index:
            macro_contrib[label] = float(total_contrib[t])

    single_stock_contrib = float(sum(total_contrib[t] for t in tickers if t not in macro_tickers))

    # ── Sector grouping ───────────────────────────────────────────
    sector_map = {
        "Tech": ["AAPL", "META", "QQQ"], "Semiconductor": ["AMD", "SMH"],
        "EV/Growth": ["TSLA", "HIMS"], "Broad/Income": ["VTI", "SCHD", "VBK"],
        "Intl/Commodity": ["VGK", "GLDM"], "Defense": ["ITA"],
    }
    sector_contrib = {}
    for sector, tkrs in sector_map.items():
        sector_contrib[sector] = sum(float(total_contrib.get(t, 0)) for t in tkrs)

    # ── Volatility context ────────────────────────────────────────
    vol_regime = "Unknown"
    if state.vol_regime and "current_regime" in state.vol_regime:
        vol_regime = state.vol_regime["current_regime"]

    # ── Build contribution table (full period) ────────────────────
    def _safe_return(t):
        """Compute total return for ticker, returning 0 if data is empty."""
        if t not in prices.columns:
            return 0.0
        px = prices[t].dropna()
        if len(px) < 2 or px.iloc[0] == 0:
            return 0.0
        return float(px.iloc[-1] / px.iloc[0] - 1)

    contrib_df = pd.DataFrame([
        {"Ticker": t, "Weight": weights.get(t, 0),
         "Return": _safe_return(t),
         "Contribution": float(total_contrib[t])}
        for t in tickers
    ]).sort_values("Contribution", ascending=False)

    # Sector table
    sector_df = pd.DataFrame([
        {"Sector": s, "Contribution": c} for s, c in sector_contrib.items()
    ]).sort_values("Contribution", ascending=False)

    sorted_total = sorted(total_contrib.items(), key=lambda x: x[1], reverse=True)

    state.pnl_attribution = {
        "portfolio_return": portfolio_return_total,
        "portfolio_return_daily": portfolio_return_daily,
        "ticker_contributions": contrib_df,
        "sector_contributions": sector_df,
        "macro_contributions": macro_contrib,
        "single_stock_contribution": single_stock_contrib,
        "top_contributors": [{"ticker": t, "contribution": float(c)} for t, c in sorted_total[:3]],
        "bottom_contributors": [{"ticker": t, "contribution": float(c)} for t, c in sorted_total[-3:]],
        "vol_regime": vol_regime,
        # New rich data
        "cumulative_contrib": cumulative_contrib,       # DataFrame: date × ticker
        "portfolio_cumulative": portfolio_cum,           # Series: date → cumulative port return
        "rolling_attribution": rolling_attribution,      # dict: {1W/1M/3M → {ticker_contrib, portfolio_return}}
    }

    return state
