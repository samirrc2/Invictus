"""
Invictus — P&L Attribution Agent
Decomposes daily portfolio move into contributing factors.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

from invictus.agents.graph_state import PortfolioState
from invictus.config import BENCHMARK_TICKER, TRADING_DAYS_PER_YEAR


def attribute_pnl(state: PortfolioState) -> PortfolioState:
    """Decompose daily P&L into ticker, sector/factor, and macro contributions."""
    returns = state.returns
    weights = state.weights
    holdings = state.holdings
    prices = state.prices

    if returns is None or weights is None or prices is None:
        raise ValueError("Returns, weights, and prices required.")

    tickers = [t for t in weights if t in returns.columns]

    # Latest daily returns
    latest_returns = returns[tickers].iloc[-1]
    prev_returns = returns[tickers].iloc[-2] if len(returns) > 2 else latest_returns * 0

    # Ticker-level contribution to portfolio return
    ticker_contrib = {}
    for t in tickers:
        w = weights.get(t, 0)
        r = latest_returns[t] if t in latest_returns.index else 0
        ticker_contrib[t] = w * r

    portfolio_return = sum(ticker_contrib.values())

    # Sort by contribution
    sorted_contrib = sorted(ticker_contrib.items(), key=lambda x: x[1], reverse=True)
    top_contributors = sorted_contrib[:3]
    bottom_contributors = sorted_contrib[-3:]

    # Macro proxy decomposition
    macro_tickers = {"QQQ": "Tech/Growth", "SPY": "Broad Market", "VTI": "Total Market",
                     "GLDM": "Gold/Safe Haven", "SCHD": "Dividend/Value", "VGK": "International"}
    macro_contrib = {}
    for t, label in macro_tickers.items():
        if t in ticker_contrib:
            macro_contrib[label] = ticker_contrib[t]

    single_stock_contrib = sum(v for t, v in ticker_contrib.items() if t not in macro_tickers)

    # Sector grouping (simplified)
    sector_map = {
        "Tech": ["AAPL", "META", "QQQ"], "Semiconductor": ["AMD", "SMH"],
        "EV/Growth": ["TSLA", "HIMS"], "Broad/Income": ["VTI", "SCHD", "VBK"],
        "Intl/Commodity": ["VGK", "GLDM"], "Defense": ["ITA"],
    }
    sector_contrib = {}
    for sector, tkrs in sector_map.items():
        sector_contrib[sector] = sum(ticker_contrib.get(t, 0) for t in tkrs)

    # Volatility context
    vol_regime = "Unknown"
    if state.vol_regime and "current_regime" in state.vol_regime:
        vol_regime = state.vol_regime["current_regime"]

    # Build contribution table
    contrib_df = pd.DataFrame([
        {"Ticker": t, "Weight": weights.get(t, 0), "Return": latest_returns.get(t, 0),
         "Contribution": ticker_contrib.get(t, 0)}
        for t in tickers
    ]).sort_values("Contribution", ascending=False)

    # Sector table
    sector_df = pd.DataFrame([
        {"Sector": s, "Contribution": c} for s, c in sector_contrib.items()
    ]).sort_values("Contribution", ascending=False)

    state.pnl_attribution = {
        "portfolio_return": portfolio_return,
        "ticker_contributions": contrib_df,
        "sector_contributions": sector_df,
        "macro_contributions": macro_contrib,
        "single_stock_contribution": single_stock_contrib,
        "top_contributors": [{"ticker": t, "contribution": c} for t, c in top_contributors],
        "bottom_contributors": [{"ticker": t, "contribution": c} for t, c in bottom_contributors],
        "vol_regime": vol_regime,
    }

    return state
