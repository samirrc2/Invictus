"""
Invictus — Stress Testing & Historical Scenario Replay Agent
Replays historical crash periods on the current portfolio to estimate losses.

Scenarios:
- COVID Crash (Feb-Mar 2020)
- 2022 Rate Shock (Jan-Jun 2022)
- Tech Drawdown (Nov 2021-Jan 2023)
- Semiconductor Selloff (Jul-Oct 2022)
- SVB Crisis (Mar 2023)

Outputs:
- Per-scenario portfolio loss ($ and %)
- Per-ticker loss in each scenario
- Worst-hit tickers per scenario
- Sector vulnerability analysis
- Scenario comparison summary
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List
import streamlit as st

from invictus.agents.graph_state import PortfolioState
from invictus.config import STRESS_SCENARIOS


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_scenario_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch prices for a specific scenario period."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers
    return prices.dropna(how="all").ffill()


def _compute_scenario_impact(
    prices: pd.DataFrame,
    tickers: list,
    shares: Dict[str, float],
    current_prices: Dict[str, float],
    total_value: float,
) -> Dict[str, Any]:
    """Compute the impact of a historical scenario on the current portfolio."""
    available = [t for t in tickers if t in prices.columns and len(prices[t].dropna()) >= 2]

    if not available:
        return {"error": "No price data available for this scenario period."}

    # Compute returns over the scenario period
    first_prices = prices[available].dropna().iloc[0]
    last_prices = prices[available].dropna().iloc[-1]
    scenario_returns = (last_prices - first_prices) / first_prices

    # Apply scenario returns to current portfolio
    ticker_impacts = []
    total_loss = 0.0

    for t in tickers:
        cur_price = current_prices.get(t, 0)
        sh = shares.get(t, 0)
        cur_value = cur_price * sh

        if t in scenario_returns.index and not np.isnan(scenario_returns[t]):
            ret = scenario_returns[t]
            loss_dollars = cur_value * ret
            total_loss += loss_dollars
            ticker_impacts.append({
                "Ticker": t,
                "Current Value": cur_value,
                "Scenario Return": ret,
                "P&L ($)": loss_dollars,
                "Stressed Value": cur_value * (1 + ret),
            })
        else:
            # No data for this ticker in the scenario period — assume flat
            ticker_impacts.append({
                "Ticker": t,
                "Current Value": cur_value,
                "Scenario Return": 0.0,
                "P&L ($)": 0.0,
                "Stressed Value": cur_value,
            })
            total_loss += 0.0

    ticker_df = pd.DataFrame(ticker_impacts).sort_values("P&L ($)")
    portfolio_return = total_loss / total_value if total_value > 0 else 0.0

    # Worst hit tickers (top 3 losses)
    worst_3 = ticker_df.head(3)[["Ticker", "Scenario Return", "P&L ($)"]].to_dict("records")

    # Best performers (top 3 gains or smallest losses)
    best_3 = ticker_df.tail(3)[["Ticker", "Scenario Return", "P&L ($)"]].to_dict("records")

    return {
        "portfolio_return": portfolio_return,
        "portfolio_pnl": total_loss,
        "stressed_value": total_value + total_loss,
        "ticker_detail": ticker_df,
        "worst_tickers": worst_3,
        "best_tickers": best_3,
        "n_days": len(prices),
    }


def run_stress_tests(state: PortfolioState) -> PortfolioState:
    """
    Run all stress test scenarios on the portfolio.
    Writes stress_results to state.
    """
    holdings = state.holdings
    weights = state.weights
    prices = state.prices

    if holdings is None or prices is None:
        raise ValueError("Holdings and prices must be populated for stress testing.")

    tickers = holdings["Ticker"].tolist() if isinstance(holdings, pd.DataFrame) else list(weights.keys())

    # Current prices and shares
    current_prices = {}
    shares = {}
    if isinstance(holdings, pd.DataFrame):
        for _, row in holdings.iterrows():
            t = row["Ticker"]
            shares[t] = row["Shares"]
            if t in prices.columns:
                current_prices[t] = float(prices[t].iloc[-1])
            else:
                current_prices[t] = 0.0

    total_value = sum(current_prices.get(t, 0) * shares.get(t, 0) for t in tickers)

    # Run each scenario
    results = {}
    for scenario_name, dates in STRESS_SCENARIOS.items():
        scenario_prices = _fetch_scenario_prices(
            tickers, dates["start"], dates["end"]
        )
        impact = _compute_scenario_impact(
            scenario_prices, tickers, shares, current_prices, total_value
        )
        impact["period"] = f"{dates['start']} to {dates['end']}"
        results[scenario_name] = impact

    # Summary table
    summary_rows = []
    for name, res in results.items():
        if "error" not in res:
            summary_rows.append({
                "Scenario": name,
                "Period": res["period"],
                "Portfolio Return": res["portfolio_return"],
                "Portfolio P&L": res["portfolio_pnl"],
                "Stressed Value": res["stressed_value"],
                "Trading Days": res["n_days"],
            })
    summary_df = pd.DataFrame(summary_rows).sort_values("Portfolio Return")

    state.stress_results = {
        "scenarios": results,
        "summary": summary_df,
        "total_value": total_value,
    }

    return state
