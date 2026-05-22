"""
Portfolio Loader & Market Data Pipeline
Loads holdings, pulls prices, computes returns/weights/P&L.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import streamlit as st

from invictus.config import (
    DEFAULT_PORTFOLIO, LOOKBACK_DAYS, BENCHMARK_TICKER, TRADING_DAYS_PER_YEAR
)


def load_portfolio_from_csv(filepath: str) -> pd.DataFrame:
    """Load portfolio from CSV with columns: Ticker, Shares, CostBasis."""
    try:
        # Initial attempt with robust bad-line handling
        df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
        
        # Strip any whitespace from column names (handles BOM and invisible chars)
        df.columns = [str(c).strip() for c in df.columns]
        
        required = {"Ticker", "Shares", "CostBasis"}
        if not required.issubset(set(df.columns)):
            # If not found, try to search for them in a case-insensitive way
            mapping = {}
            for col in df.columns:
                c_upper = col.upper()
                if "TICKER" in c_upper or "SYMBOL" in c_upper: mapping[col] = "Ticker"
                if "SHARE" in c_upper: mapping[col] = "Shares"
                if "COST" in c_upper or "BASIS" in c_upper or "PRICE" in c_upper: mapping[col] = "CostBasis"
            
            if len(mapping) >= 3:
                df = df.rename(columns=mapping)
            else:
                raise ValueError(f"CSV must contain columns: {required}. Found: {list(df.columns)}")
        
        # Enforce types and clean data
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
        df["CostBasis"] = pd.to_numeric(df["CostBasis"], errors="coerce")
        
        # Surgical drop of invalid rows
        df = df.dropna(subset=["Ticker", "Shares", "CostBasis"])
        return df[["Ticker", "Shares", "CostBasis"]]
    except Exception as e:
        raise ValueError(f"CSV Parsing Error: {str(e)}")


def load_portfolio_from_dict(portfolio: Dict = None) -> pd.DataFrame:
    """Load portfolio from config dictionary."""
    portfolio = portfolio or DEFAULT_PORTFOLIO
    rows = []
    for ticker, info in portfolio.items():
        rows.append({
            "Ticker": ticker,
            "Shares": info["shares"],
            "CostBasis": info["cost_basis"],
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(
    tickers: list,
    lookback_days: int = LOOKBACK_DAYS,
    include_benchmark: bool = True,
    live: bool = False,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for all tickers.
    Returns DataFrame with dates as index, tickers as columns.
    If live=True, the cache is effectively bypassed by the caller or uses a very short TTL.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    all_tickers = list(tickers)
    if include_benchmark and BENCHMARK_TICKER not in all_tickers:
        all_tickers.append(BENCHMARK_TICKER)

    # Fetch all at once for speed
    data = yf.download(
        all_tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    # Handle single vs multi ticker return format
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = all_tickers

    prices = prices.dropna(how="all").ffill()
    return prices


def fetch_live_prices(tickers: list) -> pd.Series:
    """
    Fetch the absolute latest prices for the given tickers.
    Uses yfinance with a 1-day interval to get the most recent trade price.
    """
    all_tickers = list(tickers)
    if BENCHMARK_TICKER not in all_tickers:
        all_tickers.append(BENCHMARK_TICKER)
    
    try:
        # download 1d of data with 1m interval for most recent trade
        data = yf.download(all_tickers, period="1d", interval="1m", progress=False)
        
        if data.empty:
            return pd.Series()
            
        if isinstance(data.columns, pd.MultiIndex):
            latest = data["Close"].iloc[-1]
        else:
            latest = pd.Series([data["Close"].iloc[-1]], index=all_tickers)
            
        return latest.dropna()
    except Exception:
        return pd.Series()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_portfolio_state(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
) -> Dict:
    """
    Compute full portfolio state:
    - current prices, market values, weights
    - daily P&L by ticker and total
    - cost basis vs current value
    - basic stats: vol, beta proxy, drawdown per ticker
    """
    tickers = holdings["Ticker"].tolist()
    latest_prices = prices[tickers].iloc[-1]
    prev_prices = prices[tickers].iloc[-2] if len(prices) > 1 else latest_prices

    # Market values and weights
    shares = holdings.set_index("Ticker")["Shares"]
    cost_basis = holdings.set_index("Ticker")["CostBasis"]
    market_values = latest_prices * shares
    total_value = market_values.sum()
    weights = market_values / total_value

    # Daily P&L
    daily_pnl = (latest_prices - prev_prices) * shares
    total_daily_pnl = daily_pnl.sum()
    daily_return_pct = total_daily_pnl / (total_value - total_daily_pnl) * 100

    # Cost basis analysis
    total_cost = (cost_basis * shares).sum()
    total_unrealized_pnl = total_value - total_cost
    unrealized_pnl_pct = (total_unrealized_pnl / total_cost) * 100

    # Per-ticker stats
    returns = compute_returns(prices[tickers])
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Drawdown per ticker
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Beta proxy (vs benchmark if available)
    benchmark_col = BENCHMARK_TICKER if BENCHMARK_TICKER in prices.columns else None
    betas = {}
    if benchmark_col:
        bench_returns = compute_returns(prices[[benchmark_col]])[benchmark_col]
        for t in tickers:
            if t in returns.columns:
                cov = returns[t].cov(bench_returns)
                var = bench_returns.var()
                betas[t] = cov / var if var > 0 else np.nan

    # Build summary table
    summary = pd.DataFrame({
        "Ticker": tickers,
        "Shares": [shares[t] for t in tickers],
        "Cost Basis": [cost_basis[t] for t in tickers],
        "Current Price": [latest_prices[t] for t in tickers],
        "Market Value": [market_values[t] for t in tickers],
        "Weight (%)": [weights[t] * 100 for t in tickers],
        "Daily P&L ($)": [daily_pnl[t] for t in tickers],
        "Unrealized P&L ($)": [(latest_prices[t] - cost_basis[t]) * shares[t] for t in tickers],
        "Unrealized P&L (%)": [((latest_prices[t] / cost_basis[t]) - 1) * 100 for t in tickers],
        "Ann. Volatility": [ann_vol[t] if t in ann_vol.index else np.nan for t in tickers],
        "Max Drawdown": [max_drawdown[t] if t in max_drawdown.index else np.nan for t in tickers],
        "Beta (vs SPY)": [betas.get(t, np.nan) for t in tickers],
    })

    return {
        "summary": summary,
        "prices": prices,
        "returns": returns,
        "weights": weights.to_dict(),
        "total_value": total_value,
        "total_daily_pnl": total_daily_pnl,
        "daily_return_pct": daily_return_pct,
        "total_cost": total_cost,
        "total_unrealized_pnl": total_unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "holdings": holdings,
    }
