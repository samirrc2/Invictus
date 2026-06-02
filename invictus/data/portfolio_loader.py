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
    if prices.empty:
        raise ValueError(
            f"yfinance returned no price data for {all_tickers}. "
            f"This can happen on Streamlit Cloud due to network restrictions or market hours. "
            f"Try again in a few minutes."
        )
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
    """Compute daily log returns.

    Uses dropna(how='all') so that a single column with NaN doesn't
    wipe out the entire row for every other ticker.  Individual NaN
    cells are left for downstream callers to handle per-series.
    """
    raw = np.log(prices / prices.shift(1))
    # Drop only rows that are ALL NaN (e.g. the first shifted row).
    # Per-column NaN (missing ticker days) are left intact so that
    # other columns' valid data is preserved.
    return raw.dropna(how="all")


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
    available = [t for t in tickers if t in prices.columns]
    if not available or len(prices) == 0:
        raise ValueError(f"No price data available for tickers: {tickers}")

    # Get last prices and drop tickers whose entire price column is NaN
    # (yfinance sometimes returns a column header with no actual data).
    latest_prices = prices[available].iloc[-1].dropna()
    available = [t for t in available if t in latest_prices.index]
    if not available:
        raise ValueError(f"All tickers have NaN prices: {tickers}")

    prev_prices = prices[available].iloc[-2] if len(prices) > 1 else latest_prices

    # Subset holdings to available tickers only — avoids NaN from
    # pandas index alignment when tickers are missing from prices.
    h = holdings.set_index("Ticker")
    shares = h.loc[h.index.isin(available), "Shares"]
    cost_basis = h.loc[h.index.isin(available), "CostBasis"]

    market_values = latest_prices[available] * shares
    total_value = market_values.sum()
    weights = market_values / total_value

    # Daily P&L
    daily_pnl = (latest_prices[available] - prev_prices) * shares
    total_daily_pnl = daily_pnl.sum()
    daily_return_pct = total_daily_pnl / (total_value - total_daily_pnl) * 100

    # Cost basis analysis
    total_cost = (cost_basis * shares).sum()
    total_unrealized_pnl = total_value - total_cost
    unrealized_pnl_pct = (total_unrealized_pnl / total_cost) * 100

    # Per-ticker stats — use `available` to avoid NaN columns
    returns = compute_returns(prices[available])
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
        for t in available:
            if t in returns.columns:
                cov = returns[t].cov(bench_returns)
                var = bench_returns.var()
                betas[t] = cov / var if var > 0 else np.nan

    # Build summary table — use `available` so no NaN lookups
    summary = pd.DataFrame({
        "Ticker": available,
        "Shares": [shares[t] for t in available],
        "Cost Basis": [cost_basis[t] for t in available],
        "Current Price": [latest_prices[t] for t in available],
        "Market Value": [market_values[t] for t in available],
        "Weight (%)": [weights[t] * 100 for t in available],
        "Daily P&L ($)": [daily_pnl[t] for t in available],
        "Unrealized P&L ($)": [(latest_prices[t] - cost_basis[t]) * shares[t] for t in available],
        "Unrealized P&L (%)": [((latest_prices[t] / cost_basis[t]) - 1) * 100 for t in available],
        "Ann. Volatility": [ann_vol[t] if t in ann_vol.index else np.nan for t in available],
        "Max Drawdown": [max_drawdown[t] if t in max_drawdown.index else np.nan for t in available],
        "Beta (vs SPY)": [betas.get(t, np.nan) for t in available],
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
