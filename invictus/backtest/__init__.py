"""
invictus.backtest
=================
Walk-forward ex-ante backtesting framework.

Replays the conviction pipeline historically: for each evaluation date,
only data available up to that point is used (no look-ahead bias).
Forward returns are measured from actual market data that followed.

Usage:
    from invictus.backtest import run_walk_forward
    results = run_walk_forward(
        tickers=["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        start="2024-01-01", end="2024-12-31",
        frequency="monthly",
    )
"""
from invictus.backtest.runner import run_walk_forward

__all__ = ["run_walk_forward"]
