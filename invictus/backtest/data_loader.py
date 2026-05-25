"""
invictus.backtest.data_loader
==============================
Point-in-time historical data fetcher for walk-forward backtesting.

Fetches ALL data once, then provides date-filtered slices to avoid
look-ahead bias. Each slice contains only information available
as of the evaluation date.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Optional, Any

from invictus.backtest.config import BacktestConfig


class HistoricalDataStore:
    """
    Fetches and caches all historical data for the backtest universe.
    Provides point-in-time slices that enforce no look-ahead bias.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._prices: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None
        self._financials: Dict[str, pd.DataFrame] = {}  # ticker -> quarterly financials
        self._loaded = False

    def load(self, progress_callback=None):
        """Fetch all data from yFinance. Call once before running backtest."""
        all_tickers = self.config.tickers + [self.config.benchmark]
        unique_tickers = list(set(all_tickers))

        if progress_callback:
            progress_callback(f"Downloading prices for {len(unique_tickers)} tickers...")

        # ── 1. Price Data ────────────────────────────────────────
        raw = yf.download(
            unique_tickers,
            start=self.config.data_start,
            end=self.config.data_end,
            auto_adjust=True,
            progress=False,
        )

        if raw is None or raw.empty:
            raise ValueError("Failed to download price data from yFinance")

        # Handle multi-ticker vs single-ticker download format
        if isinstance(raw.columns, pd.MultiIndex):
            self._prices = raw["Close"].copy()
        else:
            # Single ticker
            self._prices = raw[["Close"]].copy()
            self._prices.columns = [unique_tickers[0]]

        # Ensure index is DatetimeIndex
        self._prices.index = pd.to_datetime(self._prices.index)

        # Forward-fill then drop any fully empty
        self._prices = self._prices.ffill().dropna(how="all")

        # ── 2. Returns ───────────────────────────────────────────
        self._returns = np.log(self._prices / self._prices.shift(1)).dropna(how="all")

        # ── 3. Quarterly Financials (for fundamental signals) ────
        if progress_callback:
            progress_callback("Fetching quarterly financials...")

        for ticker in self.config.tickers:
            try:
                t = yf.Ticker(ticker)
                qf = t.quarterly_financials
                if qf is not None and not qf.empty:
                    self._financials[ticker] = qf
            except Exception:
                pass  # Skip tickers with no financial data

        self._loaded = True
        if progress_callback:
            progress_callback(f"Loaded: {len(self._prices.columns)} tickers, "
                            f"{len(self._prices)} price days, "
                            f"{len(self._financials)} with financials")

    # ── Point-in-Time Slices ──────────────────────────────────────────

    def prices_as_of(self, eval_date: date) -> pd.DataFrame:
        """Return prices up to (and including) eval_date only."""
        self._check_loaded()
        dt = pd.Timestamp(eval_date)
        return self._prices.loc[:dt].copy()

    def returns_as_of(self, eval_date: date) -> pd.DataFrame:
        """Return log-returns up to (and including) eval_date only."""
        self._check_loaded()
        dt = pd.Timestamp(eval_date)
        return self._returns.loc[:dt].copy()

    def financials_as_of(self, ticker: str, eval_date: date) -> Optional[pd.DataFrame]:
        """
        Return quarterly financials for ticker with only periods
        whose reporting date is BEFORE eval_date.

        yFinance quarterly_financials columns are period-end dates.
        We assume a ~45-day reporting lag (10-Q filings due 40-45 days
        after quarter end for large accelerated filers).
        """
        if ticker not in self._financials:
            return None

        qf = self._financials[ticker]
        reporting_lag = timedelta(days=45)

        # Filter: only include quarters whose data would have been
        # publicly available by eval_date
        available_cols = []
        for col_date in qf.columns:
            period_end = pd.Timestamp(col_date).date()
            assumed_report_date = period_end + reporting_lag
            if assumed_report_date <= eval_date:
                available_cols.append(col_date)

        if not available_cols:
            return None

        return qf[available_cols].copy()

    def forward_prices(self, eval_date: date, horizon_days: int) -> Optional[pd.Series]:
        """
        Return prices exactly horizon_days trading days after eval_date.
        Returns None if insufficient forward data.
        """
        self._check_loaded()
        dt = pd.Timestamp(eval_date)
        future = self._prices.loc[dt:]

        if len(future) <= horizon_days:
            return None

        return future.iloc[horizon_days]

    def forward_return(self, eval_date: date, horizon_days: int) -> Optional[pd.Series]:
        """
        Compute forward return for each ticker from eval_date over horizon_days.
        Returns log-return series indexed by ticker.
        """
        self._check_loaded()
        dt = pd.Timestamp(eval_date)
        future = self._prices.loc[dt:]

        if len(future) <= horizon_days:
            return None

        p0 = future.iloc[0]
        p1 = future.iloc[horizon_days]

        # Log returns, handle zeros
        valid = (p0 > 0) & (p1 > 0)
        returns = pd.Series(0.0, index=p0.index)
        returns[valid] = np.log(p1[valid] / p0[valid])

        return returns

    @property
    def full_prices(self) -> pd.DataFrame:
        """Full price history (for forward return computation)."""
        self._check_loaded()
        return self._prices

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call load() before accessing data")
