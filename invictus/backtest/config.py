"""
invictus.backtest.config
=========================
Backtest configuration — date ranges, evaluation intervals, default tickers.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import date


# ── Default Universe ────────────────────────────────────────────────────
# Large-cap liquid names with rich fundamental & flow data history
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ",
]

# ── Horizons for forward return measurement ─────────────────────────────
DEFAULT_HORIZONS = [5, 10, 21, 63]  # 1w, 2w, 1m, 3m (trading days)

# ── Price history buffer ────────────────────────────────────────────────
# We need ~6 months of price data BEFORE the backtest start for
# momentum/volatility feature computation (ML agent needs 60+ days)
LOOKBACK_BUFFER_DAYS = 180

# ── Benchmark for relative strength ────────────────────────────────────
BENCHMARK_TICKER = "SPY"


@dataclass
class BacktestConfig:
    """Configuration for a walk-forward backtest run."""

    tickers: List[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    start: str = "2024-01-01"           # First evaluation date
    end: str = "2024-12-31"             # Last evaluation date
    frequency: str = "monthly"          # "monthly" or "biweekly"
    horizons: List[int] = field(default_factory=lambda: list(DEFAULT_HORIZONS))
    benchmark: str = BENCHMARK_TICKER
    lookback_buffer: int = LOOKBACK_BUFFER_DAYS

    # Signal weights (same as synthesis_agent defaults)
    w_fundamental: float = 0.35
    w_management: float = 0.25
    w_flows: float = 0.25
    w_technical: float = 0.15

    # Hypothetical P&L parameters
    position_size: float = 0.10         # 10% of portfolio per conviction trade
    long_threshold: float = 0.60        # go long above this outperformance prob
    short_threshold: float = 0.40       # go short below this outperformance prob

    def eval_dates(self) -> List[date]:
        """Generate evaluation dates based on frequency."""
        from datetime import datetime, timedelta
        import calendar

        start_dt = datetime.strptime(self.start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(self.end, "%Y-%m-%d").date()
        dates = []

        if self.frequency == "monthly":
            # First trading day of each month
            current = start_dt.replace(day=1)
            while current <= end_dt:
                # Advance to first weekday
                d = current
                while d.weekday() >= 5:  # Sat/Sun
                    d += timedelta(days=1)
                if d <= end_dt:
                    dates.append(d)
                # Next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)

        elif self.frequency == "biweekly":
            current = start_dt
            while current <= end_dt:
                d = current
                while d.weekday() >= 5:
                    d += timedelta(days=1)
                if d <= end_dt:
                    dates.append(d)
                current += timedelta(days=14)

        return dates

    @property
    def data_start(self) -> str:
        """Earliest date we need price data from (includes lookback buffer)."""
        from datetime import datetime, timedelta
        dt = datetime.strptime(self.start, "%Y-%m-%d") - timedelta(days=self.lookback_buffer)
        return dt.strftime("%Y-%m-%d")

    @property
    def data_end(self) -> str:
        """Latest date we need price data through (includes forward return buffer)."""
        from datetime import datetime, timedelta
        dt = datetime.strptime(self.end, "%Y-%m-%d") + timedelta(days=max(self.horizons) + 10)
        return dt.strftime("%Y-%m-%d")
