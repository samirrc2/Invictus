"""
Data pipeline health collector — tracks API success rates, latency, data freshness.
"""
import time
from typing import Optional
from invictus.observability.store import insert


def log_data_fetch(source: str, ticker: str = None, status: str = "success",
                   latency_ms: float = None, records_fetched: int = None,
                   freshness_hours: float = None, error_message: str = None,
                   run_id: str = None):
    """Log a data fetch operation."""
    try:
        insert("data_health", {
            "run_id": run_id,
            "source": source,
            "ticker": ticker,
            "status": status,
            "latency_ms": latency_ms,
            "records_fetched": records_fetched,
            "freshness_hours": freshness_hours,
            "error_message": error_message[:500] if error_message else None,
        })
    except Exception:
        pass


class DataFetchTracker:
    """
    Context manager for tracking data fetches.
    Usage:
        with DataFetchTracker("yfinance", ticker="AAPL") as tracker:
            data = yf.download(...)
            tracker.records = len(data)
    """
    def __init__(self, source: str, ticker: str = None, run_id: str = None):
        self.source = source
        self.ticker = ticker
        self.run_id = run_id
        self.records = 0
        self.freshness_hours = None
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = (time.perf_counter() - self._start) * 1000 if self._start else 0
        status = "error" if exc_type else "success"
        error_msg = str(exc_val)[:500] if exc_val else None
        log_data_fetch(
            source=self.source, ticker=self.ticker, status=status,
            latency_ms=latency, records_fetched=self.records,
            freshness_hours=self.freshness_hours, error_message=error_msg,
            run_id=self.run_id,
        )
        return False
