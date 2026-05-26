"""
Invictus — FMP (Financial Modeling Prep) REST Client
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Shared client used by all agents that need market data from FMP.
Provides typed fetcher functions with:
  - Automatic API key injection from config
  - Rate limiting (300ms between calls)
  - Retry with exponential backoff (2 retries)
  - Structured error handling
  - Demo cache fallback when FMP is unavailable

Available endpoints (matching user's FMP plan):
  ✓ Insider Trading      — insider-trading/search
  ✓ Income Statements    — income-statement
  ✓ Stock News           — news/stock
  ✓ Analyst Grades       — grades-consensus
  ✓ Earnings Calendar    — earnings
  ✓ Analyst Estimates    — analyst-estimates

Restricted (higher plan needed):
  ✗ Earnings Transcripts
  ✗ Press Releases
  ✗ Institutional Ownership
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from invictus.config import FMP_API_KEY, DATA_DIR

_log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
FMP_BASE = "https://financialmodelingprep.com/stable"
DEMO_DIR = DATA_DIR / "demo"
_MIN_CALL_INTERVAL = 0.3  # seconds between API calls (rate limit)
_MAX_RETRIES = 2
_TIMEOUT = 30  # seconds

# Track last call time for rate limiting
_last_call_time = 0.0


# ══════════════════════════════════════════════════════════════════════
# CORE HTTP CLIENT
# ══════════════════════════════════════════════════════════════════════

def _rate_limit():
    """Enforce minimum interval between FMP API calls."""
    global _last_call_time
    elapsed = time.perf_counter() - _last_call_time
    if elapsed < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - elapsed)
    _last_call_time = time.perf_counter()


def fmp_available() -> bool:
    """Check if FMP API key is configured."""
    return bool(FMP_API_KEY and len(FMP_API_KEY) > 5)


def _fmp_get(endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
    """
    Make an authenticated GET request to FMP's /stable/ API.

    Args:
        endpoint: Path after /stable/, e.g. 'income-statement'
        params: Query parameters (apikey is added automatically)

    Returns:
        Parsed JSON (list or dict) on success, None on failure.
    """
    if not fmp_available():
        _log.debug("FMP: no API key configured")
        return None

    p = {"apikey": FMP_API_KEY}
    if params:
        p.update(params)
    url = f"{FMP_BASE}/{endpoint}?{urlencode(p)}"

    for attempt in range(_MAX_RETRIES + 1):
        _rate_limit()
        try:
            req = Request(url, headers={"User-Agent": "Invictus/1.0"})
            response = urlopen(req, timeout=_TIMEOUT)
            raw = response.read().decode("utf-8")

            # FMP returns plain text error messages for restricted endpoints
            if "Restricted Endpoint" in raw or "Invalid API KEY" in raw:
                _log.warning("FMP: %s → restricted or invalid key", endpoint)
                return None

            data = json.loads(raw)

            # FMP returns {"Error Message": "..."} for bad requests
            if isinstance(data, dict) and "Error Message" in data:
                _log.warning("FMP: %s → %s", endpoint, data["Error Message"])
                return None

            return data

        except HTTPError as e:
            if e.code == 429:  # Rate limited
                wait = (attempt + 1) * 2
                _log.warning("FMP: rate limited on %s, waiting %ds", endpoint, wait)
                time.sleep(wait)
                continue
            elif e.code in (401, 403):
                _log.warning("FMP: auth error on %s (HTTP %d)", endpoint, e.code)
                return None
            else:
                _log.warning("FMP: HTTP %d on %s (attempt %d)", e.code, endpoint, attempt + 1)
                if attempt < _MAX_RETRIES:
                    time.sleep((attempt + 1) * 1)
                    continue
                return None

        except (URLError, TimeoutError, OSError) as e:
            _log.warning("FMP: network error on %s: %s (attempt %d)", endpoint, e, attempt + 1)
            if attempt < _MAX_RETRIES:
                time.sleep((attempt + 1) * 1)
                continue
            return None

        except json.JSONDecodeError as e:
            _log.warning("FMP: invalid JSON from %s: %s", endpoint, e)
            return None

    return None


def _load_demo(ticker: str, filename: str) -> Any:
    """Load cached demo data as fallback."""
    path = DEMO_DIR / ticker.lower() / filename
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == 0:
            return None  # Empty cache = no data
        _log.debug("FMP: loaded demo cache %s/%s", ticker, filename)
        return data
    except Exception as e:
        _log.warning("FMP: failed to load demo %s/%s: %s", ticker, filename, e)
        return None


# ══════════════════════════════════════════════════════════════════════
# TYPED FETCHERS — each returns structured data or falls back to demo
# ══════════════════════════════════════════════════════════════════════

def fetch_insider_trading(ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch insider trading transactions from FMP.
    Falls back to demo cache if FMP is unavailable.

    Returns list of dicts with keys:
        symbol, filingDate, transactionDate, reportingCik, transactionType,
        securitiesOwned, securitiesTransacted, price, formType,
        reportingName, typeOfOwner, link
    """
    data = _fmp_get("insider-trading/search", {
        "symbol": ticker, "limit": str(limit),
    })
    if isinstance(data, list) and len(data) > 0:
        _log.info("FMP: insider trading %s → %d records", ticker, len(data))
        return data

    # Fallback to demo
    demo = _load_demo(ticker, "insider_trading.json")
    if demo and isinstance(demo, list):
        _log.info("FMP: insider trading %s → demo cache (%d records)", ticker, len(demo))
        return demo

    return []


def fetch_income_statements(ticker: str, limit: int = 4, period: str = "quarter") -> List[Dict[str, Any]]:
    """
    Fetch income statements from FMP.
    Falls back to demo cache if FMP is unavailable.

    Returns list of dicts with keys:
        date, symbol, revenue, costOfRevenue, grossProfit, grossProfitRatio,
        operatingIncome, operatingIncomeRatio, netIncome, netIncomeRatio,
        eps, epsdiluted, weightedAverageShsOut, ...
    """
    params = {"symbol": ticker, "limit": str(limit)}
    if period == "quarter":
        params["period"] = "quarter"
    data = _fmp_get("income-statement", params)
    if isinstance(data, list) and len(data) > 0:
        _log.info("FMP: income statement %s → %d periods", ticker, len(data))
        return data

    # Fallback to demo
    demo = _load_demo(ticker, "income_statement.json")
    if demo and isinstance(demo, list):
        _log.info("FMP: income statement %s → demo cache (%d periods)", ticker, len(demo))
        return demo

    return []


def fetch_stock_news(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch stock-specific news from FMP.
    Falls back to demo cache if FMP is unavailable.

    Returns list of dicts with keys:
        title, text, publishedDate, site, url, symbol, image
    """
    data = _fmp_get("news/stock", {"tickers": ticker, "limit": str(limit)})
    if isinstance(data, list) and len(data) > 0:
        _log.info("FMP: stock news %s → %d articles", ticker, len(data))
        return data

    # Fallback to demo
    demo = _load_demo(ticker, "stock_news.json")
    if demo and isinstance(demo, list):
        _log.info("FMP: stock news %s → demo cache (%d articles)", ticker, len(demo))
        return demo

    return []


def fetch_analyst_grades(ticker: str) -> Dict[str, Any]:
    """
    Fetch analyst grade consensus from FMP.
    Falls back to demo cache if FMP is unavailable.

    Returns dict with keys:
        symbol, strongBuy, buy, hold, sell, strongSell, consensus
    """
    data = _fmp_get("grades-consensus", {"symbol": ticker})
    if isinstance(data, list) and len(data) > 0:
        _log.info("FMP: analyst grades %s → %s", ticker, data[0].get("consensus", "?"))
        return data[0]
    if isinstance(data, dict) and data.get("symbol"):
        _log.info("FMP: analyst grades %s → %s", ticker, data.get("consensus", "?"))
        return data

    # Fallback to demo
    demo = _load_demo(ticker, "analyst_grades.json")
    if demo:
        if isinstance(demo, list) and len(demo) > 0:
            return demo[0]
        if isinstance(demo, dict):
            return demo

    return {}


def fetch_earnings_calendar(ticker: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Fetch earnings report history from FMP.
    Falls back to demo cache if FMP is unavailable.

    Returns list of dicts with keys:
        date, symbol, eps, epsEstimated, revenue, revenueEstimated,
        fiscalDateEnding, updatedFromDate
    """
    data = _fmp_get("earnings", {"symbol": ticker, "limit": str(limit)})
    if isinstance(data, list) and len(data) > 0:
        _log.info("FMP: earnings calendar %s → %d reports", ticker, len(data))
        return data

    # Fallback to demo
    demo = _load_demo(ticker, "earnings_calendar.json")
    if demo and isinstance(demo, list):
        _log.info("FMP: earnings calendar %s → demo cache (%d reports)", ticker, len(demo))
        return demo

    return []


def fetch_analyst_estimates(ticker: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Fetch analyst estimates (EPS & revenue forecasts) from FMP.
    Falls back to demo cache if FMP is unavailable.

    Returns list of dicts with keys:
        symbol, date, estimatedRevenueAvg, estimatedRevenueHigh,
        estimatedRevenueLow, estimatedEpsAvg, estimatedEpsHigh,
        estimatedEpsLow, numberAnalystEstimatedRevenue, ...
    """
    data = _fmp_get("analyst-estimates", {
        "symbol": ticker, "limit": str(limit),
    })
    if isinstance(data, list) and len(data) > 0:
        _log.info("FMP: analyst estimates %s → %d periods", ticker, len(data))
        return data

    # Fallback to demo
    demo = _load_demo(ticker, "analyst_estimates.json")
    if demo and isinstance(demo, list):
        _log.info("FMP: analyst estimates %s → demo cache (%d periods)", ticker, len(demo))
        return demo

    return []


def fetch_earnings_surprises(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch historical EPS surprise data from FMP.
    Falls back to demo cache if FMP is unavailable.

    Returns list of dicts with keys:
        date, symbol, actualEarningResult, estimatedEarning
    """
    data = _fmp_get("earnings-surprises", {"symbol": ticker})
    if isinstance(data, list) and len(data) > 0:
        _log.info("FMP: earnings surprises %s → %d records", ticker, len(data))
        return data

    # Fallback to demo
    demo = _load_demo(ticker, "earnings_surprises.json")
    if demo and isinstance(demo, list):
        _log.info("FMP: earnings surprises %s → demo cache (%d records)", ticker, len(demo))
        return demo

    return []


def fetch_company_profile(ticker: str) -> Dict[str, Any]:
    """
    Fetch company profile from FMP.

    Returns dict with keys:
        symbol, companyName, industry, sector, mktCap, price,
        beta, volAvg, lastDiv, range, changes, ...
    """
    data = _fmp_get("profile", {"symbol": ticker})
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    if isinstance(data, dict) and data.get("symbol"):
        return data
    return {}
