"""
Download FMP data for demo/dev mode.

DEV MODE: During development, the outlook_agent reads from invictus/data/demo/
instead of hitting FMP live. Once the product is production-ready, switch to
live FMP API calls by setting USE_DEMO_DATA = False in outlook_agent.py.

Uses urllib + certifi as recommended by FMP docs.

Run:  python scripts/download_fmp_demo.py
"""
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode

import certifi

# ── Config ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"
TICKERS = ["AAPL", "AMD", "META", "TSLA", "SMH"]
DEMO_DIR = PROJECT_ROOT / "invictus" / "data" / "demo"

if not FMP_API_KEY:
    print("ERROR: FMP_API_KEY not set in .env")
    sys.exit(1)


def fmp_get(endpoint: str, params: dict = None):
    """
    FMP API call using urllib + certifi (official FMP pattern).
    endpoint: path after /stable/, e.g. 'earning-call-transcript-latest'
    """
    p = {"apikey": FMP_API_KEY}
    if params:
        p.update(params)
    url = f"{FMP_BASE}/{endpoint}?{urlencode(p)}"
    try:
        response = urlopen(url, cafile=certifi.where(), timeout=30)
        data = json.loads(response.read().decode("utf-8"))
        if isinstance(data, dict) and "Error Message" in data:
            print(f"    ⚠ FMP error: {data['Error Message']}")
            return None
        if isinstance(data, list):
            print(f"    ✓ {endpoint} → {len(data)} records")
        elif isinstance(data, dict):
            print(f"    ✓ {endpoint} → dict")
        return data
    except Exception as e:
        print(f"    ✗ {endpoint} → {e}")
        return None


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    size = os.path.getsize(path)
    count = len(data) if isinstance(data, list) else 1
    print(f"    💾 {path.name} — {count} records, {size:,} bytes")


# ── Downloaders ──────────────────────────────────────────────────────

def download_transcripts(ticker: str):
    """Download latest earnings call transcripts."""
    print(f"  📄 Transcripts:")

    # Step 1: Get transcript dates for this ticker
    dates = fmp_get("earning-call-transcript-dates", {"symbol": ticker})

    if not dates or not isinstance(dates, list) or len(dates) == 0:
        # Fallback: try latest transcripts endpoint (no symbol filter)
        latest = fmp_get("earning-call-transcript-latest")
        if latest and isinstance(latest, list):
            # Filter for our ticker
            ticker_transcripts = [t for t in latest if t.get("symbol") == ticker]
            if ticker_transcripts:
                return ticker_transcripts[:4]
        print(f"    ❌ No transcript dates for {ticker}")
        return []

    # Step 2: Fetch individual transcripts
    transcripts = []
    for entry in dates[:4]:
        q = entry.get("quarter", entry.get("Quarter"))
        y = entry.get("year", entry.get("Year"))
        if not q or not y:
            continue
        data = fmp_get("earning-call-transcript", {
            "symbol": ticker, "quarter": q, "year": y,
        })
        if data:
            if isinstance(data, list) and len(data) > 0:
                transcripts.append(data[0])
            elif isinstance(data, dict) and data.get("content"):
                transcripts.append(data)
        time.sleep(0.3)

    return transcripts


def download_press_releases(ticker: str):
    print(f"  📰 Press Releases:")
    # Try with 'symbols' param
    data = fmp_get("news/press-releases", {"symbols": ticker, "limit": 20})
    if data and isinstance(data, list) and len(data) > 0:
        return data
    # Try with 'symbol' param
    data = fmp_get("news/press-releases", {"symbol": ticker, "limit": 20})
    if data and isinstance(data, list) and len(data) > 0:
        return data
    return []


def download_analyst_estimates(ticker: str):
    print(f"  📊 Analyst Estimates:")
    data = fmp_get("analyst-estimates", {
        "symbol": ticker, "limit": 8,
    })
    return data if isinstance(data, list) else []


def download_insider_trading(ticker: str):
    """Download insider trading transactions (available on current plan)."""
    print(f"  🕵️ Insider Trading:")
    data = fmp_get("insider-trading/search", {
        "symbol": ticker, "limit": 100,
    })
    return data if isinstance(data, list) else []


def download_income_statement(ticker: str):
    """Download quarterly income statements (available on current plan)."""
    print(f"  📈 Income Statements:")
    data = fmp_get("income-statement", {
        "symbol": ticker, "period": "quarter", "limit": 4,
    })
    return data if isinstance(data, list) else []


def download_grades(ticker: str):
    print(f"  🏷️ Analyst Grades:")
    data = fmp_get("grades-consensus", {"symbol": ticker})
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def download_earnings_surprises(ticker: str):
    print(f"  🎯 Earnings Surprises:")
    data = fmp_get("earnings-surprises", {"symbol": ticker})
    return data if isinstance(data, list) else []


def download_stock_news(ticker: str):
    print(f"  📡 Stock News:")
    data = fmp_get("news/stock", {"tickers": ticker, "limit": 20})
    if data and isinstance(data, list) and len(data) > 0:
        return data
    # Try 'symbols' param
    data = fmp_get("news/stock", {"symbols": ticker, "limit": 20})
    if data and isinstance(data, list) and len(data) > 0:
        return data
    return []


def download_earnings_calendar(ticker: str):
    print(f"  📅 Earnings Data:")
    data = fmp_get("earnings", {"symbol": ticker, "limit": 8})
    return data if isinstance(data, list) else []


def main():
    print(f"╔{'═'*58}╗")
    print(f"║  FMP Demo Data Downloader v3 (urllib + certifi)          ║")
    print(f"║  Key: {FMP_API_KEY[:8]}...{FMP_API_KEY[-4:]}                                ║")
    print(f"╚{'═'*58}╝")
    print(f"Tickers: {', '.join(TICKERS)}\n")

    # Quick plan check
    print("🔍 Connectivity check...")
    fmp_get("profile", {"symbol": "AAPL"})
    print()

    summary = {}

    for ticker in TICKERS:
        print(f"{'━'*50}")
        print(f"📥 {ticker}")
        print(f"{'━'*50}")
        td = DEMO_DIR / ticker.lower()
        td.mkdir(parents=True, exist_ok=True)
        s = {}

        t = download_transcripts(ticker);         save_json(t, td / "transcripts.json");         s["transcripts"] = len(t)
        p = download_press_releases(ticker);      save_json(p, td / "press_releases.json");      s["press_releases"] = len(p)
        e = download_analyst_estimates(ticker);    save_json(e, td / "analyst_estimates.json");    s["analyst_estimates"] = len(e)
        g = download_grades(ticker);              save_json(g, td / "analyst_grades.json");       s["analyst_grades"] = len(g)
        sr = download_earnings_surprises(ticker); save_json(sr, td / "earnings_surprises.json");  s["earnings_surprises"] = len(sr)
        n = download_stock_news(ticker);          save_json(n, td / "stock_news.json");           s["stock_news"] = len(n)
        ec = download_earnings_calendar(ticker);  save_json(ec, td / "earnings_calendar.json");   s["earnings_calendar"] = len(ec)
        it = download_insider_trading(ticker);    save_json(it, td / "insider_trading.json");     s["insider_trading"] = len(it)
        inc = download_income_statement(ticker);  save_json(inc, td / "income_statement.json");   s["income_statement"] = len(inc)

        summary[ticker] = s
        time.sleep(0.5)

    save_json({
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tickers": TICKERS,
        "summary": summary,
    }, DEMO_DIR / "manifest.json")

    print(f"\n{'═'*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'═'*60}")
    print(f"{'Ticker':<8} {'Trans':>6} {'Press':>6} {'Estim':>6} {'Grade':>6} {'Surpr':>6} {'News':>6} {'Earns':>6} {'Insdr':>6} {'IncSt':>6}")
    print(f"{'─'*76}")
    for t, c in summary.items():
        print(f"{t:<8} {c.get('transcripts',0):>6} {c.get('press_releases',0):>6} "
              f"{c.get('analyst_estimates',0):>6} {c.get('analyst_grades',0):>6} "
              f"{c.get('earnings_surprises',0):>6} {c.get('stock_news',0):>6} "
              f"{c.get('earnings_calendar',0):>6} {c.get('insider_trading',0):>6} "
              f"{c.get('income_statement',0):>6}")
    print(f"{'═'*76}")


if __name__ == "__main__":
    main()
