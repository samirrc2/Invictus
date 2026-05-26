"""
Cache real yfinance institutional + insider data for demo mode.

Run locally before deploying to Streamlit Cloud:
    python scripts/cache_flow_data.py

This fetches actual data from yfinance and saves it as JSON files
in invictus/data/demo/<ticker>/flow_data.json — the flow agent
will fall back to these when yfinance is rate-limited on cloud.
"""
import json
import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invictus.agents.flow_agent import _fetch_flow_data

DEMO_DIR = Path(__file__).resolve().parent.parent / "invictus" / "data" / "demo"
TICKERS = ["AAPL", "AMD", "META", "TSLA", "SMH"]


def _sanitize(obj):
    """Replace NaN/Inf with None so JSON serialization doesn't break."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def main():
    for ticker in TICKERS:
        print(f"Fetching flow data for {ticker}...")
        data = _fetch_flow_data(ticker)

        if data.get("status") != "Success":
            print(f"  WARNING: {ticker} returned status={data.get('status')} — skipping")
            continue

        # Sanitize NaN/Inf values before JSON serialization
        data = _sanitize(data)

        # Save to demo directory
        out_dir = DEMO_DIR / ticker.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "flow_data.json"

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        n_inst = len(data.get("institutional", []))
        n_ins = len(data.get("insiders", []))
        print(f"  Saved: {n_inst} institutions, {n_ins} insiders → {out_path}")

    print("\nDone. Commit and push the updated JSON files.")


if __name__ == "__main__":
    main()
