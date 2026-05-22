"""
Invictus — Institutional Flow Agent (v3)
Uses yfinance as the primary source for institutional and insider data.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import streamlit as st

from invictus.agents.graph_state import PortfolioState

# ── Smart Money Classification ────────────────────────────────────────

SMART_MONEY_KEYWORDS = [
    "hedge fund", "capital management", "partners", "advisors", "macro",
    "quantitative", "citadel", "bridgewater", "renaissance", "two sigma",
    "millennium", "point72", "tiger global", "coatue", "d1 capital",
    "viking", "lone pine", "pershing", "third point", "elliott",
    "baupost", "appaloosa", "ares", "oaktree",
]

PASSIVE_KEYWORDS = ["vanguard", "blackrock", "state street", "fidelity index", "ishares", "schwab", "invesco", "spdr"]

def _classify_holder(holder_name: str) -> str:
    name_lower = holder_name.lower() if holder_name else ""
    if any(kw in name_lower for kw in SMART_MONEY_KEYWORDS): return "smart_money"
    if any(kw in name_lower for kw in PASSIVE_KEYWORDS): return "passive"
    return "active"

# ── Data Fetching ─────────────────────────────────────────────────────

def _fetch_yfinance_flow_data(ticker: str) -> Dict[str, Any]:
    """Surgically fetches institutional and insider data using yfinance."""
    try:
        t = yf.Ticker(ticker)
        # 1. Institutional Holders
        inst = t.institutional_holders
        # 2. Insider Transactions
        insiders = t.insider_transactions
        
        inst_list = []
        if inst is not None and not inst.empty:
            # yfinance returns: Date Reported, Holder, pctHeld, Shares, Value, pctChange
            for _, row in inst.iterrows():
                inst_list.append({
                    "holder": row.get("Holder", "Unknown"),
                    "shares": row.get("Shares", 0),
                    "pctChange": row.get("pctChange", 0),
                    "value": row.get("Value", 0)
                })
        
        insider_list = []
        if insiders is not None and not insiders.empty:
            # yfinance returns: Shares, Value, Text, Insider, Position, Start Date, Ownership
            for _, row in insiders.iterrows():
                insider_list.append({
                    "reportingName": row.get("Insider", "Unknown"),
                    "typeOfOwner": row.get("Position", "Officer"),
                    "transactionText": row.get("Text", ""),
                    "securitiesTransacted": row.get("Shares", 0),
                    "value": row.get("Value", 0),
                    "transactionDate": str(row.get("Start Date", ""))
                })

        return {
            "institutional": inst_list,
            "insiders": insider_list,
            "source": "yfinance",
            "status": "Success" if (inst_list or insider_list) else "Data source not available"
        }
    except Exception:
        return {"status": "Data source not available"}

# ── Scoring ───────────────────────────────────────────────────────────

def _score_flows(ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates conviction scores from yfinance data."""
    if data.get("status") != "Success":
        return {"status": "Data source not available", "flow_composite": 0, "smart_money_pct": 0, "institutional_conviction": 0, "insider_alignment": 0}

    inst = data.get("institutional", [])
    insiders = data.get("insiders", [])
    
    # 1. Smart Money & Concentration
    smart_shares = sum(h["shares"] for h in inst if _classify_holder(h["holder"]) == "smart_money")
    total_shares = sum(h["shares"] for h in inst) or 1
    smart_pct = smart_shares / total_shares
    
    # 2. Institutional Conviction (Change in holdings)
    avg_change = np.mean([h["pctChange"] for h in inst if h["pctChange"] != 0]) if inst else 0
    inst_conviction = float(np.clip(avg_change / 0.05, -1.0, 1.0))

    # 3. Insider Alignment
    total_buys = sum(1 for tx in insiders if "Buy" in tx["transactionText"] or "Purchase" in tx["transactionText"])
    total_sells = sum(1 for tx in insiders if "Sale" in tx["transactionText"] or "Sell" in tx["transactionText"])
    insider_align = float(np.clip((total_buys - total_sells) / (total_buys + total_sells + 1), -1.0, 1.0))
    
    # 4. Composite
    flow_comp = float(np.clip(inst_conviction * 0.6 + insider_align * 0.4, -1.0, 1.0))
    
    return {
        "status": "Success",
        "source": "yfinance",
        "flow_composite": flow_comp,
        "smart_money_pct": smart_pct,
        "institutional_conviction": inst_conviction,
        "insider_alignment": insider_align,
        "insider_buys": total_buys,
        "insider_sells": total_sells,
        "estimated_accumulation": "strong_accumulation" if flow_comp > 0.4 else ("distribution" if flow_comp < -0.4 else "neutral"),
        "net_insider_value": sum(tx["value"] for tx in insiders if "Buy" in tx["transactionText"]) - sum(tx["value"] for tx in insiders if "Sale" in tx["transactionText"]),
        "notable_transactions": [{"type": "BUY" if "Buy" in tx["transactionText"] else "SELL", "reporter": tx["reportingName"], "value": tx["value"], "days_ago": 30} for tx in insiders if tx["value"] > 500000][:5]
    }

def analyze_flows(state: PortfolioState) -> PortfolioState:
    tickers = list(state.weights.keys()) if state.weights else []
    flow_intel = {}
    raw_results = {} # Initialize the raw results dictionary
    progress_bar = st.progress(0, text="Analyzing Institutional Flows...")
    
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Flow Intel: {ticker}")
        raw = _fetch_yfinance_flow_data(ticker)
        raw_results[ticker] = raw # Store the raw data
        flow_intel[ticker] = _score_flows(ticker, raw)
        
    succ = {k: v for k, v in flow_intel.items() if v.get("status") == "Success"}
    p_summary = {
        "avg_smart_money_pct": float(np.mean([v["smart_money_pct"] for v in succ.values()])) if succ else 0,
        "avg_insider_alignment": float(np.mean([v["insider_alignment"] for v in succ.values()])) if succ else 0,
        "accumulating_count": sum(1 for v in succ.values() if "accumulation" in v["estimated_accumulation"]),
        "distributing_count": sum(1 for v in succ.values() if v["estimated_accumulation"] == "distribution")
    }
    
    state.flow_signals = {
        "intel": flow_intel,
        "raw": raw_results, # Restored for ML agent
        "portfolio_summary": p_summary,
        "status": "Success" if succ else "Data source not available"
    }
    progress_bar.empty()
    return state
