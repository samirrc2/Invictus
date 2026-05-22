"""
Invictus — Fundamental Intelligence Agent (v3)
Uses yfinance financials as the primary source for growth and margin signals.
Replaces unreliable SEC text extraction with quantitative institutional metrics.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import streamlit as st

from invictus.agents.graph_state import PortfolioState
from invictus.config import OPENAI_API_KEY, LLM_MODEL

def _extract_yfinance_fundamental_signals(ticker: str) -> Dict[str, Any]:
    """Fetches quantitative fundamental signals using yfinance financials."""
    try:
        t = yf.Ticker(ticker)
        # 1. Get Quarterly Financials (latest 4 periods)
        qf = t.quarterly_financials
        if qf is None or qf.empty: return {"status": "Data source not available"}
        
        # 2. Extract Growth Signals (latest vs previous)
        # Note: Index names can vary, so we use flexible matching
        rev_row = qf.loc[qf.index.str.contains("Total Revenue", case=False)]
        net_row = qf.loc[qf.index.str.contains("Net Income", case=False)]
        
        if rev_row.empty: return {"status": "Incomplete financial data"}
        
        revs = rev_row.iloc[0].values
        net = net_row.iloc[0].values if not net_row.empty else [0, 0]
        
        # Growth Rate (Sequential)
        rev_growth = (revs[0] - revs[1]) / abs(revs[1]) if len(revs) > 1 and revs[1] != 0 else 0
        net_growth = (net[0] - net[1]) / abs(net[1]) if len(net) > 1 and net[1] != 0 else 0
        
        # 3. Decision Logic (Institutional Heuristics)
        f_conviction = np.clip((rev_growth + net_growth) / 2.0 / 0.10, -1.0, 1.0)
        g_momentum = np.clip(rev_growth / 0.05, -1.0, 1.0)
        r_deterioration = 0.1 if net[0] > 0 else 0.5
        
        return {
            "status": "Success",
            "fundamental_conviction": float(f_conviction),
            "fundamental_reasoning": f"Sequential Revenue Growth: {rev_growth:.1%}. Net Income Trend: {net_growth:+.1%}.",
            "guidance_momentum": float(g_momentum),
            "guidance_reasoning": f"Revenue trajectory is {'improving' if rev_growth > 0 else 'stable/declining'} sequential QoQ.",
            "risk_deterioration": float(r_deterioration),
            "risk_reasoning": f"Profitability status: {'Healthy/Positive' if net[0] > 0 else 'Risk/Negative earnings'}.",
            "supporting_drivers": [f"QoQ Revenue: {rev_growth:+.1%}", f"Net Income: {net_growth:+.1%}"],
            "risk_drivers": ["Negative earnings" if net[0] <= 0 else "N/A"]
        }
    except Exception:
        return {"status": "Data source not available"}

def run_filing_intel(state: PortfolioState) -> PortfolioState:
    tickers = list(state.weights.keys()) if state.weights else []
    results = {}
    progress_bar = st.progress(0, text="Analyzing Company Fundamentals...")
    
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Fundamental Intel: {ticker}")
        results[ticker] = _extract_yfinance_fundamental_signals(ticker)
        
    state.filing_intel = results
    progress_bar.empty()
    return state
