"""
Invictus — Fundamental Intelligence Agent (v3)
Uses yfinance financials as the primary source for growth and margin signals.
Replaces unreliable SEC text extraction with quantitative institutional metrics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT SIGNALS (consumed by synthesis_agent.py)
───────────────────────────────────────────────

1. fundamental_conviction ∈ [-1, +1]
   Formula: clip((rev_growth × 0.35 + net_growth × 0.35 + op_growth × 0.30) / 0.10, -1, 1)
   Normalization: divide by 0.10 (i.e., 10% blended growth = full conviction)
   Rationale: For US large-caps, 10% QoQ revenue growth is exceptional (~90th pctile).
              Average S&P 500 QoQ revenue growth is ~2-3%. Division by 0.10 means:
                +5% blended → +0.5 conviction (moderate positive)
                +10% blended → +1.0 conviction (maximum)
                -10% blended → -1.0 conviction (maximum negative)

2. guidance_momentum ∈ [-1, +1]
   Formula: clip(rev_growth / 0.05, -1, 1)
   Normalization: divide by 0.05 (i.e., 5% QoQ revenue growth = full momentum)
   Rationale: Revenue trajectory is a leading indicator. We use a tighter
              threshold (5% vs 10%) because guidance_momentum should be MORE
              sensitive to small changes — it represents acceleration, not level.
              This is intentionally more sensitive than fundamental_conviction.

3. risk_deterioration ∈ [0, 1]
   Formula: heuristic checklist (0.1 base + conditions)
     - Base: 0.1 (profitable) or 0.5 (unprofitable)
     - Add 0.2 if operating income declining >10%
   Rationale: Binary/discrete risk flags rather than continuous signal.
              Profitable companies start at low risk (0.1), unprofitable at
              elevated risk (0.5). Operating margin compression adds 0.2.
              This is consumed by synthesis as: risk_signal = -(risk × 2 - 1)
              mapping 0→+1, 0.5→0, 1→-1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import streamlit as st

from invictus.agents.graph_state import PortfolioState
# LLM calls go through invictus.llm — no direct API key imports needed


def _find_row(df: pd.DataFrame, keywords: List[str]) -> Optional[pd.Series]:
    """Find first row in a yfinance financials DataFrame matching any keyword."""
    if df is None or df.empty:
        return None
    for kw in keywords:
        matches = df.loc[df.index.str.contains(kw, case=False, na=False)]
        if not matches.empty:
            return matches.iloc[0]
    return None


def _safe_growth(current, previous) -> float:
    """Calculate growth rate safely, handling zeros and NaN."""
    try:
        current = float(current) if current is not None and not pd.isna(current) else 0
        previous = float(previous) if previous is not None and not pd.isna(previous) else 0
        if previous == 0 or abs(previous) < 1:
            return 0.0
        return (current - previous) / abs(previous)
    except (ValueError, TypeError):
        return 0.0


def _extract_yfinance_fundamental_signals(ticker: str) -> Dict[str, Any]:
    """Fetches quantitative fundamental signals using yfinance financials."""
    try:
        t = yf.Ticker(ticker)

        # 1. Get Quarterly Financials (latest 4 periods)
        qf = t.quarterly_financials
        if qf is None or qf.empty:
            # Try annual as fallback
            qf = t.financials
            if qf is None or qf.empty:
                return {"status": "Data source not available"}

        # 2. Extract Revenue (try multiple possible index names)
        rev_row = _find_row(qf, ["Total Revenue", "Revenue", "Net Revenue", "Total Net Revenue"])
        if rev_row is None:
            return {"status": "Incomplete financial data — no revenue found"}

        revs = rev_row.values

        # 3. Extract Net Income
        net_row = _find_row(qf, ["Net Income", "Net Income Common", "Net Income From Continuing"])
        net = net_row.values if net_row is not None else [0, 0]

        # 4. Extract Operating Income
        op_row = _find_row(qf, ["Operating Income", "Operating Profit", "EBIT"])
        op = op_row.values if op_row is not None else [0, 0]

        # 5. Extract Gross Profit
        gp_row = _find_row(qf, ["Gross Profit"])
        gp = gp_row.values if gp_row is not None else [0, 0]

        # 6. Growth Rates (Sequential QoQ)
        rev_growth = _safe_growth(revs[0], revs[1]) if len(revs) > 1 else 0
        net_growth = _safe_growth(net[0], net[1]) if len(net) > 1 else 0
        op_growth = _safe_growth(op[0], op[1]) if len(op) > 1 else 0

        # 7. Margin Analysis
        rev_latest = float(revs[0]) if revs[0] is not None and not pd.isna(revs[0]) else 1
        gross_margin = float(gp[0]) / rev_latest if gp is not None and len(gp) > 0 and rev_latest != 0 and gp[0] is not None and not pd.isna(gp[0]) else 0
        op_margin = float(op[0]) / rev_latest if op is not None and len(op) > 0 and rev_latest != 0 and op[0] is not None and not pd.isna(op[0]) else 0

        # 8. YoY comparison (if 4+ quarters available)
        yoy_rev_growth = _safe_growth(revs[0], revs[3]) if len(revs) > 3 else None

        # 9. Decision Logic — Scoring
        #
        # fundamental_conviction ∈ [-1, +1]:
        #   = clip((0.35·rev_growth + 0.35·net_growth + 0.30·op_growth) / 0.10, -1, 1)
        #   Denominator 0.10: 10% blended QoQ growth = full conviction (+1.0)
        #   Example: rev +5%, net +8%, op +3% → (0.0175+0.028+0.009)/0.10 = 0.545
        f_conviction = np.clip(
            (rev_growth * 0.35 + net_growth * 0.35 + op_growth * 0.30) / 0.10,
            -1.0, 1.0
        )

        # guidance_momentum ∈ [-1, +1]:
        #   = clip(rev_growth / 0.05, -1, 1)
        #   Denominator 0.05: 5% QoQ revenue growth = full momentum (+1.0)
        #   More sensitive than conviction — captures acceleration signal
        g_momentum = np.clip(rev_growth / 0.05, -1.0, 1.0)

        # risk_deterioration ∈ [0, 1]:
        #   Base: 0.1 (profitable) or 0.5 (unprofitable)
        #   +0.2 if operating income declining >10% QoQ
        #   Consumed by synthesis as: -(risk×2-1) maps 0→+1, 0.5→0, 1→-1
        net_latest = float(net[0]) if net[0] is not None and not pd.isna(net[0]) else 0
        r_deterioration = 0.1 if net_latest > 0 else 0.5
        if op_growth < -0.1:
            r_deterioration += 0.2
        r_deterioration = float(np.clip(r_deterioration, 0, 1))

        # Build drivers
        supporting = [f"QoQ Revenue: {rev_growth:+.1%}", f"Net Income: {net_growth:+.1%}"]
        if op_growth != 0:
            supporting.append(f"Operating Income: {op_growth:+.1%}")
        if yoy_rev_growth is not None:
            supporting.append(f"YoY Revenue: {yoy_rev_growth:+.1%}")

        risk_drivers = []
        if net_latest <= 0:
            risk_drivers.append("Negative earnings")
        if rev_growth < 0:
            risk_drivers.append("Revenue declining QoQ")
        if op_growth < -0.1:
            risk_drivers.append("Operating income deteriorating")
        if not risk_drivers:
            risk_drivers.append("No major risk flags")

        return {
            "status": "Success",
            "fundamental_conviction": float(f_conviction),
            "fundamental_reasoning": (
                f"Multi-factor growth: Revenue {rev_growth:+.1%}, Net Income {net_growth:+.1%}, "
                f"Operating Income {op_growth:+.1%}. "
                f"Gross Margin: {gross_margin:.1%}, Op Margin: {op_margin:.1%}."
            ),
            "guidance_momentum": float(g_momentum),
            "guidance_reasoning": (
                f"Revenue trajectory is {'accelerating' if rev_growth > 0.05 else 'improving' if rev_growth > 0 else 'stable' if rev_growth > -0.02 else 'declining'} QoQ."
            ),
            "risk_deterioration": float(r_deterioration),
            "risk_reasoning": (
                f"Profitability: {'Healthy' if net_latest > 0 else 'Risk/Negative'}. "
                f"Margin trend: {'expanding' if op_growth > 0 else 'compressing'}."
            ),
            "supporting_drivers": supporting,
            "risk_drivers": risk_drivers,
            "raw_metrics": {
                "revenue_growth": rev_growth,
                "net_income_growth": net_growth,
                "operating_income_growth": op_growth,
                "gross_margin": gross_margin,
                "operating_margin": op_margin,
                "yoy_revenue_growth": yoy_rev_growth,
            },
        }
    except Exception as e:
        return {"status": f"Data source not available: {e}"}


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
