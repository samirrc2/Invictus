"""
Invictus — Capital Flow Intelligence Agent (v4)

Institutional-grade analysis of capital movements around equity positions.
Uses yfinance as the primary data source for institutional holders and insider transactions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARCHITECTURE: 3 SUB-BUCKETS PER STOCK
──────────────────────────────────────

Each stock is analyzed across three independent intelligence buckets:

┌─────────────────────────────────────────────────────────────────────────┐
│ BUCKET 1: INSIDER INTELLIGENCE (weight: 0.35)                          │
│                                                                         │
│ What we look for:                                                       │
│   GREEN FLAGS:                                                          │
│     - Cluster buys: 2+ insiders buying within same reporting period     │
│     - C-suite purchases: CEO/CFO/COO buying on open market              │
│     - Large-value buys: single purchase > $500K                         │
│     - Buying into weakness: insider buys when stock is in drawdown      │
│                                                                         │
│   RED FLAGS:                                                            │
│     - C-suite large sells: CEO/CFO selling large portion                │
│     - Cluster sells: 3+ insiders selling in same period                 │
│     - Accelerating sells: sell frequency increasing                     │
│     - Sells into strength: insiders selling at/near 52w highs           │
│                                                                         │
│ Score: [-1, +1] via (green_weight - red_weight) / max(green+red, 1)    │
│ Role weights: CEO/CFO = 3x, VP/Director = 2x, Officer = 1x            │
├─────────────────────────────────────────────────────────────────────────┤
│ BUCKET 2: FUND ACCUMULATION TREND (weight: 0.40)                       │
│                                                                         │
│ What we look for:                                                       │
│   - Are top holders INCREASING or DECREASING positions?                 │
│   - Smart money (hedge funds) behavior vs passive (index funds)         │
│   - New positions opened vs positions closed                            │
│   - Average % change direction and magnitude                            │
│                                                                         │
│ Score: [-1, +1] via (non-passive holders only):                          │
│   smart_trend × 0.50 + active_trend × 0.30 + breadth × 0.20           │
│   where:                                                                │
│     smart_trend = clip(avg smart money % change / 0.05, -1, 1)          │
│     active_trend = clip(avg non-passive % change / 0.05, -1, 1)         │
│     breadth = (active_increasing - active_decreasing) / active_total    │
├─────────────────────────────────────────────────────────────────────────┤
│ BUCKET 3: CAPITAL CONCENTRATION (weight: 0.25)                         │
│                                                                         │
│ What we look for:                                                       │
│   - Top-5 holder concentration (% of total institutional shares)        │
│   - Smart money concentration (what % is held by hedge funds/quants)    │
│   - High concentration + increasing = strong conviction play            │
│   - Low concentration + decreasing = potential derisking                │
│                                                                         │
│ Score: [-1, +1] via:                                                    │
│   smart_money_pct × 2 - 0.3 (centered: 15% smart money = neutral)      │
│   Boosted if smart money is also accumulating (trend positive)          │
└─────────────────────────────────────────────────────────────────────────┘

COMPOSITE: flow_composite = 0.35·insider + 0.40·fund_trend + 0.25·concentration
           ∈ [-1, +1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import streamlit as st

_log = logging.getLogger(__name__)

from invictus.agents.graph_state import PortfolioState

# ══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

# ── Active fund overrides — checked FIRST ──
# These are active stock-picking funds run by companies that also run
# passive/index products. Without this list, "Fidelity Contrafund" would
# match "fidelity" in PASSIVE_KEYWORDS and be wrongly excluded from signal.
# Rule: if the fund name contains any of these, classify as active
# regardless of what the parent company name would suggest.
ACTIVE_FUND_OVERRIDES = [
    # Fidelity active funds (among the largest active equity funds globally)
    "contrafund", "growth company", "blue chip", "otc portfolio",
    "magellan", "low-priced stock", "puritan", "balanced",
    "fidelity select", "fidelity advisor",
    # JPMorgan active
    "jpmorgan growth", "jpmorgan equity", "jpmorgan large cap",
    "jpmorgan mid cap", "jpmorgan small cap",
    # Morgan Stanley active
    "morgan stanley inst", "morgan stanley growth",
    "counterpoint global",
    # BlackRock active (distinct from iShares ETFs)
    "blackrock equity dividend", "blackrock capital appreciation",
    "blackrock health sciences",
    # Generic active-fund keywords
    "growth fund", "value fund", "opportunity fund", "select fund",
    "aggressive", "strategic equity", "capital appreciation",
    "stock fund", "equity income",
]

# ── Passive / Index funds ──
# These manage trillions via index replication. Their position changes
# reflect fund flows and rebalancing, NOT stock-picking conviction.
# NOTE: checked AFTER ACTIVE_FUND_OVERRIDES to avoid misclassifying
# active arms of passive parents (Fidelity Contrafund, etc.).
PASSIVE_KEYWORDS = [
    "vanguard", "blackrock", "state street", "fidelity", "ishares",
    "schwab", "invesco", "spdr", "index fund", "etf trust",
    "geode capital", "northern trust", "dimensional fund",
    "legal & general", "ssga", "jpmorgan chase",  # JPM index arm
    "bank of america", "morgan stanley",  # prime brokerage/custody
    "norges bank",  # sovereign wealth
]

# ── Smart money — active stock pickers with research-driven conviction ──
# These are hedge funds, activist investors, and quant shops whose
# position changes signal deliberate allocation decisions.
# NOTE: generic terms like "capital management" or "partners" are TOO BROAD
# and will match passive managers. Only use specific firm names.
SMART_MONEY_KEYWORDS = [
    "hedge fund",
    # Named firms (top ~30 by AUM/influence)
    "citadel", "bridgewater", "renaissance", "two sigma",
    "millennium", "point72", "tiger global", "coatue", "d1 capital",
    "viking global", "lone pine", "pershing square", "third point", "elliott management",
    "baupost", "appaloosa", "ares management", "oaktree",
    "soros fund", "druckenmiller", "duquesne",
    "ackman", "icahn", "greenlight", "einhorn",
    "tiger management", "maverick capital", "farallon",
    "wellington management", "capital research",  # Large active managers
    "t. rowe price",  # Active growth
]

# Insider role hierarchy — higher = more informative signal
ROLE_WEIGHTS = {
    "ceo": 3.0, "chief executive": 3.0,
    "cfo": 3.0, "chief financial": 3.0,
    "coo": 2.5, "chief operating": 2.5,
    "cto": 2.5, "chief technology": 2.5,
    "president": 2.5,
    "evp": 2.0, "svp": 2.0, "executive vice": 2.0, "senior vice": 2.0,
    "vp": 1.5, "vice president": 1.5,
    "director": 1.5, "board": 1.5,
    "general counsel": 2.0,
    "officer": 1.0, "controller": 1.0,
}

BUY_KEYWORDS = ["Buy", "Purchase", "buy", "purchase", "Acquisition", "Exercise"]
SELL_KEYWORDS = ["Sale", "Sell", "sale", "sell", "Disposition", "disposition"]


def _classify_holder(holder_name: str) -> str:
    """
    Classify institutional holder as smart_money, passive, or active.

    Priority order (highest first):
      1. ACTIVE_FUND_OVERRIDES — catches active arms of passive parents
         e.g. "Fidelity Contrafund" → active (not passive)
      2. SMART_MONEY_KEYWORDS — hedge funds, activists, quant shops
      3. PASSIVE_KEYWORDS — index funds, ETFs, custody
      4. Default → active
    """
    name_lower = holder_name.lower() if holder_name else ""
    # 1. Active override FIRST — rescues active funds from passive parent match
    if any(kw in name_lower for kw in ACTIVE_FUND_OVERRIDES):
        return "active"
    # 2. Smart money — these are never passive regardless of name
    if any(kw in name_lower for kw in SMART_MONEY_KEYWORDS):
        return "smart_money"
    # 3. Passive — index/ETF/custody
    if any(kw in name_lower for kw in PASSIVE_KEYWORDS):
        return "passive"
    return "active"


def _get_role_weight(title: str) -> float:
    """Get informativeness weight based on insider's corporate role."""
    title_lower = title.lower() if title else ""
    for role_key, weight in ROLE_WEIGHTS.items():
        if role_key in title_lower:
            return weight
    return 1.0  # Default: unknown role gets base weight


def _is_buy(text: str) -> bool:
    return any(k in text for k in BUY_KEYWORDS)


def _is_sell(text: str) -> bool:
    return any(k in text for k in SELL_KEYWORDS)


def _time_decay(date_str: str, half_life_days: float = 90.0) -> float:
    """
    Exponential time decay based on transaction age.

    Recent transactions carry more weight than stale ones.
    Formula: decay = 2^(-age_days / half_life)
      half_life=90 days:
        1 week old  → 0.95 (fresh — nearly full weight)
        1 month old → 0.79
        3 months    → 0.50 (half weight)
        6 months    → 0.25
        1 year      → 0.06 (nearly irrelevant)

    Returns 1.0 if date cannot be parsed (conservative — don't penalize
    missing data).
    """
    if not date_str:
        return 1.0
    try:
        dt = pd.to_datetime(date_str)
        if pd.isna(dt):
            return 1.0
        age_days = (datetime.now() - dt.to_pydatetime().replace(tzinfo=None)).days
        if age_days < 0:
            return 1.0  # Future date — treat as fresh
        return float(2 ** (-age_days / half_life_days))
    except Exception:
        return 1.0


# ══════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════

def _safe_col(row, candidates, default=None):
    """Try multiple column name candidates and return the first match."""
    for c in candidates:
        val = row.get(c)
        if val is not None:
            return val
    return default if default is not None else 0


def _fetch_flow_data(ticker: str) -> Dict[str, Any]:
    """
    Fetches institutional and insider data using yfinance.

    Enrichments over raw yfinance data:
      - Insider transactions get cross-referenced with insider_roster_holders
        to compute stake context ("selling 1% of their 14% stake")
      - Institutional holders get previous % Out computed from current + % Change
        so we can show "increased from 0.5% to 0.6%"
      - shares_outstanding is fetched to convert share counts to %
    """
    import time as _t
    _fetch_start = _t.perf_counter()
    try:
        t = yf.Ticker(ticker)

        # ── 0. Shares outstanding + ownership breakdown ──
        shares_outstanding = 0
        ownership_breakdown = {}
        try:
            info = t.info or {}
            shares_outstanding = float(info.get("sharesOutstanding", 0) or 0)
            _inst_pct = info.get("heldPercentInstitutions", 0) or 0
            _ins_pct = info.get("heldPercentInsiders", 0) or 0
            _retail_pct = max(0, 1.0 - _inst_pct - _ins_pct)
            ownership_breakdown = {
                "institutional_pct": round(float(_inst_pct), 4),
                "insider_pct": round(float(_ins_pct), 4),
                "retail_pct": round(float(_retail_pct), 4),
            }
        except Exception as e:
            _log.warning("Flow: %s shares_outstanding fetch failed: %s", ticker, e)

        # ── 1. Insider Roster (total holdings per insider) ──
        # This gives us each insider's total stake so we can say
        # "CEO selling X% of their Y% stake"
        # Names differ between roster & transactions — roster may have
        # "Susan Li" while transactions have "LI SUSAN J."
        # We store by full name AND build a surname index for fuzzy matching.
        insider_roster = {}  # name_lower → {"shares_owned": N, "position": str}
        _surname_index = {}  # surname_lower → name_lower (for fuzzy matching)
        try:
            roster = t.insider_roster_holders
            if roster is not None and not roster.empty:
                for _, row in roster.iterrows():
                    name = str(_safe_col(row, ["Name", "name", "Insider"], ""))
                    shares_owned = float(_safe_col(row, [
                        "Shares Owned Directly", "sharesOwnedDirectly",
                        "Position Direct", "Shares", "shares",
                    ], 0))
                    pos = str(_safe_col(row, ["Position", "position", "Title"], ""))
                    if name:
                        key = name.lower().strip()
                        insider_roster[key] = {
                            "shares_owned": shares_owned,
                            "position": pos,
                        }
                        # Build surname index for fuzzy matching
                        # Roster name "Susan Li" → surname "li"
                        # Roster name "Mark Zuckerberg" → surname "zuckerberg"
                        parts = key.replace(".", "").split()
                        if parts:
                            _surname_index[parts[-1]] = key  # last word = surname
                            _surname_index[parts[0]] = key   # first word too (handles "LI SUSAN")
        except Exception as e:
            _log.warning("Flow: %s insider_roster fetch failed: %s", ticker, e)

        # ── 2. Institutional Holders ──
        inst_list = []
        try:
            inst = t.institutional_holders
            if inst is not None and not inst.empty:
                for _, row in inst.iterrows():
                    holder = _safe_col(row, ["Holder", "holder", "Name", "name"], "Unknown")
                    shares = float(_safe_col(row, ["Shares", "shares", "Position", "position"], 0))
                    pct_change_raw = float(_safe_col(row, ["% Change", "pctChange", "% change", "Change"], 0))
                    # Normalize: yfinance sometimes reports % change as percentage
                    # points (e.g. 15.0 meaning 15%) instead of decimals (0.15).
                    # Threshold at 5.0: a fund legitimately quadrupling its position
                    # (4.0 = 400%) is plausible. But 15.0 (1500%) almost certainly
                    # means "15 percentage points." Old threshold of 2.0 was wrong —
                    # it mishandled funds that tripled positions (3.0 → 0.03).
                    if abs(pct_change_raw) > 5.0:
                        pct_change = pct_change_raw / 100.0
                    else:
                        pct_change = pct_change_raw
                    # Cap at ±200% (a fund can legitimately triple; beyond that is noise)
                    pct_change = float(np.clip(pct_change, -2.0, 2.0))
                    value = float(_safe_col(row, ["Value", "value"], 0))
                    # % Out = this holder's shares as % of total outstanding
                    pct_held = float(_safe_col(row, ["% Out", "pctHeld", "% Held", "pct_held"], 0))

                    # Compute previous % held: pctHeld_prev = pctHeld / (1 + pctChange)
                    # This lets us show "increased from 0.5% to 0.6%"
                    if pct_change != 0 and abs(1 + pct_change) > 0.01:
                        pct_held_prev = pct_held / (1 + pct_change)
                    else:
                        pct_held_prev = pct_held

                    # Date Reported — when this filing was made
                    date_rep = _safe_col(row, ["Date Reported", "dateReported", "Date", "date"], "")
                    date_rep_str = str(date_rep)
                    try:
                        dt = pd.to_datetime(date_rep)
                        date_rep_str = dt.strftime("%b %d, %Y") if not pd.isna(dt) else str(date_rep)
                    except Exception:
                        pass

                    inst_list.append({
                        "holder": str(holder),
                        "shares": shares,
                        "pctChange": pct_change,
                        "pctChangeRaw": pct_change_raw,
                        "value": value,
                        "pctHeld": pct_held,
                        "pctHeldPrev": float(pct_held_prev),
                        "date_reported": date_rep_str,
                        "classification": _classify_holder(str(holder)),
                    })
        except Exception as e:
            _log.warning("Flow: %s institutional_holders fetch failed: %s", ticker, e)

        # ── 3. Insider Transactions ──
        insider_list = []
        try:
            insiders = t.insider_transactions
            if insiders is not None and not insiders.empty:
                for _, row in insiders.iterrows():
                    insider_name = _safe_col(row, ["Insider", "insider", "Insider Trading"], "Unknown")
                    position = _safe_col(row, ["Position", "position", "Title"], "Officer")
                    text = _safe_col(row, ["Text", "text", "Transaction", "transaction"], "")
                    shares_tx = float(_safe_col(row, ["Shares", "shares"], 0))
                    value = float(_safe_col(row, ["Value", "value"], 0))
                    date = _safe_col(row, ["Start Date", "startDate", "Date", "date"], "")

                    # Cross-reference with insider roster for stake context
                    # Transactions use "LI SUSAN J." — roster uses "Susan Li"
                    # Strategy: exact match → surname index → word overlap
                    name_key = str(insider_name).lower().strip().replace(".", "")
                    roster_entry = insider_roster.get(name_key, {})
                    if not roster_entry:
                        # Try surname index: "li susan j" → try "li", then "susan", then "j"
                        name_parts = name_key.split()
                        for part in name_parts:
                            if len(part) > 1 and part in _surname_index:
                                matched_key = _surname_index[part]
                                roster_entry = insider_roster.get(matched_key, {})
                                if roster_entry:
                                    break
                    shares_owned = roster_entry.get("shares_owned", 0)

                    # Compute stake percentages
                    # stake_pct = insider's total holding as % of company
                    stake_pct = (shares_owned / shares_outstanding * 100) if shares_outstanding > 0 and shares_owned > 0 else 0
                    # tx_pct_of_stake = this transaction as % of insider's total holding
                    tx_pct_of_stake = (shares_tx / shares_owned * 100) if shares_owned > 0 else 0

                    # Format date cleanly
                    date_str = str(date)
                    try:
                        dt = pd.to_datetime(date)
                        date_str = dt.strftime("%b %d, %Y") if not pd.isna(dt) else str(date)
                    except Exception:
                        pass

                    insider_list.append({
                        "name": str(insider_name),
                        "role": str(position) or roster_entry.get("position", "Officer"),
                        "transaction": str(text),
                        "shares": shares_tx,
                        "value": value,
                        "date": date_str,
                        "role_weight": _get_role_weight(str(position)),
                        # Enriched stake context
                        "shares_owned": shares_owned,
                        "stake_pct": round(stake_pct, 2),      # e.g. 14.2 (%)
                        "tx_pct_of_stake": round(tx_pct_of_stake, 1),  # e.g. 1.3 (%)
                    })
        except Exception as e:
            _log.warning("Flow: %s insider_transactions fetch failed: %s", ticker, e)

        _status = "Success" if (inst_list or insider_list) else "Data source not available"
        _latency = (_t.perf_counter() - _fetch_start) * 1000
        _log.info("Flow fetch %s: status=%s, inst=%d, ins=%d, %.0fms",
                  ticker, _status, len(inst_list), len(insider_list), _latency)
        # Log to observability data_health table
        try:
            from invictus.observability.store import insert, generate_run_id
            insert("data_health", {
                "run_id": generate_run_id(),
                "source": "yfinance_flow",
                "ticker": ticker,
                "status": "success" if _status == "Success" else "empty",
                "latency_ms": _latency,
                "records_fetched": len(inst_list) + len(insider_list),
                "error_message": None if _status == "Success" else "No institutional or insider data returned",
            })
        except Exception:
            pass
        return {
            "institutional": inst_list,
            "insiders": insider_list,
            "shares_outstanding": shares_outstanding,
            "ownership_breakdown": ownership_breakdown,
            "source": "yfinance",
            "status": _status,
        }
    except Exception as e:
        _latency = (_t.perf_counter() - _fetch_start) * 1000
        _log.error("Flow: %s _fetch_flow_data CRASHED: %s", ticker, e, exc_info=True)
        # Log crash to observability
        try:
            from invictus.observability.store import insert, generate_run_id
            insert("data_health", {
                "run_id": generate_run_id(),
                "source": "yfinance_flow",
                "ticker": ticker,
                "status": "error",
                "latency_ms": _latency,
                "records_fetched": 0,
                "error_message": f"{type(e).__name__}: {str(e)[:300]}",
            })
        except Exception:
            pass
        return {"institutional": [], "insiders": [], "shares_outstanding": 0, "source": "yfinance", "status": "Data source not available"}


# ══════════════════════════════════════════════════════════════════════════
# BUCKET 1: INSIDER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════

def _analyze_insider_intelligence(insiders: List[Dict]) -> Dict[str, Any]:
    """
    Intelligent insider transaction analysis with MATERIALITY-BASED scoring.

    Key insight: raw dollar value is misleading for mega-caps. A CEO selling
    $340M of AAPL sounds alarming, but if that's 1% of their 14% stake it's
    routine compensation liquidation. What matters is: what % of their stake
    did they sell?

    Scoring methodology:
      Step 1 — Materiality per transaction:
        tx_pct_of_stake tells us what fraction of the insider's own holdings
        they transacted. This drives the signal strength:
          < 2% of stake  → 0.05  (routine — comp plan / 10b5-1 schedule)
          2-5%           → 0.15  (minor — probably planned)
          5-15%          → 0.40  (moderate — worth noting)
          15-30%         → 0.70  (significant — deliberate decision)
          > 30%          → 1.00  (major — strong conviction signal)
        If stake data unavailable, falls back to value-based (less reliable).

      Step 2 — Aggregate materiality:
        avg_materiality = weighted average of all transaction materialities
        (weighted by role_weight so CEO transactions count more).

      Step 3 — Direction × magnitude:
        score = direction × avg_materiality
        where direction = (green_count - red_count) / total_count ∈ [-1, +1]

      Result: all-sells with <2% of stake each → score ≈ -0.05 (noise)
              all-sells with 30%+ of stake     → score ≈ -1.0  (alarm)
              CEO buying 20% of salary in stock → strong positive signal
    """
    if not insiders:
        return {
            "score": 0.0,
            "red_flags": [],
            "green_flags": [],
            "summary": "No insider transaction data available.",
            "buy_count": 0,
            "sell_count": 0,
            "net_value": 0,
            "notable_transactions": [],
        }

    # Classify each transaction
    buys = [tx for tx in insiders if _is_buy(tx["transaction"])]
    sells = [tx for tx in insiders if _is_sell(tx["transaction"])]

    # ── Materiality-Based Scoring ──
    # Step 1: Aggregate by insider to get true % of stake change per person
    # (same logic as the notable-transactions aggregation below).
    # Step 2: Score each insider's materiality based on how much of their
    #         stake they actually moved over the period.
    # Step 3: direction × avg_materiality → score ∈ [-1, +1]
    _insider_totals = {}  # name → {bought, sold, shares_now, stake_pct, role_weight, latest_date}
    for tx in insiders:
        nm = tx["name"]
        is_buy = _is_buy(tx["transaction"])
        if nm not in _insider_totals:
            _insider_totals[nm] = {
                "bought": 0, "sold": 0,
                "shares_now": tx.get("shares_owned", 0),
                "stake_pct": tx.get("stake_pct", 0),  # % of company
                "role_weight": tx["role_weight"],
                "latest_date": tx.get("date", ""),
            }
        if is_buy:
            _insider_totals[nm]["bought"] += tx["shares"]
        else:
            _insider_totals[nm]["sold"] += tx["shares"]
        # Track most recent transaction date for time decay
        if tx.get("date", "") > _insider_totals[nm]["latest_date"]:
            _insider_totals[nm]["latest_date"] = tx.get("date", "")
        if tx.get("shares_owned", 0) > _insider_totals[nm]["shares_now"]:
            _insider_totals[nm]["shares_now"] = tx.get("shares_owned", 0)
        if tx.get("stake_pct", 0) > _insider_totals[nm]["stake_pct"]:
            _insider_totals[nm]["stake_pct"] = tx.get("stake_pct", 0)

    def _insider_materiality(info: Dict) -> float:
        """
        Two-dimensional materiality:
          1. pct_moved: what % of THEIR OWN stake did they sell?
          2. company_weight: how much does their stake matter to the company?

        An Officer with 0.00% of AAPL selling 79% of their holdings = noise.
        A founder with 5% selling 30% = major signal.

        Formula: personal_materiality × company_relevance
          company_relevance:
            stake >= 1%   → 1.0  (significant holder — their moves matter)
            stake 0.1-1%  → 0.6  (meaningful holder)
            stake 0.01-0.1% → 0.2  (small holder — limited signal)
            stake < 0.01% → 0.05 (negligible — compensation liquidation)
        """
        net_change = info["bought"] - info["sold"]
        shares_now = info["shares_now"]
        stake_pct = info["stake_pct"]  # % of company they own

        # 1. Personal materiality: what % of their stake did they move?
        if shares_now == 0 and info["sold"] > 0:
            # Full exit — they sold everything. This is the strongest signal.
            pct_moved = 100.0
        elif shares_now > 0 and abs(net_change) > 0:
            shares_start = shares_now - net_change
            if shares_start > 0:
                pct_moved = abs(net_change) / shares_start * 100
            else:
                pct_moved = 100.0  # Entirely new position
        elif shares_now == 0 and info["bought"] > 0:
            # New position opened (bought from zero)
            pct_moved = 100.0
        else:
            return 0.1  # No share data at all — assume low

        if pct_moved < 2:
            personal = 0.05
        elif pct_moved < 5:
            personal = 0.15
        elif pct_moved < 15:
            personal = 0.40
        elif pct_moved < 30:
            personal = 0.70
        else:
            personal = 1.00

        # 2. Company relevance: does this person's stake matter?
        if stake_pct >= 1.0:
            relevance = 1.0     # Significant holder
        elif stake_pct >= 0.1:
            relevance = 0.6     # Meaningful
        elif stake_pct >= 0.01:
            relevance = 0.2     # Small — limited signal
        else:
            relevance = 0.05    # Negligible — comp liquidation

        return personal * relevance

    # Weighted-average materiality across all insiders (with time decay)
    # Time decay: half-life 90 days. A 6-month-old transaction → 25% weight.
    # This prevents stale 10-month-old transactions from carrying equal
    # weight to last week's filing.
    total_role_wt = sum(
        i["role_weight"] * _time_decay(i.get("latest_date", ""), half_life_days=90)
        for i in _insider_totals.values()
    )
    if total_role_wt > 0:
        avg_materiality = sum(
            i["role_weight"]
            * _time_decay(i.get("latest_date", ""), half_life_days=90)
            * _insider_materiality(i)
            for i in _insider_totals.values()
        ) / total_role_wt
    else:
        avg_materiality = 0.0

    # Direction: computed PER PERSON, weighted by role × materiality × decay.
    # This solves the Test 5 problem: a CEO buying 3% of their 5% stake
    # (materiality=0.4) dominates 3 officers selling tiny fractions of
    # their 0.001% stakes (materiality≈0.0025 each).
    # Time decay ensures a CEO's buy last week dominates an officer's
    # sell from 8 months ago.
    #
    # Old formula: direction = (Σ role_wt_buyers - Σ role_wt_sellers) / total
    #   → CEO buy (3.0) vs 3 officers sell (3×1.0=3.0) → direction=0 (WRONG)
    #
    # New formula: direction = (Σ rw×mat×decay buyers - Σ rw×mat×decay sellers) / total
    #   → CEO buy (3.0×0.4×0.95=1.14) vs 3 officers (3×1.0×0.0025×0.3≈0.002) → +0.99
    buy_signal = 0.0
    sell_signal = 0.0
    for info in _insider_totals.values():
        net = info["bought"] - info["sold"]
        rw = info["role_weight"]
        mat = _insider_materiality(info)
        decay = _time_decay(info.get("latest_date", ""), half_life_days=90)
        weight = rw * mat * decay  # role × materiality × freshness
        if net > 0:
            buy_signal += weight
        elif net < 0:
            sell_signal += weight
        # net == 0 → neutral, contributes nothing

    total_signal = buy_signal + sell_signal
    if total_signal > 0:
        direction = (buy_signal - sell_signal) / total_signal
    else:
        direction = 0.0

    # Score IS the direction — materiality is already baked into it.
    # No separate "direction × avg_materiality" multiplication needed.
    #
    # AAPL: CEO (0.02% stake) selling 33% of holdings
    #   → mat=0.70×0.05=0.035, rw=3.0 → signal=0.105
    #   → if only seller: direction=-1, score=-0.105 (noise ✓)
    # Founder (5% stake) selling 30% of holdings
    #   → mat=1.0×1.0=1.0, rw=3.0 → signal=3.0
    #   → if only seller: direction=-1, score=-1.0 (alarm ✓)
    # Test 5: CEO buys (mat=0.4, rw=3.0→1.2) + 3 officers sell (mat≈0.0025, rw=1.0→0.0025 each)
    #   → buy=1.2, sell=0.0075, direction=+0.99 (CEO dominates ✓)
    #
    # But we still want the MAGNITUDE to reflect avg materiality, not just
    # direction. A score of ±0.99 when all moves are immaterial is misleading.
    # Solution: scale by avg_materiality so direction=±1 but low materiality
    # produces a low final score.
    score = float(np.clip(direction * avg_materiality, -1, 1))

    # ── Red Flag Detection ──
    red_flags = []
    csuite_sells = [tx for tx in sells if tx["role_weight"] >= 2.5]
    if csuite_sells:
        names = list(set(tx["name"] for tx in csuite_sells))
        total_val = sum(tx["value"] for tx in csuite_sells)
        red_flags.append(
            f"C-suite selling: {', '.join(names[:3])} "
            f"(${total_val/1e6:.1f}M total)"
        )

    unique_sellers = set(tx["name"] for tx in sells)
    if len(unique_sellers) >= 3:
        red_flags.append(
            f"Cluster sells: {len(unique_sellers)} unique insiders selling "
            f"(coordinated exit risk)"
        )

    large_sells = [tx for tx in sells if tx["value"] > 1_000_000]
    if large_sells and not csuite_sells:  # Don't double-report with C-suite
        red_flags.append(
            f"Large dispositions: {len(large_sells)} transactions > $1M"
        )

    # Sell-to-buy ratio
    if len(sells) > 0 and len(buys) == 0:
        red_flags.append("Zero insider buying — all transactions are sells")

    # ── Green Flag Detection ──
    green_flags = []
    csuite_buys = [tx for tx in buys if tx["role_weight"] >= 2.5]
    if csuite_buys:
        names = list(set(tx["name"] for tx in csuite_buys))
        total_val = sum(tx["value"] for tx in csuite_buys)
        green_flags.append(
            f"C-suite buying: {', '.join(names[:3])} "
            f"(${total_val/1e6:.2f}M total)"
        )

    unique_buyers = set(tx["name"] for tx in buys)
    if len(unique_buyers) >= 2:
        green_flags.append(
            f"Cluster buys: {len(unique_buyers)} unique insiders buying "
            f"(internal confidence)"
        )

    large_buys = [tx for tx in buys if tx["value"] > 500_000]
    if large_buys:
        green_flags.append(
            f"High-conviction purchases: {len(large_buys)} transactions > $500K"
        )

    if len(buys) > 0 and len(sells) == 0:
        green_flags.append("Pure insider buying — zero sells in period")

    # Aggregate transactions BY INSIDER (not per-transaction)
    # This prevents the same person appearing 4x in the table.
    # For each insider: sum shares transacted, compute total % of stake,
    # determine net direction, use most recent date.
    insider_agg = {}  # name → aggregated record
    for tx in insiders:
        if tx["value"] < 10_000:
            continue  # Skip trivially small
        name = tx["name"]
        is_buy = _is_buy(tx["transaction"])
        if name not in insider_agg:
            insider_agg[name] = {
                "name": name,
                "role": tx["role"],
                "role_weight": tx["role_weight"],
                "stake_pct": tx.get("stake_pct", 0),
                "shares_owned": tx.get("shares_owned", 0),
                "total_shares_bought": 0,
                "total_shares_sold": 0,
                "total_value_bought": 0,
                "total_value_sold": 0,
                "tx_count": 0,
                "latest_date": tx.get("date", ""),
            }
        agg = insider_agg[name]
        if is_buy:
            agg["total_shares_bought"] += tx["shares"]
            agg["total_value_bought"] += tx["value"]
        else:
            agg["total_shares_sold"] += tx["shares"]
            agg["total_value_sold"] += tx["value"]
        agg["tx_count"] += 1
        # Keep most recent date
        if tx.get("date", "") > agg["latest_date"]:
            agg["latest_date"] = tx.get("date", "")
        # Keep highest stake_pct (in case one tx matched roster and another didn't)
        if tx.get("stake_pct", 0) > agg["stake_pct"]:
            agg["stake_pct"] = tx.get("stake_pct", 0)
        if tx.get("shares_owned", 0) > agg["shares_owned"]:
            agg["shares_owned"] = tx.get("shares_owned", 0)

    # Build notable list from aggregated data
    # Key math: to compute true % of stake sold over the period:
    #   shares_owned_now = from roster (current holdings)
    #   net_shares_sold = total_shares_sold - total_shares_bought
    #   shares_owned_start = shares_owned_now + net_shares_sold
    #   pct_of_stake_sold = net_shares_sold / shares_owned_start × 100
    # This correctly handles multi-transaction sell programs.
    notable = []
    for agg in sorted(insider_agg.values(),
                      key=lambda a: a["total_value_bought"] + a["total_value_sold"],
                      reverse=True)[:5]:
        net_bought = agg["total_shares_bought"]
        net_sold = agg["total_shares_sold"]
        shares_now = agg["shares_owned"]
        # Determine direction
        if net_bought > 0 and net_sold == 0:
            direction = "BUY"
        elif net_sold > 0 and net_bought == 0:
            direction = "SELL"
        elif net_bought > net_sold:
            direction = "NET BUY"
        elif net_sold > net_bought:
            direction = "NET SELL"
        else:
            direction = "MIXED"
        # Compute true % of stake reduced/increased over the period
        net_change = net_bought - net_sold  # positive = net buying
        if shares_now == 0 and net_sold > 0:
            # Full exit — sold everything
            pct_change = 100.0
        elif shares_now == 0 and net_bought > 0:
            # New position opened from zero
            pct_change = 100.0
        elif shares_now > 0 and abs(net_change) > 0:
            # Reconstruct starting position
            shares_start = shares_now - net_change  # start = now - (bought - sold)
            if shares_start > 0:
                pct_change = abs(net_change) / shares_start * 100
            else:
                pct_change = 100.0  # Entirely new position
        else:
            pct_change = 0
        notable.append({
            "type": direction,
            "name": agg["name"],
            "role": agg["role"],
            "value": agg["total_value_bought"] + agg["total_value_sold"],
            "date": agg["latest_date"],
            "role_weight": agg["role_weight"],
            "stake_pct": agg["stake_pct"],
            "pct_stake_change": round(pct_change, 1),  # true % of starting stake
            "tx_count": agg["tx_count"],
        })

    # Summary
    net_value = sum(tx["value"] for tx in buys) - sum(tx["value"] for tx in sells)
    if score > 0.3:
        summary = f"Net insider buying ({len(buys)} buys vs {len(sells)} sells). Weighted toward accumulation."
    elif score < -0.3:
        summary = f"Net insider selling ({len(sells)} sells vs {len(buys)} buys). Distribution signals present."
    else:
        summary = f"Mixed insider activity ({len(buys)} buys, {len(sells)} sells). No dominant direction."

    return {
        "score": round(score, 3),
        "red_flags": red_flags,
        "green_flags": green_flags,
        "summary": summary,
        "buy_count": len(buys),
        "sell_count": len(sells),
        "net_value": net_value,
        "notable_transactions": notable,
    }


# ══════════════════════════════════════════════════════════════════════════
# BUCKET 2: FUND ACCUMULATION TREND
# ══════════════════════════════════════════════════════════════════════════

def _analyze_fund_accumulation(institutional: List[Dict]) -> Dict[str, Any]:
    """
    Fund accumulation trend analysis.

    Scoring methodology (non-passive holders only — index fund rebalancing excluded):
      smart_trend = clip(avg_smart_money_%_change / 0.05, -1, 1)
        → 5% position increase by smart money = full bullish signal
        → If no smart money holders exist, smart_trend = 0 (no signal)
      active_trend = clip(avg_non_passive_%_change / 0.05, -1, 1)
        → captures active + smart money sentiment (excludes index funds)
      breadth = (non_passive_increasing - non_passive_decreasing) / non_passive_total
        → what fraction of active holders are adding? [-1, +1]

      Final: smart_trend × 0.50 + active_trend × 0.30 + breadth × 0.20

    Smart money gets 50% weight because:
      - Active stock pickers with research teams
      - Their position changes reflect deliberate conviction
      - Passive flows are mechanical (index rebalancing) — excluded entirely
    """
    if not institutional:
        return {
            "score": 0.0,
            "smart_money_trend": 0.0,
            "active_trend": 0.0,
            "breadth": 0.0,
            "red_flags": [],
            "green_flags": [],
            "summary": "No institutional holder data available.",
            "holders_increasing": 0,
            "holders_decreasing": 0,
            "smart_money_holders": [],
        }

    # Separate by classification
    smart_money = [h for h in institutional if h["classification"] == "smart_money"]
    active = [h for h in institutional if h["classification"] == "active"]
    passive = [h for h in institutional if h["classification"] == "passive"]
    non_passive = smart_money + active  # Everyone except index funds

    # ── Smart Money Trend (time-decay weighted) ──
    # Weighted average of % change among smart money holders.
    # More recent filings carry more weight (half-life 60 days).
    # If no smart money holders exist, smart_trend = 0 — no signal.
    smart_with_changes = [(h["pctChange"], h) for h in smart_money if h["pctChange"] != 0]
    if smart_with_changes:
        weights = [abs(c) * _time_decay(h.get("date_reported", ""), 60) or 1.0
                   for c, h in smart_with_changes]
        # Use value-weighted mean: larger positions' changes matter more
        val_weights = [h["value"] * _time_decay(h.get("date_reported", ""), 60)
                       for _, h in smart_with_changes]
        total_vw = sum(val_weights) or 1
        avg_smart_change = float(sum(c * w for (c, _), w in zip(smart_with_changes, val_weights)) / total_vw)
    else:
        avg_smart_change = 0.0  # No smart money → no signal

    # Normalize: 5% avg position change = full conviction
    smart_trend = float(np.clip(avg_smart_change / 0.05, -1.0, 1.0))

    # ── Active Trend (time-decay weighted) ──
    # Weighted average across NON-PASSIVE holders only.
    # Index fund rebalancing is excluded. More recent filings carry
    # more weight (half-life 60 days).
    active_with_changes = [(h["pctChange"], h) for h in non_passive if h["pctChange"] != 0]
    if active_with_changes:
        val_weights_a = [h["value"] * _time_decay(h.get("date_reported", ""), 60)
                         for _, h in active_with_changes]
        total_vwa = sum(val_weights_a) or 1
        avg_active_change = float(sum(c * w for (c, _), w in zip(active_with_changes, val_weights_a)) / total_vwa)
    else:
        avg_active_change = 0.0
    active_trend = float(np.clip(avg_active_change / 0.05, -1.0, 1.0))

    # ── Breadth (value-weighted) ──
    # Among non-passive holders: what fraction of CAPITAL is increasing vs
    # decreasing? Value-weighted so a $50B Berkshire position adding counts
    # far more than a $50M boutique fund. Equal-weight breadth treats them
    # the same, which a PhD would rightly call a bias toward small holders.
    #
    # Formula: breadth = (Σ value_increasing - Σ value_decreasing) / Σ value_all
    # Still count-based metrics for UI display (holders_increasing, etc.)
    increasing = sum(1 for h in non_passive if h["pctChange"] > 0)
    decreasing = sum(1 for h in non_passive if h["pctChange"] < 0)
    unchanged = sum(1 for h in non_passive if h["pctChange"] == 0)

    val_increasing = sum(h["value"] for h in non_passive if h["pctChange"] > 0)
    val_decreasing = sum(h["value"] for h in non_passive if h["pctChange"] < 0)
    val_total = sum(h["value"] for h in non_passive) or 1
    breadth = float((val_increasing - val_decreasing) / val_total)
    breadth = float(np.clip(breadth, -1, 1))

    # ── Composite Score ──
    score = smart_trend * 0.50 + active_trend * 0.30 + breadth * 0.20
    score = float(np.clip(score, -1, 1))

    # ── Flag Detection ──
    red_flags = []
    green_flags = []

    # Smart money reducing
    smart_decreasing = [h for h in smart_money if h["pctChange"] < -0.01]
    if len(smart_decreasing) >= 2:
        names = [h["holder"][:30] for h in smart_decreasing[:3]]
        red_flags.append(f"Smart money reducing: {', '.join(names)}")

    active_increasing = sum(1 for h in non_passive if h["pctChange"] > 0)
    active_decreasing = sum(1 for h in non_passive if h["pctChange"] < 0)
    if active_decreasing > active_increasing and len(active_with_changes) > 2:
        red_flags.append(
            f"Active funds derisking: {active_decreasing} reducing vs {active_increasing} increasing"
        )

    if avg_active_change < -0.03:
        red_flags.append(f"Significant active outflows: avg position change {avg_active_change:+.1%}")

    # Smart money accumulating
    smart_increasing = [h for h in smart_money if h["pctChange"] > 0.01]
    if len(smart_increasing) >= 2:
        names = [h["holder"][:30] for h in smart_increasing[:3]]
        green_flags.append(f"Smart money accumulating: {', '.join(names)}")

    if active_increasing > active_decreasing * 2 and len(active_with_changes) > 2:
        green_flags.append(
            f"Active fund accumulation: {active_increasing} adding vs {active_decreasing} reducing"
        )

    if avg_smart_change > 0.03:
        green_flags.append(f"Strong smart money inflows: avg {avg_smart_change:+.1%}")

    # Summary
    if score > 0.3:
        summary = f"Active fund accumulation. Smart money trend: {avg_smart_change:+.1%}."
    elif score < -0.3:
        summary = f"Active fund distribution. Smart money trend: {avg_smart_change:+.1%}."
    else:
        summary = f"Mixed institutional activity. Smart money trend: {avg_smart_change:+.1%}."

    # Full holder breakdown for UI trend table (sorted by value, top 10)
    # Includes from/to stake % so we can show "increased from 0.50% to 0.60%"
    holders_detail = []
    for h in sorted(institutional, key=lambda x: x["value"], reverse=True)[:10]:
        chg = h["pctChange"]
        if chg > 0.01:
            direction = "Adding"
        elif chg < -0.01:
            direction = "Reducing"
        else:
            direction = "Stable"

        pct_held = h.get("pctHeld", 0)
        pct_held_prev = h.get("pctHeldPrev", pct_held)

        # Build stake change string: "0.50% → 0.60%"
        if pct_held > 0 and abs(pct_held - pct_held_prev) > 0.0001:
            stake_change = f"{pct_held_prev:.2%} → {pct_held:.2%}"
        elif pct_held > 0:
            stake_change = f"{pct_held:.2%}"
        else:
            stake_change = ""

        holders_detail.append({
            "name": h["holder"],
            "type": h["classification"].replace("_", " ").title(),
            "value": h["value"],
            "pct_change": chg,
            "direction": direction,
            "pct_held": pct_held,
            "pct_held_prev": pct_held_prev,
            "stake_change": stake_change,
            "date_reported": h.get("date_reported", ""),
        })

    return {
        "score": round(score, 3),
        "smart_money_trend": round(smart_trend, 3),
        "active_trend": round(active_trend, 3),
        "breadth": round(breadth, 3),
        "red_flags": red_flags,
        "green_flags": green_flags,
        "summary": summary,
        # Non-passive counts only — index fund rebalancing is not a signal
        "holders_increasing": increasing,
        "holders_decreasing": decreasing,
        "holders_unchanged": unchanged,
        "holders_detail": holders_detail,
        "smart_money_holders": [h["holder"] for h in smart_money],
    }


# ══════════════════════════════════════════════════════════════════════════
# BUCKET 3: CAPITAL CONCENTRATION
# ══════════════════════════════════════════════════════════════════════════

def _analyze_concentration(institutional: List[Dict]) -> Dict[str, Any]:
    """
    Capital concentration analysis.

    What we measure:
      - Top-5 holder share of total institutional holdings
      - Smart money share of total (hedge funds + quants as % of all institutional)
      - Whether concentration is increasing or decreasing

    Scoring:
      base = smart_money_pct × 2 - 0.3
        → 15% smart money = neutral (0.0)
        → 30% smart money = strong positive (+0.3)
        → 5% smart money = negative (-0.2)
      boost: if smart money is also accumulating (positive change), add 0.2
      penalty: if smart money is reducing AND concentration high, it means
               conviction is unwinding → subtract 0.3

      Final: clip(base + adjustments, -1, 1)

    Rationale:
      High smart money concentration = institutional conviction (these are
      research-driven allocators). But concentration alone isn't enough —
      we need it to be INCREASING (or stable) to be bullish. Decreasing
      concentration with smart money exiting = distribution signal.
    """
    if not institutional:
        return {
            "score": 0.0,
            "smart_money_pct": 0.0,
            "top5_concentration": 0.0,
            "red_flags": [],
            "green_flags": [],
            "summary": "No institutional data available.",
        }

    # Total shares across all institutional holders
    total_shares = sum(h["shares"] for h in institutional) or 1

    # Smart money concentration
    smart_shares = sum(h["shares"] for h in institutional if h["classification"] == "smart_money")
    smart_pct = smart_shares / total_shares

    # Top-5 concentration (Herfindahl-style)
    sorted_by_shares = sorted(institutional, key=lambda x: x["shares"], reverse=True)
    top5_shares = sum(h["shares"] for h in sorted_by_shares[:5])
    top5_pct = top5_shares / total_shares

    # Smart money trend (are they increasing or decreasing?)
    smart_holders = [h for h in institutional if h["classification"] == "smart_money"]
    smart_changes = [h["pctChange"] for h in smart_holders if h["pctChange"] != 0]
    avg_smart_change = float(np.mean(smart_changes)) if smart_changes else 0.0

    # ── Score Calculation ──
    # Base: smart money presence (centered at 15% = neutral)
    base = float(smart_pct * 2 - 0.3)

    # Trend adjustment
    if avg_smart_change > 0.01:
        # Smart money accumulating → boost
        adjustment = 0.2
    elif avg_smart_change < -0.01:
        # Smart money reducing → penalty (stronger if they're a large presence)
        adjustment = -0.3 if smart_pct > 0.2 else -0.15
    else:
        adjustment = 0.0

    # High top-5 concentration with positive trend = conviction
    if top5_pct > 0.6 and avg_smart_change > 0:
        adjustment += 0.1

    raw_score = float(np.clip(base + adjustment, -1, 1))

    # Confidence fade-in: when smart money presence is very low (<5%),
    # the concentration signal is unreliable — mega-caps are dominated
    # by passive/index funds and low smart money is structural.
    # Instead of a hard cutoff (which creates a cliff), we linearly
    # ramp confidence from 0 at 0% to 1.0 at 5%.
    #   0% smart money → confidence=0.0 → score=0
    #   2% smart money → confidence=0.4 → score attenuated
    #   5%+ smart money → confidence=1.0 → full signal
    confidence = float(np.clip(smart_pct / 0.05, 0.0, 1.0))
    score = raw_score * confidence

    # ── Flags ──
    red_flags = []
    green_flags = []

    # NOTE: We intentionally do NOT flag "low smart money presence" as a red flag.
    # For mega-caps (AAPL, MSFT, etc.), passive/index funds dominate the holder
    # base, so low smart money % is structural — not a risk signal.
    if smart_pct > 0.2 and avg_smart_change < -0.02:
        red_flags.append(
            f"Smart money exiting: {smart_pct:.0%} concentration but reducing {avg_smart_change:+.1%}"
        )
    if top5_pct > 0.7 and avg_smart_change < 0:
        red_flags.append("High concentration unwinding — top holders reducing")

    if smart_pct > 0.2:
        green_flags.append(f"Strong smart money presence: {smart_pct:.0%} of institutional shares")
    if smart_pct > 0.1 and avg_smart_change > 0.02:
        green_flags.append(f"Smart money building positions (avg change: {avg_smart_change:+.1%})")
    if top5_pct > 0.5 and avg_smart_change > 0:
        green_flags.append(f"High-conviction holders (top 5 own {top5_pct:.0%}) and adding")

    # Summary
    if score > 0.2:
        summary = f"Strong institutional conviction. Smart money: {smart_pct:.0%}, trend: accumulating."
    elif score < -0.2:
        summary = f"Weak institutional conviction. Smart money: {smart_pct:.0%}, trend: reducing."
    else:
        summary = f"Neutral conviction profile. Smart money: {smart_pct:.0%}."

    return {
        "score": round(score, 3),
        "smart_money_pct": round(smart_pct, 3),
        "top5_concentration": round(top5_pct, 3),
        "avg_smart_change": round(avg_smart_change, 4),
        "red_flags": red_flags,
        "green_flags": green_flags,
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORING
# ══════════════════════════════════════════════════════════════════════════

def _score_flows(ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Composite flow intelligence from 3 sub-buckets.

    flow_composite = 0.35 · insider_intel + 0.40 · fund_trend + 0.25 · concentration
                   ∈ [-1, +1]

    Weights rationale:
      Fund accumulation (0.40): Most reliable — 13F filings are legally required,
        covers largest dollar volume, shows deliberate allocation decisions.
      Insider intelligence (0.35): Most informative per-transaction — insiders
        have asymmetric information. But lower volume (few transactions).
      Concentration (0.25): Contextual — tells you WHO holds the stock, not
        what they're DOING. Supporting signal, not primary.
    """
    if data.get("status") != "Success":
        return {
            "status": "Data source not available",
            "flow_composite": 0,
            "insider_intelligence": {"score": 0, "red_flags": [], "green_flags": [], "summary": "No data"},
            "fund_accumulation": {"score": 0, "red_flags": [], "green_flags": [], "summary": "No data"},
            "concentration": {"score": 0, "red_flags": [], "green_flags": [], "summary": "No data"},
            # Backward compatibility
            "smart_money_pct": 0,
            "institutional_conviction": 0,
            "insider_alignment": 0,
            "capital_participation": 0,
            "insider_buys": 0,
            "insider_sells": 0,
            "estimated_accumulation": "neutral",
            "net_insider_value": 0,
            "notable_transactions": [],
        }

    inst = data.get("institutional", [])
    insiders = data.get("insiders", [])
    ownership = data.get("ownership_breakdown", {})

    # Run each bucket
    insider_intel = _analyze_insider_intelligence(insiders)
    fund_accum = _analyze_fund_accumulation(inst)
    concentration = _analyze_concentration(inst)

    # Composite
    flow_comp = float(np.clip(
        insider_intel["score"] * 0.35 +
        fund_accum["score"] * 0.40 +
        concentration["score"] * 0.25,
        -1.0, 1.0
    ))

    # Accumulation estimate (human-readable)
    if flow_comp > 0.4:
        est_accum = "strong_accumulation"
    elif flow_comp > 0.15:
        est_accum = "moderate_accumulation"
    elif flow_comp < -0.4:
        est_accum = "distribution"
    elif flow_comp < -0.15:
        est_accum = "moderate_distribution"
    else:
        est_accum = "neutral"

    # Aggregate all flags
    all_red_flags = insider_intel["red_flags"] + fund_accum["red_flags"] + concentration["red_flags"]
    all_green_flags = insider_intel["green_flags"] + fund_accum["green_flags"] + concentration["green_flags"]

    return {
        "status": "Success",
        "source": "yfinance",
        "flow_composite": flow_comp,
        # Sub-bucket details (for per-stock rendering)
        "insider_intelligence": insider_intel,
        "fund_accumulation": fund_accum,
        "concentration": concentration,
        # Aggregate flags
        "all_red_flags": all_red_flags,
        "all_green_flags": all_green_flags,
        "estimated_accumulation": est_accum,
        # Ownership breakdown (institutional vs insider vs retail)
        "ownership_breakdown": ownership,
        # Backward compatibility fields (consumed by synthesis_agent)
        "smart_money_pct": concentration.get("smart_money_pct", 0),
        "institutional_conviction": fund_accum.get("score", 0),
        "insider_alignment": insider_intel.get("score", 0),
        "capital_participation": (1 + flow_comp) / 2,  # Map [-1,1] → [0,1] for compat
        "insider_buys": insider_intel.get("buy_count", 0),
        "insider_sells": insider_intel.get("sell_count", 0),
        "net_insider_value": insider_intel.get("net_value", 0),
        "notable_transactions": insider_intel.get("notable_transactions", []),
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def analyze_flows(state: PortfolioState) -> PortfolioState:
    """Run capital flow intelligence for all portfolio tickers."""
    tickers = list(state.weights.keys()) if state.weights else []
    flow_intel = {}
    raw_results = {}
    progress_bar = st.progress(0, text="Analyzing Capital Flows...")

    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Flow Intel: {ticker}")
        raw = _fetch_flow_data(ticker)
        raw_results[ticker] = raw
        _log.info("Flow fetch %s: status=%s, inst=%d, ins=%d",
                  ticker, raw.get("status"), len(raw.get("institutional", [])), len(raw.get("insiders", [])))
        try:
            flow_intel[ticker] = _score_flows(ticker, raw)
        except Exception as e:
            _log.error("Flow scoring CRASHED for %s: %s", ticker, e, exc_info=True)
            flow_intel[ticker] = _score_flows(ticker, {"status": "Error"})

    # Portfolio-level summary
    succ = {k: v for k, v in flow_intel.items() if v.get("status") == "Success"}

    p_summary = {
        "portfolio_flow_score": float(np.mean([v["flow_composite"] for v in succ.values()])) if succ else 0,
        "avg_smart_money_pct": float(np.mean([v["smart_money_pct"] for v in succ.values()])) if succ else 0,
        "avg_insider_alignment": float(np.mean([v["insider_alignment"] for v in succ.values()])) if succ else 0,
        "accumulating_count": sum(1 for v in succ.values() if "accumulation" in v.get("estimated_accumulation", "")),
        "distributing_count": sum(1 for v in succ.values() if "distribution" in v.get("estimated_accumulation", "")),
        # New: aggregate flag counts
        "total_red_flags": sum(len(v.get("all_red_flags", [])) for v in succ.values()),
        "total_green_flags": sum(len(v.get("all_green_flags", [])) for v in succ.values()),
    }

    state.flow_signals = {
        "intel": flow_intel,
        "raw": raw_results,
        "portfolio_summary": p_summary,
        "status": "Success" if succ else "Data source not available",
    }
    progress_bar.empty()
    return state
