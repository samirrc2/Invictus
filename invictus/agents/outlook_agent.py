"""
Invictus — Management Outlook Intelligence Agent (v1)

Extracts qualitative forward-looking signals from management's own commentary
about business trajectory, industry environment, and strategic direction.
Uses LLM to parse unstructured text into structured conviction dimensions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEV MODE — DEMO FOLDER
──────────────────────
During development, this agent reads pre-downloaded FMP data from
    invictus/data/demo/<ticker>/
instead of hitting the FMP API live. This avoids burning API credits
during iteration and ensures reproducible results.

To refresh demo data:  python scripts/download_fmp_demo.py

To switch to live API:  set USE_DEMO_DATA = False below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEPARATE FROM TRANSCRIPT ANALYSIS
──────────────────────────────────
Management Outlook  = WHAT management says (qualitative signal extraction)
Transcript Analysis = HOW management says it (linguistic credibility gate)

Final formula:
    management_score = outlook_score × credibility_multiplier

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6 SIGNAL DIMENSIONS (each scored [-1, +1])
──────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│ DIM 1: DEMAND ENVIRONMENT (weight: 0.25)                               │
│   Management's characterization of customer demand, order pipelines,   │
│   bookings trends, pipeline visibility. Are customers accelerating     │
│   or decelerating?                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ DIM 2: COMPETITIVE POSITIONING (weight: 0.15)                          │
│   Management's view on market share trajectory, competitive moat,      │
│   differentiation, pricing power. Are they gaining or ceding ground?   │
├─────────────────────────────────────────────────────────────────────────┤
│ DIM 3: STRATEGIC CONFIDENCE (weight: 0.20)                             │
│   Management's conviction in their strategy, execution quality, and    │
│   operational trajectory. Are they executing or struggling?            │
├─────────────────────────────────────────────────────────────────────────┤
│ DIM 4: MACRO / INDUSTRY OUTLOOK (weight: 0.15)                        │
│   Management's read on the broader macro and industry environment.     │
│   Tailwinds vs headwinds they see coming. External forces.             │
├─────────────────────────────────────────────────────────────────────────┤
│ DIM 5: HEADWINDS / TAILWINDS BALANCE (weight: 0.15)                   │
│   Net balance of explicitly mentioned risks vs opportunities.          │
│   Forward-looking risk acknowledgment and confidence in mitigation.    │
├─────────────────────────────────────────────────────────────────────────┤
│ DIM 6: INVESTMENT THESIS CLARITY (weight: 0.10)                       │
│   How clearly management articulates a path to value creation.         │
│   Vague platitudes vs concrete milestones. Capital allocation clarity. │
└─────────────────────────────────────────────────────────────────────────┘

COMPOSITE FORMULA
─────────────────
    outlook_score = Σ (weight_i × dim_score_i)    ∈ [-1, +1]

DATA SOURCES (current FMP plan)
───────────────────────────────
1. FMP Stock News — 20 articles of analyst/market commentary (primary)
2. FMP Analyst Grades — buy/sell/hold consensus
3. FMP Earnings Calendar — EPS beat/miss history (quantitative context)
4. yfinance News — supplementary market coverage (fallback)

FUTURE (with FMP Ultimate plan, auto-enabled when demo data exists):
5. FMP Earnings Call Transcripts — full management commentary
6. FMP Press Releases — official corporate announcements
7. FMP Analyst Estimates — forward estimates

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import logging
import time as _t
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yfinance as yf

from invictus.config import FMP_API_KEY, DATA_DIR
from invictus.llm import call_llm_json_raw, llm_available, get_provider
from invictus.fmp_client import (
    fetch_stock_news, fetch_analyst_grades, fetch_earnings_calendar,
    fmp_available,
)

_log = logging.getLogger(__name__)

DEMO_DIR = DATA_DIR / "demo"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIMENSION WEIGHTS — must sum to 1.0
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIMENSION_WEIGHTS = {
    "demand_environment":       0.25,
    "competitive_positioning":  0.15,
    "strategic_confidence":     0.20,
    "macro_industry_outlook":   0.15,
    "headwinds_tailwinds":      0.15,
    "investment_thesis_clarity": 0.10,
}

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9, "Dimension weights must sum to 1.0"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA FETCHERS — LIVE FMP + DEMO FALLBACK
# Tries FMP API live first, falls back to cached demo JSON when unavailable
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_demo_json(ticker: str, filename: str) -> Any:
    """Load a JSON file from the demo folder for a given ticker."""
    path = DEMO_DIR / ticker.lower() / filename
    if not path.exists():
        _log.debug("Demo file not found: %s", path)
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        _log.warning("Failed to load demo file %s: %s", path, e)
        return []


def _fetch_fmp_stock_news_live(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch stock news from FMP live API.
    Falls back to demo cache if FMP unavailable.
    """
    _start = _t.perf_counter()
    articles = []

    # Try live FMP first
    raw = fetch_stock_news(ticker, limit=20)
    source_tag = "FMP Live"
    if not raw:
        # Fallback to demo cache
        raw = _load_demo_json(ticker, "stock_news.json")
        source_tag = "FMP Demo"
        if not isinstance(raw, list):
            raw = []

    if isinstance(raw, list):
        for item in raw:
            title = item.get("title", "")
            text = item.get("text", "")
            date = item.get("publishedDate", "")
            publisher = item.get("site", item.get("publisher", ""))
            if title:
                articles.append({
                    "title": title,
                    "text": text[:2000],
                    "date": date,
                    "publisher": publisher,
                    "source": f"FMP Stock News ({source_tag})",
                })

    _log_data_health("fmp_stock_news", ticker,
                     "success" if articles else "no_data",
                     _t.perf_counter() - _start, len(articles))
    return articles


def _fetch_fmp_analyst_grades_live(ticker: str) -> Dict[str, Any]:
    """Fetch analyst grade consensus from FMP live, demo fallback."""
    grades = fetch_analyst_grades(ticker)
    if grades and grades.get("symbol"):
        return grades
    # Fallback to demo
    raw = _load_demo_json(ticker, "analyst_grades.json")
    if isinstance(raw, list) and raw:
        return raw[0]
    if isinstance(raw, dict):
        return raw
    return {}


def _fetch_fmp_earnings_live(ticker: str) -> List[Dict[str, Any]]:
    """Fetch earnings calendar from FMP live, demo fallback."""
    data = fetch_earnings_calendar(ticker, limit=8)
    if data:
        return data
    # Fallback to demo
    raw = _load_demo_json(ticker, "earnings_calendar.json")
    return raw if isinstance(raw, list) else []


def _fetch_fmp_transcripts_demo(ticker: str) -> List[Dict[str, Any]]:
    """
    Load FMP earnings transcripts from demo folder.
    Only available with FMP Ultimate plan ($79+). Returns [] if not downloaded.
    """
    raw = _load_demo_json(ticker, "transcripts.json")
    transcripts = []
    if isinstance(raw, list):
        for item in raw:
            content = item.get("content", "")
            if content and len(content) > 100:
                transcripts.append({
                    "content": content[:8000],
                    "date": item.get("date", ""),
                    "quarter": item.get("quarter", ""),
                    "year": item.get("year", ""),
                    "source": "FMP Earnings Transcript",
                })
    return transcripts


def _fetch_fmp_press_releases_demo(ticker: str) -> List[Dict[str, Any]]:
    """
    Load FMP press releases from demo folder.
    Only available with FMP Ultimate plan ($79+). Returns [] if not downloaded.
    """
    raw = _load_demo_json(ticker, "press_releases.json")
    releases = []
    if isinstance(raw, list):
        for item in raw:
            title = item.get("title", "")
            text = item.get("text", item.get("content", ""))
            if title or text:
                releases.append({
                    "title": title,
                    "text": text[:3000],
                    "date": item.get("date", item.get("publishedDate", "")),
                    "source": "FMP Press Release",
                })
    return releases


def _fetch_yfinance_news(ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch latest news from yfinance as supplementary context."""
    _start = _t.perf_counter()
    articles = []
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news:
            _log_data_health("yfinance_news", ticker, "no_data",
                             _t.perf_counter() - _start, 0)
            return []
        for item in news[:limit]:
            if not isinstance(item, dict):
                continue
            content = item.get("content", item)
            if isinstance(content, dict):
                title = content.get("title", "")
                summary = content.get("summary", content.get("description", ""))
            else:
                title = item.get("title", "")
                summary = item.get("summary", item.get("description", ""))
            if title:
                articles.append({
                    "title": title,
                    "text": summary or "",
                    "date": item.get("providerPublishTime", ""),
                    "source": "yfinance News",
                })
    except Exception as e:
        _log.warning("yfinance news fetch failed for %s: %s", ticker, e)
        _log_data_health("yfinance_news", ticker, "error",
                         _t.perf_counter() - _start, 0, str(e))
        return []
    _log_data_health("yfinance_news", ticker,
                     "success" if articles else "no_data",
                     _t.perf_counter() - _start, len(articles))
    return articles


def _build_earnings_context(earnings: List[Dict], grades: Dict) -> str:
    """
    Build a textual summary from earnings beat/miss history and analyst grades.
    This gives the LLM quantitative context to anchor its qualitative analysis.
    """
    parts = []

    # Earnings beat/miss streak
    if earnings:
        beats, misses = 0, 0
        for e in earnings:
            actual = e.get("epsActual")
            estimated = e.get("epsEstimated")
            if actual is not None and estimated is not None:
                if actual > estimated:
                    beats += 1
                elif actual < estimated:
                    misses += 1
        recent = earnings[0] if earnings else {}
        parts.append(
            f"[EARNINGS HISTORY — last {len(earnings)} quarters]\n"
            f"EPS Beat/Miss: {beats} beats, {misses} misses out of {len(earnings)} quarters.\n"
            f"Most recent: EPS actual={recent.get('epsActual')} vs "
            f"est={recent.get('epsEstimated')} "
            f"({recent.get('date', 'N/A')})"
        )
        # Revenue context
        rev_actual = recent.get("revenueActual")
        rev_est = recent.get("revenueEstimated")
        if rev_actual and rev_est:
            rev_beat = ((rev_actual - rev_est) / rev_est * 100) if rev_est else 0
            parts[-1] += f"\nRevenue: ${rev_actual/1e9:.1f}B actual vs ${rev_est/1e9:.1f}B est ({rev_beat:+.1f}%)"

    # Analyst consensus
    if grades:
        total = sum(grades.get(k, 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])
        consensus = grades.get("consensus", "N/A")
        parts.append(
            f"[ANALYST CONSENSUS]\n"
            f"Rating: {consensus} ({total} analysts)\n"
            f"Strong Buy: {grades.get('strongBuy', 0)}, Buy: {grades.get('buy', 0)}, "
            f"Hold: {grades.get('hold', 0)}, Sell: {grades.get('sell', 0)}, "
            f"Strong Sell: {grades.get('strongSell', 0)}"
        )

    return "\n---\n".join(parts)


def fetch_outlook_data(ticker: str) -> Dict[str, Any]:
    """
    Master fetcher — pulls FMP live data (with demo fallback) + yfinance
    and assembles context for LLM signal extraction.

    Strategy:
        - Stock news, analyst grades, earnings calendar → FMP live API
          (falls back to demo cache if FMP unavailable)
        - Transcripts, press releases → demo cache only
          (require FMP Ultimate plan, not available on current plan)
        - yfinance news → always live as supplementary source

    Returns:
        {
            "ticker": str,
            "fmp_news": [...],
            "transcripts": [...],       # empty unless FMP Ultimate plan
            "press_releases": [...],    # empty unless FMP Ultimate plan
            "yfinance_news": [...],
            "earnings": [...],
            "grades": {...},
            "context_text": str,        # assembled text for LLM
            "source_summary": str,
            "status": "Success" | "Partial" | "NoData",
        }
    """
    # ── FMP data (live API with demo fallback) ──
    fmp_news = _fetch_fmp_stock_news_live(ticker)
    grades = _fetch_fmp_analyst_grades_live(ticker)
    earnings = _fetch_fmp_earnings_live(ticker)

    # ── Restricted endpoints — demo cache only ──
    transcripts = _fetch_fmp_transcripts_demo(ticker)
    press_releases = _fetch_fmp_press_releases_demo(ticker)

    # ── yfinance (always live) ──
    yf_news = _fetch_yfinance_news(ticker)

    # ── Assemble LLM context ──
    # Priority: transcripts > press releases > FMP news > yfinance news
    context_parts = []

    # Transcripts (if available — FMP Ultimate plan)
    for tr in transcripts:
        q, y = tr.get("quarter", "?"), tr.get("year", "?")
        context_parts.append(
            f"[EARNINGS CALL TRANSCRIPT — Q{q} {y}]\n{tr['content']}\n---"
        )

    # Press releases (if available — FMP Ultimate plan)
    for pr in press_releases[:5]:
        context_parts.append(
            f"[PRESS RELEASE — {pr.get('date', 'recent')}]\n"
            f"Title: {pr['title']}\n{pr['text']}\n---"
        )

    # FMP stock news (always available)
    for article in fmp_news[:15]:
        context_parts.append(
            f"[NEWS — {article.get('publisher', '')} — {article.get('date', '')}]\n"
            f"Title: {article['title']}\n{article['text']}\n---"
        )

    # yfinance news (supplementary)
    for article in yf_news[:5]:
        context_parts.append(
            f"[NEWS — yfinance — {article.get('date', '')}]\n"
            f"Title: {article['title']}\n{article['text']}\n---"
        )

    # Earnings quantitative context
    earnings_context = _build_earnings_context(earnings, grades)
    if earnings_context:
        context_parts.append(earnings_context)

    context_text = "\n\n".join(context_parts)

    # ── Status ──
    has_transcripts = len(transcripts) > 0
    has_press = len(press_releases) > 0
    has_fmp_news = len(fmp_news) > 0
    has_yf_news = len(yf_news) > 0

    sources_used = []
    if has_transcripts:
        sources_used.append(f"FMP Transcripts ({len(transcripts)})")
    if has_press:
        sources_used.append(f"FMP Press Releases ({len(press_releases)})")
    if has_fmp_news:
        sources_used.append(f"FMP News ({len(fmp_news)})")
    if has_yf_news:
        sources_used.append(f"yfinance News ({len(yf_news)})")
    if earnings:
        sources_used.append(f"Earnings ({len(earnings)}Q)")
    if grades:
        sources_used.append(f"Grades ({grades.get('consensus', 'N/A')})")

    if has_transcripts or has_fmp_news:
        status = "Success"
    elif has_yf_news:
        status = "Partial"
    else:
        status = "NoData"

    return {
        "ticker": ticker,
        "fmp_news": fmp_news,
        "transcripts": transcripts,
        "press_releases": press_releases,
        "yfinance_news": yf_news,
        "earnings": earnings,
        "grades": grades,
        "context_text": context_text,
        "source_summary": " + ".join(sources_used) if sources_used else "No data",
        "status": status,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM SIGNAL EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTLOOK_EXTRACTION_PROMPT = """You are an institutional equity research analyst extracting qualitative forward-looking signals from management commentary and market coverage.

Analyze the following news, analyst commentary, and earnings data for {ticker}. Extract ONLY what is being said about the FUTURE of the business — not backward-looking results.

{context}

Score each dimension from -1.0 (very bearish) to +1.0 (very bullish). Provide 1-2 direct evidence quotes per dimension.

Output strictly in this JSON format:
{{
  "demand_environment": {{
    "score": <float -1.0 to +1.0>,
    "signal": "Accelerating | Stable | Decelerating | Mixed",
    "evidence": ["quote 1", "quote 2"],
    "reasoning": "1-2 sentence explanation"
  }},
  "competitive_positioning": {{
    "score": <float -1.0 to +1.0>,
    "signal": "Strengthening | Stable | Weakening | Mixed",
    "evidence": ["quote 1", "quote 2"],
    "reasoning": "1-2 sentence explanation"
  }},
  "strategic_confidence": {{
    "score": <float -1.0 to +1.0>,
    "signal": "High | Moderate | Low | Mixed",
    "evidence": ["quote 1", "quote 2"],
    "reasoning": "1-2 sentence explanation"
  }},
  "macro_industry_outlook": {{
    "score": <float -1.0 to +1.0>,
    "signal": "Favorable | Neutral | Challenging | Mixed",
    "evidence": ["quote 1", "quote 2"],
    "reasoning": "1-2 sentence explanation"
  }},
  "headwinds_tailwinds": {{
    "score": <float -1.0 to +1.0>,
    "signal": "Net Tailwind | Balanced | Net Headwind",
    "evidence": ["quote 1", "quote 2"],
    "reasoning": "1-2 sentence explanation of key headwinds and tailwinds mentioned"
  }},
  "investment_thesis_clarity": {{
    "score": <float -1.0 to +1.0>,
    "signal": "Clear & Specific | Moderate | Vague | Absent",
    "evidence": ["quote 1", "quote 2"],
    "reasoning": "1-2 sentence explanation"
  }},
  "overall_tone": "Bullish | Cautiously Optimistic | Neutral | Cautiously Pessimistic | Bearish",
  "key_themes": ["theme 1", "theme 2", "theme 3"],
  "notable_guidance_changes": ["any explicit guidance raise/cut/reiteration"]
}}

RULES:
- Score 0.0 = genuinely neutral or insufficient evidence
- Evidence quotes must be SHORT (under 20 words each) and directly from the text
- If no evidence exists for a dimension, score 0.0 and say "No evidence in available commentary"
- Do NOT infer from financial numbers alone — this is QUALITATIVE signal extraction
- Distinguish between management's own view vs analyst/media interpretation
- Use earnings beat/miss context to calibrate confidence, not to score directly
"""


def _extract_outlook_signals_llm(
    ticker: str, context_text: str
) -> Dict[str, Any]:
    """
    Use LLM to extract structured qualitative signals from management commentary.
    Returns the 6-dimension signal structure or a fallback dict on failure.
    """
    if not llm_available():
        _log.warning("No LLM API key — falling back to dictionary for %s", ticker)
        return _dictionary_fallback(ticker, context_text)

    if not context_text or len(context_text) < 50:
        _log.info("Insufficient context for LLM extraction for %s", ticker)
        return _no_data_result(ticker)

    _start = _t.perf_counter()
    prompt = OUTLOOK_EXTRACTION_PROMPT.format(
        ticker=ticker, context=context_text[:12000]
    )

    try:
        raw = call_llm_json_raw(prompt)
        analysis = json.loads(raw)
        _latency = (_t.perf_counter() - _start) * 1000

        result = _normalize_llm_output(analysis, ticker)
        result["source"] = f"LLM ({get_provider()})"
        result["status"] = "Success"

        _log_llm_call(
            ticker, prompt, raw, _latency,
            None, result["outlook_score"],
        )
        return result

    except Exception as e:
        _latency = (_t.perf_counter() - _start) * 1000
        _log.error("LLM extraction failed for %s: %s", ticker, e, exc_info=True)
        _log_llm_call(ticker, prompt, "", _latency, None, None,
                      fallback_reason=str(e))
        return _dictionary_fallback(ticker, context_text)


def _normalize_llm_output(analysis: dict, ticker: str) -> Dict[str, Any]:
    """
    Validate LLM output, clamp scores, compute composite.

    Composite formula:
        outlook_score = Σ (weight_i × score_i)  ∈ [-1, +1]
    """
    dimensions = {}
    for dim_name, weight in DIMENSION_WEIGHTS.items():
        dim_data = analysis.get(dim_name, {})
        if not isinstance(dim_data, dict):
            dim_data = {}

        raw_score = dim_data.get("score", 0)
        try:
            score = float(np.clip(float(raw_score), -1.0, 1.0))
        except (ValueError, TypeError):
            score = 0.0

        dimensions[dim_name] = {
            "score": score,
            "weight": weight,
            "weighted_contribution": round(score * weight, 4),
            "signal": dim_data.get("signal", "N/A"),
            "evidence": dim_data.get("evidence", [])[:3],
            "reasoning": dim_data.get("reasoning", "No reasoning provided"),
        }

    outlook_score = sum(d["weighted_contribution"] for d in dimensions.values())
    outlook_score = float(np.clip(outlook_score, -1.0, 1.0))

    return {
        "ticker": ticker,
        "dimensions": dimensions,
        "outlook_score": round(outlook_score, 4),
        "overall_tone": analysis.get("overall_tone", "Neutral"),
        "key_themes": analysis.get("key_themes", [])[:5],
        "notable_guidance_changes": analysis.get("notable_guidance_changes", [])[:5],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DICTIONARY FALLBACK (LLM-free)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Keyword dictionaries — single words + short phrases that match real news text ──
# Each list uses common financial language that appears in FMP/yfinance articles.

DEMAND_POSITIVE = [
    "growth", "revenue growth", "strong", "momentum", "accelerat", "outperform",
    "beat", "exceed", "record", "surge", "soar", "rally", "uptick", "robust",
    "upside", "upgrade", "buy", "bullish", "optimis", "rebound",
]
DEMAND_NEGATIVE = [
    "decline", "slow", "weak", "miss", "disappoint", "downgrade", "sell",
    "bearish", "pessimis", "contraction", "downturn", "slump", "plunge",
    "drop", "fell", "losses", "underperform", "concern", "warn",
]

COMPETITIVE_POSITIVE = [
    "leader", "dominat", "innovat", "pioneer", "advantage", "premium",
    "ecosystem", "loyal", "brand", "differentiat", "patent", "moat",
    "market share", "pricing power", "best-in-class",
]
COMPETITIVE_NEGATIVE = [
    "competition", "competitor", "rival", "threat", "disrupt", "commodit",
    "erode", "losing", "challenge", "pressure", "substitute", "alternative",
]

STRATEGIC_POSITIVE = [
    "invest", "expand", "launch", "partner", "acquisit", "transform",
    "strateg", "initiative", "roadmap", "milestone", "execut", "deliver",
    "progress", "plan", "confident", "guidance",
]
STRATEGIC_NEGATIVE = [
    "restructur", "layoff", "cut", "delay", "pivot", "uncertain",
    "setback", "reorganiz", "downsize", "impairment", "writedown",
]

MACRO_POSITIVE = [
    "recovery", "tailwind", "favorable", "opportunity", "secular",
    "stimulus", "easing", "supportive", "accommodat",
]
MACRO_NEGATIVE = [
    "tariff", "geopolitic", "recession", "inflation", "headwind",
    "regulat", "sanction", "trade war", "rate hike", "macro risk",
    "uncertainty", "volatility", "crisis", "conflict",
]

HEADWIND_TAILWIND_POSITIVE = [
    "margin", "efficien", "cost saving", "productiv", "leverage",
    "profit", "earning", "dividend", "cash flow", "free cash",
    "return", "yield", "income",
]
HEADWIND_TAILWIND_NEGATIVE = [
    "cost", "expense", "debt", "liability", "impair", "loss",
    "charge", "forex", "currency", "supply chain", "shortage",
    "wage", "freight",
]

THESIS_POSITIVE = [
    "buyback", "repurchas", "shareholder", "capital return", "dividend",
    "allocat", "target", "outlook", "visibility", "conviction",
    "value", "undervalued", "bargain", "attractive",
]
THESIS_NEGATIVE = [
    "overvalued", "expensive", "rich valuation", "bubble", "specul",
    "unclear", "opaque", "vague", "no guidance", "withdraw",
]


def _score_dimension_dict(
    text: str, positives: list, negatives: list
) -> Tuple[float, str, List[str]]:
    """
    Score a single dimension using substring keyword matching.

    Uses substring matching so "invest" catches "investing", "investment",
    "investor" etc. — much more effective on real news text than exact
    multi-word phrase matching.

    Formula:
        net = pos_count - neg_count
        raw = net / max(pos_count + neg_count, 1)   ← normalized [-1, +1]
        score = clip(raw × intensity, -1, 1)
        intensity = min(total_hits / 3, 1.0)         ← dampens low-evidence scores
    """
    text_lower = text.lower()

    pos_hits = [(kw, text_lower.count(kw)) for kw in positives if kw in text_lower]
    neg_hits = [(kw, text_lower.count(kw)) for kw in negatives if kw in text_lower]

    total_pos = sum(c for _, c in pos_hits)
    total_neg = sum(c for _, c in neg_hits)
    total_hits = total_pos + total_neg

    if total_hits == 0:
        return 0.0, "Neutral", []

    # Direction: which way does the evidence lean
    raw = (total_pos - total_neg) / max(total_hits, 1)
    # Intensity: dampen score when evidence is thin (< 3 hits)
    intensity = min(total_hits / 3.0, 1.0)
    score = float(np.clip(raw * intensity, -1.0, 1.0))

    if score > 0.15:
        signal = "Positive"
    elif score < -0.15:
        signal = "Negative"
    else:
        signal = "Neutral"

    # Build readable evidence: top hits sorted by count
    all_hits = sorted(pos_hits + neg_hits, key=lambda x: x[1], reverse=True)
    evidence = [f"{kw} ({c}x)" for kw, c in all_hits[:4]]
    return score, signal, evidence


def _dictionary_fallback(ticker: str, context_text: str) -> Dict[str, Any]:
    """LLM-free fallback: score each dimension using keyword dictionaries."""
    if not context_text or len(context_text) < 50:
        return _no_data_result(ticker)

    dim_configs = {
        "demand_environment":        (DEMAND_POSITIVE, DEMAND_NEGATIVE),
        "competitive_positioning":   (COMPETITIVE_POSITIVE, COMPETITIVE_NEGATIVE),
        "strategic_confidence":      (STRATEGIC_POSITIVE, STRATEGIC_NEGATIVE),
        "macro_industry_outlook":    (MACRO_POSITIVE, MACRO_NEGATIVE),
        "headwinds_tailwinds":       (HEADWIND_TAILWIND_POSITIVE, HEADWIND_TAILWIND_NEGATIVE),
        "investment_thesis_clarity": (THESIS_POSITIVE, THESIS_NEGATIVE),
    }

    dimensions = {}
    for dim_name, (pos_list, neg_list) in dim_configs.items():
        weight = DIMENSION_WEIGHTS[dim_name]
        score, signal, evidence = _score_dimension_dict(
            context_text, pos_list, neg_list
        )
        dimensions[dim_name] = {
            "score": round(score, 4),
            "weight": weight,
            "weighted_contribution": round(score * weight, 4),
            "signal": signal,
            "evidence": evidence,
            "reasoning": ", ".join(evidence) if evidence else "No keyword matches found",
        }

    outlook_score = sum(d["weighted_contribution"] for d in dimensions.values())
    outlook_score = float(np.clip(outlook_score, -1.0, 1.0))

    return {
        "ticker": ticker,
        "dimensions": dimensions,
        "outlook_score": round(outlook_score, 4),
        "overall_tone": "Neutral",
        "key_themes": [],
        "notable_guidance_changes": [],
        "source": "Dictionary Fallback",
        "status": "Success",
    }


def _no_data_result(ticker: str) -> Dict[str, Any]:
    """Return a zero-scored result when no data is available."""
    dimensions = {}
    for dim_name, weight in DIMENSION_WEIGHTS.items():
        dimensions[dim_name] = {
            "score": 0.0,
            "weight": weight,
            "weighted_contribution": 0.0,
            "signal": "No Data",
            "evidence": [],
            "reasoning": "No management commentary available",
        }
    return {
        "ticker": ticker,
        "dimensions": dimensions,
        "outlook_score": 0.0,
        "overall_tone": "Neutral",
        "key_themes": [],
        "notable_guidance_changes": [],
        "source": "N/A",
        "status": "NoData",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRANSCRIPT ANALYSIS — CREDIBILITY GATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CREDIBILITY_PROMPT = """You are a forensic linguistics analyst evaluating the CREDIBILITY of management's communication style. This is NOT about what they say — it's about HOW they say it.

Analyze the following management commentary and market coverage for {ticker}:

{context}

Evaluate these 4 credibility dimensions on a scale of 0.0 (very low credibility) to 1.0 (very high credibility):

Output strictly in this JSON format:
{{
  "hedging_density": {{
    "score": <float 0.0 to 1.0>,
    "reasoning": "How much hedging language (may, could, potentially, uncertain) vs assertive language"
  }},
  "specificity": {{
    "score": <float 0.0 to 1.0>,
    "reasoning": "Does management give specific numbers, dates, targets or vague generalities?"
  }},
  "forward_backward_ratio": {{
    "score": <float 0.0 to 1.0>,
    "reasoning": "Ratio of forward-looking statements to backward-looking explanations. High = confident about future."
  }},
  "dodge_detection": {{
    "score": <float 0.0 to 1.0>,
    "reasoning": "Does management directly address hard questions or deflect? Look for topic pivots, non-answers."
  }},
  "overall_credibility": "High | Moderate | Low",
  "red_flags": ["any concerning linguistic patterns"],
  "green_flags": ["any confidence-building linguistic patterns"]
}}

SCORING GUIDE:
- hedging_density: 1.0 = very direct/assertive, 0.0 = extremely hedged
- specificity: 1.0 = concrete numbers/dates, 0.0 = all platitudes
- forward_backward_ratio: 1.0 = mostly forward-looking, 0.0 = all backward-looking excuses
- dodge_detection: 1.0 = addresses everything directly, 0.0 = constant deflection
"""

CREDIBILITY_WEIGHTS = {
    "hedging_density":         0.25,
    "specificity":             0.30,
    "forward_backward_ratio":  0.25,
    "dodge_detection":         0.20,
}

# Keyword sets for dictionary credibility fallback
HEDGE_WORDS = [
    "may", "might", "could", "potentially", "possibly", "uncertain",
    "we believe", "we think", "we hope", "it appears", "somewhat",
    "relatively", "approximately", "in our view", "arguably",
]
ASSERTIVE_WORDS = [
    "will", "we are confident", "we expect", "committed to", "on track",
    "our plan is", "we have achieved", "clearly", "definitively",
    "strong conviction", "we are certain",
]
SPECIFIC_MARKERS = [
    "$", "%", "million", "billion", "quarter", "Q1", "Q2", "Q3", "Q4",
    "by 2025", "by 2026", "by 2027", "target of", "goal of",
    "basis points", "year-over-year", "sequential",
]
VAGUE_MARKERS = [
    "going forward", "in the future", "at some point", "over time",
    "we're excited", "great opportunity", "tremendous potential",
    "synergies", "well-positioned", "optimistic about",
]


def _extract_credibility_llm(ticker: str, context_text: str) -> Dict[str, Any]:
    """Use LLM to assess linguistic credibility of management commentary."""
    if not llm_available():
        return _dictionary_credibility(ticker, context_text)
    if not context_text or len(context_text) < 50:
        return _no_credibility_result(ticker)

    _start = _t.perf_counter()
    prompt = CREDIBILITY_PROMPT.format(
        ticker=ticker, context=context_text[:10000]
    )

    try:
        raw = call_llm_json_raw(prompt)
        analysis = json.loads(raw)
        _latency = (_t.perf_counter() - _start) * 1000

        result = _normalize_credibility_output(analysis, ticker)
        result["source"] = f"LLM ({get_provider()})"
        result["status"] = "Success"

        _log_llm_call(
            ticker, prompt, raw, _latency,
            None, result["credibility_multiplier"],
            agent_name="transcript_analysis",
        )
        return result

    except Exception as e:
        _log.error("Credibility LLM failed for %s: %s", ticker, e, exc_info=True)
        return _dictionary_credibility(ticker, context_text)


def _normalize_credibility_output(analysis: dict, ticker: str) -> Dict[str, Any]:
    """
    Validate credibility LLM output and compute multiplier.

    Formula:
        raw_credibility = Σ (weight_i × score_i)  ∈ [0, 1]
        credibility_multiplier = 0.5 + 0.5 × raw_credibility  ∈ [0.5, 1.0]

    Rationale: credibility=0 halves the outlook score (harsh penalty),
    credibility=1 leaves it untouched. This is a gate, not a boost.
    """
    sub_dims = {}
    for dim_name, weight in CREDIBILITY_WEIGHTS.items():
        dim_data = analysis.get(dim_name, {})
        if not isinstance(dim_data, dict):
            dim_data = {}
        raw_score = dim_data.get("score", 0.5)
        try:
            score = float(np.clip(float(raw_score), 0.0, 1.0))
        except (ValueError, TypeError):
            score = 0.5
        sub_dims[dim_name] = {
            "score": round(score, 4),
            "weight": weight,
            "weighted_contribution": round(score * weight, 4),
            "reasoning": dim_data.get("reasoning", ""),
        }

    raw_credibility = sum(d["weighted_contribution"] for d in sub_dims.values())
    credibility_multiplier = round(0.5 + 0.5 * raw_credibility, 4)

    return {
        "ticker": ticker,
        "sub_dimensions": sub_dims,
        "raw_credibility": round(raw_credibility, 4),
        "credibility_multiplier": credibility_multiplier,
        "overall_credibility": analysis.get("overall_credibility", "Moderate"),
        "red_flags": analysis.get("red_flags", [])[:5],
        "green_flags": analysis.get("green_flags", [])[:5],
    }


def _dictionary_credibility(ticker: str, context_text: str) -> Dict[str, Any]:
    """Dictionary-based credibility scoring as LLM fallback."""
    if not context_text or len(context_text) < 50:
        return _no_credibility_result(ticker)

    text_lower = context_text.lower()

    hedge_count = sum(1 for kw in HEDGE_WORDS if kw in text_lower)
    assert_count = sum(1 for kw in ASSERTIVE_WORDS if kw in text_lower)
    hedging_score = float(np.clip(
        assert_count / max(hedge_count + assert_count, 1), 0, 1
    ))

    specific_count = sum(1 for kw in SPECIFIC_MARKERS if kw in text_lower)
    vague_count = sum(1 for kw in VAGUE_MARKERS if kw in text_lower)
    specificity_score = float(np.clip(
        specific_count / max(specific_count + vague_count, 1), 0, 1
    ))

    forward_words = sum(
        1 for w in ["will", "expect", "plan", "target", "goal", "outlook", "forecast"]
        if w in text_lower
    )
    backward_words = sum(
        1 for w in ["was", "were", "had", "last quarter", "previously", "declined"]
        if w in text_lower
    )
    fb_score = float(np.clip(
        forward_words / max(forward_words + backward_words, 1), 0, 1
    ))

    dodge_score = 0.5  # can't detect from keywords

    sub_dims = {
        "hedging_density": {
            "score": round(hedging_score, 4),
            "weight": CREDIBILITY_WEIGHTS["hedging_density"],
            "weighted_contribution": round(hedging_score * CREDIBILITY_WEIGHTS["hedging_density"], 4),
            "reasoning": f"Dictionary: {assert_count} assertive vs {hedge_count} hedge",
        },
        "specificity": {
            "score": round(specificity_score, 4),
            "weight": CREDIBILITY_WEIGHTS["specificity"],
            "weighted_contribution": round(specificity_score * CREDIBILITY_WEIGHTS["specificity"], 4),
            "reasoning": f"Dictionary: {specific_count} specific vs {vague_count} vague",
        },
        "forward_backward_ratio": {
            "score": round(fb_score, 4),
            "weight": CREDIBILITY_WEIGHTS["forward_backward_ratio"],
            "weighted_contribution": round(fb_score * CREDIBILITY_WEIGHTS["forward_backward_ratio"], 4),
            "reasoning": f"Dictionary: {forward_words} forward vs {backward_words} backward",
        },
        "dodge_detection": {
            "score": round(dodge_score, 4),
            "weight": CREDIBILITY_WEIGHTS["dodge_detection"],
            "weighted_contribution": round(dodge_score * CREDIBILITY_WEIGHTS["dodge_detection"], 4),
            "reasoning": "Dictionary: dodge detection unavailable, default moderate",
        },
    }

    raw_credibility = sum(d["weighted_contribution"] for d in sub_dims.values())
    credibility_multiplier = round(0.5 + 0.5 * raw_credibility, 4)

    return {
        "ticker": ticker,
        "sub_dimensions": sub_dims,
        "raw_credibility": round(raw_credibility, 4),
        "credibility_multiplier": credibility_multiplier,
        "overall_credibility": "Moderate",
        "red_flags": [],
        "green_flags": [],
        "source": "Dictionary Fallback",
        "status": "Success",
    }


def _no_credibility_result(ticker: str) -> Dict[str, Any]:
    """Return neutral credibility when no data available."""
    sub_dims = {}
    for dim_name, weight in CREDIBILITY_WEIGHTS.items():
        sub_dims[dim_name] = {
            "score": 0.5, "weight": weight,
            "weighted_contribution": round(0.5 * weight, 4),
            "reasoning": "No data — default moderate",
        }
    return {
        "ticker": ticker,
        "sub_dimensions": sub_dims,
        "raw_credibility": 0.5,
        "credibility_multiplier": 0.75,
        "overall_credibility": "Moderate",
        "red_flags": [], "green_flags": [],
        "source": "N/A", "status": "NoData",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MASTER PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_management_outlook(ticker: str) -> Dict[str, Any]:
    """
    Management Outlook pipeline for a single ticker.

    ── ARCHITECTURE: 3 INDEPENDENT LAYERS ──────────────────────────────

    This agent produces TWO of the three independent layers that feed
    into the synthesis engine as PEERS:

        Layer 1: Capital Flows        → flow_composite ∈ [-1, +1]
                 (produced by flow_agent.py — NOT this agent)
                 Measures: what institutional money IS DOING

        Layer 2: Management Outlook   → outlook_score ∈ [-1, +1]
                 (produced HERE)
                 Measures: WHAT management says about the future
                 6 qualitative dimensions, LLM-extracted

        Layer 3: Transcript Analysis  → credibility_mult ∈ [0.5, 1.0]
                 (produced HERE)
                 Measures: HOW credibly management communicates
                 4 linguistic dimensions, gates Layer 2 only

    ── WHY THREE INDEPENDENT LAYERS ────────────────────────────────────

    Each layer measures a fundamentally different type of information:

        Flows       = revealed preference (actions, not words)
        Outlook     = stated preference (forward-looking claims)
        Credibility = meta-signal (quality of stated preference)

    A maths PhD would reject nesting flows inside management outlook
    because institutional behavior is an INDEPENDENT information source.
    A fund selling while management talks bullish is not "bad management
    outlook" — it's two signals DISAGREEING, which is itself informative.
    Keeping them as peers lets the synthesis engine detect and exploit
    that disagreement.

    ── MANAGEMENT SIGNAL FORMULA ───────────────────────────────────────

    This agent outputs:

        management_signal = outlook_score × credibility_multiplier

    Where:
        outlook_score ∈ [-1, +1]         — weighted sum of 6 dimensions
        credibility_multiplier ∈ [0.5, 1.0] — linguistic quality gate

    Credibility gates outlook because it's a META-signal about the
    quality of the outlook, not an independent information source.
    If management is hedgy/vague, their outlook claims carry less weight.

    The synthesis engine then combines:
        conviction = f(fundamental, management_signal, flow_composite, technical)
    where management_signal and flow_composite are independent peers.

    Args:
        ticker: Stock symbol.

    Returns:
        {
            "ticker": str,
            "outlook": { ... 6-dimension outlook ... },
            "credibility": { ... 4-dimension credibility ... },
            "management_signal": float ∈ [-1, +1],
            "data_sources": str,
            "status": str,
        }
    """
    _log.info("Starting Management Outlook analysis for %s", ticker)

    # ── Step 1: Fetch data ──
    data = fetch_outlook_data(ticker)

    # ── Step 2: Management Outlook (Layer 2) ──
    outlook = _extract_outlook_signals_llm(ticker, data["context_text"])

    # ── Step 3: Transcript Credibility (Layer 3 — gates Layer 2) ──
    credibility = _extract_credibility_llm(ticker, data["context_text"])

    # ── Step 4: Management Signal = Outlook × Credibility ──
    outlook_score = outlook.get("outlook_score", 0.0)
    cred_mult = credibility.get("credibility_multiplier", 0.75)
    management_signal = round(outlook_score * cred_mult, 4)
    management_signal = float(np.clip(management_signal, -1.0, 1.0))

    result = {
        "ticker": ticker,
        "outlook": outlook,
        "credibility": credibility,
        "management_signal": management_signal,
        "outlook_score_raw": outlook_score,
        "credibility_multiplier": cred_mult,
        "formula": f"mgmt_signal = {outlook_score:.4f} × {cred_mult:.4f} = {management_signal:.4f}",
        "data_sources": data["source_summary"],
        "earnings_context": {
            "grades": data.get("grades", {}),
            "earnings_history": data.get("earnings", []),
        },
        "status": data["status"],
        # Legacy compatibility — feed into synthesis engine
        "management_confidence": management_signal,
        "analyst_pressure": max(0, -management_signal),
    }

    _log.info(
        "Management Outlook %s: outlook=%.3f × cred=%.3f = mgmt_signal=%.3f [%s]",
        ticker, outlook_score, cred_mult, management_signal, data["source_summary"],
    )
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBSERVABILITY HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _log_data_health(
    source: str, ticker: str, status: str, elapsed: float,
    records: int, error: str = None,
):
    try:
        from invictus.observability.store import insert
        insert("data_health", {
            "source": source, "ticker": ticker, "status": status,
            "latency_ms": round(elapsed * 1000, 1),
            "records_fetched": records,
            "error_message": error[:500] if error else None,
        })
    except Exception:
        pass


def _log_llm_call(
    ticker: str, prompt: str, raw_response: str, latency_ms: float,
    response_obj=None, score=None, fallback_reason: str = None,
    agent_name: str = "outlook_agent",
):
    try:
        from invictus.observability.collectors.llm_collector import log_llm_call
        usage = getattr(response_obj, "usage", None) if response_obj else None
        log_llm_call(
            agent_name=agent_name, ticker=ticker, model=LLM_MODEL,
            prompt=prompt, response=raw_response,
            tokens_in=usage.prompt_tokens if usage else len(prompt) // 4,
            tokens_out=usage.completion_tokens if usage else len(raw_response) // 4,
            latency_ms=latency_ms, sentiment_score=score,
            fallback_used=bool(fallback_reason),
            fallback_reason=fallback_reason,
        )
    except Exception:
        pass
