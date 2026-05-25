"""
Invictus — Earnings & Sentiment Intelligence Agent (v3)
Uses yfinance news context as the primary source for management tone and market sentiment.
Bypasses restricted transcript APIs with high-fidelity real-time news analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT SIGNALS (consumed by synthesis_agent.py)
───────────────────────────────────────────────

1. management_confidence ∈ [-1, +1]
   LLM source: Directly scored by GPT on [-1, +1] scale.
   Dictionary fallback formula:
       raw = (positive_hits - negative_hits) / max(word_count / 500, 1)
       score = clip(raw, -1, +1)
   Normalization: word_count / 500 — i.e., per 500 words of news text,
       we expect about 1 net keyword hit as "neutral". More = bullish, less = bearish.
   Rationale: News articles average ~300-500 words. Normalizing per 500 words
       makes the score independent of article count (10 articles with 1 hit each
       = same as 1 article with 10 hits, scaled appropriately).

2. analyst_pressure ∈ [0, 1]
   LLM source: Directly scored by GPT on [0, 1] scale.
   Dictionary fallback formula:
       raw = pressure_hits / max(word_count / 1000, 1)
       score = clip(raw, 0, 1)
   Normalization: word_count / 1000 — more conservative threshold because
       pressure keywords are rarer than sentiment keywords in typical coverage.
       1 pressure keyword per 1000 words = baseline neutral.
   Rationale: Analyst skepticism keywords ("concern", "downside risk", etc.)
       are naturally less frequent than general sentiment words. Using 1000
       instead of 500 avoids inflating the pressure score for typical coverage.
       This means the pressure signal requires more concentrated skepticism
       to reach high values — appropriate since false-positive pressure
       signals are more costly than false-negative ones.

Note: The different normalization bases (500 vs 1000) are INTENTIONAL.
      Confidence measures general tone (common words) while pressure
      measures specific skepticism (rare words). Different base rates
      require different normalization to produce comparable [0,1] outputs.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import yfinance as yf
import streamlit as st
import json
import numpy as np
from typing import Dict, Any, List, Optional

from invictus.agents.graph_state import PortfolioState
from invictus.llm import call_llm_json, llm_available, get_provider

# ── Sentiment Dictionaries (LLM-free fallback) ────────────────────────

CONFIDENCE_POSITIVE = [
    "strong quarter", "exceeded", "beat expectations", "confident", "optimistic",
    "record revenue", "momentum", "expanding", "outperform", "raising guidance",
    "ahead of plan", "strong execution", "market share gains", "double digit growth",
    "significant improvement", "exceptional", "robust demand", "accelerating",
]

CONFIDENCE_NEGATIVE = [
    "challenging", "headwind", "uncertainty", "cautious", "soft demand",
    "below expectations", "missed", "disappointed", "weakness", "lowering guidance",
    "restructuring", "impairment", "macro uncertainty", "demand softness",
    "cost pressure", "margin compression", "supply chain",
]

ANALYST_PRESSURE_KEYWORDS = [
    "concern", "worried", "skeptical", "downside risk", "bear case",
    "guidance cut", "margin pressure", "competitive threat",
    "customer churn", "deterioration", "sustainability",
]


def _fetch_yfinance_sentiment_context(ticker: str) -> str:
    """Fetches latest institutional news context from yfinance."""
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news:
            return ""
        articles = news[:10]
        context = []
        for item in articles:
            # yfinance news format can vary — handle both structures
            if isinstance(item, dict):
                content = item.get("content", item)
                if isinstance(content, dict):
                    title = content.get("title", "")
                    summary = content.get("summary", content.get("description", ""))
                else:
                    title = item.get("title", "")
                    summary = item.get("summary", item.get("description", ""))
            else:
                continue
            if title:
                context.append(f"Title: {title}\nSummary: {summary}\n---")
        return "\n".join(context)
    except Exception:
        return ""


def _dictionary_sentiment(text: str) -> Dict[str, Any]:
    """LLM-free sentiment scoring using keyword dictionaries."""
    if not text:
        return {
            "management_confidence": 0.0,
            "confidence_reasoning": "No news context available for analysis.",
            "analyst_pressure": 0.0,
            "pressure_reasoning": "No data available.",
            "sentiment_trend": "Stable",
            "tone_drivers": ["Insufficient data"],
            "analyst_concerns": ["No data available"],
            "status": "Success",
            "source": "Dictionary Fallback",
        }

    text_lower = text.lower()
    word_count = max(len(text_lower.split()), 1)

    pos_hits = sum(1 for kw in CONFIDENCE_POSITIVE if kw in text_lower)
    neg_hits = sum(1 for kw in CONFIDENCE_NEGATIVE if kw in text_lower)
    pressure_hits = sum(1 for kw in ANALYST_PRESSURE_KEYWORDS if kw in text_lower)

    conf_raw = (pos_hits - neg_hits) / max(word_count / 500.0, 1.0)
    management_confidence = float(np.clip(conf_raw, -1.0, 1.0))
    analyst_pressure = float(np.clip(pressure_hits / max(word_count / 1000.0, 1.0), 0.0, 1.0))

    if pos_hits > neg_hits + 2:
        trend = "Improving"
    elif neg_hits > pos_hits + 2:
        trend = "Deteriorating"
    else:
        trend = "Stable"

    tone_drivers = []
    if pos_hits > neg_hits:
        tone_drivers.append(f"Net positive sentiment ({pos_hits} positive vs {neg_hits} negative)")
    elif neg_hits > pos_hits:
        tone_drivers.append(f"Net negative sentiment ({neg_hits} negative vs {pos_hits} positive)")
    else:
        tone_drivers.append("Balanced sentiment")
    if pressure_hits > 2:
        tone_drivers.append(f"Elevated market skepticism ({pressure_hits} pressure indicators)")

    concerns = []
    if analyst_pressure > 0.3:
        concerns.append("High analyst skepticism detected")
    if neg_hits > pos_hits:
        concerns.append("Defensive/negative tone in coverage")
    if not concerns:
        concerns.append("No major concerns flagged")

    return {
        "management_confidence": management_confidence,
        "confidence_reasoning": f"Dictionary analysis: {pos_hits} positive, {neg_hits} negative indicators. Net: {management_confidence:+.2f}.",
        "analyst_pressure": analyst_pressure,
        "pressure_reasoning": f"{pressure_hits} pressure keywords detected.",
        "sentiment_trend": trend,
        "tone_drivers": tone_drivers[:3],
        "analyst_concerns": concerns[:3],
        "status": "Success",
        "source": "News Dictionary",
    }


def _analyze_sentiment_with_llm(ticker: str, context: str) -> Dict[str, Any]:
    """Uses LLM to assess sentiment, management tone, and analyst pressure."""
    if not llm_available():
        try:
            from invictus.observability.collectors.llm_collector import log_llm_call
            log_llm_call("earnings_agent", ticker=ticker, fallback_used=True, fallback_reason="No API key")
        except Exception:
            pass
        return {"status": "LLM API Key missing"}
    if not context:
        return {"status": "Insufficient news context"}

    prompt = f"""You are a specialist in institutional sentiment analysis for equity research.
Analyze the following latest news and reports for {ticker}.
Assess the management's confidence, sentiment trends, and market/analyst skepticism.

News Context: {context}

Output strictly in the following JSON format:
{{
  "management": {{
    "score": "Score from -1.0 to +1.0 (Pessimistic to Highly Confident)",
    "reasoning": "Explain the tone found in management quotes or news reporting."
  }},
  "analyst": {{
    "score": "Score from 0.0 to 1.0 (Compliant to Highly Skeptical)",
    "reasoning": "Explain the nature of market skepticism or analyst concerns."
  }},
  "sentiment_trend": "Improving, Stable, or Deteriorating",
  "tone_drivers": ["list of 2-3 key observations about management's language or news sentiment"],
  "analyst_concerns": ["list of 2-3 topics the market is most skeptical about"]
}}
"""
    try:
        import time as _t
        _start = _t.perf_counter()
        analysis = call_llm_json(prompt)
        _latency = (_t.perf_counter() - _start) * 1000

        result = {
            "management_confidence": float(analysis["management"]["score"]),
            "confidence_reasoning": analysis["management"]["reasoning"],
            "analyst_pressure": float(analysis["analyst"]["score"]),
            "pressure_reasoning": analysis["analyst"]["reasoning"],
            "sentiment_trend": analysis["sentiment_trend"],
            "tone_drivers": analysis["tone_drivers"],
            "analyst_concerns": analysis["analyst_concerns"],
            "status": "Success",
            "source": f"Yahoo News (LLM — {get_provider()})",
        }

        try:
            from invictus.observability.collectors.llm_collector import log_llm_call
            log_llm_call(
                agent_name="earnings_agent", ticker=ticker, model=get_provider(),
                prompt=prompt, response=str(analysis),
                tokens_in=len(prompt) // 4,
                tokens_out=len(str(analysis)) // 4,
                latency_ms=_latency,
                sentiment_score=result["management_confidence"],
            )
        except Exception:
            pass

        return result
    except Exception as e:
        try:
            from invictus.observability.collectors.llm_collector import log_llm_call
            log_llm_call("earnings_agent", ticker=ticker, model=get_provider(), fallback_used=True, fallback_reason=str(e)[:200])
        except Exception:
            pass
        return {"status": f"LLM failed: {str(e)}"}


def run_earnings_intel(state: PortfolioState) -> PortfolioState:
    tickers = list(state.weights.keys()) if state.weights else []
    results = {}
    progress_bar = st.progress(0, text="Analyzing Management Tone...")

    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Sentiment Intel: {ticker}")
        context = _fetch_yfinance_sentiment_context(ticker)

        # Try LLM first, fall back to dictionary
        llm_result = _analyze_sentiment_with_llm(ticker, context)
        if llm_result and llm_result.get("status") == "Success":
            results[ticker] = llm_result
        else:
            # Dictionary fallback — always works
            results[ticker] = _dictionary_sentiment(context)

    state.earnings_intel = results
    progress_bar.empty()
    return state
