"""
Invictus — Earnings & Sentiment Intelligence Agent (v3)
Uses yfinance news context as the primary source for management tone and market sentiment.
Bypasses restricted transcript APIs with high-fidelity real-time news analysis.
"""
import yfinance as yf
import streamlit as st
import json
from typing import Dict, Any, List, Optional

from invictus.agents.graph_state import PortfolioState
from invictus.config import OPENAI_API_KEY, LLM_MODEL

def _fetch_yfinance_sentiment_context(ticker: str) -> str:
    """Fetches latest institutional news context from yfinance."""
    try:
        t = yf.Ticker(ticker)
        news = t.news[:10] # Top 10 latest articles
        context = []
        for item in news:
            content = item.get("content", {})
            title = content.get("title", "")
            summary = content.get("summary", "")
            context.append(f"Title: {title}\nSummary: {summary}\n---")
        return "\n".join(context)
    except Exception:
        return ""

def _analyze_sentiment_with_llm(ticker: str, context: str) -> Dict[str, Any]:
    """Uses LLM to assess sentiment, management tone, and analyst pressure."""
    if not OPENAI_API_KEY: return {"status": "LLM API Key missing"}
    if not context: return {"status": "Insufficient news context"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
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
        response = client.chat.completions.create(
            model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0, response_format={"type": "json_object"}
        )
        analysis = json.loads(response.choices[0].message.content)
        
        return {
            "management_confidence": float(analysis["management"]["score"]),
            "confidence_reasoning": analysis["management"]["reasoning"],
            "analyst_pressure": float(analysis["analyst"]["score"]),
            "pressure_reasoning": analysis["analyst"]["reasoning"],
            "sentiment_trend": analysis["sentiment_trend"],
            "tone_drivers": analysis["tone_drivers"],
            "analyst_concerns": analysis["analyst_concerns"],
            "status": "Success",
            "source": "Yahoo News Sentiment"
        }
    except Exception as e:
        return {"status": f"Analysis failed: {str(e)}"}

def run_earnings_intel(state: PortfolioState) -> PortfolioState:
    tickers = list(state.weights.keys()) if state.weights else []
    results = {}
    progress_bar = st.progress(0, text="Analyzing Management Tone...")
    
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Sentiment Intel: {ticker}")
        context = _fetch_yfinance_sentiment_context(ticker)
        results[ticker] = _analyze_sentiment_with_llm(ticker, context)
        
    state.earnings_intel = results
    progress_bar.empty()
    return state
