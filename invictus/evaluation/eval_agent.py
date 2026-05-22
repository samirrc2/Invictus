"""
Invictus — LLM Evaluation & Prompt A/B Testing Agent
Evaluates commentary quality across prompt variants using rule-based checks.

Metrics:
- Factual consistency (references real numbers from data)
- Numerical grounding (contains specific percentages/values)
- Completeness (covers key sections: performance, risk, attribution, action)
- Clarity (readability, sentence length)
- Hallucination risk (mentions tickers/data not in portfolio)
- Output length
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from invictus.agents.graph_state import PortfolioState


def _check_numerical_grounding(text: str) -> Dict[str, Any]:
    """Check if commentary contains specific numbers."""
    numbers = re.findall(r'[\d]+\.[\d]+%|[\d]+%|\$[\d,]+', text)
    return {
        "score": min(len(numbers) / 5, 1.0),  # expect at least 5 numbers
        "count": len(numbers),
        "numbers_found": numbers[:10],
    }


def _check_completeness(text: str) -> Dict[str, Any]:
    """Check if commentary covers key topics."""
    topics = {
        "performance": ["return", "up", "down", "performance", "gain", "loss", "moved"],
        "risk": ["risk", "volatility", "var", "drawdown", "sharpe", "vol"],
        "attribution": ["contributor", "driven", "attribution", "sector", "factor"],
        "action": ["monitor", "recommend", "watch", "consider", "review", "action"],
    }
    covered = {}
    for topic, keywords in topics.items():
        text_lower = text.lower()
        covered[topic] = any(kw in text_lower for kw in keywords)

    score = sum(covered.values()) / len(topics)
    return {"score": score, "topics_covered": covered}


def _check_clarity(text: str) -> Dict[str, Any]:
    """Check readability metrics."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    avg_sentence_len = len(words) / max(len(sentences), 1)

    # Penalize very long sentences
    clarity_score = 1.0
    if avg_sentence_len > 35:
        clarity_score -= 0.3
    if avg_sentence_len > 50:
        clarity_score -= 0.3

    return {
        "score": max(clarity_score, 0.1),
        "avg_sentence_length": avg_sentence_len,
        "total_sentences": len(sentences),
        "total_words": len(words),
    }


def _check_hallucination_risk(text: str, valid_tickers: List[str]) -> Dict[str, Any]:
    """Check for potential hallucinations — mentions of tickers not in portfolio."""
    # Find ticker-like patterns (all caps, 2-5 chars)
    mentioned = set(re.findall(r'\b[A-Z]{2,5}\b', text))
    # Filter to likely ticker mentions
    common_words = {"THE", "AND", "FOR", "THIS", "THAT", "WITH", "FROM", "ARE", "WAS",
                    "HAS", "HAD", "NOT", "BUT", "ALL", "CAN", "HER", "ONE", "OUR",
                    "NEW", "NOW", "WAY", "MAY", "DAY", "TOO", "ANY", "WHO", "BOY",
                    "DID", "GET", "HIM", "HIS", "HOW", "MAN", "OLD", "SEE", "TOP",
                    "KEY", "LOW", "HIGH", "NET", "PER", "GDP", "YOY", "QOQ", "BPS",
                    "NAV", "AUM", "ETF", "IPO", "CEO", "CFO", "CTO", "COO", "FCF",
                    "ROE", "ROA", "EPS", "P&L", "RISK", "BRIEF", "DATA", "PORTFOLIO",
                    "SUMMARY", "SECTOR", "REGIME", "DETAILED", "MANAGER", "CONTEXT",
                    "ANALYSIS", "ITEMS", "PERFORMANCE", "RECOMMENDATION"}
    potential_tickers = mentioned - common_words
    valid_set = set(valid_tickers)
    unknown_tickers = potential_tickers - valid_set

    score = 1.0 - min(len(unknown_tickers) * 0.2, 0.8)
    return {
        "score": max(score, 0.2),
        "unknown_mentions": list(unknown_tickers),
        "valid_mentions": list(potential_tickers & valid_set),
    }


def _check_factual_consistency(text: str, context: Dict) -> Dict[str, Any]:
    """Check if commentary is consistent with provided data."""
    issues = []

    # Check direction consistency
    port_ret = context.get("portfolio_return", 0)
    if port_ret > 0 and any(w in text.lower() for w in ["down", "loss", "declined", "fell"]):
        if not any(w in text.lower() for w in ["detractor", "drag", "underperform"]):
            issues.append("Commentary says 'down' but portfolio was up")
    if port_ret < 0 and any(w in text.lower() for w in ["gained", "rallied", "surged"]):
        if not any(w in text.lower() for w in ["contributor", "gained"]):
            issues.append("Commentary says 'up' but portfolio was down")

    score = 1.0 - min(len(issues) * 0.3, 0.9)
    return {"score": max(score, 0.1), "issues": issues}


def evaluate_commentary(state: PortfolioState) -> PortfolioState:
    """Evaluate all commentary variants and produce comparison."""
    commentary = state.commentary
    if not commentary:
        raise ValueError("No commentary to evaluate. Run commentary generation first.")

    styles = commentary.get("_styles", [])
    source = commentary.get("_source", "template")

    # Get context for factual checks
    pnl = state.pnl_attribution or {}
    tickers = []
    if state.holdings is not None and hasattr(state.holdings, "Ticker"):
        tickers = state.holdings["Ticker"].tolist()
    elif state.weights:
        tickers = list(state.weights.keys())

    context = {"portfolio_return": pnl.get("portfolio_return", 0)}

    # Evaluate each style
    eval_rows = []
    eval_details = {}

    for style in styles:
        text = commentary.get(style, "")
        if not text:
            continue

        numerical = _check_numerical_grounding(text)
        completeness = _check_completeness(text)
        clarity = _check_clarity(text)
        hallucination = _check_hallucination_risk(text, tickers)
        factual = _check_factual_consistency(text, context)

        overall = np.mean([
            numerical["score"],
            completeness["score"],
            clarity["score"],
            hallucination["score"],
            factual["score"],
        ])

        eval_rows.append({
            "Prompt Style": style.replace("_", " ").title(),
            "Numerical Grounding": numerical["score"],
            "Completeness": completeness["score"],
            "Clarity": clarity["score"],
            "Hallucination Safety": hallucination["score"],
            "Factual Consistency": factual["score"],
            "Overall Score": overall,
            "Word Count": clarity["total_words"],
            "Numbers Found": numerical["count"],
        })

        eval_details[style] = {
            "numerical": numerical,
            "completeness": completeness,
            "clarity": clarity,
            "hallucination": hallucination,
            "factual": factual,
            "overall": overall,
        }

    eval_df = pd.DataFrame(eval_rows).sort_values("Overall Score", ascending=False)
    best_prompt = eval_df.iloc[0]["Prompt Style"] if len(eval_df) > 0 else "N/A"

    state.eval_results = {
        "eval_table": eval_df,
        "eval_details": eval_details,
        "best_prompt": best_prompt,
        "source": source,
        "n_variants": len(styles),
    }

    return state
