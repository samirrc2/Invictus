"""
Conviction engine collector — tracks conviction scores, signal contributions, stability.
"""
from typing import Dict, Any
from invictus.observability.store import insert


def log_conviction(ticker: str, synthesis: Dict[str, Any], run_id: str = None):
    """Log a conviction synthesis result for a ticker."""
    try:
        ci = synthesis.get("confidence_intervals", {})
        agree = synthesis.get("signal_agreement", {})
        signals = synthesis.get("signals_detail", {})

        # Extract individual signal scores
        filing_score = signals.get("fundamentals", {}).get("score", None)
        earnings_score = signals.get("management", {}).get("score", None)
        flow_score = signals.get("flows", {}).get("score", None)
        ml_score = signals.get("technical", {}).get("score", None)

        insert("conviction_scores", {
            "run_id": run_id,
            "ticker": ticker,
            "composite_score": synthesis.get("composite_score"),
            "outperformance_prob": synthesis.get("outperformance_probability"),
            "conviction_level": synthesis.get("conviction_level"),
            "signal_confidence": synthesis.get("signal_confidence"),
            "dominant_driver": synthesis.get("dominant_driver"),
            "ci_width": ci.get("ci_width"),
            "ci_mean": ci.get("mean_probability"),
            "signal_agreement": agree.get("agreement_label"),
            "bullish_count": agree.get("bullish_count"),
            "bearish_count": agree.get("bearish_count"),
            "filing_score": filing_score,
            "earnings_score": earnings_score,
            "flow_score": flow_score,
            "ml_score": ml_score,
        })
    except Exception:
        pass
