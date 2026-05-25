"""
Drift Analyzer — detects feature drift, prediction drift, conviction stability.
"""
from typing import Dict, Any, List
from invictus.observability.store import query


def analyze_ml_drift() -> Dict[str, Any]:
    """Analyze ML prediction drift and ensemble agreement."""
    total = query("SELECT COUNT(*) as cnt FROM ml_predictions")[0]["cnt"]
    if total == 0:
        return {"status": "no_data", "total_predictions": 0}

    # Prediction confidence distribution
    conf_dist = query(
        "SELECT "
        "  SUM(CASE WHEN accumulation_prob > 0.7 THEN 1 ELSE 0 END) as high_conf, "
        "  SUM(CASE WHEN accumulation_prob BETWEEN 0.4 AND 0.6 THEN 1 ELSE 0 END) as low_conf, "
        "  SUM(CASE WHEN accumulation_prob < 0.3 THEN 1 ELSE 0 END) as strong_sell, "
        "  AVG(accumulation_prob) as avg_prob, "
        "  COUNT(*) as total "
        "FROM ml_predictions"
    )[0]

    # Ensemble agreement: how often do LR and RF agree on direction?
    agreement = query(
        "SELECT COUNT(*) as cnt FROM ml_predictions "
        "WHERE lr_prob IS NOT NULL AND rf_prob IS NOT NULL "
        "AND ((lr_prob > 0.5 AND rf_prob > 0.5) OR (lr_prob < 0.5 AND rf_prob < 0.5))"
    )[0]["cnt"]
    total_with_both = query(
        "SELECT COUNT(*) as cnt FROM ml_predictions "
        "WHERE lr_prob IS NOT NULL AND rf_prob IS NOT NULL"
    )[0]["cnt"]
    agreement_rate = agreement / max(total_with_both, 1)

    # Per-ticker prediction stability (same ticker, multiple runs)
    stability = query(
        "SELECT ticker, COUNT(*) as runs, "
        "AVG(accumulation_prob) as avg_p, "
        "MAX(accumulation_prob) - MIN(accumulation_prob) as range_p "
        "FROM ml_predictions GROUP BY ticker HAVING runs > 1 "
        "ORDER BY range_p DESC LIMIT 10"
    )

    # CV score trending
    cv_scores = query(
        "SELECT cv_score, created_at FROM ml_predictions "
        "WHERE cv_score IS NOT NULL ORDER BY created_at"
    )

    return {
        "total_predictions": total,
        "avg_probability": conf_dist["avg_prob"],
        "high_confidence_pct": conf_dist["high_conf"] / max(total, 1),
        "low_confidence_pct": conf_dist["low_conf"] / max(total, 1),
        "ensemble_agreement_rate": agreement_rate,
        "ticker_stability": stability,
        "cv_score_history": cv_scores,
    }


def analyze_conviction_stability() -> Dict[str, Any]:
    """Analyze conviction score stability and signal contribution patterns."""
    total = query("SELECT COUNT(*) as cnt FROM conviction_scores")[0]["cnt"]
    if total == 0:
        return {"status": "no_data", "total_scores": 0}

    # Per-ticker conviction stability
    stability = query(
        "SELECT ticker, COUNT(*) as runs, "
        "AVG(composite_score) as avg_score, "
        "MAX(composite_score) - MIN(composite_score) as score_range, "
        "AVG(outperformance_prob) as avg_prob "
        "FROM conviction_scores GROUP BY ticker "
        "ORDER BY score_range DESC"
    )

    # Signal agreement distribution
    agreement_dist = query(
        "SELECT signal_agreement, COUNT(*) as cnt "
        "FROM conviction_scores WHERE signal_agreement IS NOT NULL "
        "GROUP BY signal_agreement ORDER BY cnt DESC"
    )

    # Dominant driver distribution (which signal dominates most often?)
    driver_dist = query(
        "SELECT dominant_driver, COUNT(*) as cnt "
        "FROM conviction_scores WHERE dominant_driver IS NOT NULL "
        "GROUP BY dominant_driver ORDER BY cnt DESC"
    )

    # CI width distribution (are intervals informative?)
    ci_stats = query(
        "SELECT AVG(ci_width) as avg_width, MAX(ci_width) as max_width, "
        "MIN(ci_width) as min_width FROM conviction_scores WHERE ci_width IS NOT NULL"
    )

    # Signal score correlations (which signals move together?)
    signal_data = query(
        "SELECT filing_score, earnings_score, flow_score, ml_score "
        "FROM conviction_scores "
        "WHERE filing_score IS NOT NULL AND earnings_score IS NOT NULL"
    )

    return {
        "total_scores": total,
        "ticker_stability": stability,
        "agreement_distribution": {r["signal_agreement"]: r["cnt"] for r in agreement_dist},
        "dominant_driver_distribution": {r["dominant_driver"]: r["cnt"] for r in driver_dist},
        "ci_width_stats": ci_stats[0] if ci_stats else {},
        "signal_data_points": len(signal_data),
    }
