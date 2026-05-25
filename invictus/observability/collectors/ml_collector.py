"""
ML model collector — tracks predictions, ensemble agreement, CV scores.
"""
from typing import Dict, Any, Optional
from invictus.observability.store import insert, insert_batch


def log_ml_predictions(predictions: list, run_id: str = None,
                       cv_score: float = None, n_features: int = None,
                       n_samples: int = None, model_type: str = None):
    """
    Log ML predictions for all tickers.
    predictions: list of dicts with keys: ticker, accumulation_prob, lr_prob, rf_prob, xgb_prob, signal_strength
    """
    rows = []
    for pred in predictions:
        rows.append({
            "run_id": run_id,
            "ticker": pred.get("ticker", ""),
            "accumulation_prob": pred.get("accumulation_prob"),
            "lr_prob": pred.get("lr_prob"),
            "rf_prob": pred.get("rf_prob"),
            "xgb_prob": pred.get("xgb_prob"),
            "signal_strength": pred.get("signal_strength"),
            "cv_score": cv_score,
            "n_features": n_features,
            "n_samples": n_samples,
            "model_type": model_type,
        })
    try:
        insert_batch("ml_predictions", rows)
    except Exception:
        pass


def log_single_prediction(ticker: str, prob: float, lr_prob: float = None,
                          rf_prob: float = None, xgb_prob: float = None,
                          run_id: str = None):
    """Log a single ML prediction."""
    try:
        insert("ml_predictions", {
            "run_id": run_id,
            "ticker": ticker,
            "accumulation_prob": prob,
            "lr_prob": lr_prob,
            "rf_prob": rf_prob,
            "xgb_prob": xgb_prob,
        })
    except Exception:
        pass
