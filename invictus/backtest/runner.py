"""
invictus.backtest.runner
=========================
Walk-forward backtest engine.

For each evaluation date, runs a lightweight conviction mini-pipeline
using ONLY data available as of that date. Produces conviction scores
that are then compared against actual forward returns.

Design: Does NOT use the full orchestrator/LangGraph pipeline.
Instead, directly calls the mathematical signal functions to:
  1. Compute fundamental signals from point-in-time financials
  2. Run Bayesian ML model on point-in-time price features
  3. Synthesize signals into conviction probability
  4. Record conviction with evaluation date for later analysis

LLM-dependent agents (outlook, commentary) are skipped because:
  - Running them 120+ times would cost $50+ in API calls
  - Historical transcript data isn't point-in-time accessible
  - The fundamental + ML signals are the quantitative core anyway
"""
import numpy as np
import pandas as pd
from datetime import date
from typing import Dict, List, Any, Optional, Callable

from invictus.backtest.config import BacktestConfig
from invictus.backtest.data_loader import HistoricalDataStore
from invictus.backtest.analyzer import BacktestAnalyzer


# ══════════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION (lightweight versions of agent logic)
# ══════════════════════════════════════════════════════════════════════════

def _compute_fundamental_signal(
    financials: Optional[pd.DataFrame],
) -> Dict[str, float]:
    """
    Compute fundamental conviction from point-in-time quarterly financials.
    Mirrors filing_agent.py logic exactly:
      fundamental_conviction = clip((0.35·rev + 0.35·net + 0.30·op) / 0.10, -1, 1)
      guidance_momentum = clip(rev_growth / 0.05, -1, 1)
      risk_deterioration = heuristic [0, 1]
    """
    if financials is None or financials.empty or len(financials.columns) < 2:
        return {
            "fundamental_conviction": 0.0,
            "guidance_momentum": 0.0,
            "risk_deterioration": 0.3,
            "status": "insufficient_data",
        }

    def _find_row(df, keywords):
        for kw in keywords:
            matches = df.loc[df.index.str.contains(kw, case=False, na=False)]
            if not matches.empty:
                return matches.iloc[0]
        return None

    def _safe_growth(current, previous):
        try:
            c = float(current) if current is not None and not pd.isna(current) else 0
            p = float(previous) if previous is not None and not pd.isna(previous) else 0
            if p == 0 or abs(p) < 1:
                return 0.0
            return (c - p) / abs(p)
        except (ValueError, TypeError):
            return 0.0

    rev_row = _find_row(financials, ["Total Revenue", "Revenue", "Net Revenue"])
    if rev_row is None:
        return {"fundamental_conviction": 0.0, "guidance_momentum": 0.0,
                "risk_deterioration": 0.3, "status": "no_revenue"}

    revs = rev_row.values
    net_row = _find_row(financials, ["Net Income", "Net Income Common"])
    net = net_row.values if net_row is not None else [0, 0]
    op_row = _find_row(financials, ["Operating Income", "Operating Profit", "EBIT"])
    op = op_row.values if op_row is not None else [0, 0]

    rev_growth = _safe_growth(revs[0], revs[1]) if len(revs) > 1 else 0
    net_growth = _safe_growth(net[0], net[1]) if len(net) > 1 else 0
    op_growth = _safe_growth(op[0], op[1]) if len(op) > 1 else 0

    # fundamental_conviction = clip((0.35·rev + 0.35·net + 0.30·op) / 0.10, -1, 1)
    f_conv = float(np.clip(
        (rev_growth * 0.35 + net_growth * 0.35 + op_growth * 0.30) / 0.10,
        -1.0, 1.0
    ))
    # guidance_momentum = clip(rev_growth / 0.05, -1, 1)
    g_mom = float(np.clip(rev_growth / 0.05, -1.0, 1.0))

    # risk_deterioration
    net_latest = float(net[0]) if net[0] is not None and not pd.isna(net[0]) else 0
    r_det = 0.1 if net_latest > 0 else 0.5
    if op_growth < -0.1:
        r_det += 0.2
    r_det = float(np.clip(r_det, 0, 1))

    return {
        "fundamental_conviction": f_conv,
        "guidance_momentum": g_mom,
        "risk_deterioration": r_det,
        "status": "ok",
    }


def _compute_ml_signal(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str,
    benchmark: str = "SPY",
) -> Dict[str, float]:
    """
    Compute Bayesian ML accumulation signal from point-in-time price data.
    Imports and calls the actual ml_agent functions directly.
    """
    try:
        from invictus.agents.ml_agent import (
            _compute_features, _compute_signal_values, _bayesian_update,
        )

        if ticker not in returns.columns or len(returns[ticker].dropna()) < 60:
            return {"accumulation_prob": 0.5, "status": "insufficient_data"}

        # Build features for this single ticker
        weights = {ticker: 1.0}
        if benchmark in returns.columns:
            weights[benchmark] = 0.0  # include for relative strength calc

        features = _compute_features(returns, prices, weights)
        if features.empty:
            return {"accumulation_prob": 0.5, "status": "no_features"}

        ticker_row = features[features["Ticker"] == ticker]
        if ticker_row.empty:
            return {"accumulation_prob": 0.5, "status": "ticker_not_found"}

        row = ticker_row.iloc[0]
        px = prices[ticker].dropna()
        price_level = float(px.iloc[-1]) if len(px) > 0 else 100.0

        signals = _compute_signal_values(row, price_level)
        result = _bayesian_update(signals)

        return {
            "accumulation_prob": float(result["posterior"]),
            "status": "ok",
        }
    except Exception as e:
        return {"accumulation_prob": 0.5, "status": f"error: {e}"}


def _synthesize_conviction(
    fundamental: Dict[str, float],
    ml_signal: Dict[str, float],
    config: BacktestConfig,
    signal_quality: float = 0.6,
) -> Dict[str, Any]:
    """
    Combine fundamental + ML signals into conviction probability.
    Mirrors synthesis_agent._calculate_stock_conviction logic.

    Note: management and flow signals are set to neutral (0.0) since
    we don't run LLM-dependent agents or historical 13F replay.
    The weights are redistributed proportionally to fundamental + technical.

    signal_quality: shrinkage factor [0,1]. Lower = more shrinkage toward 0.5.
      - 0.6  = conservative (2/4 signals → heavy shrinkage, narrow spread)
      - 0.85 = full-signal proxy (assumes missing signals ≈ neutral, not unknown)
    """
    # Extract base scores
    f_score = fundamental.get("fundamental_conviction", 0.0)
    g_mom = fundamental.get("guidance_momentum", 0.0)
    r_det = fundamental.get("risk_deterioration", 0.3)

    # Filing signal (same formula as synthesis_agent)
    r_signal = -(r_det * 2 - 1)  # [0,1] → [+1,-1]
    base_f = f_score * 0.40 + g_mom * 0.30 + r_signal * 0.30

    # ML/technical signal
    accum_prob = ml_signal.get("accumulation_prob", 0.5)
    base_tech = (accum_prob - 0.5) * 2  # [0,1] → [-1,+1]

    # Management and flow signals unavailable — set neutral
    base_mgmt = 0.0
    base_flow = 0.0

    # Clip all to [-1, +1]
    base_scores = {
        "fundamental": float(np.clip(base_f, -1, 1)),
        "management": float(np.clip(base_mgmt, -1, 1)),
        "flows": float(np.clip(base_flow, -1, 1)),
        "technical": float(np.clip(base_tech, -1, 1)),
    }

    # Weights (use config defaults, same as synthesis_agent)
    weights = {
        "fundamental": config.w_fundamental,
        "management": config.w_management,
        "flows": config.w_flows,
        "technical": config.w_technical,
    }

    # Weighted composite
    composite = sum(base_scores[s] * weights[s] for s in base_scores)

    # Signal agreement (simplified — count same-sign signals)
    signs = [1 if v > 0.05 else -1 if v < -0.05 else 0
             for v in base_scores.values()]
    non_zero = [s for s in signs if s != 0]
    if non_zero:
        agreement = abs(sum(non_zero)) / len(non_zero)
    else:
        agreement = 0.5

    # Agreement multiplier: M = 0.8 + 0.4·agreement ∈ [0.8, 1.2]
    composite *= (0.8 + agreement * 0.4)

    # Probability via calibrated logistic: P = σ(3·C)
    prob = float(1 / (1 + np.exp(-3 * composite)))

    # Quality shrinkage: pull probability toward 0.5 based on data completeness
    # P_final = P·Q + 0.5·(1−Q) where Q = signal_quality
    prob_final = (prob * signal_quality) + (0.5 * (1 - signal_quality))

    # Dominant driver
    abs_scores = {k: abs(v * weights[k]) for k, v in base_scores.items()}
    dominant = max(abs_scores, key=abs_scores.get)

    return {
        "composite_score": float(composite),
        "outperformance_prob": float(prob_final),
        "base_scores": base_scores,
        "weights_used": weights,
        "agreement_score": float(agreement),
        "dominant_driver": dominant,
        "signal_quality": signal_quality,
        "signals_available": ["fundamental", "technical"],
        "signals_unavailable": ["management", "flows"],
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_walk_forward(
    tickers: Optional[List[str]] = None,
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    frequency: str = "monthly",
    signal_quality: float = 0.6,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Run a complete walk-forward ex-ante backtest.

    For each evaluation date:
      1. Slice all data to that date only (no look-ahead)
      2. Compute fundamental signals from point-in-time financials
      3. Run Bayesian ML model on point-in-time price features
      4. Synthesize into conviction probability
      5. Record conviction + actual forward returns

    signal_quality: shrinkage factor for conviction probabilities.
      0.6  = conservative (heavy shrinkage, fewer trades)
      0.85 = full-signal proxy (lighter shrinkage, more trades)

    Returns a comprehensive results dict ready for analysis.
    """
    config = BacktestConfig(
        tickers=tickers or BacktestConfig().tickers,
        start=start,
        end=end,
        frequency=frequency,
    )

    eval_dates = config.eval_dates()
    if not eval_dates:
        return {"status": "error", "message": "No evaluation dates in range"}

    # ── Step 1: Load all data ─────────────────────────────────────
    if progress_callback:
        progress_callback("Loading historical data...")

    store = HistoricalDataStore(config)
    store.load(progress_callback)

    # ── Step 2: Walk-forward loop ─────────────────────────────────
    all_convictions: List[Dict[str, Any]] = []
    n_dates = len(eval_dates)

    for i, eval_dt in enumerate(eval_dates):
        if progress_callback:
            progress_callback(
                f"Evaluating {eval_dt.isoformat()} ({i+1}/{n_dates})..."
            )

        # Point-in-time data slices
        prices = store.prices_as_of(eval_dt)
        returns = store.returns_as_of(eval_dt)

        for ticker in config.tickers:
            # Skip if no price data for this ticker as of eval_date
            if ticker not in prices.columns or prices[ticker].dropna().empty:
                continue

            # 2a. Fundamental signal (point-in-time financials)
            financials = store.financials_as_of(ticker, eval_dt)
            fundamental = _compute_fundamental_signal(financials)

            # 2b. ML/technical signal (point-in-time price features)
            ml_signal = _compute_ml_signal(
                returns, prices, ticker, config.benchmark
            )

            # 2c. Synthesize
            conviction = _synthesize_conviction(
                fundamental, ml_signal, config, signal_quality
            )

            # 2d. Record
            all_convictions.append({
                "eval_date": eval_dt.isoformat(),
                "ticker": ticker,
                "outperformance_prob": conviction["outperformance_prob"],
                "composite_score": conviction["composite_score"],
                "fundamental_score": conviction["base_scores"]["fundamental"],
                "technical_score": conviction["base_scores"]["technical"],
                "agreement": conviction["agreement_score"],
                "dominant_driver": conviction["dominant_driver"],
            })

    if not all_convictions:
        return {"status": "error", "message": "No convictions generated"}

    # ── Step 3: Compute forward returns ───────────────────────────
    if progress_callback:
        progress_callback("Computing forward returns...")

    for entry in all_convictions:
        eval_dt = date.fromisoformat(entry["eval_date"])
        ticker = entry["ticker"]

        for h in config.horizons:
            fwd = store.forward_return(eval_dt, h)
            if fwd is not None and ticker in fwd.index:
                entry[f"fwd_{h}d"] = float(fwd[ticker])
            else:
                entry[f"fwd_{h}d"] = None

    # ── Step 4: Analyze ───────────────────────────────────────────
    if progress_callback:
        progress_callback("Analyzing results...")

    analyzer = BacktestAnalyzer(config)
    results = analyzer.analyze(all_convictions)

    results["config"] = {
        "tickers": config.tickers,
        "start": config.start,
        "end": config.end,
        "frequency": config.frequency,
        "horizons": config.horizons,
        "eval_dates": [d.isoformat() for d in eval_dates],
        "n_eval_dates": n_dates,
        "n_convictions": len(all_convictions),
        "signal_quality": signal_quality,
        "quality_mode": "conservative" if signal_quality <= 0.7 else "full_proxy",
        "signals_used": ["fundamental (filing_agent)", "technical (ml_agent)"],
        "signals_unavailable": ["management (needs LLM)", "flows (needs 13F replay)"],
    }
    results["raw_convictions"] = all_convictions
    results["status"] = "ok"

    if progress_callback:
        progress_callback("Backtest complete!")

    return results
