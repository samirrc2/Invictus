"""
Invictus — Conviction Synthesis Engine (v2)
Institutional-grade signal aggregation with dynamic weighting,
regime-conditional adjustments, Monte Carlo confidence intervals,
and signal agreement/disagreement analysis.

Aggregates independent signals from Fundamentals, Management, Flows,
and ML predictions to generate explainable conviction probabilities.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from invictus.agents.graph_state import PortfolioState

# ── Base Weighting Configuration ─────────────────────────────────────
# Replicating institutional "Football Field" weights
BASE_SIGNAL_WEIGHTS = {
    "fundamental": 0.35,   # Filing intelligence (MD&A, Risk, Growth)
    "management": 0.25,    # Earnings tone & analyst pressure
    "flows": 0.25,         # Institutional & Insider activity
    "technical": 0.15,     # ML model / technical signals
}

# Regime-conditional weight adjustments
REGIME_ADJUSTMENTS = {
    "High": {
        # In high volatility: flows & fundamentals matter more, management tone less reliable
        "fundamental": +0.05, "management": -0.10, "flows": +0.05, "technical": 0.0,
    },
    "Medium": {
        # Normal regime: use base weights
        "fundamental": 0.0, "management": 0.0, "flows": 0.0, "technical": 0.0,
    },
    "Low": {
        # In low volatility: technical signals & momentum matter more
        "fundamental": -0.05, "management": 0.0, "flows": -0.05, "technical": +0.10,
    },
}


def _get_dynamic_weights(
    vol_regime: Optional[str],
    signal_quality: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute dynamic signal weights based on:
    1. Current volatility regime
    2. Signal quality/availability
    """
    weights = dict(BASE_SIGNAL_WEIGHTS)

    # Apply regime adjustments
    if vol_regime and vol_regime in REGIME_ADJUSTMENTS:
        adjustments = REGIME_ADJUSTMENTS[vol_regime]
        for signal, adj in adjustments.items():
            weights[signal] = max(0.05, weights[signal] + adj)

    # Apply signal quality adjustments
    # If a signal source has low quality, redistribute its weight
    for signal, quality in signal_quality.items():
        if quality < 0.3:  # Low quality signal
            penalty = weights[signal] * (1 - quality) * 0.5
            weights[signal] -= penalty
            # Redistribute to other signals
            other_signals = [s for s in weights if s != signal]
            if other_signals:
                bonus = penalty / len(other_signals)
                for s in other_signals:
                    weights[s] += bonus

    # Renormalize to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


# ── Signal Quality Assessment ─────────────────────────────────────────

def _assess_signal_quality(
    filing: Dict, earnings: Dict, flows: Dict, ml_pred: Optional[Dict],
) -> Dict[str, float]:
    """
    Assess the quality/reliability of each signal source.
    Returns quality scores [0, 1] for each signal type.
    """
    quality = {}

    # Fundamental quality
    f_status = filing.get("status", "")
    if "Success" in f_status:
        quality["fundamental"] = 0.9 if "Quantitative" not in f_status else 0.6
    else:
        quality["fundamental"] = 0.1

    # Management/Earnings quality
    e_status = earnings.get("status", "")
    if e_status == "Success":
        source = earnings.get("source", "")
        quality["management"] = 0.9 if "Transcript" in source else 0.5  # News fallback is weaker
    else:
        quality["management"] = 0.1

    # Flow quality
    f_status_flow = flows.get("status", "")
    if f_status_flow == "Success":
        source = flows.get("source", "")
        quality["flows"] = 0.8 if source in ("finnhub", "fmp") else 0.4
    else:
        quality["flows"] = 0.1

    # Technical/ML quality
    if ml_pred and ml_pred.get("model_confidence"):
        mc = ml_pred["model_confidence"]
        quality["technical"] = mc.get("model_confidence", 0.5) if isinstance(mc, dict) else 0.5
    else:
        quality["technical"] = 0.2

    return quality


# ── Signal Agreement Analysis ─────────────────────────────────────────

def _analyze_signal_agreement(
    scores: Dict[str, float],
    signal_quality: Dict[str, float],
) -> Dict[str, Any]:
    """
    Analyze whether signals agree or disagree (convergence/divergence).
    Strong agreement across independent sources increases conviction.
    """
    available_scores = {k: v for k, v in scores.items() if signal_quality.get(k, 0) > 0.2}

    if len(available_scores) < 2:
        return {
            "agreement_score": 0.5,
            "agreement_label": "INSUFFICIENT DATA",
            "signal_spread": 0.0,
            "bullish_count": 0,
            "bearish_count": 0,
            "confirming_signals": [],
            "diverging_signals": [],
        }

    values = list(available_scores.values())
    mean_signal = np.mean(values)
    signal_spread = float(np.std(values))

    # Count directional alignment
    bullish = {k: v for k, v in available_scores.items() if v > 0.05}
    bearish = {k: v for k, v in available_scores.items() if v < -0.05}
    neutral = {k: v for k, v in available_scores.items() if -0.05 <= v <= 0.05}

    # Agreement score: 1.0 = perfect agreement, 0.0 = complete disagreement
    if len(values) > 1:
        max_spread = 2.0  # theoretical max spread for [-1, 1] range
        agreement = 1.0 - (signal_spread / max_spread)
    else:
        agreement = 0.5

    # Boost agreement if all signals point same direction
    if len(bullish) == len(available_scores) or len(bearish) == len(available_scores):
        agreement = min(1.0, agreement + 0.2)

    # Identify confirming and diverging signals
    confirming = []
    diverging = []
    direction = "bullish" if mean_signal > 0 else "bearish"
    for k, v in available_scores.items():
        if (direction == "bullish" and v > 0) or (direction == "bearish" and v < 0):
            confirming.append(k)
        elif abs(v) > 0.05:
            diverging.append(k)

    # Label
    if agreement >= 0.8:
        label = "STRONG CONVERGENCE"
    elif agreement >= 0.6:
        label = "MODERATE AGREEMENT"
    elif agreement >= 0.4:
        label = "MIXED SIGNALS"
    else:
        label = "SIGNAL DIVERGENCE"

    return {
        "agreement_score": round(float(agreement), 3),
        "agreement_label": label,
        "signal_spread": round(signal_spread, 3),
        "mean_signal": round(float(mean_signal), 3),
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "confirming_signals": confirming,
        "diverging_signals": diverging,
    }


# ── Monte Carlo Confidence Intervals ─────────────────────────────────

def _monte_carlo_confidence(
    base_scores: Dict[str, float],
    signal_quality: Dict[str, float],
    weights: Dict[str, float],
    n_simulations: int = 5000,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation to estimate confidence intervals
    around the conviction probability. Each simulation perturbs
    signal scores based on their quality (lower quality = more noise).
    """
    simulated_composites = []

    for _ in range(n_simulations):
        perturbed_composite = 0.0
        for signal, base_score in base_scores.items():
            quality = signal_quality.get(signal, 0.5)
            weight = weights.get(signal, 0.0)

            # Noise inversely proportional to quality
            noise_std = 0.3 * (1 - quality)
            perturbed_score = base_score + np.random.normal(0, noise_std)
            perturbed_score = np.clip(perturbed_score, -1.0, 1.0)

            perturbed_composite += perturbed_score * weight

        simulated_composites.append(perturbed_composite)

    simulated_composites = np.array(simulated_composites)

    # Convert to probabilities
    simulated_probs = 1 / (1 + np.exp(-3 * simulated_composites))

    # Confidence intervals
    ci_5 = float(np.percentile(simulated_probs, 5))
    ci_25 = float(np.percentile(simulated_probs, 25))
    ci_50 = float(np.percentile(simulated_probs, 50))
    ci_75 = float(np.percentile(simulated_probs, 75))
    ci_95 = float(np.percentile(simulated_probs, 95))

    return {
        "ci_5": round(ci_5, 3),
        "ci_25": round(ci_25, 3),
        "ci_50": round(ci_50, 3),
        "ci_75": round(ci_75, 3),
        "ci_95": round(ci_95, 3),
        "ci_width": round(ci_95 - ci_5, 3),
        "mean_probability": round(float(np.mean(simulated_probs)), 3),
        "probability_std": round(float(np.std(simulated_probs)), 3),
    }


# ── Per-Stock Conviction ──────────────────────────────────────────────

def _calculate_stock_conviction(
    ticker: str,
    filing: Dict,
    earnings: Dict,
    flows: Dict,
    ml_pred: Optional[Dict],
    vol_regime: Optional[str],
    horizon: str,
) -> Dict[str, Any]:
    """
    Synthesizes all signals for a single stock into a conviction probability.
    Uses dynamic weighting, signal agreement, and Monte Carlo intervals.
    """
    # 1. Assess signal quality
    signal_quality = _assess_signal_quality(filing, earnings, flows, ml_pred)

    # 2. Get dynamic weights
    weights = _get_dynamic_weights(vol_regime, signal_quality)

    # 3. Extract raw scores (Standardized to -1 to +1)

    # Filing Intel
    f_score = float(filing.get("fundamental_conviction", 0))
    g_mom = float(filing.get("guidance_momentum", 0))
    r_det = float(filing.get("risk_deterioration", 0))
    base_f = (f_score + g_mom - r_det) / 2.0

    # Earnings Intel
    m_conf = float(earnings.get("management_confidence", 0))
    a_pres = float(earnings.get("analyst_pressure", 0))
    base_e = m_conf - (a_pres * 0.5)

    # Flow Intel
    flow_composite = float(flows.get("flow_composite", 0))
    if flow_composite == 0:
        # Backward compat: compute from individual scores
        i_conv = float(flows.get("institutional_conviction", 0))
        i_align = float(flows.get("insider_alignment", 0))
        c_part = float(flows.get("capital_participation", 0))
        flow_composite = (i_conv * 0.5) + (i_align * 0.3) + (c_part * 0.2)

    # Technical/ML signal
    base_tech = 0.0
    if ml_pred:
        accum_prob = float(ml_pred.get("accumulation_prob", 0.5))
        base_tech = (accum_prob - 0.5) * 2  # Map [0, 1] to [-1, 1]

    # Bundle base scores
    base_scores = {
        "fundamental": float(np.clip(base_f, -1, 1)),
        "management": float(np.clip(base_e, -1, 1)),
        "flows": float(np.clip(flow_composite, -1, 1)),
        "technical": float(np.clip(base_tech, -1, 1)),
    }

    # 4. Signal agreement analysis
    agreement = _analyze_signal_agreement(base_scores, signal_quality)

    # 5. Weighted composite score
    composite_score = sum(
        base_scores[signal] * weights[signal]
        for signal in base_scores
    )

    # 6. Agreement multiplier (strong agreement boosts, divergence dampens)
    agreement_mult = 0.8 + (agreement["agreement_score"] * 0.4)  # [0.8, 1.2]
    composite_score *= agreement_mult

    # 7. Probability mapping (calibrated logistic)
    prob = 1 / (1 + np.exp(-3 * composite_score))

    # 8. Signal confidence adjustment
    avg_quality = np.mean(list(signal_quality.values()))
    prob_final = (prob * avg_quality) + (0.5 * (1 - avg_quality))

    # 9. Monte Carlo confidence intervals
    mc_intervals = _monte_carlo_confidence(base_scores, signal_quality, weights)

    # 10. Conviction level labeling
    level = "NEUTRAL"
    if prob_final >= 0.78:
        level = "STRONG CONVICTION"
    elif prob_final >= 0.68:
        level = "HIGH"
    elif prob_final >= 0.58:
        level = "MODERATE POSITIVE"
    elif prob_final >= 0.45:
        level = "NEUTRAL"
    elif prob_final >= 0.35:
        level = "MODERATE NEGATIVE"
    else:
        level = "LOW / RISK"

    # 11. Determine dominant driver
    driver_scores = {k: abs(v) * weights.get(k, 0) for k, v in base_scores.items()}
    dominant_driver = max(driver_scores, key=driver_scores.get) if driver_scores else "none"

    return {
        "ticker": ticker,
        "composite_score": float(np.round(composite_score, 3)),
        "outperformance_probability": float(np.round(prob_final, 3)),
        "conviction_level": level,
        "signal_confidence": round(float(avg_quality), 2),
        "signal_agreement": agreement,
        "confidence_intervals": mc_intervals,
        "dynamic_weights": {k: round(v, 3) for k, v in weights.items()},
        "signal_quality": {k: round(v, 3) for k, v in signal_quality.items()},
        "dominant_driver": dominant_driver,
        "available_sources": [
            s.replace("_", " ").title()
            for s, q in signal_quality.items() if q > 0.2
        ],
        "drivers": filing.get("supporting_drivers", []) + earnings.get("tone_drivers", []),
        "risks": filing.get("risk_drivers", []) + earnings.get("analyst_concerns", []),
        "signals_detail": {
            "fundamentals": {
                "score": f_score,
                "reasoning": filing.get("fundamental_reasoning", "N/A"),
                "quality": signal_quality.get("fundamental", 0),
            },
            "guidance": {
                "score": g_mom,
                "reasoning": filing.get("guidance_reasoning", "N/A"),
            },
            "risk_env": {
                "score": r_det,
                "reasoning": filing.get("risk_reasoning", "N/A"),
            },
            "management": {
                "score": m_conf,
                "reasoning": earnings.get("confidence_reasoning", "N/A"),
                "quality": signal_quality.get("management", 0),
            },
            "analyst": {
                "score": a_pres,
                "reasoning": earnings.get("pressure_reasoning", "N/A"),
            },
            "flows": {
                "score": flow_composite,
                "reasoning": f"Smart money: {flows.get('smart_money_pct', 0):.0%}, "
                             f"Insider: {flows.get('insider_alignment', 0):+.2f}, "
                             f"Accumulation: {flows.get('estimated_accumulation', 'N/A')}",
                "quality": signal_quality.get("flows", 0),
            },
            "technical": {
                "score": base_tech,
                "reasoning": f"ML accumulation probability: {ml_pred.get('accumulation_prob', 'N/A') if ml_pred else 'N/A'}",
                "quality": signal_quality.get("technical", 0),
            },
        },
    }


# ── Portfolio-Level Synthesis ─────────────────────────────────────────

def _compute_portfolio_conviction(
    synthesis_results: Dict[str, Dict],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Aggregate per-stock convictions into portfolio-level metrics.
    Weight-adjusted to reflect actual portfolio allocation.
    """
    if not synthesis_results:
        return {"overall_conviction": 0.5, "portfolio_signal": "NEUTRAL"}

    # Weight-adjusted conviction
    total_weight = sum(weights.get(t, 0) for t in synthesis_results)
    if total_weight == 0:
        total_weight = 1

    weighted_prob = sum(
        r["outperformance_probability"] * weights.get(t, 1.0 / len(synthesis_results))
        for t, r in synthesis_results.items()
    ) / total_weight

    # Conviction distribution
    levels = [r["conviction_level"] for r in synthesis_results.values()]
    strong = sum(1 for l in levels if l == "STRONG CONVICTION")
    high = sum(1 for l in levels if l == "HIGH")
    low_risk = sum(1 for l in levels if "LOW" in l or "RISK" in l)

    # Portfolio signal
    if weighted_prob >= 0.65:
        portfolio_signal = "BULLISH"
    elif weighted_prob >= 0.55:
        portfolio_signal = "MODERATELY BULLISH"
    elif weighted_prob >= 0.45:
        portfolio_signal = "NEUTRAL"
    elif weighted_prob >= 0.35:
        portfolio_signal = "MODERATELY BEARISH"
    else:
        portfolio_signal = "BEARISH"

    # Dispersion (how spread out are the convictions)
    probs = [r["outperformance_probability"] for r in synthesis_results.values()]
    dispersion = float(np.std(probs))

    return {
        "overall_conviction": float(np.round(weighted_prob, 3)),
        "portfolio_signal": portfolio_signal,
        "conviction_dispersion": round(dispersion, 3),
        "strong_conviction_count": strong,
        "high_conviction_count": high,
        "low_risk_count": low_risk,
        "conviction_distribution": {
            "STRONG CONVICTION": strong,
            "HIGH": high,
            "MODERATE POSITIVE": sum(1 for l in levels if l == "MODERATE POSITIVE"),
            "NEUTRAL": sum(1 for l in levels if l == "NEUTRAL"),
            "MODERATE NEGATIVE": sum(1 for l in levels if l == "MODERATE NEGATIVE"),
            "LOW / RISK": low_risk,
        },
    }


def run_conviction_synthesis(state: PortfolioState) -> PortfolioState:
    """
    The main node for Conviction Synthesis (v2).
    Enhanced with dynamic weighting, regime awareness, and Monte Carlo intervals.
    """
    tickers = state.weights.keys() if state.weights else []
    if not tickers:
        return state

    # Ensure dependencies exist
    filing_intel = state.filing_intel or {}
    earnings_intel = state.earnings_intel or {}
    flow_intel = state.flow_signals.get("intel", {}) if state.flow_signals else {}

    # Get ML predictions per ticker (if available)
    ml_data = {}
    if state.ml_predictions and "prediction_table" in state.ml_predictions:
        pred_table = state.ml_predictions["prediction_table"]
        if isinstance(pred_table, pd.DataFrame):
            for _, row in pred_table.iterrows():
                ml_data[row["Ticker"]] = {
                    "accumulation_prob": row.get("Accumulation Prob", 0.5),
                    "signal_strength": row.get("Signal Strength", "WEAK"),
                }
        # Also get model confidence
        model_conf = state.ml_predictions.get("model_confidence", {})
    else:
        model_conf = {}

    # Get current volatility regime
    vol_regime = None
    if state.vol_regime and "current_regime" in state.vol_regime:
        vol_regime = state.vol_regime["current_regime"]

    synthesis_results = {}
    for ticker in tickers:
        f = filing_intel.get(ticker, {"status": "Data source not available"})
        e = earnings_intel.get(ticker, {"status": "Data source not available"})
        fl = flow_intel.get(ticker, {"status": "Data source not available"})
        ml = ml_data.get(ticker)

        synthesis_results[ticker] = _calculate_stock_conviction(
            ticker, f, e, fl, ml, vol_regime, state.selected_horizon
        )

    # Portfolio-level aggregation
    portfolio_conviction = _compute_portfolio_conviction(synthesis_results, state.weights)

    state.conviction_synthesis = {
        "results": synthesis_results,
        "horizon": state.selected_horizon,
        "vol_regime_used": vol_regime,
        "portfolio_conviction": portfolio_conviction,
        # Backward compat
        "overall_portfolio_conviction": portfolio_conviction["overall_conviction"],
    }

    return state
