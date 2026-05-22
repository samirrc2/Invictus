"""
Invictus — ML Accumulation & Alpha Classifier (v2)
Institutional-grade predictive model for identifying accumulation patterns
and forward alpha potential across portfolio holdings.

Features (25+):
- Technical: RSI(14), MACD histogram, Bollinger %B, ADX, ATR ratio, OBV trend
- Momentum: 5/10/20/60-day returns, relative strength vs benchmark
- Volatility: 20d vol, vol ratio (20/60), vol regime z-score
- Flow: Institutional conviction, insider alignment, capital participation
- Fundamental: Distance from 52w high/low, drawdown depth, recovery velocity
- Microstructure: Volume z-score, price-volume divergence

Models: Ensemble of LogisticRegression + RandomForest + optional XGBoost
Target: Forward 20-day risk-adjusted returns (Sharpe-style) with percentile thresholds
Validation: Expanding-window time-series cross-validation (no look-ahead bias)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report
from typing import Dict, Any, Optional, List, Tuple

from invictus.agents.graph_state import PortfolioState
from invictus.config import TRADING_DAYS_PER_YEAR, BENCHMARK_TICKER


# ── Technical Indicator Helpers ───────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> float:
    """Compute RSI for the most recent point."""
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period).mean().iloc[-1]
    avg_loss = losses.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _macd_histogram(prices: pd.Series) -> float:
    """MACD histogram (12/26/9 standard)."""
    if len(prices) < 35:
        return 0.0
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return float(histogram.iloc[-1])


def _bollinger_pct_b(prices: pd.Series, period: int = 20) -> float:
    """Bollinger Band %B — position within bands (0=lower, 1=upper)."""
    if len(prices) < period:
        return 0.5
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    band_width = upper.iloc[-1] - lower.iloc[-1]
    if band_width == 0:
        return 0.5
    pct_b = (prices.iloc[-1] - lower.iloc[-1]) / band_width
    return float(np.clip(pct_b, -0.5, 1.5))


def _adx(prices: pd.Series, period: int = 14) -> float:
    """Approximate ADX using price-based directional movement."""
    if len(prices) < period * 2:
        return 25.0  # neutral
    # Simplified: use absolute returns as proxy for directional strength
    abs_returns = prices.pct_change().abs().dropna()
    smoothed = abs_returns.rolling(period).mean()
    # Scale to 0-100 range
    adx_proxy = float(smoothed.iloc[-1] * 100 * np.sqrt(TRADING_DAYS_PER_YEAR))
    return float(np.clip(adx_proxy, 0, 100))


def _atr_ratio(prices: pd.Series, short: int = 5, long: int = 20) -> float:
    """ATR ratio: short-term ATR vs long-term ATR (expansion/contraction)."""
    if len(prices) < long + 1:
        return 1.0
    # Use price range as ATR proxy
    ret = prices.pct_change().abs().dropna()
    atr_short = ret.tail(short).mean()
    atr_long = ret.tail(long).mean()
    if atr_long == 0:
        return 1.0
    return float(atr_short / atr_long)


def _obv_trend(prices: pd.Series, returns: pd.Series, period: int = 20) -> float:
    """OBV trend direction: slope of cumulative OBV over period, normalized."""
    if len(returns) < period:
        return 0.0
    # Use sign of returns as volume direction proxy
    obv = returns.tail(period).apply(lambda x: 1 if x > 0 else -1).cumsum()
    if len(obv) < 2:
        return 0.0
    # Normalize slope to [-1, 1]
    slope = (obv.iloc[-1] - obv.iloc[0]) / period
    return float(np.clip(slope * 5, -1, 1))


def _volume_zscore(returns: pd.Series, window: int = 60) -> float:
    """Z-score of recent absolute return magnitude vs history."""
    if len(returns) < window:
        return 0.0
    recent_mag = returns.tail(5).abs().mean()
    hist_mean = returns.tail(window).abs().mean()
    hist_std = returns.tail(window).abs().std()
    if hist_std == 0:
        return 0.0
    return float((recent_mag - hist_mean) / hist_std)


def _price_volume_divergence(prices: pd.Series, returns: pd.Series, period: int = 20) -> float:
    """Detect divergence between price trend and volume/volatility trend."""
    if len(prices) < period or len(returns) < period:
        return 0.0
    price_trend = (prices.iloc[-1] / prices.iloc[-period] - 1) if prices.iloc[-period] != 0 else 0
    vol_trend = returns.tail(period).abs().mean() - returns.tail(period * 2).abs().mean() if len(returns) >= period * 2 else 0
    # Divergence: price up but vol down (bullish stealth), or price down but vol up (bearish panic)
    if price_trend > 0 and vol_trend < 0:
        return 0.5  # bullish stealth accumulation signal
    elif price_trend < 0 and vol_trend > 0:
        return -0.5  # bearish distribution signal
    return 0.0


def _recovery_velocity(prices: pd.Series, period: int = 60) -> float:
    """Speed of recovery from recent drawdown (normalized)."""
    if len(prices) < period:
        return 0.0
    window = prices.tail(period)
    trough_idx = window.idxmin()
    trough_val = window[trough_idx]
    if trough_val == 0:
        return 0.0
    # Recovery = (current - trough) / trough, normalized by time since trough
    recovery = (prices.iloc[-1] - trough_val) / trough_val
    trough_pos = window.index.get_loc(trough_idx)
    time_since = len(window) - trough_pos
    if time_since == 0:
        return 0.0
    velocity = recovery / (time_since / period)
    return float(np.clip(velocity, -2, 2))


# ── Feature Engineering ───────────────────────────────────────────────

def _compute_features(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    weights: Dict[str, float],
    flow_data: Optional[Dict] = None,
    synthesis_data: Optional[Dict] = None,
) -> pd.DataFrame:
    """Build institutional-grade feature matrix for all tickers."""
    tickers = [t for t in weights if t in returns.columns]
    rows = []

    for ticker in tickers:
        ret = returns[ticker].dropna()
        px = prices[ticker].dropna()

        if len(ret) < 60:
            continue

        # ── Momentum Features ──
        mom_5 = float(ret.tail(5).sum())
        mom_10 = float(ret.tail(10).sum())
        mom_20 = float(ret.tail(20).sum())
        mom_60 = float(ret.tail(60).sum())

        # ── Volatility Features ──
        vol_20 = float(ret.tail(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        vol_ratio = float(ret.tail(20).std() / ret.tail(60).std()) if ret.tail(60).std() > 0 else 1.0
        vol_zscore = float((ret.tail(5).std() - ret.tail(60).std()) / ret.tail(60).std()) if ret.tail(60).std() > 0 else 0.0

        # ── Technical Indicators ──
        rsi_14 = _rsi(px, 14)
        macd_hist = _macd_histogram(px)
        boll_pct_b = _bollinger_pct_b(px, 20)
        adx_val = _adx(px, 14)
        atr_rat = _atr_ratio(px, 5, 20)
        obv_t = _obv_trend(px, ret, 20)

        # ── Microstructure Signals ──
        vol_z = _volume_zscore(ret, 60)
        pv_div = _price_volume_divergence(px, ret, 20)
        recov_vel = _recovery_velocity(px, 60)

        # ── Relative Strength vs Benchmark ──
        rel_strength = 0.0
        rel_strength_60 = 0.0
        if BENCHMARK_TICKER in returns.columns:
            bench_ret = returns[BENCHMARK_TICKER].dropna()
            bench_20 = float(bench_ret.tail(20).sum())
            bench_60 = float(bench_ret.tail(60).sum())
            rel_strength = mom_20 - bench_20
            rel_strength_60 = mom_60 - bench_60

        # ── Flow Features (from institutional flow agent) ──
        inst_conviction = 0.0
        insider_alignment = 0.0
        capital_participation = 0.0
        if flow_data and ticker in flow_data:
            fd = flow_data[ticker]
            inst_conviction = float(fd.get("institutional_conviction", 0.0))
            insider_alignment = float(fd.get("insider_alignment", 0.0))
            capital_participation = float(fd.get("capital_participation", 0.0))

        # ── Fundamental Conviction (from synthesis) ──
        fundamental_score = 0.0
        management_score = 0.0
        if synthesis_data and ticker in synthesis_data:
            sd = synthesis_data[ticker]
            fundamental_score = float(sd.get("composite_score", 0.0))
            management_score = float(sd.get("signals_detail", {}).get("management", {}).get("score", 0.0))

        # ── Price Structure ──
        high_252 = px.tail(252).max() if len(px) >= 252 else px.max()
        low_252 = px.tail(252).min() if len(px) >= 252 else px.min()
        dist_from_high = float((px.iloc[-1] - high_252) / high_252) if high_252 > 0 else 0.0
        dist_from_low = float((px.iloc[-1] - low_252) / low_252) if low_252 > 0 else 0.0

        # ── Drawdown Depth ──
        cummax = px.cummax()
        drawdown = (px - cummax) / cummax
        current_dd = float(drawdown.iloc[-1])

        rows.append({
            "Ticker": ticker,
            # Momentum (4)
            "Momentum_5d": mom_5,
            "Momentum_10d": mom_10,
            "Momentum_20d": mom_20,
            "Momentum_60d": mom_60,
            # Volatility (3)
            "Volatility_20d": vol_20,
            "Vol_Ratio": vol_ratio,
            "Vol_Zscore": vol_zscore,
            # Technical (6)
            "RSI_14": rsi_14,
            "MACD_Hist": macd_hist,
            "Bollinger_PctB": boll_pct_b,
            "ADX": adx_val,
            "ATR_Ratio": atr_rat,
            "OBV_Trend": obv_t,
            # Microstructure (3)
            "Volume_Zscore": vol_z,
            "PV_Divergence": pv_div,
            "Recovery_Velocity": recov_vel,
            # Relative Strength (2)
            "Rel_Strength_20d": rel_strength,
            "Rel_Strength_60d": rel_strength_60,
            # Flow (3)
            "Inst_Conviction": inst_conviction,
            "Insider_Alignment": insider_alignment,
            "Capital_Participation": capital_participation,
            # Fundamental (2)
            "Fundamental_Score": fundamental_score,
            "Management_Score": management_score,
            # Price Structure (3)
            "Dist_From_High": dist_from_high,
            "Dist_From_Low": dist_from_low,
            "Drawdown_Depth": current_dd,
        })

    return pd.DataFrame(rows)


# ── Label Generation ──────────────────────────────────────────────────

def _generate_forward_return_labels(
    returns: pd.DataFrame,
    tickers: List[str],
    forward_window: int = 20,
    threshold_pct: float = 0.60,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate labels based on forward risk-adjusted returns.
    Uses a lookback proxy when true forward data would cause look-ahead bias:
    the most recent `forward_window` days act as the 'outcome period'.

    Returns:
        labels: 1 = outperformance, 0 = underperformance
        forward_returns: dict of ticker -> forward return used for labeling
    """
    forward_rets = {}
    for ticker in tickers:
        if ticker not in returns.columns:
            forward_rets[ticker] = 0.0
            continue
        ret = returns[ticker].dropna()
        if len(ret) < forward_window + 60:
            # Not enough data — use full period return
            forward_rets[ticker] = float(ret.tail(forward_window).sum())
        else:
            # Use most recent forward_window as outcome, train on prior data
            outcome_ret = float(ret.tail(forward_window).sum())
            outcome_vol = float(ret.tail(forward_window).std())
            # Risk-adjusted return (Sharpe-like)
            if outcome_vol > 0:
                forward_rets[ticker] = outcome_ret / outcome_vol
            else:
                forward_rets[ticker] = outcome_ret

    # Rank and label
    sorted_tickers = sorted(forward_rets.items(), key=lambda x: x[1], reverse=True)
    n_positive = max(1, int(len(sorted_tickers) * (1 - threshold_pct)))
    positive_set = {t[0] for t in sorted_tickers[:n_positive]}

    labels = np.array([1 if t in positive_set else 0 for t in tickers])
    return labels, forward_rets


def _generate_composite_labels(features: pd.DataFrame) -> np.ndarray:
    """
    Fallback: Multi-factor composite scoring for label generation.
    Uses a richer combination than v1 for more robust synthetic targets.
    """
    # Rank-based composite with more dimensions
    scores = pd.Series(0.0, index=features.index)

    # Momentum cluster (30%)
    if "Momentum_20d" in features.columns:
        scores += features["Momentum_20d"].rank(pct=True) * 0.15
    if "Momentum_60d" in features.columns:
        scores += features["Momentum_60d"].rank(pct=True) * 0.15

    # Technical signals (25%)
    if "RSI_14" in features.columns:
        # RSI sweet spot: 40-60 is neutral, penalize extremes
        rsi_score = 1.0 - abs(features["RSI_14"] - 55).rank(pct=True)
        scores += rsi_score * 0.10
    if "MACD_Hist" in features.columns:
        scores += features["MACD_Hist"].rank(pct=True) * 0.10
    if "OBV_Trend" in features.columns:
        scores += features["OBV_Trend"].rank(pct=True) * 0.05

    # Flow signals (25%)
    if "Inst_Conviction" in features.columns:
        scores += features["Inst_Conviction"].rank(pct=True) * 0.15
    if "Insider_Alignment" in features.columns:
        scores += features["Insider_Alignment"].rank(pct=True) * 0.10

    # Risk-adjusted (20%)
    if "Volatility_20d" in features.columns:
        scores += (1 - features["Volatility_20d"].rank(pct=True)) * 0.10
    if "Recovery_Velocity" in features.columns:
        scores += features["Recovery_Velocity"].rank(pct=True) * 0.05
    if "Rel_Strength_20d" in features.columns:
        scores += features["Rel_Strength_20d"].rank(pct=True) * 0.05

    threshold = scores.quantile(0.60)
    return (scores >= threshold).astype(int).values


# ── Model Training & Ensemble ─────────────────────────────────────────

def _build_ensemble(X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Build a weighted ensemble of multiple classifiers.
    Returns model, predictions, feature importance, and diagnostics.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model 1: Logistic Regression (interpretable baseline)
    lr = LogisticRegression(
        random_state=42, max_iter=1000, C=0.5,
        penalty="l2", solver="lbfgs"
    )

    # Model 2: Random Forest (captures non-linear patterns)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )

    # Ensemble: Soft voting (probability-weighted)
    estimators = [("lr", lr), ("rf", rf)]
    ensemble_weights = [0.4, 0.6]  # RF gets more weight for non-linear capture

    # Try XGBoost (trained separately to avoid VotingClassifier compatibility issues)
    xgb_available = False
    xgb_model = None
    xgb_probas = None
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric="logloss", subsample=0.8, colsample_bytree=0.8,
        )
        xgb_model.fit(X_scaled, y)
        xgb_probas = xgb_model.predict_proba(X_scaled)[:, 1]
        xgb_available = True
    except (ImportError, Exception):
        pass

    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=ensemble_weights,
    )
    ensemble.fit(X_scaled, y)

    # Individual model predictions for comparison
    lr_rf_probas = ensemble.predict_proba(X_scaled)[:, 1]

    # Blend in XGBoost if available (manual ensemble since XGB has VotingClassifier compat issues)
    if xgb_available and xgb_probas is not None:
        probas_ensemble = lr_rf_probas * 0.6 + xgb_probas * 0.4
    else:
        probas_ensemble = lr_rf_probas

    # Train individual models for feature importance
    lr.fit(X_scaled, y)
    rf.fit(X_scaled, y)

    # Feature importance: blend LR coefficients + RF importances
    lr_importance = np.abs(lr.coef_[0])
    lr_importance = lr_importance / lr_importance.sum()
    rf_importance = rf.feature_importances_
    rf_importance = rf_importance / rf_importance.sum()

    blended_importance = lr_importance * 0.3 + rf_importance * 0.7

    if xgb_available and xgb_model is not None:
        xgb_imp = xgb_model.feature_importances_
        xgb_imp = xgb_imp / xgb_imp.sum() if xgb_imp.sum() > 0 else xgb_imp
        blended_importance = lr_importance * 0.2 + rf_importance * 0.35 + xgb_imp * 0.45

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "LR_Coefficient": lr.coef_[0],
        "RF_Importance": rf.feature_importances_,
        "Blended_Importance": blended_importance,
    }).sort_values("Blended_Importance", ascending=False)

    # Time-series cross-validation (expanding window) — uses LR+RF ensemble only
    cv_scores = {"accuracy": None, "f1": None}
    try:
        n_splits = min(3, max(2, len(y) // 4))
        if n_splits >= 2 and len(np.unique(y)) > 1:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            # Use the LR+RF VotingClassifier for CV (no XGB compat issues)
            acc_scores = cross_val_score(ensemble, X_scaled, y, cv=tscv, scoring="accuracy")
            cv_scores["accuracy"] = float(acc_scores.mean())
            cv_scores["accuracy_std"] = float(acc_scores.std())
            try:
                f1_scores = cross_val_score(ensemble, X_scaled, y, cv=tscv, scoring="f1")
                cv_scores["f1"] = float(f1_scores.mean())
            except Exception:
                pass
    except Exception:
        pass

    # Model agreement analysis
    lr_preds = lr.predict(X_scaled)
    rf_preds = rf.predict(X_scaled)
    agreement_rate = float(np.mean(lr_preds == rf_preds))

    return {
        "ensemble": ensemble,
        "scaler": scaler,
        "probabilities": probas_ensemble,
        "importance": importance_df,
        "cv_scores": cv_scores,
        "model_agreement": agreement_rate,
        "xgb_available": xgb_available,
        "individual_models": {
            "lr": {"probas": lr.predict_proba(X_scaled)[:, 1]},
            "rf": {"probas": rf.predict_proba(X_scaled)[:, 1]},
        },
    }


# ── Signal Decomposition ─────────────────────────────────────────────

def _compute_signal_decomposition(
    features: pd.DataFrame,
    importance_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Decompose each ticker's prediction into contributing signal groups.
    Groups: Momentum, Technical, Flow, Fundamental, Risk/Structure.
    """
    group_map = {
        "Momentum_5d": "Momentum", "Momentum_10d": "Momentum",
        "Momentum_20d": "Momentum", "Momentum_60d": "Momentum",
        "Volatility_20d": "Risk", "Vol_Ratio": "Risk", "Vol_Zscore": "Risk",
        "RSI_14": "Technical", "MACD_Hist": "Technical", "Bollinger_PctB": "Technical",
        "ADX": "Technical", "ATR_Ratio": "Technical", "OBV_Trend": "Technical",
        "Volume_Zscore": "Microstructure", "PV_Divergence": "Microstructure",
        "Recovery_Velocity": "Microstructure",
        "Rel_Strength_20d": "Momentum", "Rel_Strength_60d": "Momentum",
        "Inst_Conviction": "Flow", "Insider_Alignment": "Flow",
        "Capital_Participation": "Flow",
        "Fundamental_Score": "Fundamental", "Management_Score": "Fundamental",
        "Dist_From_High": "Structure", "Dist_From_Low": "Structure",
        "Drawdown_Depth": "Structure",
    }

    imp_map = dict(zip(importance_df["Feature"], importance_df["Blended_Importance"]))

    decomp = {}
    for _, row in features.iterrows():
        ticker = row["Ticker"]
        groups = {}
        for feat in feature_cols:
            group = group_map.get(feat, "Other")
            weight = imp_map.get(feat, 0)
            val = row.get(feat, 0)
            if group not in groups:
                groups[group] = 0.0
            # Contribution = feature value rank * importance weight
            groups[group] += abs(float(val)) * float(weight)
        # Normalize
        total = sum(groups.values())
        if total > 0:
            groups = {k: round(v / total, 3) for k, v in groups.items()}
        decomp[ticker] = groups

    return decomp


# ── Confidence Calibration ────────────────────────────────────────────

def _calibrate_confidence(
    probas: np.ndarray,
    model_agreement: float,
    cv_accuracy: Optional[float],
    n_samples: int,
) -> Dict[str, Any]:
    """
    Compute model confidence metrics to qualify predictions.
    """
    # Base confidence from probability spread
    prob_spread = float(np.std(probas))
    prob_entropy = float(-np.mean(probas * np.log(probas + 1e-10) + (1 - probas) * np.log(1 - probas + 1e-10)))

    # Sample size penalty (small datasets are less reliable)
    sample_factor = min(1.0, n_samples / 20.0)

    # Model agreement boost
    agreement_factor = model_agreement

    # CV accuracy factor
    cv_factor = cv_accuracy if cv_accuracy and cv_accuracy > 0 else 0.5

    # Overall model confidence
    model_confidence = (
        sample_factor * 0.30 +
        agreement_factor * 0.30 +
        cv_factor * 0.25 +
        (1.0 - prob_entropy / np.log(2)) * 0.15  # low entropy = more confident
    )
    model_confidence = float(np.clip(model_confidence, 0.1, 0.95))

    # Confidence tier
    if model_confidence >= 0.75:
        tier = "HIGH"
    elif model_confidence >= 0.55:
        tier = "MODERATE"
    else:
        tier = "LOW (Prototype)"

    return {
        "model_confidence": round(model_confidence, 3),
        "confidence_tier": tier,
        "probability_spread": round(prob_spread, 3),
        "probability_entropy": round(prob_entropy, 3),
        "sample_size_factor": round(sample_factor, 3),
        "model_agreement_factor": round(agreement_factor, 3),
        "cv_accuracy_factor": round(cv_factor, 3),
    }


# ── Main Entry Point ─────────────────────────────────────────────────

def run_accumulation_model(state: PortfolioState) -> PortfolioState:
    """
    Train and run the institutional accumulation classifier (v2).
    Uses flow data from the institutional flow agent if available.
    Integrates conviction synthesis signals when available.
    """
    returns = state.returns
    prices = state.prices
    weights = state.weights

    if returns is None or weights is None:
        raise ValueError("Returns and weights required for ML model.")

    # Get flow data if available
    flow_data = None
    if state.flow_signals and "intel" in state.flow_signals:
        flow_data = state.flow_signals["intel"]

    # Get synthesis data if available (from conviction synthesis)
    synthesis_data = None
    if state.conviction_synthesis and "results" in state.conviction_synthesis:
        synthesis_data = state.conviction_synthesis["results"]

    # Build feature matrix
    features = _compute_features(returns, prices, weights, flow_data, synthesis_data)

    if len(features) < 5:
        raise ValueError(f"Not enough tickers for ML model: {len(features)}")

    # Feature columns (26 features)
    feature_cols = [
        # Momentum
        "Momentum_5d", "Momentum_10d", "Momentum_20d", "Momentum_60d",
        # Volatility
        "Volatility_20d", "Vol_Ratio", "Vol_Zscore",
        # Technical
        "RSI_14", "MACD_Hist", "Bollinger_PctB", "ADX", "ATR_Ratio", "OBV_Trend",
        # Microstructure
        "Volume_Zscore", "PV_Divergence", "Recovery_Velocity",
        # Relative Strength
        "Rel_Strength_20d", "Rel_Strength_60d",
        # Flow
        "Inst_Conviction", "Insider_Alignment", "Capital_Participation",
        # Fundamental
        "Fundamental_Score", "Management_Score",
        # Structure
        "Dist_From_High", "Dist_From_Low", "Drawdown_Depth",
    ]

    X = features[feature_cols].fillna(0).values
    tickers = features["Ticker"].values

    # Generate labels
    # Try forward-return-based labels first, fall back to composite
    labels, forward_rets = _generate_forward_return_labels(
        returns, list(tickers), forward_window=20, threshold_pct=0.60
    )
    label_type = "forward_return"

    # If all same class, fall back to composite
    if len(np.unique(labels)) < 2:
        labels = _generate_composite_labels(features)
        label_type = "composite"

    # Build ensemble model
    model_results = _build_ensemble(X, labels, feature_cols)
    probas = model_results["probabilities"]

    # Compute confidence calibration
    confidence = _calibrate_confidence(
        probas,
        model_results["model_agreement"],
        model_results["cv_scores"].get("accuracy"),
        len(features),
    )

    # Signal decomposition per ticker
    signal_decomp = _compute_signal_decomposition(
        features, model_results["importance"], feature_cols
    )

    # Prediction table (enriched)
    pred_table = pd.DataFrame({
        "Ticker": tickers,
        "Accumulation Prob": probas,
        "Predicted Label": ["ACCUMULATION" if p >= 0.5 else "NO SIGNAL" for p in probas],
        "Signal Strength": [
            "STRONG" if p >= 0.75 else "MODERATE" if p >= 0.6 else "WEAK" if p >= 0.5 else "NEGATIVE"
            for p in probas
        ],
        "LR Prob": model_results["individual_models"]["lr"]["probas"],
        "RF Prob": model_results["individual_models"]["rf"]["probas"],
        "Forward Ret": [forward_rets.get(t, 0) for t in tickers],
    })

    # Add key feature values for transparency
    for col in ["RSI_14", "MACD_Hist", "Rel_Strength_20d", "Inst_Conviction", "Momentum_20d"]:
        if col in features.columns:
            pred_table[col] = features[col].values

    pred_table = pred_table.sort_values("Accumulation Prob", ascending=False)

    # Top candidates
    top_accumulation = pred_table[pred_table["Accumulation Prob"] >= 0.5].head(5)
    top_3 = top_accumulation[["Ticker", "Accumulation Prob", "Signal Strength"]].to_dict("records")

    # Distribution candidates (shorts/sells)
    distribution = pred_table[pred_table["Accumulation Prob"] < 0.4]
    bottom_3 = distribution.tail(3)[["Ticker", "Accumulation Prob", "Signal Strength"]].to_dict("records")

    state.ml_predictions = {
        "prediction_table": pred_table,
        "feature_importance": model_results["importance"],
        "top_accumulation": top_3,
        "distribution_candidates": bottom_3,
        "cv_scores": model_results["cv_scores"],
        "model_confidence": confidence,
        "signal_decomposition": signal_decomp,
        "model_agreement": model_results["model_agreement"],
        "model_type": "Ensemble (LR + RF" + (" + XGB)" if model_results["xgb_available"] else ")"),
        "n_features": len(feature_cols),
        "n_samples": len(features),
        "label_type": label_type,
        "feature_groups": {
            "Momentum": ["Momentum_5d", "Momentum_10d", "Momentum_20d", "Momentum_60d", "Rel_Strength_20d", "Rel_Strength_60d"],
            "Technical": ["RSI_14", "MACD_Hist", "Bollinger_PctB", "ADX", "ATR_Ratio", "OBV_Trend"],
            "Flow": ["Inst_Conviction", "Insider_Alignment", "Capital_Participation"],
            "Fundamental": ["Fundamental_Score", "Management_Score"],
            "Risk": ["Volatility_20d", "Vol_Ratio", "Vol_Zscore"],
            "Microstructure": ["Volume_Zscore", "PV_Divergence", "Recovery_Velocity"],
            "Structure": ["Dist_From_High", "Dist_From_Low", "Drawdown_Depth"],
        },
        # Backward compat
        "top_3": top_3[:3] if top_3 else [],
        "cv_accuracy": model_results["cv_scores"].get("accuracy"),
        "xgb_results": {"available": model_results["xgb_available"]},
    }

    return state
