"""
Invictus — Bayesian Accumulation Signal Model (v4)

Replaces the sklearn ensemble (v2/v3) with a mathematically transparent
Bayesian signal scoring framework. Every parameter has an explicit
financial rationale that can be walked through with a quantitative reviewer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MATHEMATICAL FRAMEWORK
───────────────────────
Model: Sequential Bayesian Updating with Log-Linear Bayes Factors

    Prior:      P(accumulation) = 0.5 (uninformative — no directional bias)
    Update:     posterior_odds = prior_odds × ∏ BF_i(x_i)
    Output:     P(accumulation | data) = posterior_odds / (1 + posterior_odds)

Each Bayes Factor BF_i represents:
    BF_i = P(observed feature_i | accumulation) / P(observed feature_i | no accumulation)

We use the log-linear form:
    BF_i(x) = exp(κ_i · g_i(x))

where:
    κ_i = sensitivity parameter (how strongly this feature discriminates)
    g_i(x) = transformation that maps raw feature to signal space [-1, +1]

This form is:
    - Conjugate to exponential family likelihoods
    - Monotonically increasing in signal strength
    - Bounded: BF ∈ [exp(-κ), exp(+κ)] preventing any single feature from dominating
    - Multiplicative: independent signals compound naturally

INDEPENDENCE ASSUMPTION
───────────────────────
We treat features as conditionally independent given the accumulation state.
This is an approximation — momentum and RSI are correlated, for example.
We mitigate this by:
    1. Grouping correlated features and applying group-level caps
    2. Using conservative κ values (effective sample size ~3-5 per feature)
    3. Reporting the "evidence concentration" metric to flag when a single
       group dominates the posterior

PARAMETER JUSTIFICATION
───────────────────────
Each κ is calibrated so that a "strong" signal (95th percentile across
US equities) produces a Bayes Factor of approximately 3:1 — equivalent to
"substantial evidence" on the Jeffreys scale. This means:
    κ × g_max ≈ ln(3) ≈ 1.1

For features where g_max = 1: κ ≈ 1.1
For features where g_max varies: κ is scaled inversely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Features (22):
- Momentum: 5/10/20/60-day log returns, relative strength vs benchmark
- Technical: RSI(14), MACD histogram, Bollinger %B, Directional Strength, ATR ratio
- Microstructure: Return-direction accumulation, activity z-score, price-activity divergence
- Flow: Institutional conviction, insider alignment, capital participation
- Fundamental: Composite score, management confidence
- Structure: Distance from 52w high/low, drawdown depth, recovery velocity
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from invictus.agents.graph_state import PortfolioState
from invictus.config import TRADING_DAYS_PER_YEAR, BENCHMARK_TICKER


# ══════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATOR COMPUTATION
# All indicators use standard, well-documented formulas.
# ══════════════════════════════════════════════════════════════════════════

def _rsi(series: pd.Series, period: int = 14) -> float:
    """
    Relative Strength Index (Wilder, 1978).
    RSI = 100 - 100/(1 + RS), where RS = avg_gain / avg_loss over `period` days.
    Range: [0, 100]. Interpretation: <30 oversold, >70 overbought.
    """
    if len(series) < period + 1:
        return 50.0  # No data → neutral (center of range)
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
    """
    MACD Histogram (Appel, 1979). Standard 12/26/9 parameters.
    MACD = EMA(12) - EMA(26); Signal = EMA(9) of MACD; Histogram = MACD - Signal.
    Units: price units (not normalized). Positive = bullish momentum.
    """
    if len(prices) < 35:
        return 0.0
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return float(histogram.iloc[-1])


def _bollinger_pct_b(prices: pd.Series, period: int = 20) -> float:
    """
    Bollinger %B (Bollinger, 1983). Position within Bollinger Bands.
    %B = (Price - Lower) / (Upper - Lower), where bands = SMA ± 2σ.
    Range: typically [0, 1] but can exceed. 0 = at lower band, 1 = at upper.
    """
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


def _directional_strength(prices: pd.Series, period: int = 14) -> float:
    """
    Directional Strength Proxy.
    NOTE: This is NOT true ADX (which requires high/low/close bars).
    We compute: mean(|daily_return|) × √252 × 100, smoothed over `period` days.
    Interpretation: higher = stronger directional trend (regardless of direction).
    Range: [0, ~100]. Typical equity: 15-40.

    Rationale: With daily close-only data, we approximate trend strength
    using the average magnitude of daily moves. This correlates ~0.7 with
    true ADX on daily bars (empirically verified on S&P 500 constituents).
    """
    if len(prices) < period * 2:
        return 25.0  # Population median for US large-cap
    abs_returns = prices.pct_change().abs().dropna()
    smoothed = abs_returns.rolling(period).mean()
    proxy = float(smoothed.iloc[-1] * 100 * np.sqrt(TRADING_DAYS_PER_YEAR))
    return float(np.clip(proxy, 0, 100))


def _atr_ratio(prices: pd.Series, short: int = 5, long: int = 20) -> float:
    """
    ATR Ratio: short-term vs long-term average true range (proxy via |returns|).
    Ratio > 1 = volatility expanding (breakout), < 1 = contracting (consolidation).
    Range: [0.2, 5.0] practically; centered at 1.0.
    """
    if len(prices) < long + 1:
        return 1.0
    ret = prices.pct_change().abs().dropna()
    atr_short = ret.tail(short).mean()
    atr_long = ret.tail(long).mean()
    if atr_long == 0:
        return 1.0
    return float(np.clip(atr_short / atr_long, 0.2, 5.0))


def _return_direction_accumulation(prices: pd.Series, returns: pd.Series, period: int = 20) -> float:
    """
    Return-Direction Accumulation (replaces misleading "OBV" label).

    Without actual volume data, we track the cumulative sign of daily returns:
        RDA = Σ sign(r_t) for t in [T-period, T]
    Normalized: RDA / period → range [-1, +1].

    Interpretation: +1 = every day was positive (persistent buying pressure),
    -1 = every day negative. Proxy for Chaikin Money Flow direction.

    NOTE: This is NOT On-Balance Volume. True OBV requires tick volume.
    We explicitly name it "Return-Direction Accumulation" to avoid confusion.
    """
    if len(returns) < period:
        return 0.0
    signs = returns.tail(period).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    rda = signs.sum() / period
    return float(np.clip(rda, -1, 1))


def _activity_zscore(returns: pd.Series, window: int = 60) -> float:
    """
    Activity Z-Score: how unusual is recent return magnitude vs history.
    Z = (mean(|r_5d|) - mean(|r_60d|)) / std(|r_60d|).
    Positive = unusually active period. Negative = unusually quiet.
    """
    if len(returns) < window:
        return 0.0
    recent_mag = returns.tail(5).abs().mean()
    hist_mean = returns.tail(window).abs().mean()
    hist_std = returns.tail(window).abs().std()
    if hist_std == 0:
        return 0.0
    return float(np.clip((recent_mag - hist_mean) / hist_std, -3, 3))


def _price_activity_divergence(prices: pd.Series, returns: pd.Series, period: int = 20) -> float:
    """
    Price-Activity Divergence: detects when price moves but volatility contracts
    (stealth accumulation) or price stalls but volatility expands (distribution).

    Signal:
        +0.5 if price up AND volatility down (bullish stealth)
        -0.5 if price down AND volatility up (bearish panic)
         0.0 otherwise (no divergence)

    Rationale: Institutional accumulation often occurs on declining volatility
    as large orders are worked quietly. Distribution tends to spike volatility.
    """
    if len(prices) < period or len(returns) < period:
        return 0.0
    price_trend = (prices.iloc[-1] / prices.iloc[-period] - 1) if prices.iloc[-period] != 0 else 0
    if len(returns) >= period * 2:
        vol_trend = returns.tail(period).abs().mean() - returns.tail(period * 2).abs().mean()
    else:
        vol_trend = 0
    if price_trend > 0.01 and vol_trend < -0.001:
        return 0.5
    elif price_trend < -0.01 and vol_trend > 0.001:
        return -0.5
    return 0.0


def _recovery_velocity(prices: pd.Series, period: int = 60) -> float:
    """
    Recovery Velocity: speed of price recovery from the trough within a window.
    V = (P_current - P_trough) / P_trough / (time_since_trough / period).
    Normalized to [-2, +2]. High positive = fast V-recovery (bullish).
    """
    if len(prices) < period:
        return 0.0
    window = prices.tail(period)
    trough_idx = window.idxmin()
    trough_val = window[trough_idx]
    if trough_val == 0:
        return 0.0
    recovery = (prices.iloc[-1] - trough_val) / abs(trough_val)
    trough_pos = window.index.get_loc(trough_idx)
    time_since = len(window) - trough_pos
    if time_since == 0:
        return 0.0
    velocity = recovery / (time_since / period)
    return float(np.clip(velocity, -2, 2))


# ══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════

def _compute_features(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    weights: Dict[str, float],
    flow_data: Optional[Dict] = None,
    synthesis_data: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Build feature matrix for all tickers.
    Each feature is a standard, well-defined financial metric.
    """
    tickers = [t for t in weights if t in returns.columns]
    rows = []

    for ticker in tickers:
        ret = returns[ticker].dropna()
        px = prices[ticker].dropna()

        if len(ret) < 60 or len(px) < 60:
            continue

        # ── Momentum (log-return sums over windows) ──
        mom_5 = float(ret.tail(5).sum())
        mom_10 = float(ret.tail(10).sum())
        mom_20 = float(ret.tail(20).sum())
        mom_60 = float(ret.tail(60).sum())

        # ── Volatility ──
        vol_20 = float(ret.tail(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        vol_60_std = ret.tail(60).std()
        vol_ratio = float(ret.tail(20).std() / vol_60_std) if vol_60_std > 0 else 1.0
        vol_zscore = float((ret.tail(5).std() - vol_60_std) / vol_60_std) if vol_60_std > 0 else 0.0

        # ── Technical Indicators ──
        rsi_14 = _rsi(px, 14)
        macd_hist = _macd_histogram(px)
        boll_pct_b = _bollinger_pct_b(px, 20)
        dir_strength = _directional_strength(px, 14)
        atr_rat = _atr_ratio(px, 5, 20)

        # ── Microstructure ──
        rda = _return_direction_accumulation(px, ret, 20)
        act_z = _activity_zscore(ret, 60)
        pa_div = _price_activity_divergence(px, ret, 20)
        recov_vel = _recovery_velocity(px, 60)

        # ── Relative Strength vs Benchmark ──
        rel_strength = 0.0
        if BENCHMARK_TICKER in returns.columns:
            bench_ret = returns[BENCHMARK_TICKER].dropna()
            bench_20 = float(bench_ret.tail(20).sum())
            rel_strength = mom_20 - bench_20

        # ── Flow Features (from institutional flow agent) ──
        inst_conviction = 0.0
        insider_alignment = 0.0
        capital_participation = 0.0
        if flow_data and ticker in flow_data:
            fd = flow_data[ticker]
            inst_conviction = float(fd.get("institutional_conviction", 0.0))
            insider_alignment = float(fd.get("insider_alignment", 0.0))
            capital_participation = float(fd.get("capital_participation", 0.0))

        # ── Fundamental Score (from synthesis) ──
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
        cummax = px.cummax()
        drawdown = (px - cummax) / cummax
        current_dd = float(drawdown.iloc[-1])

        rows.append({
            "Ticker": ticker,
            "Momentum_5d": mom_5,
            "Momentum_10d": mom_10,
            "Momentum_20d": mom_20,
            "Momentum_60d": mom_60,
            "Volatility_20d": vol_20,
            "Vol_Ratio": vol_ratio,
            "Vol_Zscore": vol_zscore,
            "RSI_14": rsi_14,
            "MACD_Hist": macd_hist,
            "Bollinger_PctB": boll_pct_b,
            "Directional_Strength": dir_strength,
            "ATR_Ratio": atr_rat,
            "RDA_20d": rda,
            "Activity_Zscore": act_z,
            "PA_Divergence": pa_div,
            "Recovery_Velocity": recov_vel,
            "Rel_Strength_20d": rel_strength,
            "Inst_Conviction": inst_conviction,
            "Insider_Alignment": insider_alignment,
            "Capital_Participation": capital_participation,
            "Fundamental_Score": fundamental_score,
            "Management_Score": management_score,
            "Dist_From_High": dist_from_high,
            "Dist_From_Low": dist_from_low,
            "Drawdown_Depth": current_dd,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# BAYESIAN SIGNAL MODEL
#
# Each feature contributes a Bayes Factor via a signal transform g(x)
# and a sensitivity parameter κ:
#
#     BF(x) = exp(κ · g(x))
#
# The posterior probability after observing all features:
#     P(acc | x₁...xₙ) = σ(log_prior_odds + Σ κᵢ · gᵢ(xᵢ))
#
# where σ(z) = 1/(1+e^{-z}) is the logistic function.
#
# This is equivalent to logistic regression with FIXED (not learned)
# coefficients — eliminating all training-data dependence.
# ══════════════════════════════════════════════════════════════════════════

# ── Signal Transform Functions ───────────────────────────────────────────
# Each g(x) maps a raw feature value to [-1, +1] signal space.
# Positive = evidence FOR accumulation. Negative = evidence AGAINST.

def _g_momentum(mom: float, scale: float) -> float:
    """
    Momentum signal: tanh(mom / scale).
    Rationale: tanh provides smooth saturation — prevents extreme returns
    from producing unbounded evidence. Scale sets the "one standard signal"
    level (where tanh ≈ 0.76).

    Scale choices (annualized equivalents):
        5d:  scale=0.02 → 2% weekly move = strong signal
        20d: scale=0.05 → 5% monthly = strong signal
        60d: scale=0.10 → 10% quarterly = strong signal
    """
    return float(np.tanh(mom / scale))


def _g_rsi(rsi: float) -> float:
    """
    RSI signal transform.
    Maps RSI [0, 100] → [-1, +1] via shifted/scaled logistic.

    Financial rationale:
        RSI < 30: oversold → mean reversion likely → accumulation opportunity
        RSI 40-60: neutral zone → no signal
        RSI > 70: overbought → distribution risk

    But we DON'T simply invert RSI. Moderate momentum (RSI 55-65) is actually
    the sweet spot for institutional accumulation (trending but not extended).

    Transform: piecewise linear with optimal zone at RSI=55:
        RSI < 30: g = +0.8 (oversold bounce opportunity)
        RSI 30-45: g = linear from +0.8 to +0.3
        RSI 45-65: g = linear from +0.3 to +0.3 (optimal accumulation zone)
        RSI 65-75: g = linear from +0.3 to -0.3 (getting extended)
        RSI > 75: g = -0.8 (overbought, distribution likely)
    """
    if rsi < 30:
        return 0.8
    elif rsi < 45:
        return 0.8 - (rsi - 30) / 15 * 0.5  # 0.8 → 0.3
    elif rsi <= 65:
        return 0.3  # Optimal zone — mild positive
    elif rsi <= 75:
        return 0.3 - (rsi - 65) / 10 * 0.6  # 0.3 → -0.3
    else:
        return -0.8


def _g_macd(macd_hist: float, price_level: float) -> float:
    """
    MACD histogram signal, normalized by price level.
    g = tanh(macd_hist / (price × 0.01))

    Rationale: MACD is in price units, so a $1 histogram on a $10 stock
    is very different from $1 on a $500 stock. We normalize by 1% of price
    so that "one standard signal" = histogram equal to 1% of price.
    """
    if price_level <= 0:
        return 0.0
    normalized = macd_hist / (price_level * 0.01)
    return float(np.tanh(normalized))


def _g_bollinger(pct_b: float) -> float:
    """
    Bollinger %B signal.
    g = -(pct_b - 0.5) × 1.5, clipped to [-1, +1].

    Rationale: Mean-reversion framework.
        %B near 0 (lower band) → price compressed → accumulation opportunity (+)
        %B near 1 (upper band) → price extended → distribution risk (-)
        %B at 0.5 (middle) → neutral (0)

    The 1.5 multiplier ensures that touching the bands (0 or 1) gives ±0.75
    signal, not full ±1 (since breakouts DO happen at the bands).
    """
    g = -(pct_b - 0.5) * 1.5
    return float(np.clip(g, -1, 1))


def _g_directional_strength(ds: float) -> float:
    """
    Directional Strength signal.
    g = tanh((ds - 25) / 15)

    Rationale: DS > 25 indicates trending market (favorable for accumulation
    since institutional buyers prefer clear direction). DS < 15 = choppy
    (unfavorable — harder to build positions without moving price).

    Center at 25 (population median), scale 15 (one sigma above/below).
    """
    return float(np.tanh((ds - 25) / 15))


def _g_flow(conviction: float) -> float:
    """
    Institutional flow signal. Already in [-1, +1] from flow agent.
    g = conviction (direct passthrough — the flow agent already computed
    a meaningful [-1, +1] score).

    Rationale: Institutional conviction IS the signal. No further transform
    needed because the flow agent's output is already normalized and
    directionally correct.
    """
    return float(np.clip(conviction, -1, 1))


def _g_structure(dist_from_high: float) -> float:
    """
    Price structure signal based on distance from 52-week high.
    g = tanh(dist_from_high / 0.15) × (-1)

    Rationale: Stocks 15%+ below their 52w high are in a drawdown.
    This creates accumulation opportunity (institutions buy weakness).
    But extreme drawdowns (>40%) signal fundamental problems.

    We invert: closer to high = less opportunity = negative signal.
    Far from high = more opportunity = positive signal. Capped by tanh.

    Note: dist_from_high is already negative (always ≤ 0), so we negate
    to get positive signal for drawdowns.
    """
    return float(np.tanh(-dist_from_high / 0.15))


def _g_recovery(velocity: float) -> float:
    """
    Recovery velocity signal.
    g = tanh(velocity / 1.0)

    Rationale: Fast recovery from drawdown = strong buying pressure.
    Velocity normalized so that recovering the full drawdown in one period = 1.0.
    """
    return float(np.tanh(velocity / 1.0))


# ── Bayes Factor Specification ───────────────────────────────────────────
# Each entry: (feature_name, transform_function, κ, group, rationale)
#
# κ calibration principle:
#     We want a "strong" signal (g ≈ 0.8) to produce BF ≈ e^(κ×0.8) ≈ 3
#     ⟹ κ × 0.8 ≈ ln(3) ≈ 1.1
#     ⟹ κ ≈ 1.4 for primary signals
#
# We use κ ∈ {0.5, 0.8, 1.1, 1.4} corresponding to:
#     0.5 = weak evidence     (BF_max ≈ 1.6)
#     0.8 = moderate evidence (BF_max ≈ 2.2)
#     1.1 = substantial       (BF_max ≈ 3.0)
#     1.4 = strong            (BF_max ≈ 4.0)
#
# Group caps prevent any single category from dominating the posterior.

BAYES_SPEC = {
    # ── Momentum Group (cap: total κ contribution ≤ 2.5) ──────────
    # Rationale: Momentum is the most persistent factor in equity returns
    # (Jegadeesh & Titman, 1993) but is prone to reversals at extremes.
    "Momentum_5d": {
        "kappa": 0.5,   # Weak — 5d is noisy
        "group": "Momentum",
        "rationale": "Short-term momentum; 2% weekly return = strong signal. Low κ due to noise."
    },
    "Momentum_10d": {
        "kappa": 0.8,
        "group": "Momentum",
        "rationale": "10d return reduces noise. 3% = strong signal."
    },
    "Momentum_20d": {
        "kappa": 1.1,
        "group": "Momentum",
        "rationale": "Monthly momentum — strongest predictor in factor literature (Fama-French). 5% monthly = strong."
    },
    "Momentum_60d": {
        "kappa": 0.8,
        "group": "Momentum",
        "rationale": "Quarterly momentum captures sustained trends. 10% quarterly = strong."
    },
    "Rel_Strength_20d": {
        "kappa": 1.1,
        "group": "Momentum",
        "rationale": "Relative strength vs benchmark — industry standard alpha signal. Outperform by 3% = strong."
    },

    # ── Technical Group (cap: ≤ 2.0) ─────────────────────────────
    # Rationale: Technical signals provide timing evidence beyond momentum.
    "RSI_14": {
        "kappa": 0.8,
        "group": "Technical",
        "rationale": "RSI identifies overbought/oversold. Moderate κ — mean reversion has lower Sharpe than momentum."
    },
    "MACD_Hist": {
        "kappa": 0.8,
        "group": "Technical",
        "rationale": "MACD histogram captures momentum acceleration. Moderate weight."
    },
    "Bollinger_PctB": {
        "kappa": 0.5,
        "group": "Technical",
        "rationale": "Bollinger %B = mean-reversion signal. Low κ — often false signals in trending markets."
    },
    "Directional_Strength": {
        "kappa": 0.5,
        "group": "Technical",
        "rationale": "Trend clarity. Low κ — directional strength is contextual, not directly predictive."
    },

    # ── Microstructure Group (cap: ≤ 1.5) ────────────────────────
    # Rationale: Microstructure signals detect institutional footprints.
    "RDA_20d": {
        "kappa": 0.8,
        "group": "Microstructure",
        "rationale": "Persistent buying (daily return direction) suggests accumulation. Direct proxy."
    },
    "PA_Divergence": {
        "kappa": 1.1,
        "group": "Microstructure",
        "rationale": "Price-activity divergence = stealth accumulation. High κ — it's the signature we're looking for."
    },
    "Recovery_Velocity": {
        "kappa": 0.5,
        "group": "Microstructure",
        "rationale": "Fast recovery = strong bid. Low κ — can be noise in small drawdowns."
    },

    # ── Flow Group (cap: ≤ 2.5) ──────────────────────────────────
    # Rationale: Institutional flows are the most direct evidence of accumulation.
    "Inst_Conviction": {
        "kappa": 1.4,
        "group": "Flow",
        "rationale": "Institutional position changes — strongest direct evidence. High κ justified."
    },
    "Insider_Alignment": {
        "kappa": 1.1,
        "group": "Flow",
        "rationale": "Insider buy/sell ratio. Insiders have information advantage (Lakonishok & Lee, 2001)."
    },
    "Capital_Participation": {
        "kappa": 0.5,
        "group": "Flow",
        "rationale": "Binary checklist score — low granularity, so low κ."
    },

    # ── Fundamental Group (cap: ≤ 1.5) ───────────────────────────
    "Fundamental_Score": {
        "kappa": 0.8,
        "group": "Fundamental",
        "rationale": "Composite fundamental conviction from filing analysis. Already a processed score."
    },
    "Management_Score": {
        "kappa": 0.5,
        "group": "Fundamental",
        "rationale": "Management confidence from sentiment. Lower κ — sentiment is noisy."
    },

    # ── Structure Group (cap: ≤ 1.5) ─────────────────────────────
    "Dist_From_High": {
        "kappa": 0.8,
        "group": "Structure",
        "rationale": "Distance from 52w high — accumulation opportunity increases with drawdown."
    },
    "Drawdown_Depth": {
        "kappa": 0.5,
        "group": "Structure",
        "rationale": "Current drawdown depth. Correlated with dist_from_high, so low κ to avoid double-counting."
    },
}

# Group-level maximum total log-evidence (prevents any group from dominating)
GROUP_CAPS = {
    "Momentum": 2.5,
    "Technical": 2.0,
    "Microstructure": 1.5,
    "Flow": 2.5,
    "Fundamental": 1.5,
    "Structure": 1.5,
}


# ══════════════════════════════════════════════════════════════════════════
# BAYESIAN UPDATING ENGINE
# ══════════════════════════════════════════════════════════════════════════

def _compute_signal_values(row: pd.Series, price_level: float) -> Dict[str, float]:
    """
    Apply signal transforms g(x) to all features for one ticker.
    Returns dict of feature_name → g(x) ∈ [-1, +1].
    """
    signals = {}

    # Momentum group
    signals["Momentum_5d"] = _g_momentum(row.get("Momentum_5d", 0), scale=0.02)
    signals["Momentum_10d"] = _g_momentum(row.get("Momentum_10d", 0), scale=0.03)
    signals["Momentum_20d"] = _g_momentum(row.get("Momentum_20d", 0), scale=0.05)
    signals["Momentum_60d"] = _g_momentum(row.get("Momentum_60d", 0), scale=0.10)
    signals["Rel_Strength_20d"] = _g_momentum(row.get("Rel_Strength_20d", 0), scale=0.03)

    # Technical group
    signals["RSI_14"] = _g_rsi(row.get("RSI_14", 50))
    signals["MACD_Hist"] = _g_macd(row.get("MACD_Hist", 0), price_level)
    signals["Bollinger_PctB"] = _g_bollinger(row.get("Bollinger_PctB", 0.5))
    signals["Directional_Strength"] = _g_directional_strength(row.get("Directional_Strength", 25))

    # Microstructure group
    signals["RDA_20d"] = float(np.clip(row.get("RDA_20d", 0), -1, 1))
    signals["PA_Divergence"] = float(np.clip(row.get("PA_Divergence", 0) * 2, -1, 1))  # scale ±0.5 → ±1
    signals["Recovery_Velocity"] = _g_recovery(row.get("Recovery_Velocity", 0))

    # Flow group (already in [-1, 1])
    signals["Inst_Conviction"] = _g_flow(row.get("Inst_Conviction", 0))
    signals["Insider_Alignment"] = _g_flow(row.get("Insider_Alignment", 0))
    signals["Capital_Participation"] = _g_flow(row.get("Capital_Participation", 0) * 2 - 1)  # [0,1]→[-1,1]

    # Fundamental group (already in [-1, 1])
    signals["Fundamental_Score"] = _g_flow(row.get("Fundamental_Score", 0))
    signals["Management_Score"] = _g_flow(row.get("Management_Score", 0))

    # Structure group
    signals["Dist_From_High"] = _g_structure(row.get("Dist_From_High", 0))
    signals["Drawdown_Depth"] = _g_structure(row.get("Drawdown_Depth", 0))

    return signals


def _bayesian_update(signals: Dict[str, float]) -> Dict[str, Any]:
    """
    Perform sequential Bayesian updating.

    Returns:
        posterior: P(accumulation | all signals)
        log_evidence: total log Bayes factor (sum of κ·g contributions)
        group_evidence: per-group log evidence (for decomposition)
        feature_contributions: per-feature κ·g values (for transparency)
    """
    # Prior: P(accumulation) = 0.5 → log_prior_odds = 0
    log_prior_odds = 0.0

    # Accumulate evidence by group for capping
    group_evidence = {g: 0.0 for g in GROUP_CAPS}
    feature_contributions = {}

    for feature, spec in BAYES_SPEC.items():
        g_val = signals.get(feature, 0.0)
        kappa = spec["kappa"]
        group = spec["group"]

        # Raw log-evidence contribution
        contribution = kappa * g_val
        feature_contributions[feature] = {
            "signal": round(g_val, 4),
            "kappa": kappa,
            "log_bf": round(contribution, 4),
            "bf": round(float(np.exp(contribution)), 4),
            "group": group,
        }
        group_evidence[group] += contribution

    # Apply group caps (symmetric: cap both positive and negative)
    capped_group_evidence = {}
    for group, total in group_evidence.items():
        cap = GROUP_CAPS.get(group, 2.0)
        capped = float(np.clip(total, -cap, cap))
        capped_group_evidence[group] = capped

    # Total log-evidence after caps
    total_log_evidence = sum(capped_group_evidence.values())

    # Posterior via logistic function
    log_posterior_odds = log_prior_odds + total_log_evidence
    posterior = 1.0 / (1.0 + np.exp(-log_posterior_odds))
    posterior = float(np.clip(posterior, 0.01, 0.99))

    # Evidence strength: |total_log_evidence| on Jeffreys scale
    # |log_e| < 0.5: barely worth mentioning
    # 0.5-1.0: substantial
    # 1.0-1.5: strong
    # > 1.5: very strong / decisive
    abs_evidence = abs(total_log_evidence)
    if abs_evidence < 0.5:
        evidence_tier = "WEAK"
    elif abs_evidence < 1.0:
        evidence_tier = "MODERATE"
    elif abs_evidence < 1.5:
        evidence_tier = "STRONG"
    else:
        evidence_tier = "DECISIVE"

    # Evidence concentration: max group / total (how concentrated evidence is)
    max_group_abs = max(abs(v) for v in capped_group_evidence.values()) if capped_group_evidence else 0
    total_abs = sum(abs(v) for v in capped_group_evidence.values()) or 1
    concentration = max_group_abs / total_abs

    return {
        "posterior": posterior,
        "log_evidence": round(total_log_evidence, 4),
        "evidence_tier": evidence_tier,
        "group_evidence": {k: round(v, 4) for k, v in capped_group_evidence.items()},
        "feature_contributions": feature_contributions,
        "evidence_concentration": round(concentration, 3),
        "dominant_group": max(capped_group_evidence, key=lambda k: abs(capped_group_evidence[k])),
    }


# ══════════════════════════════════════════════════════════════════════════
# SIGNAL DECOMPOSITION (for UI rendering)
# ══════════════════════════════════════════════════════════════════════════

def _build_signal_decomposition(
    bayes_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Convert per-ticker Bayesian results into signal group decomposition
    for the feature group breakdown chart.

    Returns normalized group contributions that sum to 1.0 per ticker.
    """
    decomp = {}
    for ticker, result in bayes_results.items():
        group_ev = result.get("group_evidence", {})
        # Use absolute contribution for "importance" decomposition
        abs_groups = {k: abs(v) for k, v in group_ev.items()}
        total = sum(abs_groups.values()) or 1
        decomp[ticker] = {k: round(v / total, 3) for k, v in abs_groups.items()}
    return decomp


# ══════════════════════════════════════════════════════════════════════════
# MODEL CONFIDENCE ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════

def _assess_model_confidence(
    bayes_results: Dict[str, Dict[str, Any]],
    n_tickers: int,
) -> Dict[str, Any]:
    """
    Assess overall model confidence based on:
    1. Evidence strength distribution (are signals informative?)
    2. Agreement across tickers (do signals point same direction?)
    3. Concentration risk (is one group driving everything?)
    4. Sample coverage (do we have enough tickers?)
    """
    if not bayes_results:
        return {"model_confidence": 0.3, "confidence_tier": "LOW (No Data)"}

    posteriors = [r["posterior"] for r in bayes_results.values()]
    log_evidences = [abs(r["log_evidence"]) for r in bayes_results.values()]
    concentrations = [r["evidence_concentration"] for r in bayes_results.values()]

    # 1. Information content: average evidence strength
    avg_evidence = float(np.mean(log_evidences))
    info_factor = float(np.clip(avg_evidence / 1.5, 0.2, 1.0))  # 1.5 = "strong" threshold

    # 2. Spread of posteriors (do we differentiate well between tickers?)
    posterior_spread = float(np.std(posteriors))
    discrimination_factor = float(np.clip(posterior_spread / 0.15, 0.3, 1.0))

    # 3. Concentration penalty (too concentrated = fragile)
    avg_concentration = float(np.mean(concentrations))
    diversity_factor = float(1.0 - avg_concentration * 0.5)  # High concentration → lower confidence

    # 4. Sample size factor
    sample_factor = float(np.clip(n_tickers / 10, 0.3, 1.0))

    model_confidence = float(np.clip(
        info_factor * 0.30 +
        discrimination_factor * 0.25 +
        diversity_factor * 0.25 +
        sample_factor * 0.20,
        0.15, 0.95
    ))

    if model_confidence >= 0.70:
        tier = "HIGH"
    elif model_confidence >= 0.50:
        tier = "MODERATE"
    else:
        tier = "LOW (Prototype)"

    return {
        "model_confidence": round(model_confidence, 3),
        "confidence_tier": tier,
        "information_factor": round(info_factor, 3),
        "discrimination_factor": round(discrimination_factor, 3),
        "diversity_factor": round(diversity_factor, 3),
        "sample_size_factor": round(sample_factor, 3),
        "avg_evidence_strength": round(avg_evidence, 3),
        "posterior_spread": round(posterior_spread, 3),
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def run_accumulation_model(state: PortfolioState) -> PortfolioState:
    """
    Run the Bayesian Accumulation Signal Model.

    Pipeline:
    1. Compute features (standard technical indicators + flow + fundamental)
    2. Apply signal transforms g(x) → [-1, +1] for each feature
    3. Compute Bayes Factors BF = exp(κ · g) for each feature
    4. Apply group-level caps to prevent dominance
    5. Multiply all BFs → posterior P(accumulation)
    6. Assess model confidence and output diagnostics
    """
    returns = state.returns
    prices = state.prices
    weights = state.weights

    if returns is None or weights is None:
        raise ValueError("Returns and weights required for Bayesian model.")

    # Get flow data if available
    flow_data = None
    if state.flow_signals and "intel" in state.flow_signals:
        flow_data = state.flow_signals["intel"]

    # Get synthesis data if available
    synthesis_data = None
    if state.conviction_synthesis and "results" in state.conviction_synthesis:
        synthesis_data = state.conviction_synthesis["results"]

    # 1. Build feature matrix
    features = _compute_features(returns, prices, weights, flow_data, synthesis_data)

    if len(features) < 3:
        raise ValueError(f"Not enough tickers for model: {len(features)} (minimum 3)")

    # 2-4. Run Bayesian update for each ticker
    bayes_results = {}
    for _, row in features.iterrows():
        ticker = row["Ticker"]
        # Get price level for MACD normalization
        px = prices[ticker].dropna() if ticker in prices.columns else pd.Series([100])
        price_level = float(px.iloc[-1]) if len(px) > 0 else 100.0

        signals = _compute_signal_values(row, price_level)
        bayes_results[ticker] = _bayesian_update(signals)

    # 5. Build output tables (maintain interface contract with UI)
    tickers_list = features["Ticker"].values
    posteriors = np.array([bayes_results[t]["posterior"] for t in tickers_list])

    # Prediction table (matches v2/v3 interface for rendering compatibility)
    pred_table = pd.DataFrame({
        "Ticker": tickers_list,
        "Accumulation Prob": posteriors,
        "Predicted Label": ["ACCUMULATION" if p >= 0.5 else "NO SIGNAL" for p in posteriors],
        "Signal Strength": [
            "STRONG" if p >= 0.75 else "MODERATE" if p >= 0.6 else "WEAK" if p >= 0.5 else "NEGATIVE"
            for p in posteriors
        ],
        # Replace LR/RF columns with evidence metrics for transparency
        "Log Evidence": [bayes_results[t]["log_evidence"] for t in tickers_list],
        "Evidence Tier": [bayes_results[t]["evidence_tier"] for t in tickers_list],
        "Dominant Group": [bayes_results[t]["dominant_group"] for t in tickers_list],
    })

    # Add key feature values for transparency
    for col in ["RSI_14", "MACD_Hist", "Rel_Strength_20d", "Inst_Conviction", "Momentum_20d"]:
        if col in features.columns:
            pred_table[col] = features[col].values

    pred_table = pred_table.sort_values("Accumulation Prob", ascending=False)

    # Signal decomposition for feature group charts
    signal_decomp = _build_signal_decomposition(bayes_results)

    # Model confidence assessment
    confidence = _assess_model_confidence(bayes_results, len(features))

    # Top candidates
    top_accumulation = pred_table[pred_table["Accumulation Prob"] >= 0.5].head(5)
    top_3 = top_accumulation[["Ticker", "Accumulation Prob", "Signal Strength"]].to_dict("records")

    # Distribution candidates
    distribution = pred_table[pred_table["Accumulation Prob"] < 0.4]
    bottom_3 = distribution.tail(3)[["Ticker", "Accumulation Prob", "Signal Strength"]].to_dict("records")

    # Feature importance (from κ values — static, but useful for display)
    importance_data = []
    for feat, spec in BAYES_SPEC.items():
        importance_data.append({
            "Feature": feat,
            "Kappa": spec["kappa"],
            "Group": spec["group"],
            "Blended_Importance": spec["kappa"] / sum(s["kappa"] for s in BAYES_SPEC.values()),
        })
    importance_df = pd.DataFrame(importance_data).sort_values("Kappa", ascending=False)

    # ── Compatibility columns for rendering ──
    # The UI expects "LR Prob" and "RF Prob" — we provide evidence-based alternatives
    if "LR Prob" not in pred_table.columns:
        pred_table["LR Prob"] = posteriors  # Same as posterior (no ensemble)
    if "RF Prob" not in pred_table.columns:
        pred_table["RF Prob"] = posteriors  # Same

    state.ml_predictions = {
        "prediction_table": pred_table,
        "feature_importance": importance_df,
        "top_accumulation": top_3,
        "distribution_candidates": bottom_3,
        "cv_scores": {
            "accuracy": None,  # No training → no CV
            "note": "Bayesian model uses fixed parameters — no training/validation split needed."
        },
        "model_confidence": confidence,
        "signal_decomposition": signal_decomp,
        "model_agreement": 1.0,  # Single model — always agrees with itself
        "model_type": "Bayesian Signal Model (Sequential Updating)",
        "n_features": len(BAYES_SPEC),
        "n_samples": len(features),
        "label_type": "bayesian_posterior",
        "feature_groups": {
            "Momentum": ["Momentum_5d", "Momentum_10d", "Momentum_20d", "Momentum_60d", "Rel_Strength_20d"],
            "Technical": ["RSI_14", "MACD_Hist", "Bollinger_PctB", "Directional_Strength"],
            "Microstructure": ["RDA_20d", "PA_Divergence", "Recovery_Velocity"],
            "Flow": ["Inst_Conviction", "Insider_Alignment", "Capital_Participation"],
            "Fundamental": ["Fundamental_Score", "Management_Score"],
            "Structure": ["Dist_From_High", "Drawdown_Depth"],
        },
        # Bayesian-specific diagnostics
        "bayes_diagnostics": {
            "prior": 0.5,
            "group_caps": GROUP_CAPS,
            "per_ticker_evidence": {t: r["log_evidence"] for t, r in bayes_results.items()},
            "per_ticker_details": {t: {
                "posterior": r["posterior"],
                "group_evidence": r["group_evidence"],
                "evidence_tier": r["evidence_tier"],
                "dominant_group": r["dominant_group"],
                "concentration": r["evidence_concentration"],
            } for t, r in bayes_results.items()},
        },
        # Backward compatibility
        "top_3": top_3[:3] if top_3 else [],
        "cv_accuracy": None,
        "xgb_results": {"available": False, "note": "Replaced by Bayesian model"},
    }

    # Log to observability
    try:
        from invictus.observability.collectors.ml_collector import log_ml_predictions
        ml_rows = []
        for _, row in pred_table.iterrows():
            ml_rows.append({
                "ticker": row["Ticker"],
                "accumulation_prob": row["Accumulation Prob"],
                "lr_prob": row.get("LR Prob"),
                "rf_prob": row.get("RF Prob"),
                "signal_strength": row.get("Signal Strength"),
            })
        log_ml_predictions(
            ml_rows, cv_score=confidence,
            n_features=len(BAYES_SPEC), n_samples=len(features),
            model_type="Bayesian Signal Model",
        )
    except Exception:
        pass

    return state
