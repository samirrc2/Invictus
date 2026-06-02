"""
Invictus — Volatility Regime Detection Agent
Identifies current market volatility regime using rolling vol and KMeans clustering.

Outputs:
- Rolling 20-day portfolio volatility series
- Regime labels: Low / Medium / High
- Current regime + days in current regime
- Regime transition history
- Annualized vol per regime
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, Any

from invictus.agents.graph_state import PortfolioState
from invictus.config import ROLLING_VOL_WINDOW, VOL_REGIME_CLUSTERS, TRADING_DAYS_PER_YEAR


def detect_vol_regime(state: PortfolioState) -> PortfolioState:
    """
    Detect volatility regime using rolling vol + KMeans.
    Writes vol_regime to state.
    """
    returns = state.returns
    weights = state.weights

    if returns is None or weights is None:
        raise ValueError("Returns and weights required for vol regime detection.")

    # Compute portfolio returns — fillna(0) so per-cell NaN doesn't
    # propagate to the entire portfolio return for that row.
    tickers = [t for t in weights if t in returns.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    port_returns = returns[tickers].fillna(0).dot(w)

    # Rolling volatility (annualized)
    rolling_vol = port_returns.rolling(window=ROLLING_VOL_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rolling_vol = rolling_vol.dropna()

    if len(rolling_vol) < ROLLING_VOL_WINDOW * 2:
        raise ValueError(f"Not enough data for regime detection: {len(rolling_vol)} points")

    # KMeans clustering on rolling vol
    vol_values = rolling_vol.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=VOL_REGIME_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vol_values)

    # Sort clusters by centroid so 0=Low, 1=Medium, 2=High
    centroids = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centroids)
    label_map = {sorted_indices[i]: i for i in range(VOL_REGIME_CLUSTERS)}
    sorted_labels = np.array([label_map[l] for l in labels])

    regime_names = {0: "Low", 1: "Medium", 2: "High"}
    sorted_centroids = np.sort(centroids)

    # Build regime series
    regime_series = pd.Series(
        [regime_names[l] for l in sorted_labels],
        index=rolling_vol.index,
        name="Regime",
    )

    # Current regime
    current_regime = regime_series.iloc[-1]
    current_vol = float(rolling_vol.iloc[-1])

    # Days in current regime
    days_in_regime = 0
    for i in range(len(regime_series) - 1, -1, -1):
        if regime_series.iloc[i] == current_regime:
            days_in_regime += 1
        else:
            break

    # Regime transitions (last 10)
    transitions = []
    prev = regime_series.iloc[0]
    for i in range(1, len(regime_series)):
        curr = regime_series.iloc[i]
        if curr != prev:
            transitions.append({
                "date": regime_series.index[i].strftime("%Y-%m-%d"),
                "from": prev,
                "to": curr,
            })
            prev = curr
    recent_transitions = transitions[-10:] if transitions else []

    # Regime statistics
    regime_stats = {}
    for rid, name in regime_names.items():
        mask = sorted_labels == rid
        if mask.any():
            regime_vols = rolling_vol.values[mask]
            regime_stats[name] = {
                "mean_vol": float(np.mean(regime_vols)),
                "min_vol": float(np.min(regime_vols)),
                "max_vol": float(np.max(regime_vols)),
                "days": int(mask.sum()),
                "pct_time": float(mask.sum() / len(sorted_labels)),
                "centroid": float(sorted_centroids[rid]),
            }

    # Regime distribution
    regime_counts = regime_series.value_counts()

    state.vol_regime = {
        "rolling_vol": rolling_vol,
        "regime_series": regime_series,
        "current_regime": current_regime,
        "current_vol": current_vol,
        "days_in_regime": days_in_regime,
        "recent_transitions": recent_transitions,
        "regime_stats": regime_stats,
        "regime_counts": regime_counts.to_dict(),
        "centroids": {regime_names[i]: float(sorted_centroids[i]) for i in range(VOL_REGIME_CLUSTERS)},
    }

    return state
