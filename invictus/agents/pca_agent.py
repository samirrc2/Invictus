"""
Invictus — PCA Risk Factor Decomposition Agent
Decomposes portfolio risk into principal components to identify
dominant risk factors and concentration of exposures.

Outputs:
- Explained variance per component
- Cumulative explained variance
- Factor loadings (which tickers drive each component)
- Risk concentration assessment
- Plain-English interpretation
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

from invictus.agents.graph_state import PortfolioState
from invictus.config import PCA_COMPONENTS


def run_pca(state: PortfolioState) -> PortfolioState:
    """
    Run PCA on portfolio ticker returns.
    Writes pca_results to state.
    """
    returns = state.returns
    weights = state.weights

    if returns is None or weights is None:
        raise ValueError("Returns and weights must be populated before PCA.")

    tickers = [t for t in weights if t in returns.columns]
    ret_matrix = returns[tickers].dropna()

    if len(ret_matrix) < 30:
        raise ValueError(f"Not enough data for PCA: {len(ret_matrix)} days (need 30+)")

    # Standardize returns
    scaler = StandardScaler()
    scaled = scaler.fit_transform(ret_matrix)

    # Fit PCA
    n_components = min(PCA_COMPONENTS, len(tickers))
    pca = PCA(n_components=n_components)
    pca.fit(scaled)

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Factor loadings: how much each ticker loads onto each component
    loadings = pd.DataFrame(
        pca.components_.T,
        index=tickers,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    # Dominant tickers per component
    dominant_tickers = {}
    for i in range(n_components):
        col = f"PC{i+1}"
        sorted_loadings = loadings[col].abs().sort_values(ascending=False)
        top_3 = sorted_loadings.head(3)
        dominant_tickers[col] = {
            "tickers": top_3.index.tolist(),
            "loadings": [float(loadings.loc[t, col]) for t in top_3.index],
            "abs_loadings": [float(v) for v in top_3.values],
        }

    # Risk concentration assessment
    pc1_var = float(explained_var[0])
    if pc1_var > 0.60:
        concentration = "HIGH"
        assessment = (
            f"PC1 explains {pc1_var:.0%} of portfolio variance. "
            f"Risk is highly concentrated in a single factor — likely broad market beta. "
            f"Diversification benefit is limited."
        )
    elif pc1_var > 0.40:
        concentration = "MODERATE"
        assessment = (
            f"PC1 explains {pc1_var:.0%} of portfolio variance. "
            f"There is moderate factor concentration. "
            f"The portfolio has some diversification but a dominant risk driver exists."
        )
    else:
        concentration = "LOW"
        assessment = (
            f"PC1 explains {pc1_var:.0%} of portfolio variance. "
            f"Risk is well-distributed across multiple factors. "
            f"Good diversification across independent risk drivers."
        )

    # Interpret what each component likely represents
    component_labels = _interpret_components(loadings, tickers)

    # Full PCA on all components for scree plot
    pca_full = PCA()
    pca_full.fit(scaled)

    state.pca_results = {
        "n_components": n_components,
        "explained_variance": [float(v) for v in explained_var],
        "cumulative_variance": [float(v) for v in cumulative_var],
        "loadings": loadings,
        "dominant_tickers": dominant_tickers,
        "concentration": concentration,
        "assessment": assessment,
        "component_labels": component_labels,
        "full_explained_variance": [float(v) for v in pca_full.explained_variance_ratio_],
        "eigenvalues": [float(v) for v in pca_full.explained_variance_],
    }

    return state


def _interpret_components(loadings: pd.DataFrame, tickers: list) -> Dict[str, str]:
    """
    Heuristic interpretation of what each PC likely represents.
    Based on which tickers load heavily onto each component.
    """
    # Sector classification (simplified)
    sector_map = {
        "AAPL": "Tech", "AMD": "Semi", "META": "Tech", "TSLA": "EV/Growth",
        "HIMS": "Health/Growth", "SMH": "Semi ETF", "QQQ": "Tech ETF",
        "VTI": "Broad Market", "SCHD": "Dividend", "GLDM": "Gold",
        "VGK": "Intl Europe", "ITA": "Defense", "VBK": "Small Cap Growth",
        "SPY": "Broad Market",
    }

    labels = {}
    for col in loadings.columns:
        top = loadings[col].abs().sort_values(ascending=False).head(4)
        top_tickers = top.index.tolist()
        sectors = [sector_map.get(t, "Other") for t in top_tickers]

        # Check for common patterns
        if any("ETF" in s or "Market" in s for s in sectors[:2]):
            labels[col] = "Broad Market / Beta"
        elif any("Semi" in s for s in sectors[:2]):
            labels[col] = "Semiconductor / Tech Momentum"
        elif "Gold" in sectors[:2]:
            labels[col] = "Safe Haven / Inflation Hedge"
        elif any("Intl" in s for s in sectors[:2]):
            labels[col] = "International / FX Exposure"
        elif any("Dividend" in s or "Defense" in s for s in sectors[:2]):
            labels[col] = "Defensive / Income"
        elif any("Growth" in s for s in sectors[:2]):
            labels[col] = "Growth / Speculative"
        else:
            labels[col] = f"Factor driven by {', '.join(top_tickers[:2])}"

    return labels
