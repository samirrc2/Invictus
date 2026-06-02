"""
Invictus — Portfolio Risk Agent
Computes comprehensive risk metrics for the portfolio.

Metrics:
- Portfolio volatility (annualized)
- VaR 95% (parametric, historical, Monte Carlo)
- CVaR / Expected Shortfall 95%
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Omega Ratio
- Max Drawdown
- Return distribution stats (skew, kurtosis, Jarque-Bera)
- Sector / ticker concentration (HHI)
- Correlation matrix
- Per-ticker contribution to portfolio risk (MCTR)
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Dict, Any

from invictus.agents.graph_state import PortfolioState
from invictus.config import (
    VAR_CONFIDENCE,
    CVAR_CONFIDENCE,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
)


def _portfolio_returns(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Compute weighted portfolio return series."""
    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return pd.Series(dtype=float)
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()  # normalize
    # fillna(0): if a ticker has no return for a given day, treat as
    # 0% return rather than propagating NaN to the entire portfolio.
    return returns[tickers].fillna(0).dot(w)


def _annualized_vol(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def _annualized_return(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    return clean.mean() * TRADING_DAYS_PER_YEAR


def _sharpe_ratio(returns: pd.Series) -> float:
    ann_ret = _annualized_return(returns)
    ann_vol = _annualized_vol(returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - RISK_FREE_RATE) / ann_vol


def _sortino_ratio(returns: pd.Series) -> float:
    ann_ret = _annualized_return(returns)
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (ann_ret - RISK_FREE_RATE) / downside_vol


def _calmar_ratio(returns: pd.Series) -> float:
    ann_ret = _annualized_return(returns)
    max_dd = _max_drawdown(returns)
    if max_dd == 0:
        return 0.0
    return ann_ret / abs(max_dd)


def _omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gains / losses relative to threshold."""
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess <= 0].sum())
    if losses == 0:
        return float("inf")
    return gains / losses


def _max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    cumulative = (1 + clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def _max_drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the full drawdown series for charting."""
    clean = returns.dropna()
    if len(clean) == 0:
        return pd.Series(dtype=float)
    cumulative = (1 + clean).cumprod()
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max


def _var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level."""
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    return np.percentile(clean, (1 - confidence) * 100)


def _var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR."""
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    z = scipy_stats.norm.ppf(1 - confidence)
    return clean.mean() + z * clean.std()


def _var_monte_carlo(
    returns: pd.Series, confidence: float = 0.95, n_sims: int = 10000
) -> float:
    """Monte Carlo VaR using bootstrapped returns."""
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    np.random.seed(42)
    simulated = np.random.choice(clean.values, size=n_sims, replace=True)
    return np.percentile(simulated, (1 - confidence) * 100)


def _cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) — average of losses beyond VaR."""
    var = _var_historical(returns, confidence)
    tail = returns[returns <= var]
    return tail.mean() if len(tail) > 0 else var


def _concentration_hhi(weights: Dict[str, float]) -> float:
    """Herfindahl-Hirschman Index for portfolio concentration."""
    w = np.array(list(weights.values()))
    w = w / w.sum()
    return float(np.sum(w**2))


def _marginal_contribution_to_risk(
    returns: pd.DataFrame, weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Marginal Contribution to Risk (MCTR) by ticker.
    MCTR_i = w_i * (Cov * w)_i / sigma_p
    """
    tickers = [t for t in weights if t in returns.columns]
    if not tickers or len(returns[tickers].dropna(how="all")) < 2:
        return pd.DataFrame(columns=[
            "Ticker", "Weight", "MCTR", "Risk Contribution", "% of Portfolio Risk"
        ])

    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    cov_matrix = returns[tickers].cov().values * TRADING_DAYS_PER_YEAR
    port_var = w @ cov_matrix @ w
    port_vol = np.sqrt(port_var)

    if port_vol == 0:
        return pd.DataFrame({
            "Ticker": tickers, "Weight": w,
            "MCTR": np.zeros(len(w)),
            "Risk Contribution": np.zeros(len(w)),
            "% of Portfolio Risk": np.zeros(len(w)),
        })

    # Marginal contribution
    mctr = (cov_matrix @ w) / port_vol  # risk contribution per unit weight
    ctr = w * mctr  # total contribution = weight * marginal
    pct_ctr = ctr / port_vol  # percentage of total risk

    return pd.DataFrame({
        "Ticker": tickers,
        "Weight": w,
        "MCTR": mctr,
        "Risk Contribution": ctr,
        "% of Portfolio Risk": pct_ctr,
    }).sort_values("Risk Contribution", ascending=False)


def _return_distribution_stats(returns: pd.Series) -> Dict[str, float]:
    """Compute distribution statistics."""
    clean = returns.dropna()
    if len(clean) < 3:
        return {
            "skewness": 0.0, "kurtosis": 0.0,
            "jarque_bera_stat": 0.0, "jarque_bera_pval": 1.0,
            "is_normal": True,
            "mean_daily": float(clean.mean()) if len(clean) else 0.0,
            "std_daily": float(clean.std()) if len(clean) else 0.0,
            "min_daily": float(clean.min()) if len(clean) else 0.0,
            "max_daily": float(clean.max()) if len(clean) else 0.0,
        }
    skew = scipy_stats.skew(clean)
    kurt = scipy_stats.kurtosis(clean)
    jb_stat, jb_pval = scipy_stats.jarque_bera(clean)
    return {
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pval": float(jb_pval),
        "is_normal": jb_pval > 0.05,
        "mean_daily": float(clean.mean()),
        "std_daily": float(clean.std()),
        "min_daily": float(clean.min()),
        "max_daily": float(clean.max()),
    }


def compute_risk(state: PortfolioState) -> PortfolioState:
    """
    Main risk computation entry point.
    Reads returns + weights from state, computes all risk metrics,
    writes results back to state.
    """
    returns = state.returns
    weights = state.weights

    if returns is None or weights is None:
        raise ValueError("Returns and weights must be populated before risk computation.")

    if isinstance(returns, pd.DataFrame) and len(returns) == 0:
        raise ValueError(
            f"Returns DataFrame is empty (shape {returns.shape}). "
            f"This usually means yfinance returned price data with NaN columns "
            f"that caused all rows to be dropped during log-return computation. "
            f"Check that all tickers have valid price history."
        )

    # ── Diagnostic: inspect weight values for NaN ──────────────
    nan_weights = {k: v for k, v in weights.items() if pd.isna(v)}
    if nan_weights:
        raise ValueError(
            f"NaN weight values detected for: {nan_weights}. "
            f"This means prices were missing for those tickers in portfolio_loader. "
            f"All weights: {weights}"
        )

    # ── Diagnostic: check returns for column-level NaN density ──
    nan_pct = {c: float(returns[c].isna().mean()) for c in returns.columns}
    heavy_nan = {c: f"{p:.0%}" for c, p in nan_pct.items() if p > 0.5}

    # Portfolio-level return series
    port_returns = _portfolio_returns(returns, weights)

    if len(port_returns.dropna()) == 0:
        raise ValueError(
            f"Portfolio return series is empty after weighting. "
            f"Returns shape: {returns.shape}, weight tickers: {list(weights.keys())}, "
            f"weight values: {weights}, "
            f"NaN density per column: {nan_pct}, "
            f"port_returns sample: {port_returns.head(5).tolist()}, "
            f"port_returns NaN count: {port_returns.isna().sum()}/{len(port_returns)}"
        )

    # Core risk metrics
    risk_metrics = {
        # Returns
        "annualized_return": _annualized_return(port_returns),
        "daily_return_mean": float(port_returns.mean()),
        # Volatility
        "annualized_volatility": _annualized_vol(port_returns),
        "daily_volatility": float(port_returns.std()),
        # VaR
        "var_95_historical": _var_historical(port_returns, VAR_CONFIDENCE),
        "var_95_parametric": _var_parametric(port_returns, VAR_CONFIDENCE),
        "var_95_monte_carlo": _var_monte_carlo(port_returns, VAR_CONFIDENCE),
        # CVaR
        "cvar_95": _cvar(port_returns, CVAR_CONFIDENCE),
        # Ratios
        "sharpe_ratio": _sharpe_ratio(port_returns),
        "sortino_ratio": _sortino_ratio(port_returns),
        "calmar_ratio": _calmar_ratio(port_returns),
        "omega_ratio": _omega_ratio(port_returns),
        # Drawdown
        "max_drawdown": _max_drawdown(port_returns),
        "drawdown_series": _max_drawdown_series(port_returns),
        # Concentration
        "hhi_concentration": _concentration_hhi(weights),
        # Distribution
        "distribution_stats": _return_distribution_stats(port_returns),
        # Portfolio return series (for other agents)
        "portfolio_returns": port_returns,
    }

    # Correlation matrix
    tickers = [t for t in weights if t in returns.columns]
    corr_matrix = returns[tickers].corr()

    # Per-ticker risk contribution
    ticker_risk = _marginal_contribution_to_risk(returns, weights)

    # Write to state
    state.risk_metrics = risk_metrics
    state.correlation_matrix = corr_matrix
    state.ticker_risk = ticker_risk

    return state
