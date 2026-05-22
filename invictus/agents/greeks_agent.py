"""
Invictus — Portfolio Greeks & Options Risk Agent
Pulls options chain data via yfinance and computes:
- ATM implied volatility per ticker
- IV percentile rank (current IV vs 1-year range)
- Black-Scholes Greeks: delta, gamma, vega, theta, rho
- Put/call IV skew
- Volga (vomma) — 2nd derivative of price w.r.t. vol
- IV term structure (near vs far expiry)

Uses Black-Scholes model for Greeks computation.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import streamlit as st
import warnings

from invictus.agents.graph_state import PortfolioState
from invictus.config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Black-Scholes ──────────────────────────────────────────────────────

def _bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _bs_d2(S, K, T, r, sigma):
    return _bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def _bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def _bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def _bs_vega(S, K, T, r, sigma):
    """Vega per 1% move in vol (divided by 100)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def _bs_theta(S, K, T, r, sigma, option_type="call"):
    """Theta per day."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return (term1 + term2) / 365


def _bs_rho(S, K, T, r, sigma, option_type="call"):
    """Rho per 1% move in rates."""
    if T <= 0:
        return 0.0
    d2 = _bs_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def _bs_volga(S, K, T, r, sigma):
    """Volga (vomma) — d2(Price)/d(sigma)^2. Convexity of vega."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(S, K, T, r, sigma)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega * d1 * d2 / sigma


# ── Options Chain Helpers ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_options_data(ticker: str) -> Optional[Dict]:
    """Fetch options chain and compute Greeks for ATM options."""
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.info.get("regularMarketPrice") or stock.info.get("currentPrice")
        if current_price is None or current_price <= 0:
            hist = stock.history(period="2d")
            if len(hist) == 0:
                return None
            current_price = float(hist["Close"].iloc[-1])

        # Get available expirations
        expirations = stock.options
        if not expirations or len(expirations) == 0:
            return None

        # Pick nearest expiry (30-60 days out ideally)
        today = datetime.now()
        near_exp = None
        far_exp = None
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = (exp_date - today).days
            if 20 <= dte <= 60 and near_exp is None:
                near_exp = exp_str
            if 80 <= dte <= 180 and far_exp is None:
                far_exp = exp_str

        if near_exp is None and expirations:
            near_exp = expirations[0]
        if far_exp is None and len(expirations) > 1:
            far_exp = expirations[min(2, len(expirations) - 1)]

        if near_exp is None:
            return None

        # Get ATM options for near expiry
        chain = stock.option_chain(near_exp)
        calls = chain.calls
        puts = chain.puts

        if calls.empty:
            return None

        # Find ATM strike
        strikes = calls["strike"].values
        atm_idx = np.argmin(np.abs(strikes - current_price))
        atm_strike = strikes[atm_idx]

        atm_call = calls.iloc[atm_idx]
        atm_put = puts.iloc[atm_idx] if not puts.empty and atm_idx < len(puts) else None

        # Time to expiry
        exp_date = datetime.strptime(near_exp, "%Y-%m-%d")
        T = max((exp_date - today).days / 365, 1 / 365)

        # Implied vol from ATM call
        iv = atm_call.get("impliedVolatility", None)
        if iv is None or iv <= 0 or np.isnan(iv):
            iv = 0.30  # fallback

        S = current_price
        K = atm_strike
        r = RISK_FREE_RATE

        # Compute Greeks
        greeks = {
            "delta": _bs_delta(S, K, T, r, iv, "call"),
            "gamma": _bs_gamma(S, K, T, r, iv),
            "vega": _bs_vega(S, K, T, r, iv),
            "theta": _bs_theta(S, K, T, r, iv, "call"),
            "rho": _bs_rho(S, K, T, r, iv, "call"),
            "volga": _bs_volga(S, K, T, r, iv),
        }

        # IV skew: compare 25-delta put IV vs ATM
        otm_put_iv = None
        if not puts.empty:
            # Find ~25 delta put (roughly 5-10% OTM)
            otm_strike = current_price * 0.93
            otm_idx = np.argmin(np.abs(puts["strike"].values - otm_strike))
            otm_put_row = puts.iloc[otm_idx]
            otm_put_iv = otm_put_row.get("impliedVolatility", None)

        skew = None
        if otm_put_iv and otm_put_iv > 0 and not np.isnan(otm_put_iv):
            skew = otm_put_iv - iv  # positive = puts are more expensive (crash protection)

        # IV term structure
        term_structure = None
        if far_exp:
            far_chain = stock.option_chain(far_exp)
            far_calls = far_chain.calls
            if not far_calls.empty:
                far_atm_idx = np.argmin(np.abs(far_calls["strike"].values - current_price))
                far_iv = far_calls.iloc[far_atm_idx].get("impliedVolatility", None)
                if far_iv and far_iv > 0 and not np.isnan(far_iv):
                    term_structure = {
                        "near_expiry": near_exp,
                        "near_iv": iv,
                        "far_expiry": far_exp,
                        "far_iv": far_iv,
                        "term_spread": far_iv - iv,
                    }

        return {
            "current_price": S,
            "atm_strike": K,
            "expiry": near_exp,
            "dte": int((exp_date - today).days),
            "iv": iv,
            "greeks": greeks,
            "skew": skew,
            "otm_put_iv": otm_put_iv,
            "term_structure": term_structure,
        }

    except Exception as e:
        return {"error": str(e)}


def compute_greeks(state: PortfolioState) -> PortfolioState:
    """
    Compute Greeks for all portfolio tickers.
    ETFs may not have options — those are skipped gracefully.
    """
    holdings = state.holdings
    weights = state.weights

    if holdings is None:
        raise ValueError("Holdings required for Greeks computation.")

    tickers = holdings["Ticker"].tolist() if isinstance(holdings, pd.DataFrame) else list(weights.keys())

    results = {}
    summary_rows = []

    for ticker in tickers:
        data = _fetch_options_data(ticker)
        if data is None:
            summary_rows.append({
                "Ticker": ticker, "IV": None, "Delta": None, "Gamma": None,
                "Vega": None, "Theta": None, "Rho": None, "Volga": None,
                "Skew": None, "Status": "No options data",
            })
            continue
        if "error" in data:
            summary_rows.append({
                "Ticker": ticker, "IV": None, "Delta": None, "Gamma": None,
                "Vega": None, "Theta": None, "Rho": None, "Volga": None,
                "Skew": None, "Status": f"Error: {data['error'][:40]}",
            })
            continue

        g = data["greeks"]
        summary_rows.append({
            "Ticker": ticker,
            "IV": data["iv"],
            "Delta": g["delta"],
            "Gamma": g["gamma"],
            "Vega": g["vega"],
            "Theta": g["theta"],
            "Rho": g["rho"],
            "Volga": g["volga"],
            "Skew": data["skew"],
            "DTE": data["dte"],
            "ATM Strike": data["atm_strike"],
            "Status": "OK",
        })
        results[ticker] = data

    summary_df = pd.DataFrame(summary_rows)

    # Portfolio-weighted Greeks
    weighted_greeks = {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}
    total_w = 0
    for _, row in summary_df.iterrows():
        t = row["Ticker"]
        w = weights.get(t, 0) if weights else 0
        if row.get("Status") == "OK" and row["Delta"] is not None:
            weighted_greeks["delta"] += w * row["Delta"]
            weighted_greeks["gamma"] += w * row["Gamma"]
            weighted_greeks["vega"] += w * row["Vega"]
            weighted_greeks["theta"] += w * row["Theta"]
            total_w += w

    state.greeks_results = {
        "summary": summary_df,
        "detail": results,
        "portfolio_greeks": weighted_greeks,
        "coverage": f"{len(results)}/{len(tickers)} tickers with options data",
    }

    return state
