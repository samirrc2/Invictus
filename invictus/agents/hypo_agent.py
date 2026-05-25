"""
Invictus — Hypothetical Portfolio Simulator Agent
Computes risk metrics for a hypothetical portfolio (existing + new positions)
and generates AI commentary with 3 pros and 3 cons.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from invictus.config import (
    TRADING_DAYS_PER_YEAR, RISK_FREE_RATE, VAR_CONFIDENCE,
)


def _annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def _annualized_return(returns: pd.Series) -> float:
    return returns.mean() * TRADING_DAYS_PER_YEAR


def _sharpe(returns: pd.Series) -> float:
    vol = _annualized_vol(returns)
    if vol == 0:
        return 0.0
    return (_annualized_return(returns) - RISK_FREE_RATE) / vol


def _max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def _var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    return np.percentile(returns, (1 - confidence) * 100)


def _hhi(weights: Dict[str, float]) -> float:
    w = np.array(list(weights.values()))
    w = w / w.sum()
    return float(np.sum(w ** 2))


def _portfolio_returns(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    tickers = [t for t in weights if t in returns.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    return returns[tickers].dot(w)


def compute_before_after(
    current_state: Dict[str, Any],
    current_risk: Dict[str, Any],
    new_positions: Dict[str, float],
    prices: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute before/after risk metrics when adding new positions.

    Args:
        current_state: portfolio_state from session (summary, weights, total_value, etc.)
        current_risk: risk_state from session (risk_metrics dict)
        new_positions: {ticker: dollar_amount} for new investments
        prices: price history DataFrame (must include new tickers)

    Returns:
        Dict with before/after metrics and deltas.
    """
    rm = (current_risk or {}).get("risk_metrics") or {}

    # ── Before metrics (from existing risk state) ──
    before = {
        "sharpe_ratio": rm.get("sharpe_ratio", 0),
        "annualized_volatility": rm.get("annualized_volatility", 0),
        "max_drawdown": rm.get("max_drawdown", 0),
        "var_95": rm.get("var_95_historical", 0),
        "hhi_concentration": rm.get("hhi_concentration", 0),
        "annualized_return": rm.get("annualized_return", 0),
        "total_value": current_state["total_value"],
    }

    # ── Build hypothetical portfolio ──
    # Current holdings: ticker -> market_value
    existing_weights_dollar = {}
    summary = current_state["summary"]
    for _, row in summary.iterrows():
        existing_weights_dollar[row["Ticker"]] = row["Market Value"]

    # Add new positions
    for ticker, amount in new_positions.items():
        if amount > 0:
            existing_weights_dollar[ticker] = existing_weights_dollar.get(ticker, 0) + amount

    # Compute new total and weights
    new_total = sum(existing_weights_dollar.values())
    new_weights = {t: v / new_total for t, v in existing_weights_dollar.items()}

    # Compute returns for hypothetical portfolio
    returns = np.log(prices / prices.shift(1)).dropna()
    available_tickers = [t for t in new_weights if t in returns.columns]

    if not available_tickers:
        return {"error": "No price data available for hypothetical portfolio"}

    # Filter weights to available tickers and renormalize
    filtered_weights = {t: new_weights[t] for t in available_tickers}
    w_sum = sum(filtered_weights.values())
    filtered_weights = {t: v / w_sum for t, v in filtered_weights.items()}

    hypo_returns = _portfolio_returns(returns, filtered_weights)

    # ── After metrics ──
    after = {
        "sharpe_ratio": _sharpe(hypo_returns),
        "annualized_volatility": _annualized_vol(hypo_returns),
        "max_drawdown": _max_drawdown(hypo_returns),
        "var_95": _var_historical(hypo_returns, VAR_CONFIDENCE),
        "hhi_concentration": _hhi(filtered_weights),
        "annualized_return": _annualized_return(hypo_returns),
        "total_value": new_total,
    }

    # ── Deltas ──
    deltas = {k: after[k] - before[k] for k in before}

    # ── Weight changes ──
    old_weights = current_state.get("weights", {})
    weight_changes = []
    all_tickers = set(list(old_weights.keys()) + list(filtered_weights.keys()))
    for t in sorted(all_tickers):
        old_w = old_weights.get(t, 0)
        new_w = filtered_weights.get(t, 0)
        if abs(new_w - old_w) > 0.001:
            weight_changes.append({
                "Ticker": t,
                "Before (%)": old_w * 100,
                "After (%)": new_w * 100,
                "Change (pp)": (new_w - old_w) * 100,
            })

    return {
        "before": before,
        "after": after,
        "deltas": deltas,
        "new_weights": filtered_weights,
        "weight_changes": weight_changes,
        "new_total": new_total,
        "investment_total": sum(new_positions.values()),
    }


def _materiality_check(comparison: Dict[str, Any]) -> bool:
    """Return True if the investment is large enough to have material impact."""
    inv = comparison.get("investment_total", 0)
    total = comparison.get("new_total", 1)
    return (inv / max(total, 1)) > 0.02  # at least 2% of portfolio


def generate_pros_cons(
    comparison: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate pros and cons ranked by impact magnitude.
    Pure portfolio analytics — does this investment fit well within
    the existing portfolio? Filters out negligible changes and detects
    immaterial investment sizes.

    Returns:
        Dict with 'rows' (paired pro/con per row for aligned display),
        'ranked_metrics' (only material ones), 'verdict', 'is_material'.
    """
    before = comparison["before"]
    after = comparison["after"]
    deltas = comparison["deltas"]
    is_material = _materiality_check(comparison)

    # ── If investment is immaterial, short-circuit ──
    if not is_material:
        inv = comparison.get("investment_total", 0)
        total = comparison.get("new_total", 1)
        pct = inv / max(total, 1) * 100
        return {
            "rows": [],
            "ranked_metrics": [],
            "verdict": "IMMATERIAL",
            "verdict_detail": (
                f"The proposed investment of ${inv:,.0f} represents only {pct:.1f}% "
                f"of your portfolio (${total:,.0f}). This allocation is too small "
                f"to have a material impact on portfolio risk characteristics. "
                f"Consider increasing the position size for a meaningful simulation."
            ),
            "is_material": False,
        }

    # ── Metric definitions ──
    # (key, label, format, scale_factor, lower_is_better, min_threshold)
    # min_threshold: below this absolute delta, the change is noise
    metric_defs = [
        ("sharpe_ratio",          "Sharpe Ratio",         "{:.2f}",  50,  False, 0.005),
        ("annualized_volatility", "Annualized Volatility", "{:.1%}", 100, True,  0.001),
        ("max_drawdown",          "Max Drawdown",         "{:.1%}", 100, False,  0.001),
        ("var_95",                "Value-at-Risk (95%)",  "{:.2%}", 200, False,  0.0005),
        ("hhi_concentration",     "Concentration (HHI)",  "{:.3f}",  30, True,   0.001),
        ("annualized_return",     "Annualized Return",    "{:.1%}",  10, False,  0.002),
    ]

    # ── Build observations, skip negligible ──
    observations: List[tuple] = []
    for key, label, fmt, scale, lower_better, threshold in metric_defs:
        d = deltas[key]
        if abs(d) < threshold:
            continue  # skip noise — AI won't comment on negligible changes

        if lower_better:
            is_good = d <= 0
        else:
            is_good = d >= 0

        impact = abs(d) * scale
        b_str = fmt.format(before[key])
        a_str = fmt.format(after[key])

        # Build rich metric-specific commentary (explains the metric + impact)
        if key == "sharpe_ratio":
            if is_good:
                text = (
                    f"The Sharpe Ratio measures how much return you earn per unit of "
                    f"risk taken — a higher value means better risk-adjusted performance. "
                    f"Adding these positions moves your Sharpe from {b_str} to {a_str} "
                    f"({d:+.2f}), meaning the hypothetical portfolio generates more "
                    f"return for each percentage point of volatility you accept. "
                    f"This is a positive signal for portfolio efficiency."
                )
            else:
                text = (
                    f"The Sharpe Ratio measures return earned per unit of risk — "
                    f"a declining Sharpe indicates deteriorating risk-adjusted efficiency. "
                    f"Your Sharpe drops from {b_str} to {a_str} ({d:+.2f}), meaning "
                    f"the new positions dilute your portfolio's return-per-risk ratio. "
                    f"You are taking on proportionally more risk relative to the "
                    f"additional return these positions contribute."
                )
        elif key == "annualized_volatility":
            rel = abs(d) / max(before[key], 0.001) * 100
            if is_good:
                text = (
                    f"Annualized volatility measures the expected range of portfolio "
                    f"returns over a year — lower volatility means more predictable outcomes. "
                    f"Volatility decreases from {b_str} to {a_str} ({rel:.1f}% relative reduction), "
                    f"indicating the new positions provide diversification benefit. "
                    f"Their return patterns partially offset existing holdings, "
                    f"smoothing overall portfolio performance."
                )
            else:
                text = (
                    f"Annualized volatility captures the expected swing in portfolio "
                    f"returns — higher values mean wider potential outcomes. "
                    f"Adding these positions increases volatility from {b_str} to {a_str} "
                    f"({rel:.1f}% relative increase), meaning your portfolio becomes "
                    f"less predictable. The new holdings amplify rather than "
                    f"diversify existing risk exposures."
                )
        elif key == "max_drawdown":
            if is_good:
                text = (
                    f"Max drawdown is the largest peak-to-trough decline your portfolio "
                    f"would have experienced historically — it measures worst-case pain. "
                    f"The drawdown improves from {b_str} to {a_str}, meaning the "
                    f"hypothetical portfolio would have lost less during the worst "
                    f"historical period. The new positions provide a cushion "
                    f"during severe market stress."
                )
            else:
                text = (
                    f"Max drawdown captures the deepest historical loss from peak to "
                    f"trough — it shows how much you could lose in the worst scenario. "
                    f"Drawdown deepens from {b_str} to {a_str}, meaning these positions "
                    f"would have amplified losses during the worst historical period. "
                    f"The portfolio becomes more vulnerable to severe "
                    f"market downturns with this allocation."
                )
        elif key == "var_95":
            if is_good:
                text = (
                    f"Value-at-Risk (95%) estimates the maximum daily loss you'd expect "
                    f"to exceed only 5% of trading days — it quantifies tail risk. "
                    f"VaR improves from {b_str} to {a_str}, meaning the expected "
                    f"worst-day losses decrease. The new positions reduce your "
                    f"exposure to extreme single-day adverse moves."
                )
            else:
                text = (
                    f"Value-at-Risk (95%) estimates the daily loss exceeded only 5% of "
                    f"trading days — higher magnitude means greater tail risk exposure. "
                    f"VaR worsens from {b_str} to {a_str}, meaning you face larger "
                    f"potential losses on adverse days. The new positions increase "
                    f"your portfolio's sensitivity to extreme market events."
                )
        elif key == "hhi_concentration":
            if is_good:
                text = (
                    f"The Herfindahl-Hirschman Index measures portfolio concentration — "
                    f"lower HHI means capital is spread more evenly across holdings. "
                    f"HHI drops from {b_str} to {a_str}, indicating better "
                    f"diversification. The new positions reduce your dependence on "
                    f"any single holding, lowering idiosyncratic risk from "
                    f"individual stock moves."
                )
            else:
                text = (
                    f"The Herfindahl-Hirschman Index measures how concentrated your "
                    f"portfolio is — higher HHI means more capital in fewer positions. "
                    f"HHI rises from {b_str} to {a_str}, meaning the portfolio becomes "
                    f"more concentrated. A larger share of your capital depends on "
                    f"fewer stocks, increasing the impact of any single position's "
                    f"underperformance."
                )
        elif key == "annualized_return":
            if is_good:
                text = (
                    f"Annualized return projects the portfolio's expected yearly gain "
                    f"based on historical daily performance. Expected return increases "
                    f"from {b_str} to {a_str}, meaning the new positions have "
                    f"historically outperformed relative to your existing mix. "
                    f"This does not guarantee future returns but indicates "
                    f"favorable backward-looking momentum."
                )
            else:
                text = (
                    f"Annualized return projects expected yearly gain based on historical "
                    f"daily performance. Return decreases from {b_str} to {a_str}, "
                    f"meaning the new positions have historically underperformed relative "
                    f"to your existing portfolio mix. The drag on expected return "
                    f"should be weighed against any diversification or risk "
                    f"benefits the positions provide."
                )
        else:
            text = f"{label} moves from {b_str} to {a_str}."

        observations.append((impact, is_good, key, label, fmt, text))

    # Sort by impact
    observations.sort(key=lambda x: x[0], reverse=True)

    # Split into pros and cons, take top 3 each
    pros = []
    cons = []
    for impact, is_good, key, label, fmt, text in observations:
        entry = {"metric": label, "text": text, "impact": impact, "key": key}
        if is_good and len(pros) < 3:
            pros.append(entry)
        elif not is_good and len(cons) < 3:
            cons.append(entry)

    # ── Build paired rows (pro[i] alongside con[i]) for aligned rendering ──
    max_rows = max(len(pros), len(cons))
    rows = []
    for i in range(max_rows):
        rows.append({
            "pro": pros[i] if i < len(pros) else None,
            "con": cons[i] if i < len(cons) else None,
        })

    # ── Ranked metrics (only material ones) ──
    ranked_metrics = [
        {"key": key, "label": label, "impact": impact, "is_positive": is_good, "fmt": fmt}
        for impact, is_good, key, label, fmt, _ in observations
    ]

    # ── Verdict ──
    pro_impact = sum(p["impact"] for p in pros)
    con_impact = sum(c["impact"] for c in cons)
    n_pros = len(pros)
    n_cons = len(cons)

    if n_pros == 0 and n_cons == 0:
        verdict = "NEUTRAL"
        verdict_detail = "No material metric changes detected at this investment size."
    elif n_cons == 0 or pro_impact > con_impact * 1.5:
        verdict = "FAVORABLE"
        verdict_detail = (
            f"Net positive — {pros[0]['metric']} leads the improvement"
            + (f", supported by {pros[1]['metric']}." if n_pros > 1 else ".")
        )
    elif n_pros == 0 or con_impact > pro_impact * 1.5:
        verdict = "UNFAVORABLE"
        verdict_detail = (
            f"Net negative — {cons[0]['metric']} is the primary concern"
            + (f", compounded by {cons[1]['metric']}." if n_cons > 1 else ".")
        )
    else:
        verdict = "MIXED"
        verdict_detail = (
            f"Trade-off — improvement in {pros[0]['metric']} "
            f"offset by pressure on {cons[0]['metric']}."
        )

    # ── All observations keyed by metric for UI lookup ──
    all_commentary = {
        key: {"text": text, "is_good": is_good, "metric": label}
        for _, is_good, key, label, _, text in observations
    }

    return {
        "rows": rows,
        "ranked_metrics": ranked_metrics,
        "all_commentary": all_commentary,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "is_material": True,
        "pro_count": n_pros,
        "con_count": n_cons,
    }
