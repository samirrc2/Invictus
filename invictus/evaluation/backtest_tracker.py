"""
invictus.evaluation.backtest_tracker
=====================================
Conviction vs Forward Returns backtest engine.
Fetches actual forward returns via yfinance and compares against
logged conviction signals to measure prediction accuracy.

Metrics:
- Hit rate: % of bullish convictions with positive forward returns
- Spearman rank correlation: conviction score vs actual return
- Calibration curve: predicted probability vs actual hit rate by decile
- Hypothetical P&L: equal-weight long/short based on conviction signals

DISCLAIMER: Results are based on limited data and are NOT investment advice.
Past performance does not guarantee future results.
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from invictus.observability.store import query, query_one


def _fetch_forward_returns(
    ticker: str,
    from_date: str,
    horizons: List[int] = None,
) -> Dict[int, Optional[float]]:
    """
    Fetch forward returns for a ticker from a given date.
    Horizons in trading days (5, 10, 30).
    Returns {horizon_days: return_pct} or None if data unavailable.
    """
    if horizons is None:
        horizons = [5, 10, 30]

    try:
        import yfinance as yf
        from datetime import datetime as dt

        # Parse the from_date
        if isinstance(from_date, str):
            # Handle various date formats from SQLite
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
                try:
                    start_dt = dt.strptime(from_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                return {h: None for h in horizons}
        else:
            start_dt = from_date

        # Fetch enough data to cover longest horizon + buffer
        max_horizon = max(horizons)
        end_dt = start_dt + timedelta(days=max_horizon * 2)  # 2x buffer for weekends
        now = datetime.now()
        if end_dt > now:
            end_dt = now

        data = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data is None or data.empty or len(data) < 2:
            return {h: None for h in horizons}

        # Handle multi-level columns from yfinance
        if hasattr(data.columns, 'levels') and data.columns.nlevels > 1:
            close = data["Close"].iloc[:, 0] if data["Close"].shape[1] > 0 else data["Close"]
        else:
            close = data["Close"]

        base_price = close.iloc[0]
        results = {}

        for h in horizons:
            if len(close) > h:
                fwd_price = close.iloc[h]
                results[h] = (fwd_price - base_price) / base_price
            else:
                # Use whatever data we have
                fwd_price = close.iloc[-1]
                results[h] = (fwd_price - base_price) / base_price

        return results

    except Exception:
        return {h: None for h in horizons}


def run_backtest(
    horizons: List[int] = None,
    min_conviction_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Full backtest: conviction scores vs forward returns.

    Steps:
    1. Query all logged conviction scores with timestamps
    2. For each (ticker, timestamp), fetch actual forward returns
    3. Compute hit rate, rank correlation, calibration, hypothetical P&L

    Returns comprehensive backtest results.
    """
    if horizons is None:
        horizons = [5, 10, 30]

    # Get all conviction scores with timestamps
    convictions = query(
        "SELECT ticker, composite_score, outperformance_prob, "
        "  conviction_level, created_at, run_id, "
        "  filing_score, earnings_score, flow_score, ml_score "
        "FROM conviction_scores "
        "WHERE composite_score IS NOT NULL "
        "ORDER BY created_at ASC"
    )

    if not convictions:
        return {"status": "no_data", "message": "No conviction scores logged yet."}

    # Deduplicate: keep the latest conviction per (ticker, run_id)
    seen = set()
    unique_convictions = []
    for c in convictions:
        key = (c["ticker"], c["run_id"])
        if key not in seen:
            seen.add(key)
            unique_convictions.append(c)

    # Fetch forward returns for each conviction
    backtest_rows = []
    skipped = 0

    for conv in unique_convictions:
        ticker = conv["ticker"]
        created = conv["created_at"]

        if not created:
            skipped += 1
            continue

        fwd_returns = _fetch_forward_returns(ticker, created, horizons)

        row = {
            "ticker": ticker,
            "created_at": created,
            "composite_score": conv["composite_score"],
            "outperformance_prob": conv["outperformance_prob"],
            "conviction_level": conv["conviction_level"],
        }

        for h in horizons:
            row[f"fwd_{h}d"] = fwd_returns.get(h)

        backtest_rows.append(row)

    if not backtest_rows:
        return {
            "status": "insufficient_history",
            "message": f"Could not fetch forward returns for any conviction. {skipped} scores skipped (missing timestamps).",
            "total_convictions": len(unique_convictions),
            "skipped": skipped,
        }

    # ── Compute Metrics ──────────────────────────────────────────
    results_by_horizon = {}

    for h in horizons:
        # Filter rows with valid forward returns for this horizon
        valid = [r for r in backtest_rows if r.get(f"fwd_{h}d") is not None]
        if not valid:
            results_by_horizon[h] = {"status": "no_data", "valid_count": 0}
            continue

        # Hit rate: bullish conviction (score > 0) → positive forward return
        bullish = [r for r in valid if r["composite_score"] > min_conviction_threshold]
        bearish = [r for r in valid if r["composite_score"] < -min_conviction_threshold]

        bullish_hits = sum(1 for r in bullish if r[f"fwd_{h}d"] > 0)
        bearish_hits = sum(1 for r in bearish if r[f"fwd_{h}d"] < 0)

        total_directional = len(bullish) + len(bearish)
        total_hits = bullish_hits + bearish_hits
        hit_rate = total_hits / max(total_directional, 1)

        # Spearman rank correlation
        scores = [r["composite_score"] for r in valid]
        returns = [r[f"fwd_{h}d"] for r in valid]
        spearman = _spearman_correlation(scores, returns)

        # Calibration curve (bucket by predicted probability)
        calibration = _compute_calibration(valid, h)

        # Hypothetical P&L
        pnl = _compute_hypothetical_pnl(valid, h, min_conviction_threshold)

        results_by_horizon[h] = {
            "valid_count": len(valid),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "bullish_hit_rate": bullish_hits / max(len(bullish), 1),
            "bearish_hit_rate": bearish_hits / max(len(bearish), 1),
            "overall_hit_rate": hit_rate,
            "spearman_rho": spearman,
            "calibration": calibration,
            "hypothetical_pnl": pnl,
        }

    # Per-ticker breakdown
    ticker_results = _compute_per_ticker(backtest_rows, horizons)

    return {
        "status": "ok",
        "total_convictions": len(unique_convictions),
        "evaluated": len(backtest_rows),
        "skipped_too_recent": skipped,
        "horizons": results_by_horizon,
        "per_ticker": ticker_results,
        "disclaimer": (
            "DISCLAIMER: These backtest results are based on limited historical data "
            "and are NOT investment advice. Past performance does not guarantee future results. "
            "Sample sizes may be too small for statistical significance."
        ),
    }


def _spearman_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation between two lists."""
    if len(x) < 3:
        return 0.0

    def _rank(arr):
        sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks

    rank_x = _rank(x)
    rank_y = _rank(y)
    n = len(x)

    d_sq = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
    return rho


def _compute_calibration(
    rows: List[Dict], horizon: int, n_buckets: int = 5
) -> List[Dict[str, Any]]:
    """
    Calibration curve: bucket predictions by probability, compare to actual hit rate.
    """
    # Use outperformance_prob if available, otherwise convert composite_score
    for r in rows:
        if r.get("outperformance_prob") is not None:
            r["_prob"] = r["outperformance_prob"]
        else:
            # Sigmoid mapping: P = 1/(1+e^{-3C})
            r["_prob"] = 1 / (1 + np.exp(-3 * r["composite_score"]))

    # Sort by predicted probability
    sorted_rows = sorted(rows, key=lambda r: r["_prob"])
    bucket_size = max(1, len(sorted_rows) // n_buckets)

    calibration = []
    for i in range(0, len(sorted_rows), bucket_size):
        bucket = sorted_rows[i:i + bucket_size]
        if not bucket:
            continue
        pred_mean = sum(r["_prob"] for r in bucket) / len(bucket)
        actual_hits = sum(1 for r in bucket if r[f"fwd_{horizon}d"] > 0)
        actual_rate = actual_hits / len(bucket)

        calibration.append({
            "predicted_prob": round(pred_mean, 3),
            "actual_hit_rate": round(actual_rate, 3),
            "count": len(bucket),
            "calibration_error": round(abs(pred_mean - actual_rate), 3),
        })

    return calibration


def _compute_hypothetical_pnl(
    rows: List[Dict],
    horizon: int,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Hypothetical equal-weight long/short P&L.
    Long tickers with conviction > threshold, short those < -threshold.
    """
    trades = []
    for r in rows:
        fwd = r[f"fwd_{horizon}d"]
        if fwd is None:
            continue
        score = r["composite_score"]
        if score > threshold:
            # Long: profit if forward return positive
            trades.append({
                "ticker": r["ticker"],
                "direction": "LONG",
                "conviction": score,
                "return": fwd,
                "pnl_contribution": fwd,  # equal weight
            })
        elif score < -threshold:
            # Short: profit if forward return negative
            trades.append({
                "ticker": r["ticker"],
                "direction": "SHORT",
                "conviction": score,
                "return": fwd,
                "pnl_contribution": -fwd,  # inverted for short
            })

    if not trades:
        return {"status": "no_trades", "total_return": 0}

    total_return = sum(t["pnl_contribution"] for t in trades) / len(trades)
    winners = sum(1 for t in trades if t["pnl_contribution"] > 0)
    win_rate = winners / len(trades)

    # Cumulative P&L curve (ordered by conviction date)
    cumulative = []
    running = 0
    for t in trades:
        running += t["pnl_contribution"] / len(trades)
        cumulative.append(round(running, 6))

    return {
        "total_return": total_return,
        "trade_count": len(trades),
        "long_count": sum(1 for t in trades if t["direction"] == "LONG"),
        "short_count": sum(1 for t in trades if t["direction"] == "SHORT"),
        "win_rate": win_rate,
        "avg_winner": np.mean([t["pnl_contribution"] for t in trades if t["pnl_contribution"] > 0]) if winners else 0,
        "avg_loser": np.mean([t["pnl_contribution"] for t in trades if t["pnl_contribution"] <= 0]) if (len(trades) - winners) else 0,
        "cumulative_pnl": cumulative,
        "best_trade": max(trades, key=lambda t: t["pnl_contribution"]),
        "worst_trade": min(trades, key=lambda t: t["pnl_contribution"]),
    }


def _compute_per_ticker(
    rows: List[Dict],
    horizons: List[int],
) -> List[Dict[str, Any]]:
    """Per-ticker backtest summary."""
    from collections import defaultdict
    ticker_data = defaultdict(list)
    for r in rows:
        ticker_data[r["ticker"]].append(r)

    results = []
    for ticker, entries in ticker_data.items():
        summary = {
            "ticker": ticker,
            "conviction_count": len(entries),
            "avg_conviction": np.mean([e["composite_score"] for e in entries]),
        }
        for h in horizons:
            valid = [e for e in entries if e.get(f"fwd_{h}d") is not None]
            if valid:
                fwd_returns = [e[f"fwd_{h}d"] for e in valid]
                scores = [e["composite_score"] for e in valid]
                bullish = [e for e in valid if e["composite_score"] > 0]
                bullish_wins = sum(1 for e in bullish if e[f"fwd_{h}d"] > 0)
                summary[f"hit_rate_{h}d"] = bullish_wins / max(len(bullish), 1) if bullish else None
                summary[f"avg_fwd_{h}d"] = np.mean(fwd_returns)
                summary[f"count_{h}d"] = len(valid)
            else:
                summary[f"hit_rate_{h}d"] = None
                summary[f"avg_fwd_{h}d"] = None
                summary[f"count_{h}d"] = 0
        results.append(summary)

    return sorted(results, key=lambda r: r["conviction_count"], reverse=True)
