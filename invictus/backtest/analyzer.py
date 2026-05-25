"""
invictus.backtest.analyzer
============================
Backtest results analyzer — hit rates, Spearman correlation,
calibration curves, hypothetical P&L, per-ticker breakdowns.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy import stats as scipy_stats

from invictus.backtest.config import BacktestConfig


class BacktestAnalyzer:
    """Analyze walk-forward backtest results."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def analyze(self, convictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Full analysis of backtest results.
        Returns dict with per-horizon metrics, per-ticker breakdown,
        calibration data, and hypothetical P&L.
        """
        df = pd.DataFrame(convictions)

        results = {
            "summary": self._summary(df),
            "per_horizon": {},
            "per_ticker": self._per_ticker_analysis(df),
            "conviction_timeline": self._conviction_timeline(df),
            "disclaimer": (
                "Walk-forward backtest using ex-ante data only. "
                "Management and flow signals unavailable (set to neutral). "
                "Actual live performance would include all 4 signal sources. "
                "Past performance does not guarantee future results."
            ),
        }

        for h in self.config.horizons:
            col = f"fwd_{h}d"
            if col in df.columns:
                valid = df.dropna(subset=[col])
                if len(valid) >= 5:
                    results["per_horizon"][h] = self._horizon_analysis(valid, h)

        return results

    # ── Summary ──────────────────────────────────────────────────────

    def _summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """High-level summary statistics."""
        return {
            "total_convictions": len(df),
            "unique_tickers": df["ticker"].nunique(),
            "eval_dates": df["eval_date"].nunique(),
            "avg_probability": float(df["outperformance_prob"].mean()),
            "avg_composite": float(df["composite_score"].mean()),
            "bullish_count": int((df["outperformance_prob"] > 0.55).sum()),
            "bearish_count": int((df["outperformance_prob"] < 0.45).sum()),
            "neutral_count": int(
                ((df["outperformance_prob"] >= 0.45) &
                 (df["outperformance_prob"] <= 0.55)).sum()
            ),
        }

    # ── Per-Horizon Analysis ─────────────────────────────────────────

    def _horizon_analysis(
        self, df: pd.DataFrame, horizon: int
    ) -> Dict[str, Any]:
        """Full analysis for a specific forward-return horizon."""
        col = f"fwd_{horizon}d"
        probs = df["outperformance_prob"].values
        fwd = df[col].values

        # ── Hit Rates ─────────────────────────────────────────
        # Bullish: conviction > 0.55 AND forward return > 0
        bullish_mask = probs > self.config.long_threshold
        bearish_mask = probs < self.config.short_threshold
        neutral_mask = ~bullish_mask & ~bearish_mask

        bullish_hits = (fwd[bullish_mask] > 0).sum() if bullish_mask.sum() > 0 else 0
        bearish_hits = (fwd[bearish_mask] < 0).sum() if bearish_mask.sum() > 0 else 0

        bullish_total = int(bullish_mask.sum())
        bearish_total = int(bearish_mask.sum())

        bullish_hr = bullish_hits / bullish_total if bullish_total > 0 else 0
        bearish_hr = bearish_hits / bearish_total if bearish_total > 0 else 0

        # Overall: correct direction calls
        directional_mask = bullish_mask | bearish_mask
        if directional_mask.sum() > 0:
            correct = (
                (bullish_mask & (fwd > 0)) |
                (bearish_mask & (fwd < 0))
            ).sum()
            overall_hr = correct / directional_mask.sum()
        else:
            overall_hr = 0.5

        # ── Spearman Rank Correlation ─────────────────────────
        if len(probs) >= 5:
            rho, p_val = scipy_stats.spearmanr(probs, fwd)
        else:
            rho, p_val = 0.0, 1.0

        # ── Information Coefficient (Pearson) ─────────────────
        if len(probs) >= 5:
            ic, ic_pval = scipy_stats.pearsonr(probs, fwd)
        else:
            ic, ic_pval = 0.0, 1.0

        # ── Calibration Curve ─────────────────────────────────
        calibration = self._calibration_curve(probs, fwd)

        # ── Hypothetical P&L ──────────────────────────────────
        hypo_pnl = self._hypothetical_pnl(df, horizon)

        # ── Return by Conviction Quintile ─────────────────────
        quintile_returns = self._quintile_analysis(probs, fwd)

        return {
            "horizon_days": horizon,
            "n_observations": len(df),
            "bullish_count": bullish_total,
            "bearish_count": bearish_total,
            "neutral_count": int(neutral_mask.sum()),
            "overall_hit_rate": float(overall_hr),
            "bullish_hit_rate": float(bullish_hr),
            "bearish_hit_rate": float(bearish_hr),
            "spearman_rho": float(rho),
            "spearman_pval": float(p_val),
            "information_coefficient": float(ic),
            "ic_pval": float(ic_pval),
            "avg_fwd_return": float(fwd.mean()),
            "avg_fwd_bullish": float(fwd[bullish_mask].mean()) if bullish_total > 0 else 0,
            "avg_fwd_bearish": float(fwd[bearish_mask].mean()) if bearish_total > 0 else 0,
            "calibration": calibration,
            "hypothetical_pnl": hypo_pnl,
            "quintile_returns": quintile_returns,
        }

    def _calibration_curve(
        self, probs: np.ndarray, fwd: np.ndarray, n_bins: int = 5
    ) -> List[Dict[str, float]]:
        """
        Calibration: group predictions into bins, compare predicted probability
        with actual hit rate (positive forward return).
        """
        if len(probs) < n_bins:
            return []

        # Use quantile-based bins for balanced counts
        try:
            bins = pd.qcut(probs, q=n_bins, duplicates="drop")
        except ValueError:
            return []

        result = []
        for bin_label in sorted(bins.unique()):
            mask = bins == bin_label
            if mask.sum() == 0:
                continue
            predicted = float(probs[mask].mean())
            actual = float((fwd[mask] > 0).mean())
            avg_ret = float(fwd[mask].mean())
            result.append({
                "predicted_prob": predicted,
                "actual_hit_rate": actual,
                "avg_return": avg_ret,
                "count": int(mask.sum()),
                "bin": str(bin_label),
            })

        return result

    def _hypothetical_pnl(
        self, df: pd.DataFrame, horizon: int
    ) -> Dict[str, Any]:
        """
        Hypothetical P&L from a simple long/short strategy.
        Long when conviction > threshold, short when < threshold.
        Equal-weight positions, no leverage.
        """
        col = f"fwd_{horizon}d"
        trades = []

        for _, row in df.iterrows():
            prob = row["outperformance_prob"]
            ret = row[col]
            if pd.isna(ret):
                continue

            if prob > self.config.long_threshold:
                trades.append({
                    "ticker": row["ticker"],
                    "eval_date": row["eval_date"],
                    "direction": "LONG",
                    "conviction": prob,
                    "return": float(ret),
                    "pnl": float(ret) * self.config.position_size,
                })
            elif prob < self.config.short_threshold:
                trades.append({
                    "ticker": row["ticker"],
                    "eval_date": row["eval_date"],
                    "direction": "SHORT",
                    "conviction": prob,
                    "return": float(ret),
                    "pnl": float(-ret) * self.config.position_size,
                })

        if not trades:
            return {"trade_count": 0}

        trade_df = pd.DataFrame(trades)
        pnls = trade_df["pnl"].values
        cum_pnl = np.cumsum(pnls)

        long_trades = trade_df[trade_df["direction"] == "LONG"]
        short_trades = trade_df[trade_df["direction"] == "SHORT"]

        return {
            "trade_count": len(trades),
            "long_count": len(long_trades),
            "short_count": len(short_trades),
            "total_return": float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0,
            "win_rate": float((pnls > 0).mean()),
            "avg_win": float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0,
            "avg_loss": float(pnls[pnls < 0].mean()) if (pnls < 0).any() else 0,
            "profit_factor": float(
                pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum())
            ) if (pnls < 0).any() and pnls[pnls < 0].sum() != 0 else float("inf"),
            "max_drawdown": float(self._max_drawdown(cum_pnl)),
            "sharpe": float(
                pnls.mean() / pnls.std() * np.sqrt(12)  # annualize monthly
            ) if pnls.std() > 0 else 0,
            "cumulative_pnl": cum_pnl.tolist(),
            "trades": trades,
        }

    def _max_drawdown(self, cum_pnl: np.ndarray) -> float:
        """Compute maximum drawdown from cumulative P&L curve."""
        if len(cum_pnl) == 0:
            return 0.0
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - running_max
        return float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    def _quintile_analysis(
        self, probs: np.ndarray, fwd: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Average forward return by conviction quintile."""
        if len(probs) < 10:
            return []

        try:
            quintiles = pd.qcut(probs, q=5, labels=["Q1 (Bear)", "Q2", "Q3", "Q4", "Q5 (Bull)"],
                               duplicates="drop")
        except ValueError:
            return []

        result = []
        for q_label in ["Q1 (Bear)", "Q2", "Q3", "Q4", "Q5 (Bull)"]:
            mask = quintiles == q_label
            if mask.sum() == 0:
                continue
            result.append({
                "quintile": q_label,
                "avg_return": float(fwd[mask].mean()),
                "hit_rate": float((fwd[mask] > 0).mean()),
                "count": int(mask.sum()),
                "avg_conviction": float(probs[mask].mean()),
            })

        return result

    # ── Per-Ticker Analysis ──────────────────────────────────────────

    def _per_ticker_analysis(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Per-ticker conviction accuracy breakdown."""
        results = []

        for ticker in sorted(df["ticker"].unique()):
            tdf = df[df["ticker"] == ticker]
            entry = {
                "ticker": ticker,
                "n_evaluations": len(tdf),
                "avg_conviction": float(tdf["outperformance_prob"].mean()),
                "conviction_std": float(tdf["outperformance_prob"].std()),
                "avg_fundamental": float(tdf["fundamental_score"].mean()),
                "avg_technical": float(tdf["technical_score"].mean()),
            }

            for h in self.config.horizons:
                col = f"fwd_{h}d"
                if col in tdf.columns:
                    valid = tdf.dropna(subset=[col])
                    if len(valid) > 0:
                        fwd = valid[col].values
                        probs = valid["outperformance_prob"].values
                        bullish = probs > self.config.long_threshold
                        bearish = probs < self.config.short_threshold

                        entry[f"avg_fwd_{h}d"] = float(fwd.mean())

                        # Hit rate for directional calls
                        directional = bullish | bearish
                        if directional.sum() > 0:
                            correct = ((bullish & (fwd > 0)) | (bearish & (fwd < 0))).sum()
                            entry[f"hit_rate_{h}d"] = float(correct / directional.sum())
                        else:
                            entry[f"hit_rate_{h}d"] = None
                    else:
                        entry[f"avg_fwd_{h}d"] = None
                        entry[f"hit_rate_{h}d"] = None

            results.append(entry)

        return results

    # ── Conviction Timeline ──────────────────────────────────────────

    def _conviction_timeline(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Monthly conviction heatmap data: eval_date × ticker → probability.
        """
        timeline = []
        for _, row in df.iterrows():
            timeline.append({
                "eval_date": row["eval_date"],
                "ticker": row["ticker"],
                "prob": float(row["outperformance_prob"]),
                "composite": float(row["composite_score"]),
            })
        return timeline
