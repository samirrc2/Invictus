"""
Invictus — LangGraph State Schema
Central state object that flows through the orchestration graph.
Every agent reads from and writes to this shared state.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import pandas as pd


class PortfolioState(BaseModel):
    """Immutable state container passed through the LangGraph workflow."""

    class Config:
        arbitrary_types_allowed = True

    # ── Step 3: Portfolio Data ─────────────────────────────────────────
    holdings: Optional[Any] = Field(default=None, description="Holdings DataFrame")
    prices: Optional[Any] = Field(default=None, description="Price history DataFrame")
    returns: Optional[Any] = Field(default=None, description="Returns DataFrame")
    weights: Optional[Dict[str, float]] = Field(default=None, description="Ticker weights")
    total_value: Optional[float] = None
    total_daily_pnl: Optional[float] = None
    daily_return_pct: Optional[float] = None
    total_cost: Optional[float] = None
    total_unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    summary: Optional[Any] = Field(default=None, description="Portfolio summary DataFrame")

    # ── Step 4: Risk Metrics ───────────────────────────────────────────
    risk_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Portfolio-level risk: vol, VaR, CVaR, Sharpe, Sortino, max_dd, etc.",
    )
    ticker_risk: Optional[Any] = Field(
        default=None, description="Per-ticker risk contribution DataFrame"
    )
    correlation_matrix: Optional[Any] = Field(
        default=None, description="Correlation matrix DataFrame"
    )

    # ── Step 5: PCA ────────────────────────────────────────────────────
    pca_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="PCA components, explained variance, loadings, interpretation",
    )

    # ── Step 6: Volatility Regime ──────────────────────────────────────
    vol_regime: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current regime, regime history, transition timestamps",
    )

    # ── Step 7: Stress Testing ─────────────────────────────────────────
    stress_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-scenario projected loss, sector vulnerability",
    )

    # ── Step 8: Greeks / Options Risk ──────────────────────────────────
    greeks_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Delta, gamma, vega, theta, IV per ticker",
    )

    # ── Step 9: Institutional Flows ────────────────────────────────────
    flow_signals: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Institutional ownership changes, insider activity, signal table",
    )

    # ── Step 10: ML Accumulation ───────────────────────────────────────
    ml_predictions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Accumulation probabilities, feature importance, top picks",
    )

    # ── Step 11: RAG / 10-K ───────────────────────────────────────────
    rag_insights: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-ticker structured insights from 10-K filings",
    )

    # ── Step 12: P&L Attribution ───────────────────────────────────────
    pnl_attribution: Optional[Dict[str, Any]] = Field(
        default=None,
        description="P&L decomposition: ticker, sector, factor, macro, news",
    )

    # ── Step 13: Commentary ────────────────────────────────────────────
    commentary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generated commentary variants: concise, detailed, risk-mgr, PM",
    )

    # ── Step 14: Evaluation ────────────────────────────────────────────
    eval_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Eval scores, best prompt, failure examples, comparison table",
    )

    # ── Step 15: Predictive Intelligence ──────────────────────────────
    selected_horizon: str = Field(default="1 year", description="Predictive horizon")
    filing_intel: Optional[Dict[str, Any]] = Field(
        default=None, description="Structural & directional fundamental signals"
    )
    earnings_intel: Optional[Dict[str, Any]] = Field(
        default=None, description="Management confidence & analyst pressure signals"
    )
    conviction_synthesis: Optional[Dict[str, Any]] = Field(
        default=None, description="Composite conviction scores and probabilities"
    )

    # ── Orchestration Metadata ─────────────────────────────────────────
    errors: List[str] = Field(default_factory=list, description="Errors from any node")
    completed_nodes: List[str] = Field(
        default_factory=list, description="Nodes that have completed"
    )
    current_node: Optional[str] = Field(default=None, description="Currently executing node")

    def mark_complete(self, node_name: str) -> "PortfolioState":
        """Mark a node as completed."""
        self.completed_nodes.append(node_name)
        self.current_node = None
        return self

    def add_error(self, node_name: str, error: str) -> "PortfolioState":
        """Record an error from a node."""
        self.errors.append(f"[{node_name}] {error}")
        return self
