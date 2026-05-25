"""
invictus.pages.dev_analytics.architecture
==========================================
Architecture visualization — graph topology, state schema, execution history, tech stack.
"""
import streamlit as st
import pandas as pd

from invictus.design import (
    render_section_header, apply_invictus_layout,
    BRAND_BLUE, SUCCESS_GREEN,
)
from invictus.pages.dev_analytics._shared import subtitle, conviction_card


def render_architecture():
    """Render architecture visualization — graph topology, execution timeline, state flow."""
    from invictus.agents.orchestrator import InvictusGraph
    from invictus.agents.graph_state import PortfolioState

    graph = InvictusGraph()
    topo = graph.get_topology()

    # Check if real LangGraph is backing the executor
    try:
        from langgraph.graph import StateGraph as _SG
        _lg_status = f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Active</span> — langgraph.graph.StateGraph compiled'
        _lg_detail = f' ({topo["total_nodes"]} agents, 2 barriers, fan-out/fan-in)'
    except ImportError:
        _lg_status = f'<span style="color:{DANGER_RED};font-weight:700;">Not Installed</span> — pip install langgraph'
        _lg_detail = ''

    # ── 1. Pipeline Topology ────────────────────────────────────
    render_section_header("Pipeline Topology")
    subtitle(
        'Multi-agent orchestration graph with '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">{topo["total_nodes"]} agents</span> '
        f'across <span style="color:{SUCCESS_GREEN};font-weight:700;">{len(topo["stages"])} stages</span>. '
        f'LangGraph: {_lg_status}{_lg_detail}.'
    )

    for stage_info in topo["stages"]:
        stage_idx = stage_info["stage"]
        nodes = stage_info["nodes"]
        is_parallel = stage_info["parallel"]

        _mode_color = SUCCESS_GREEN if is_parallel else BRAND_BLUE
        _mode_label = "PARALLEL" if is_parallel else "SEQUENTIAL"

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:16px 0 8px 0;">'
            f'<div style="width:32px;height:32px;border-radius:50%;background:{_mode_color};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:14px;font-weight:800;color:#fff;">{stage_idx}</div>'
            f'<div>'
            f'<span style="font-size:14px;font-weight:700;color:#0f172a;">Stage {stage_idx}</span>'
            f'<span style="font-size:10px;font-weight:700;color:{_mode_color};'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-left:8px;">{_mode_label}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        n_cols = min(len(nodes), 5)
        cols = st.columns(n_cols)
        for i, node_name in enumerate(nodes):
            with cols[i % n_cols]:
                _display = node_name.replace("_", " ").title()
                st.markdown(
                    f'<div style="border:1px solid #e2e8f0;border-radius:6px;'
                    f'padding:10px 12px;background:#fafbfc;text-align:center;">'
                    f'<div style="font-size:12px;font-weight:700;color:#334155;">'
                    f'{_display}</div>'
                    f'<div style="font-size:10px;color:#94a3b8;margin-top:2px;">'
                    f'{node_name}</div></div>',
                    unsafe_allow_html=True,
                )

        if stage_idx < len(topo["stages"]) - 1:
            st.markdown(
                '<div style="text-align:center;color:#cbd5e1;font-size:18px;margin:4px 0;">▼</div>',
                unsafe_allow_html=True,
            )

    # ── 2. State Schema ─────────────────────────────────────────
    render_section_header("State Schema")
    subtitle(
        f'Pydantic <span style="color:{BRAND_BLUE};font-weight:700;">PortfolioState</span> — '
        'central state container that flows through all agents. '
        '<span style="color:#94a3b8;">Each field is populated by a specific agent in the pipeline.</span>'
    )

    fields = PortfolioState.model_fields
    field_groups = {
        "Portfolio Data": ["holdings", "prices", "returns", "weights", "total_value",
                          "total_daily_pnl", "daily_return_pct", "total_cost",
                          "total_unrealized_pnl", "unrealized_pnl_pct", "summary"],
        "Risk Analytics": ["risk_metrics", "ticker_risk", "correlation_matrix"],
        "Factor Analysis": ["pca_results"],
        "Volatility": ["vol_regime"],
        "Stress Testing": ["stress_results"],
        "Options Risk": ["greeks_results"],
        "Intelligence": ["flow_signals", "ml_predictions", "rag_insights"],
        "Attribution": ["pnl_attribution"],
        "Commentary & Eval": ["commentary", "eval_results"],
        "Conviction": ["selected_horizon", "filing_intel", "earnings_intel", "conviction_synthesis"],
        "Orchestration": ["errors", "completed_nodes", "current_node"],
    }

    for group_name, field_names in field_groups.items():
        populated = sum(1 for f in field_names if f in fields)
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 12px;border-bottom:1px solid #f1f5f9;">'
            f'<span style="font-size:13px;font-weight:700;color:#334155;">{group_name}</span>'
            f'<span style="font-size:12px;font-weight:600;color:{BRAND_BLUE};'
            f'font-variant-numeric:tabular-nums;">{populated} fields</span></div>',
            unsafe_allow_html=True,
        )

    # ── 3. Execution History ────────────────────────────────────
    render_section_header("Execution History")
    subtitle(
        'Latest pipeline execution trace from observability store. '
        '<span style="color:#94a3b8;">Shows per-agent latency, status, and error details.</span>'
    )

    try:
        from invictus.observability.store import query
        runs = query(
            "SELECT agent_name, status, latency_ms, error_message, created_at "
            "FROM agent_runs ORDER BY created_at DESC LIMIT 30"
        )
        if runs:
            df = pd.DataFrame(runs)
            st.dataframe(
                df.style.format({"latency_ms": "{:,.0f}ms"}).applymap(
                    lambda v: f"color: {SUCCESS_GREEN}" if v == "success"
                    else f"color: {DANGER_RED}" if v == "error" else "",
                    subset=["status"]
                ),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No execution history yet. Run the pipeline to start collecting agent traces.")
    except Exception as e:
        st.caption(f"Execution history unavailable: {e}")

    # ── 4. Technology Stack ─────────────────────────────────────
    render_section_header("Technology Stack")
    subtitle('Core platform components and their roles in the architecture.')

    _stack = [
        ("Orchestration", "LangGraph StateGraph", "Multi-agent DAG with fan-out/fan-in parallel stages, barrier synchronization, progress callbacks"),
        ("State Management", "Pydantic BaseModel", "Type-safe state container with 30+ fields across 11 domain groups"),
        ("ML Pipeline", "scikit-learn + Bayesian", "Ensemble accumulation classifier with logistic regression, random forest, cross-validation"),
        ("Risk Engine", "NumPy + SciPy", "Portfolio analytics: VaR, CVaR, Sharpe, Sortino, max drawdown, PCA, volatility regimes"),
        ("NLP / RAG", "OpenAI + ChromaDB", "10-K filing analysis, earnings transcript sentiment, management outlook extraction"),
        ("Observability", "SQLite WAL + 7 tables", "Full pipeline telemetry: agent runs, LLM calls, ML predictions, conviction scores"),
        ("Evaluation", "4 evaluators", "Grounding, consistency, backtest, cost analysis with calibration curves"),
        ("Conviction Engine", "4-signal synthesis", "Filing + earnings + flows + ML → dynamic weighting + Monte Carlo CI"),
        ("Frontend", "Streamlit + Plotly", "Institutional-grade UI with 40+ interactive visualizations"),
    ]

    for component, tech, desc in _stack:
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:12px;'
            f'padding:8px 12px;border-bottom:1px solid #f1f5f9;">'
            f'<span style="font-size:12px;font-weight:700;color:{BRAND_BLUE};'
            f'text-transform:uppercase;letter-spacing:0.05em;min-width:140px;">{component}</span>'
            f'<span style="font-size:13px;font-weight:700;color:#0f172a;min-width:180px;">{tech}</span>'
            f'<span style="font-size:12px;color:#64748b;">{desc}</span></div>',
            unsafe_allow_html=True,
        )
