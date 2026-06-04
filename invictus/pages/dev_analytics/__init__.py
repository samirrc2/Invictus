"""
invictus.pages.dev_analytics
============================
Dev Analytics — AI Platform Observability & Evaluation Console.
Gated behind ?dev=invictus. Internal only.

Modular package: each sub-tab lives in its own module.
"""
import streamlit as st

from invictus.design import render_section_header, BRAND_BLUE
from invictus.pages.dev_analytics._shared import subtitle, conviction_card


# ── Tab → Module mapping (lazy imports to keep startup fast) ────────────

def render(sub: str):
    """Render the Dev Analytics console for the given sub-tab."""
    from invictus.observability.store import get_db_stats

    db_stats = get_db_stats()
    total_records = sum(db_stats.values())

    # Gate tabs that need data
    _needs_data = sub not in (
        "Error Log", "Visitor Log", "Cost Analytics", "Eval Metrics",
        "Backtest", "Architecture", "Conviction Intelligence",
    )
    if total_records == 0 and _needs_data:
        st.info("No observability data yet. Run the pipeline to start collecting.")
        for table, cnt in db_stats.items():
            st.caption(f"{table}: {cnt} rows")
        return

    # ── Route to sub-module ────────────────────────────────────────
    if sub == "Error Log":
        from invictus.pages.dev_analytics.error_log import render_error_log
        render_error_log()

    elif sub == "Visitor Log":
        from invictus.pages.dev_analytics.visitor_log import render_visitor_log
        render_visitor_log()

    elif sub == "Architecture":
        from invictus.pages.dev_analytics.architecture import render_architecture
        render_architecture()

    elif sub == "Agent Performance":
        from invictus.pages.dev_analytics.agent_perf import render_agent_performance
        render_agent_performance()

    elif sub == "LLM Quality":
        from invictus.pages.dev_analytics.llm_quality import render_llm_quality
        render_llm_quality()

    elif sub == "ML Monitoring":
        from invictus.pages.dev_analytics.ml_monitoring import render_ml_monitoring
        render_ml_monitoring()

    elif sub == "Conviction Analytics":
        from invictus.pages.dev_analytics.conviction_analytics import render_conviction_analytics
        render_conviction_analytics()

    elif sub == "Conviction Intelligence":
        from invictus.pages.dev_analytics.conviction_intel import render_conviction_intelligence
        render_conviction_intelligence()

    elif sub == "Session Analytics":
        from invictus.pages.dev_analytics.session import render_session_analytics
        render_session_analytics()

    elif sub == "Data Health":
        from invictus.pages.dev_analytics.data_health import render_data_health
        render_data_health()

    elif sub == "Cost Analytics":
        from invictus.pages.dev_analytics.cost import render_cost_analytics
        render_cost_analytics()

    elif sub == "Eval Metrics":
        from invictus.pages.dev_analytics.eval_metrics import render_eval_metrics
        render_eval_metrics()

    elif sub == "Backtest":
        from invictus.pages.dev_analytics.backtest import render_backtest
        render_backtest()

    # ── DB Overview (always at bottom) ────────────────────────────
    render_section_header("Observability Database")
    subtitle(
        'SQLite WAL-mode store with '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">{total_records:,}</span> '
        'total records across '
        f'<span style="color:{BRAND_BLUE};font-weight:700;">{len(db_stats)}</span> tables.'
    )
    db_cols = st.columns(min(6, max(len(db_stats), 1)))
    for i, (table, cnt) in enumerate(db_stats.items()):
        with db_cols[i % len(db_cols)]:
            conviction_card(table.replace("_", " ").title(), f"{cnt:,}")
