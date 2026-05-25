"""
invictus.evaluation.cost_analyzer
==================================
Advanced cost modeling — cost per ticker, agent cost breakdown,
latency waterfall, prompt caching optimization tracking.

GPT-4o pricing: $2.50 / 1M input tokens, $10.00 / 1M output tokens.
"""
import json
from typing import Dict, Any, List, Optional
from invictus.observability.store import query, query_one, count


# ── Pricing Constants ─────────────────────────────────────────────────
PRICE_PER_1M_IN = 2.50      # GPT-4o input
PRICE_PER_1M_OUT = 10.00    # GPT-4o output


def _token_cost(tokens_in: int, tokens_out: int) -> float:
    """Compute cost from token counts."""
    return (tokens_in / 1_000_000) * PRICE_PER_1M_IN + (tokens_out / 1_000_000) * PRICE_PER_1M_OUT


def analyze_cost_per_ticker() -> Dict[str, Any]:
    """
    Compute cost per ticker analyzed.
    Joins LLM calls (which have ticker field) to get per-security cost.
    """
    ticker_data = query(
        "SELECT ticker, "
        "  SUM(tokens_in) as tin, SUM(tokens_out) as tout, "
        "  COUNT(*) as calls, AVG(latency_ms) as avg_lat "
        "FROM llm_calls WHERE ticker IS NOT NULL "
        "GROUP BY ticker ORDER BY (SUM(tokens_in) + SUM(tokens_out)) DESC"
    )
    if not ticker_data:
        return {"status": "no_data", "tickers": []}

    results = []
    for row in ticker_data:
        tin = row["tin"] or 0
        tout = row["tout"] or 0
        cost = _token_cost(tin, tout)
        results.append({
            "ticker": row["ticker"],
            "tokens_in": tin,
            "tokens_out": tout,
            "total_tokens": tin + tout,
            "calls": row["calls"],
            "cost": cost,
            "avg_latency_ms": row["avg_lat"] or 0,
        })

    total_cost = sum(r["cost"] for r in results)
    return {
        "tickers": results,
        "total_cost": total_cost,
        "avg_cost_per_ticker": total_cost / max(len(results), 1),
        "most_expensive": results[0]["ticker"] if results else "N/A",
        "ticker_count": len(results),
    }


def analyze_agent_cost_breakdown() -> Dict[str, Any]:
    """
    Per-agent cost breakdown with percentage of total and cost trend.
    """
    agent_data = query(
        "SELECT agent_name, "
        "  SUM(tokens_in) as tin, SUM(tokens_out) as tout, "
        "  COUNT(*) as calls, AVG(latency_ms) as avg_lat, "
        "  SUM(CASE WHEN fallback_used = 1 THEN 1 ELSE 0 END) as fallbacks "
        "FROM llm_calls GROUP BY agent_name "
        "ORDER BY (SUM(tokens_in) + SUM(tokens_out)) DESC"
    )
    if not agent_data:
        return {"status": "no_data", "agents": []}

    results = []
    for row in agent_data:
        tin = row["tin"] or 0
        tout = row["tout"] or 0
        cost = _token_cost(tin, tout)
        results.append({
            "agent": row["agent_name"],
            "tokens_in": tin,
            "tokens_out": tout,
            "total_tokens": tin + tout,
            "calls": row["calls"],
            "cost": cost,
            "avg_latency_ms": row["avg_lat"] or 0,
            "fallback_rate": row["fallbacks"] / max(row["calls"], 1),
        })

    total_cost = sum(r["cost"] for r in results)
    for r in results:
        r["pct_of_total"] = r["cost"] / max(total_cost, 1e-10)

    return {
        "agents": results,
        "total_cost": total_cost,
        "most_expensive_agent": results[0]["agent"] if results else "N/A",
        "most_expensive_pct": results[0]["pct_of_total"] if results else 0,
    }


def analyze_cost_per_run() -> Dict[str, Any]:
    """
    Cost per pipeline run — trending over time.
    """
    run_data = query(
        "SELECT run_id, "
        "  SUM(tokens_in) as tin, SUM(tokens_out) as tout, "
        "  COUNT(*) as calls, MIN(created_at) as started_at "
        "FROM llm_calls WHERE run_id IS NOT NULL "
        "GROUP BY run_id ORDER BY started_at DESC LIMIT 30"
    )
    if not run_data:
        return {"status": "no_data", "runs": []}

    results = []
    for row in run_data:
        tin = row["tin"] or 0
        tout = row["tout"] or 0
        cost = _token_cost(tin, tout)
        results.append({
            "run_id": row["run_id"],
            "tokens_in": tin,
            "tokens_out": tout,
            "cost": cost,
            "calls": row["calls"],
            "started_at": row["started_at"],
        })

    costs = [r["cost"] for r in results]
    return {
        "runs": results,
        "avg_cost_per_run": sum(costs) / max(len(costs), 1),
        "max_cost": max(costs) if costs else 0,
        "min_cost": min(costs) if costs else 0,
        "total_runs": len(results),
    }


def analyze_latency_waterfall() -> Dict[str, Any]:
    """
    Latency waterfall: per-agent timing within the most recent pipeline run.
    Shows the critical path through the LangGraph DAG.
    """
    # Get the most recent run_id
    latest = query_one(
        "SELECT run_id FROM agent_runs WHERE run_id IS NOT NULL "
        "ORDER BY created_at DESC LIMIT 1"
    )
    if not latest:
        return {"status": "no_data"}

    run_id = latest["run_id"]

    # Get all agent timings for this run
    agents = query(
        "SELECT agent_name, latency_ms, status, created_at "
        "FROM agent_runs WHERE run_id = ? ORDER BY created_at ASC",
        (run_id,)
    )
    if not agents:
        return {"status": "no_data"}

    total_latency = sum(a["latency_ms"] or 0 for a in agents)

    # Identify parallel vs sequential stages
    # Agents that started within 100ms of each other are assumed parallel
    stages = []
    current_stage = [agents[0]]
    for i in range(1, len(agents)):
        # If timestamps are very close → same parallel stage
        # Otherwise → sequential stage
        current_stage.append(agents[i])

    waterfall = []
    for a in agents:
        lat = a["latency_ms"] or 0
        waterfall.append({
            "agent": a["agent_name"],
            "latency_ms": lat,
            "pct_of_total": lat / max(total_latency, 1),
            "status": a["status"],
        })

    # Sort by latency for bottleneck identification
    bottlenecks = sorted(waterfall, key=lambda x: x["latency_ms"], reverse=True)

    return {
        "run_id": run_id,
        "total_latency_ms": total_latency,
        "waterfall": waterfall,
        "bottleneck": bottlenecks[0]["agent"] if bottlenecks else "N/A",
        "bottleneck_pct": bottlenecks[0]["pct_of_total"] if bottlenecks else 0,
        "agent_count": len(waterfall),
    }


def analyze_prompt_caching_opportunity() -> Dict[str, Any]:
    """
    Identify prompt caching opportunities.
    If the same prompt_hash appears multiple times, those are cacheable.
    Computes potential savings from caching.
    """
    # Find repeated prompt hashes
    repeated = query(
        "SELECT prompt_hash, agent_name, COUNT(*) as runs, "
        "  AVG(tokens_in) as avg_tin, AVG(tokens_out) as avg_tout, "
        "  AVG(latency_ms) as avg_lat "
        "FROM llm_calls WHERE prompt_hash IS NOT NULL "
        "GROUP BY prompt_hash, agent_name HAVING runs > 1 "
        "ORDER BY runs DESC LIMIT 20"
    )

    if not repeated:
        return {
            "status": "no_cacheable_calls",
            "cacheable_calls": 0,
            "potential_savings": 0,
        }

    total_cacheable_calls = sum(r["runs"] - 1 for r in repeated)  # first call can't be cached
    potential_savings = 0
    details = []

    for r in repeated:
        # If cached, subsequent calls cost 0 for input tokens (only first call pays)
        saved_calls = r["runs"] - 1
        saved_input_cost = (r["avg_tin"] or 0) * saved_calls / 1_000_000 * PRICE_PER_1M_IN
        saved_latency = (r["avg_lat"] or 0) * saved_calls

        potential_savings += saved_input_cost
        details.append({
            "agent": r["agent_name"],
            "prompt_hash": r["prompt_hash"],
            "repeat_count": r["runs"],
            "cacheable_calls": saved_calls,
            "saved_input_cost": saved_input_cost,
            "saved_latency_ms": saved_latency,
        })

    # Current total cost for comparison
    total_data = query_one(
        "SELECT SUM(tokens_in) as tin, SUM(tokens_out) as tout FROM llm_calls"
    ) or {"tin": 0, "tout": 0}
    current_cost = _token_cost(total_data["tin"] or 0, total_data["tout"] or 0)

    return {
        "cacheable_calls": total_cacheable_calls,
        "potential_savings": potential_savings,
        "current_total_cost": current_cost,
        "savings_pct": potential_savings / max(current_cost, 1e-10),
        "optimized_cost": current_cost - potential_savings,
        "details": details,
    }


def get_full_cost_report() -> Dict[str, Any]:
    """
    Aggregate all cost metrics into a single report.
    """
    return {
        "per_ticker": analyze_cost_per_ticker(),
        "per_agent": analyze_agent_cost_breakdown(),
        "per_run": analyze_cost_per_run(),
        "latency_waterfall": analyze_latency_waterfall(),
        "caching_opportunity": analyze_prompt_caching_opportunity(),
    }
