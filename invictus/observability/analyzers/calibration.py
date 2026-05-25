"""
Calibration Analyzer — agent performance, session analytics, system health.
"""
from typing import Dict, Any, List
from invictus.observability.store import query


def analyze_agent_performance() -> Dict[str, Any]:
    """Analyze agent execution performance — latency, success rates, bottlenecks."""
    total = query("SELECT COUNT(*) as cnt FROM agent_runs")[0]["cnt"]
    if total == 0:
        return {"status": "no_data", "total_runs": 0}

    # Per-agent stats
    agent_stats = query(
        "SELECT agent_name, "
        "  COUNT(*) as runs, "
        "  AVG(latency_ms) as avg_latency, "
        "  MAX(latency_ms) as max_latency, "
        "  MIN(latency_ms) as min_latency, "
        "  SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes, "
        "  SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors, "
        "  SUM(fallback_used) as fallbacks "
        "FROM agent_runs GROUP BY agent_name "
        "ORDER BY avg_latency DESC"
    )

    # Overall success rate
    success_count = query(
        "SELECT COUNT(*) as cnt FROM agent_runs WHERE status = 'success'"
    )[0]["cnt"]

    # Error classification
    error_types = query(
        "SELECT error_type, COUNT(*) as cnt FROM agent_runs "
        "WHERE error_type IS NOT NULL GROUP BY error_type ORDER BY cnt DESC"
    )

    # Pipeline bottleneck (slowest agent)
    bottleneck = agent_stats[0] if agent_stats else None

    # Latency trending (last 20 pipeline runs)
    latency_trend = query(
        "SELECT run_id, SUM(latency_ms) as total_latency, "
        "MIN(created_at) as started_at "
        "FROM agent_runs GROUP BY run_id "
        "ORDER BY started_at DESC LIMIT 20"
    )

    return {
        "total_runs": total,
        "success_rate": success_count / max(total, 1),
        "agent_stats": agent_stats,
        "error_types": {r["error_type"]: r["cnt"] for r in error_types},
        "bottleneck": bottleneck,
        "latency_trend": latency_trend,
    }


def analyze_session_analytics() -> Dict[str, Any]:
    """Analyze user session behavior — page views, feature adoption, pipeline usage."""
    total = query("SELECT COUNT(*) as cnt FROM session_events")[0]["cnt"]
    if total == 0:
        return {"status": "no_data", "total_events": 0}

    # Page view distribution
    page_views = query(
        "SELECT page, sub_tab, COUNT(*) as views "
        "FROM session_events WHERE event_type = 'page_view' "
        "GROUP BY page, sub_tab ORDER BY views DESC"
    )

    # Feature adoption
    features = query(
        "SELECT detail, COUNT(*) as uses "
        "FROM session_events WHERE event_type = 'feature_use' "
        "GROUP BY detail ORDER BY uses DESC"
    )

    # Pipeline stats
    pipeline_starts = query(
        "SELECT COUNT(*) as cnt FROM session_events WHERE event_type = 'pipeline_start'"
    )[0]["cnt"]
    pipeline_completes = query(
        "SELECT COUNT(*) as cnt FROM session_events WHERE event_type = 'pipeline_complete'"
    )[0]["cnt"]
    pipeline_avg_time = query(
        "SELECT AVG(duration_ms) as avg FROM session_events "
        "WHERE event_type = 'pipeline_complete' AND duration_ms IS NOT NULL"
    )[0]["avg"]

    # Demo vs manual
    demo_runs = query(
        "SELECT COUNT(*) as cnt FROM session_events "
        "WHERE event_type = 'pipeline_start' AND detail = 'demo'"
    )[0]["cnt"]

    # Unique sessions
    sessions = query(
        "SELECT COUNT(DISTINCT session_id) as cnt FROM session_events "
        "WHERE session_id IS NOT NULL"
    )[0]["cnt"]

    return {
        "total_events": total,
        "unique_sessions": sessions,
        "page_views": page_views,
        "feature_adoption": features,
        "pipeline_starts": pipeline_starts,
        "pipeline_completes": pipeline_completes,
        "pipeline_completion_rate": pipeline_completes / max(pipeline_starts, 1),
        "pipeline_avg_time_ms": pipeline_avg_time,
        "demo_mode_pct": demo_runs / max(pipeline_starts, 1),
    }


def analyze_data_health() -> Dict[str, Any]:
    """Analyze data pipeline health — API success rates, latency, freshness."""
    total = query("SELECT COUNT(*) as cnt FROM data_health")[0]["cnt"]
    if total == 0:
        return {"status": "no_data", "total_fetches": 0}

    # Per-source stats
    source_stats = query(
        "SELECT source, "
        "  COUNT(*) as fetches, "
        "  AVG(latency_ms) as avg_latency, "
        "  SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes, "
        "  SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors, "
        "  AVG(records_fetched) as avg_records "
        "FROM data_health GROUP BY source"
    )

    # Overall success rate
    success = query(
        "SELECT COUNT(*) as cnt FROM data_health WHERE status = 'success'"
    )[0]["cnt"]

    return {
        "total_fetches": total,
        "success_rate": success / max(total, 1),
        "source_stats": source_stats,
    }
