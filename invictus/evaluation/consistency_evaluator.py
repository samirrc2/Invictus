"""
invictus.evaluation.consistency_evaluator
==========================================
Consistency & reliability evaluation.
Measures output stability across runs, cross-agent agreement,
determinism scoring, and composite workflow reliability.
"""
import statistics
from typing import Dict, Any, List, Optional
from invictus.observability.store import query, query_one, count


def analyze_answer_stability() -> Dict[str, Any]:
    """
    For the same ticker across multiple runs, how stable are conviction scores?
    Computes coefficient of variation (CV) per ticker.
    CV < 0.1 = highly stable, CV > 0.3 = unstable.
    """
    # Need at least 2 runs per ticker
    ticker_stats = query(
        "SELECT ticker, COUNT(*) as runs, "
        "  AVG(composite_score) as mean_score, "
        "  MIN(composite_score) as min_score, "
        "  MAX(composite_score) as max_score, "
        "  AVG(outperformance_prob) as mean_prob, "
        "  MIN(outperformance_prob) as min_prob, "
        "  MAX(outperformance_prob) as max_prob "
        "FROM conviction_scores GROUP BY ticker HAVING runs > 1 "
        "ORDER BY runs DESC"
    )

    if not ticker_stats:
        return {"status": "insufficient_data", "message": "Need 2+ runs per ticker"}

    results = []
    for row in ticker_stats:
        # Fetch all scores for this ticker to compute std
        scores = query(
            "SELECT composite_score FROM conviction_scores WHERE ticker = ? "
            "ORDER BY created_at",
            (row["ticker"],)
        )
        score_values = [s["composite_score"] for s in scores if s["composite_score"] is not None]

        if len(score_values) >= 2:
            mean = statistics.mean(score_values)
            std = statistics.stdev(score_values)
            cv = std / abs(mean) if abs(mean) > 1e-6 else 0
        else:
            mean = score_values[0] if score_values else 0
            std = 0
            cv = 0

        score_range = row["max_score"] - row["min_score"] if row["max_score"] and row["min_score"] else 0

        results.append({
            "ticker": row["ticker"],
            "runs": row["runs"],
            "mean_score": mean,
            "std_score": std,
            "cv": cv,
            "score_range": score_range,
            "stability": "HIGH" if cv < 0.1 else "MEDIUM" if cv < 0.3 else "LOW",
            "prob_range": (row["max_prob"] or 0) - (row["min_prob"] or 0),
        })

    # Aggregate stability score
    cvs = [r["cv"] for r in results]
    avg_cv = statistics.mean(cvs) if cvs else 0
    stability_score = max(0, 1.0 - avg_cv)  # higher is better

    return {
        "tickers": results,
        "avg_cv": avg_cv,
        "stability_score": stability_score,
        "stability_grade": "A" if stability_score > 0.9 else "B" if stability_score > 0.7 else "C" if stability_score > 0.5 else "D",
        "most_stable": min(results, key=lambda x: x["cv"])["ticker"] if results else "N/A",
        "least_stable": max(results, key=lambda x: x["cv"])["ticker"] if results else "N/A",
    }


def analyze_cross_agent_consistency() -> Dict[str, Any]:
    """
    Check that the synthesis agent's conviction direction agrees with
    individual signal agents. Flag cases where signals strongly disagree
    with the final conviction.

    Example flag: filing_score = -0.8 (bearish), earnings_score = -0.5 (bearish),
    but composite_score = +0.6 (bullish). Something is wrong.
    """
    data = query(
        "SELECT ticker, run_id, composite_score, conviction_level, "
        "  filing_score, earnings_score, flow_score, ml_score, "
        "  signal_agreement, bullish_count, bearish_count "
        "FROM conviction_scores "
        "WHERE filing_score IS NOT NULL AND earnings_score IS NOT NULL"
    )

    if not data:
        return {"status": "no_data"}

    inconsistencies = []
    total_checked = 0

    for row in data:
        total_checked += 1
        composite = row["composite_score"] or 0
        filing = row["filing_score"] or 0
        earnings = row["earnings_score"] or 0
        flow = row["flow_score"] or 0
        ml = row["ml_score"] or 0

        # Count how many signals agree with composite direction
        composite_bullish = composite > 0
        signals = [filing, earnings, flow, ml]
        signal_directions = [s > 0 for s in signals if abs(s) > 0.1]  # ignore near-zero

        if not signal_directions:
            continue

        agree_count = sum(1 for d in signal_directions if d == composite_bullish)
        agreement_rate = agree_count / len(signal_directions)

        # Flag if majority of signals disagree with composite
        if agreement_rate < 0.5 and abs(composite) > 0.3:
            inconsistencies.append({
                "ticker": row["ticker"],
                "run_id": row["run_id"],
                "composite": composite,
                "composite_direction": "BULLISH" if composite_bullish else "BEARISH",
                "filing": filing,
                "earnings": earnings,
                "flow": flow,
                "ml": ml,
                "agreement_rate": agreement_rate,
                "signal_agreement_label": row["signal_agreement"],
            })

    consistency_rate = 1 - (len(inconsistencies) / max(total_checked, 1))

    return {
        "total_checked": total_checked,
        "inconsistencies": inconsistencies[:20],
        "inconsistency_count": len(inconsistencies),
        "consistency_rate": consistency_rate,
        "grade": "A" if consistency_rate > 0.95 else "B" if consistency_rate > 0.85 else "C" if consistency_rate > 0.70 else "D",
    }


def analyze_determinism() -> Dict[str, Any]:
    """
    Expanded determinism scoring beyond sentiment.
    Checks: conviction level variance, numerical output variance,
    and LLM response length variance for same prompts.
    """
    # 1. LLM determinism — same prompt_hash → response length variance
    llm_det = query(
        "SELECT prompt_hash, agent_name, "
        "  COUNT(*) as runs, "
        "  AVG(response_length) as avg_len, "
        "  MAX(response_length) - MIN(response_length) as len_range, "
        "  COUNT(DISTINCT ROUND(sentiment_score, 2)) as unique_sentiments "
        "FROM llm_calls "
        "WHERE prompt_hash IS NOT NULL "
        "GROUP BY prompt_hash, agent_name HAVING runs > 1 "
        "ORDER BY runs DESC LIMIT 20"
    )

    # 2. Conviction determinism — same ticker across runs
    conv_det = query(
        "SELECT ticker, "
        "  COUNT(*) as runs, "
        "  COUNT(DISTINCT conviction_level) as unique_levels, "
        "  MAX(composite_score) - MIN(composite_score) as score_range "
        "FROM conviction_scores "
        "GROUP BY ticker HAVING runs > 1 "
        "ORDER BY score_range DESC"
    )

    # LLM determinism score
    if llm_det:
        stable_prompts = sum(1 for r in llm_det if r["len_range"] < r["avg_len"] * 0.2)
        llm_determinism = stable_prompts / len(llm_det)
    else:
        llm_determinism = 1.0  # no data = assume deterministic

    # Conviction determinism score
    if conv_det:
        stable_tickers = sum(1 for r in conv_det if r["unique_levels"] <= 1)
        conv_determinism = stable_tickers / len(conv_det)
    else:
        conv_determinism = 1.0

    # Composite
    composite_determinism = (llm_determinism * 0.4 + conv_determinism * 0.6)

    return {
        "llm_determinism": llm_determinism,
        "conviction_determinism": conv_determinism,
        "composite_determinism": composite_determinism,
        "llm_details": llm_det[:10],
        "conviction_details": conv_det[:10],
        "grade": "A" if composite_determinism > 0.9 else "B" if composite_determinism > 0.7 else "C",
    }


def analyze_workflow_reliability() -> Dict[str, Any]:
    """
    Composite workflow reliability score combining:
    - Pipeline completion rate
    - Agent success rate
    - Fallback dependency rate
    - Data fetch success rate
    """
    # Pipeline completion
    starts = count("session_events", "event_type = 'pipeline_start'")
    completes = count("session_events", "event_type = 'pipeline_complete'")
    completion_rate = completes / max(starts, 1)

    # Agent success
    total_agents = count("agent_runs")
    success_agents = count("agent_runs", "status = 'success'")
    agent_success_rate = success_agents / max(total_agents, 1)

    # Fallback dependency
    total_llm = count("llm_calls")
    fallback_llm = count("llm_calls", "fallback_used = 1")
    fallback_rate = fallback_llm / max(total_llm, 1)

    # Data fetch health
    total_fetches = count("data_health")
    success_fetches = count("data_health", "status = 'success'")
    data_success_rate = success_fetches / max(total_fetches, 1)

    # Weighted composite (pipeline completion is most important)
    weights = {
        "completion": 0.35,
        "agent_success": 0.25,
        "no_fallback": 0.20,
        "data_health": 0.20,
    }
    composite = (
        completion_rate * weights["completion"]
        + agent_success_rate * weights["agent_success"]
        + (1 - fallback_rate) * weights["no_fallback"]
        + data_success_rate * weights["data_health"]
    )

    # Error breakdown
    error_types = query(
        "SELECT error_type, COUNT(*) as cnt FROM agent_runs "
        "WHERE error_type IS NOT NULL GROUP BY error_type ORDER BY cnt DESC LIMIT 10"
    )

    return {
        "composite_reliability": composite,
        "pipeline_completion_rate": completion_rate,
        "agent_success_rate": agent_success_rate,
        "fallback_rate": fallback_rate,
        "data_success_rate": data_success_rate,
        "pipeline_starts": starts,
        "pipeline_completes": completes,
        "error_breakdown": {r["error_type"]: r["cnt"] for r in error_types},
        "grade": "A" if composite > 0.95 else "B" if composite > 0.85 else "C" if composite > 0.70 else "D",
    }


def get_full_consistency_report() -> Dict[str, Any]:
    """
    Aggregate all consistency/reliability metrics.
    """
    stability = analyze_answer_stability()
    cross_agent = analyze_cross_agent_consistency()
    determinism = analyze_determinism()
    reliability = analyze_workflow_reliability()

    # Overall score
    scores = []
    if stability.get("stability_score") is not None:
        scores.append(stability["stability_score"])
    if cross_agent.get("consistency_rate") is not None:
        scores.append(cross_agent["consistency_rate"])
    if determinism.get("composite_determinism") is not None:
        scores.append(determinism["composite_determinism"])
    if reliability.get("composite_reliability") is not None:
        scores.append(reliability["composite_reliability"])

    overall = sum(scores) / max(len(scores), 1) if scores else 0

    return {
        "answer_stability": stability,
        "cross_agent_consistency": cross_agent,
        "determinism": determinism,
        "workflow_reliability": reliability,
        "overall_score": overall,
        "overall_grade": "A" if overall > 0.9 else "B" if overall > 0.7 else "C" if overall > 0.5 else "D",
    }
