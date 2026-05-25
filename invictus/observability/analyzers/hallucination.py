"""
Hallucination & LLM Quality Analyzer
Detects: sentiment bias, output instability, fallback dependency, numerical grounding issues.
"""
from typing import Dict, Any, List
from invictus.observability.store import query


def analyze_llm_quality() -> Dict[str, Any]:
    """Compute LLM quality metrics from logged calls."""
    total = query("SELECT COUNT(*) as cnt FROM llm_calls")[0]["cnt"]
    if total == 0:
        return {"status": "no_data", "total_calls": 0}

    # Fallback rate
    fallback_count = query("SELECT COUNT(*) as cnt FROM llm_calls WHERE fallback_used = 1")[0]["cnt"]
    fallback_rate = fallback_count / total

    # Sentiment distribution (bias detection)
    sentiment_data = query(
        "SELECT sentiment_score FROM llm_calls WHERE sentiment_score IS NOT NULL"
    )
    sentiments = [r["sentiment_score"] for r in sentiment_data]
    if sentiments:
        import statistics
        sent_mean = statistics.mean(sentiments)
        sent_std = statistics.stdev(sentiments) if len(sentiments) > 1 else 0
        # Bias: if mean is far from 0, LLM is biased
        sentiment_bias = "positive" if sent_mean > 0.3 else "negative" if sent_mean < -0.3 else "neutral"
        # Uniformity: if std is too low, LLM gives same answer every time
        sentiment_uniform = sent_std < 0.1
    else:
        sent_mean = 0
        sent_std = 0
        sentiment_bias = "unknown"
        sentiment_uniform = False

    # Latency stats
    latency_data = query(
        "SELECT AVG(latency_ms) as avg_lat, MAX(latency_ms) as max_lat, "
        "MIN(latency_ms) as min_lat FROM llm_calls"
    )[0]

    # Model distribution
    model_dist = query(
        "SELECT model, COUNT(*) as cnt FROM llm_calls WHERE model IS NOT NULL GROUP BY model"
    )

    # Token usage
    token_data = query(
        "SELECT SUM(tokens_in) as total_in, SUM(tokens_out) as total_out, "
        "AVG(tokens_in) as avg_in, AVG(tokens_out) as avg_out FROM llm_calls"
    )[0]

    # Determinism check: same prompt_hash → different sentiment scores
    determinism = query(
        "SELECT prompt_hash, COUNT(DISTINCT ROUND(sentiment_score, 2)) as unique_scores, "
        "COUNT(*) as runs FROM llm_calls "
        "WHERE prompt_hash IS NOT NULL AND sentiment_score IS NOT NULL "
        "GROUP BY prompt_hash HAVING runs > 1"
    )
    unstable_prompts = [r for r in determinism if r["unique_scores"] > 1]

    return {
        "total_calls": total,
        "fallback_rate": fallback_rate,
        "fallback_count": fallback_count,
        "sentiment_mean": sent_mean,
        "sentiment_std": sent_std,
        "sentiment_bias": sentiment_bias,
        "sentiment_uniform": sentiment_uniform,
        "latency_avg_ms": latency_data["avg_lat"],
        "latency_max_ms": latency_data["max_lat"],
        "model_distribution": {r["model"]: r["cnt"] for r in model_dist},
        "tokens_total_in": token_data["total_in"] or 0,
        "tokens_total_out": token_data["total_out"] or 0,
        "tokens_avg_in": token_data["avg_in"] or 0,
        "tokens_avg_out": token_data["avg_out"] or 0,
        "unstable_prompt_count": len(unstable_prompts),
        "determinism_score": 1 - (len(unstable_prompts) / max(len(determinism), 1)) if determinism else 1.0,
    }
