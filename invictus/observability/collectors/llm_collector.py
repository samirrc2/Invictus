"""
LLM call collector — tracks every OpenAI/LLM call with prompt, response, tokens, latency.
"""
import time
import hashlib
from typing import Dict, Any, Optional
from invictus.observability.store import insert


def log_llm_call(
    agent_name: str,
    ticker: str = None,
    model: str = None,
    prompt: str = "",
    response: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: float = 0,
    fallback_used: bool = False,
    fallback_reason: str = None,
    sentiment_score: float = None,
    confidence_score: float = None,
    run_id: str = None,
):
    """Log a single LLM call with full metadata."""
    try:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16] if prompt else None
        insert("llm_calls", {
            "run_id": run_id,
            "agent_name": agent_name,
            "ticker": ticker,
            "model": model,
            "prompt_hash": prompt_hash,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "fallback_used": 1 if fallback_used else 0,
            "fallback_reason": fallback_reason,
            "sentiment_score": sentiment_score,
            "confidence_score": confidence_score,
        })
    except Exception:
        pass


class LLMTracker:
    """
    Context manager for tracking LLM calls.
    Usage:
        with LLMTracker("earnings_agent", ticker="AAPL") as tracker:
            result = openai_call(...)
            tracker.set_response(result, tokens_in=X, tokens_out=Y)
    """
    def __init__(self, agent_name: str, ticker: str = None, model: str = None, run_id: str = None):
        self.agent_name = agent_name
        self.ticker = ticker
        self.model = model
        self.run_id = run_id
        self.prompt = ""
        self.response = ""
        self.tokens_in = 0
        self.tokens_out = 0
        self.fallback_used = False
        self.fallback_reason = None
        self.sentiment_score = None
        self.confidence_score = None
        self._start = None

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def set_response(self, response: str, tokens_in: int = 0, tokens_out: int = 0):
        self.response = response
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out

    def set_fallback(self, reason: str):
        self.fallback_used = True
        self.fallback_reason = reason

    def set_scores(self, sentiment: float = None, confidence: float = None):
        self.sentiment_score = sentiment
        self.confidence_score = confidence

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = (time.perf_counter() - self._start) * 1000 if self._start else 0
        log_llm_call(
            agent_name=self.agent_name,
            ticker=self.ticker,
            model=self.model,
            prompt=self.prompt,
            response=self.response,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            latency_ms=latency,
            fallback_used=self.fallback_used,
            fallback_reason=self.fallback_reason,
            sentiment_score=self.sentiment_score,
            confidence_score=self.confidence_score,
            run_id=self.run_id,
        )
        return False  # don't suppress exceptions
