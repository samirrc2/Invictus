"""
Agent execution collector — wraps agent runs to capture timing, status, errors.
"""
import time
from typing import Callable
from invictus.observability.store import insert, generate_run_id


def track_agent_run(agent_name: str, run_id: str = None):
    """
    Decorator that wraps an agent function to track execution.
    Usage:
        @track_agent_run("compute_risk")
        def compute_risk(state): ...
    """
    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            rid = run_id or generate_run_id()
            start = time.perf_counter()
            status = "success"
            error_type = None
            error_msg = None
            fallback = 0

            try:
                result = fn(*args, **kwargs)
                # Check if result has errors
                if hasattr(result, 'errors') and result.errors:
                    for err in result.errors:
                        if agent_name in err:
                            status = "error"
                            error_msg = err[:500]
                return result
            except Exception as e:
                status = "error"
                error_type = type(e).__name__
                error_msg = str(e)[:500]
                raise
            finally:
                latency = (time.perf_counter() - start) * 1000
                try:
                    insert("agent_runs", {
                        "run_id": rid,
                        "agent_name": agent_name,
                        "status": status,
                        "latency_ms": latency,
                        "error_type": error_type,
                        "error_message": error_msg,
                        "fallback_used": fallback,
                    })
                except Exception:
                    pass
        return wrapper
    return decorator


def log_agent_run(agent_name: str, run_id: str, status: str,
                  latency_ms: float, error_type: str = None,
                  error_message: str = None, fallback: bool = False):
    """Manual logging for agents not using the decorator."""
    try:
        insert("agent_runs", {
            "run_id": run_id,
            "agent_name": agent_name,
            "status": status,
            "latency_ms": latency_ms,
            "error_type": error_type,
            "error_message": error_message[:500] if error_message else None,
            "fallback_used": 1 if fallback else 0,
        })
    except Exception:
        pass
