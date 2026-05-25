"""
Session collector — tracks page views, feature usage, pipeline completion.
"""
import time
from typing import Optional
from invictus.observability.store import insert


def log_event(event_type: str, page: str = None, sub_tab: str = None,
              detail: str = None, duration_ms: float = None,
              session_id: str = None):
    """Log a session event."""
    try:
        insert("session_events", {
            "session_id": session_id,
            "event_type": event_type,
            "page": page,
            "sub_tab": sub_tab,
            "detail": detail,
            "duration_ms": duration_ms,
        })
    except Exception:
        pass


def log_page_view(page: str, sub_tab: str = None, session_id: str = None):
    """Log a page view event."""
    log_event("page_view", page=page, sub_tab=sub_tab, session_id=session_id)


def log_pipeline_start(session_id: str = None, mode: str = "manual"):
    """Log pipeline execution start."""
    log_event("pipeline_start", detail=mode, session_id=session_id)


def log_pipeline_complete(session_id: str = None, duration_ms: float = None, mode: str = "manual"):
    """Log pipeline execution completion."""
    log_event("pipeline_complete", detail=mode, duration_ms=duration_ms, session_id=session_id)


def log_feature_use(feature: str, detail: str = None, session_id: str = None):
    """Log a feature usage event (conviction intel run, allocation simulation, etc.)."""
    log_event("feature_use", detail=f"{feature}: {detail}" if detail else feature,
              session_id=session_id)
