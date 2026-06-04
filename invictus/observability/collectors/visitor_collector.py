"""
invictus.observability.collectors.visitor_collector
====================================================
Tracks visitor sessions — IP, geolocation, user agent, pages visited,
pipeline runs, and tickers loaded.

IP geolocation via ip-api.com (free, no key, 45 req/min).
Caches per session to avoid redundant API calls.
"""
import streamlit as st
import uuid
import json
from datetime import datetime, timezone
from typing import Optional

from invictus.observability.store import insert, query, query_one


# ── Session Initialization ────────────────────────────────────────────

def _get_session_id() -> str:
    """Get or create a unique session ID stored in Streamlit session state."""
    if "visitor_session_id" not in st.session_state:
        st.session_state.visitor_session_id = str(uuid.uuid4())[:12]
    return st.session_state.visitor_session_id


def _get_client_ip() -> str:
    """Extract client IP from Streamlit request headers."""
    try:
        headers = st.context.headers
        # Streamlit Cloud uses X-Forwarded-For
        forwarded = headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        # Fallback headers
        for h in ("X-Real-Ip", "Cf-Connecting-Ip", "Remote-Addr"):
            val = headers.get(h, "")
            if val:
                return val.strip()
    except Exception:
        pass
    return "unknown"


def _get_user_agent() -> str:
    """Extract user agent from Streamlit request headers."""
    try:
        return st.context.headers.get("User-Agent", "unknown")[:300]
    except Exception:
        return "unknown"


def _get_referrer() -> str:
    """Extract referrer from Streamlit request headers."""
    try:
        return st.context.headers.get("Referer", "")[:500]
    except Exception:
        return ""


def _geolocate_ip(ip: str) -> dict:
    """
    Geolocate an IP address via ip-api.com (free, no key).
    Returns dict with city, region, country, lat, lon, isp.
    Cached per session so we only call once per visitor.
    """
    if "visitor_geo_cache" in st.session_state:
        return st.session_state.visitor_geo_cache

    geo = {"city": "", "region": "", "country": "", "lat": 0, "lon": 0, "isp": ""}

    if not ip or ip in ("unknown", "127.0.0.1", "::1"):
        st.session_state.visitor_geo_cache = geo
        return geo

    try:
        import urllib.request
        url = f"http://ip-api.com/json/{ip}?fields=city,regionName,country,lat,lon,isp,status"
        req = urllib.request.Request(url, headers={"User-Agent": "Invictus/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "success":
                geo = {
                    "city": data.get("city", ""),
                    "region": data.get("regionName", ""),
                    "country": data.get("country", ""),
                    "lat": data.get("lat", 0),
                    "lon": data.get("lon", 0),
                    "isp": data.get("isp", ""),
                }
    except Exception:
        pass  # geolocation is best-effort, never crash

    st.session_state.visitor_geo_cache = geo
    return geo


# ── Public API ────────────────────────────────────────────────────────

def track_visit(page: str = "landing"):
    """
    Record a visitor session. Call once on app load.
    Subsequent calls for the same session update last_active instead
    of creating duplicate rows.
    """
    session_id = _get_session_id()
    now = datetime.now(timezone.utc).isoformat()

    # Check if session already logged
    existing = query_one(
        "SELECT id, pipeline_runs, tickers_loaded FROM visitor_log WHERE session_id = ?",
        (session_id,),
    )

    if existing:
        # Update last_active and page
        from invictus.observability.store import get_connection
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE visitor_log SET last_active = ?, page = ? WHERE session_id = ?",
                (now, page, session_id),
            )
            conn.commit()
        finally:
            conn.close()
        return

    # First visit for this session — collect everything
    ip = _get_client_ip()
    geo = _geolocate_ip(ip)
    ua = _get_user_agent()
    ref = _get_referrer()

    insert("visitor_log", {
        "session_id": session_id,
        "ip_address": ip,
        "city": geo["city"],
        "region": geo["region"],
        "country": geo["country"],
        "lat": geo["lat"],
        "lon": geo["lon"],
        "isp": geo["isp"],
        "user_agent": ua,
        "referrer": ref,
        "page": page,
        "tickers_loaded": "",
        "pipeline_runs": 0,
        "is_demo": 0,
        "session_start": now,
        "last_active": now,
    })


def track_pipeline_run(is_demo: bool = False, tickers: list = None):
    """Increment pipeline run count for current session."""
    session_id = _get_session_id()
    tickers_str = ",".join(tickers) if tickers else ""
    now = datetime.now(timezone.utc).isoformat()

    from invictus.observability.store import get_connection
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE visitor_log SET pipeline_runs = pipeline_runs + 1, "
            "is_demo = ?, tickers_loaded = ?, last_active = ? "
            "WHERE session_id = ?",
            (1 if is_demo else 0, tickers_str, now, session_id),
        )
        conn.commit()
    except Exception:
        pass  # best-effort
    finally:
        conn.close()


def get_visitor_analytics() -> dict:
    """
    Query visitor analytics for Dev Console display.
    Returns summary stats + recent visitor list.
    """
    total = query_one("SELECT COUNT(*) as cnt FROM visitor_log") or {"cnt": 0}
    unique_ips = query_one(
        "SELECT COUNT(DISTINCT ip_address) as cnt FROM visitor_log "
        "WHERE ip_address != 'unknown'"
    ) or {"cnt": 0}
    unique_countries = query_one(
        "SELECT COUNT(DISTINCT country) as cnt FROM visitor_log "
        "WHERE country != ''"
    ) or {"cnt": 0}
    total_pipeline_runs = query_one(
        "SELECT COALESCE(SUM(pipeline_runs), 0) as cnt FROM visitor_log"
    ) or {"cnt": 0}

    # Recent visitors (last 100)
    recent = query(
        "SELECT session_id, ip_address, city, region, country, isp, "
        "user_agent, referrer, page, tickers_loaded, pipeline_runs, "
        "is_demo, session_start, last_active, created_at "
        "FROM visitor_log ORDER BY created_at DESC LIMIT 100"
    )

    # Top countries
    top_countries = query(
        "SELECT country, COUNT(*) as visits FROM visitor_log "
        "WHERE country != '' GROUP BY country ORDER BY visits DESC LIMIT 10"
    )

    # Top cities
    top_cities = query(
        "SELECT city, country, COUNT(*) as visits FROM visitor_log "
        "WHERE city != '' GROUP BY city, country ORDER BY visits DESC LIMIT 10"
    )

    # Visits by day (last 30 days)
    daily = query(
        "SELECT DATE(created_at) as day, COUNT(*) as visits FROM visitor_log "
        "WHERE created_at >= DATE('now', '-30 days') "
        "GROUP BY DATE(created_at) ORDER BY day"
    )

    return {
        "total_sessions": total["cnt"],
        "unique_ips": unique_ips["cnt"],
        "unique_countries": unique_countries["cnt"],
        "total_pipeline_runs": total_pipeline_runs["cnt"],
        "recent_visitors": recent,
        "top_countries": top_countries,
        "top_cities": top_cities,
        "daily_visits": daily,
    }
