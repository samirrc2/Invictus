"""
Invictus — Lightweight Analytics Tracker
Stores session, page view, and click events in a local SQLite database.
"""
import sqlite3
import uuid
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

DB_PATH = Path(__file__).resolve().parent.parent.parent / "analytics.db"


def _get_conn() -> sqlite3.Connection:
    """Get a SQLite connection, creating tables if needed."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            user_agent TEXT,
            ip_hint TEXT
        );
        CREATE TABLE IF NOT EXISTS page_views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            page TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            duration_sec REAL
        );
        CREATE TABLE IF NOT EXISTS click_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            page TEXT NOT NULL,
            element TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pv_ts ON page_views(timestamp);
        CREATE INDEX IF NOT EXISTS idx_pv_page ON page_views(page);
        CREATE INDEX IF NOT EXISTS idx_ce_page ON click_events(page);
    """)
    return conn


def create_session() -> str:
    """Create a new analytics session, return session_id."""
    sid = str(uuid.uuid4())[:12]
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, started_at, last_active) VALUES (?, ?, ?)",
        (sid, now, now),
    )
    conn.commit()
    conn.close()
    return sid


def track_page_view(session_id: str, page: str, duration_sec: float = 0):
    """Record a page view."""
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO page_views (session_id, page, timestamp, duration_sec) VALUES (?, ?, ?, ?)",
        (session_id, page, now, duration_sec),
    )
    conn.execute(
        "UPDATE sessions SET last_active = ? WHERE session_id = ?",
        (now, session_id),
    )
    conn.commit()
    conn.close()


def track_click(session_id: str, page: str, element: str):
    """Record a click/interaction event."""
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO click_events (session_id, page, element, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, page, element, now),
    )
    conn.commit()
    conn.close()


# ── Aggregation queries for the Dev Console ──────────────────────────


def get_summary_stats(days: int = 30) -> Dict[str, Any]:
    """High-level dashboard numbers."""
    conn = _get_conn()
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    total_sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE started_at >= ?", (cutoff,)
    ).fetchone()[0]

    total_page_views = conn.execute(
        "SELECT COUNT(*) FROM page_views WHERE timestamp >= ?", (cutoff,)
    ).fetchone()[0]

    total_clicks = conn.execute(
        "SELECT COUNT(*) FROM click_events WHERE timestamp >= ?", (cutoff,)
    ).fetchone()[0]

    unique_pages = conn.execute(
        "SELECT COUNT(DISTINCT page) FROM page_views WHERE timestamp >= ?", (cutoff,)
    ).fetchone()[0]

    avg_duration = conn.execute(
        "SELECT AVG(duration_sec) FROM page_views WHERE timestamp >= ? AND duration_sec > 0",
        (cutoff,),
    ).fetchone()[0] or 0

    conn.close()
    return {
        "total_sessions": total_sessions,
        "total_page_views": total_page_views,
        "total_clicks": total_clicks,
        "unique_pages_visited": unique_pages,
        "avg_page_duration_sec": round(avg_duration, 1),
    }


def get_page_popularity(days: int = 30) -> List[Dict]:
    """Page view counts ranked by popularity."""
    conn = _get_conn()
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT page, COUNT(*) as views, COUNT(DISTINCT session_id) as unique_sessions
           FROM page_views WHERE timestamp >= ?
           GROUP BY page ORDER BY views DESC""",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [{"page": r[0], "views": r[1], "unique_sessions": r[2]} for r in rows]


def get_click_heatmap(days: int = 30) -> List[Dict]:
    """Click counts by page + element."""
    conn = _get_conn()
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT page, element, COUNT(*) as clicks
           FROM click_events WHERE timestamp >= ?
           GROUP BY page, element ORDER BY clicks DESC LIMIT 50""",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [{"page": r[0], "element": r[1], "clicks": r[2]} for r in rows]


def get_daily_traffic(days: int = 30) -> List[Dict]:
    """Daily session and page view counts."""
    conn = _get_conn()
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT DATE(timestamp) as day, COUNT(*) as views, COUNT(DISTINCT session_id) as sessions
           FROM page_views WHERE timestamp >= ?
           GROUP BY day ORDER BY day""",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [{"date": r[0], "page_views": r[1], "sessions": r[2]} for r in rows]


def get_session_timeline(limit: int = 50) -> List[Dict]:
    """Recent sessions with their page journey."""
    conn = _get_conn()
    sessions = conn.execute(
        "SELECT session_id, started_at, last_active FROM sessions ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()

    result = []
    for sid, started, last_active in sessions:
        pages = conn.execute(
            "SELECT page, timestamp FROM page_views WHERE session_id = ? ORDER BY timestamp",
            (sid,),
        ).fetchall()
        clicks = conn.execute(
            "SELECT COUNT(*) FROM click_events WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        result.append({
            "session_id": sid,
            "started_at": started,
            "last_active": last_active,
            "pages_visited": [p[0] for p in pages],
            "page_count": len(pages),
            "click_count": clicks,
        })

    conn.close()
    return result
