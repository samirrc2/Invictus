"""
invictus.observability.store
============================
SQLite-based observability data store.
Thread-safe, auto-creates schema on first use.
"""
import sqlite3
import os
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

from invictus.observability.schema import TABLES, INDEXES

# DB lives next to the project root
_DB_DIR = Path(__file__).parent.parent.parent / "data"
_DB_PATH = _DB_DIR / "invictus_observability.db"


def _ensure_db():
    """Create DB directory and schema if needed."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    for table_sql in TABLES.values():
        conn.execute(table_sql)
    for idx_sql in INDEXES:
        conn.execute(idx_sql)
    conn.commit()
    conn.close()


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection. Always ensures schema is up-to-date."""
    _ensure_db()
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def generate_run_id() -> str:
    """Generate a unique run ID for pipeline execution."""
    return str(uuid.uuid4())[:12]


def insert(table: str, data: Dict[str, Any]) -> int:
    """Insert a row into a table. Returns the row ID."""
    conn = get_connection()
    try:
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        cursor = conn.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
            list(data.values()),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_batch(table: str, rows: List[Dict[str, Any]]) -> int:
    """Insert multiple rows. Returns count inserted."""
    if not rows:
        return 0
    conn = get_connection()
    try:
        cols = ", ".join(rows[0].keys())
        placeholders = ", ".join(["?"] * len(rows[0]))
        conn.executemany(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
            [list(r.values()) for r in rows],
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Run a SELECT query and return list of dicts."""
    conn = get_connection()
    try:
        cursor = conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        conn.close()


def query_one(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    """Run a SELECT query and return the first row or None."""
    results = query(sql, params)
    return results[0] if results else None


def count(table: str, where: str = "1=1", params: tuple = ()) -> int:
    """Count rows in a table with optional WHERE clause."""
    result = query_one(f"SELECT COUNT(*) as cnt FROM {table} WHERE {where}", params)
    return result["cnt"] if result else 0


@contextmanager
def timed_operation(table: str, base_data: Dict[str, Any]):
    """
    Context manager that times an operation and inserts the result.
    Usage:
        with timed_operation("agent_runs", {"agent_name": "risk", "run_id": rid}) as ctx:
            do_work()
            ctx["status"] = "success"  # or "error"
    """
    ctx = {**base_data, "status": "success", "error_type": None, "error_message": None}
    start = time.perf_counter()
    try:
        yield ctx
    except Exception as e:
        ctx["status"] = "error"
        ctx["error_type"] = type(e).__name__
        ctx["error_message"] = str(e)[:500]
        raise
    finally:
        ctx["latency_ms"] = (time.perf_counter() - start) * 1000
        try:
            insert(table, ctx)
        except Exception:
            pass  # observability should never crash the app


def get_db_stats() -> Dict[str, int]:
    """Return row counts for all tables."""
    stats = {}
    for table_name in TABLES:
        stats[table_name] = count(table_name)
    return stats
