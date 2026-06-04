"""
invictus.observability.schema
=============================
SQLite table definitions for the observability system.
"""

TABLES = {
    "agent_runs": """
        CREATE TABLE IF NOT EXISTS agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'success',
            latency_ms REAL,
            error_type TEXT,
            error_message TEXT,
            fallback_used INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "llm_calls": """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            agent_name TEXT NOT NULL,
            ticker TEXT,
            model TEXT,
            prompt_hash TEXT,
            prompt_length INTEGER,
            response_length INTEGER,
            tokens_in INTEGER,
            tokens_out INTEGER,
            latency_ms REAL,
            fallback_used INTEGER DEFAULT 0,
            fallback_reason TEXT,
            sentiment_score REAL,
            confidence_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "ml_predictions": """
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            ticker TEXT NOT NULL,
            accumulation_prob REAL,
            lr_prob REAL,
            rf_prob REAL,
            xgb_prob REAL,
            signal_strength TEXT,
            cv_score REAL,
            n_features INTEGER,
            n_samples INTEGER,
            model_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "conviction_scores": """
        CREATE TABLE IF NOT EXISTS conviction_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            ticker TEXT NOT NULL,
            composite_score REAL,
            outperformance_prob REAL,
            conviction_level TEXT,
            signal_confidence REAL,
            dominant_driver TEXT,
            ci_width REAL,
            ci_mean REAL,
            signal_agreement TEXT,
            bullish_count INTEGER,
            bearish_count INTEGER,
            filing_score REAL,
            earnings_score REAL,
            flow_score REAL,
            ml_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "session_events": """
        CREATE TABLE IF NOT EXISTS session_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            event_type TEXT NOT NULL,
            page TEXT,
            sub_tab TEXT,
            detail TEXT,
            duration_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "data_health": """
        CREATE TABLE IF NOT EXISTS data_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            source TEXT NOT NULL,
            ticker TEXT,
            status TEXT NOT NULL DEFAULT 'success',
            latency_ms REAL,
            records_fetched INTEGER,
            freshness_hours REAL,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "eval_results": """
        CREATE TABLE IF NOT EXISTS eval_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            eval_type TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            detail_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "visitor_log": """
        CREATE TABLE IF NOT EXISTS visitor_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            ip_address TEXT,
            city TEXT,
            region TEXT,
            country TEXT,
            lat REAL,
            lon REAL,
            isp TEXT,
            user_agent TEXT,
            referrer TEXT,
            page TEXT,
            tickers_loaded TEXT,
            pipeline_runs INTEGER DEFAULT 0,
            is_demo INTEGER DEFAULT 0,
            session_start TIMESTAMP,
            last_active TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_agent_runs_name ON agent_runs(agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_agent_runs_ts ON agent_runs(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_llm_calls_agent ON llm_calls(agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_llm_calls_ts ON llm_calls(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_ml_pred_ticker ON ml_predictions(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_conviction_ticker ON conviction_scores(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_session_type ON session_events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_data_health_source ON data_health(source)",
    "CREATE INDEX IF NOT EXISTS idx_eval_results_type ON eval_results(eval_type)",
    "CREATE INDEX IF NOT EXISTS idx_eval_results_ts ON eval_results(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_visitor_ip ON visitor_log(ip_address)",
    "CREATE INDEX IF NOT EXISTS idx_visitor_session ON visitor_log(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_visitor_ts ON visitor_log(created_at)",
]
