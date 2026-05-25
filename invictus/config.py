"""
Invictus Equity Portfolio Intelligence Platform — Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "invictus" / "data"
STATIC_DIR = PROJECT_ROOT / "invictus" / "static"
RAG_DIR = PROJECT_ROOT / "invictus" / "rag"
MODELS_DIR = PROJECT_ROOT / "invictus" / "models"

# Load .env BEFORE reading any keys — guarantees keys are available
# regardless of which module imports config first.
load_dotenv(PROJECT_ROOT / ".env")

# ── API Keys (set via environment or .env) ─────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

# ── Market Data ────────────────────────────────────────────────────────
BENCHMARK_TICKER = "SPY"
RISK_FREE_RATE = 0.05  # annualized, update as needed
TRADING_DAYS_PER_YEAR = 252
LOOKBACK_DAYS = 365  # 1 year of history by default

# ── Predictive Intelligence Settings ──────────────────────────────────
PREDICTIVE_HORIZONS = ["3 months", "6 months", "1 year", "3 years"]
DEFAULT_HORIZON = "1 year"
EARNINGS_LOOKBACK_QUARTERS = 4
FILING_DELTA_QUARTERS = 4

# ── ML Model Settings ─────────────────────────────────────────────────
ML_FORWARD_WINDOW = 20          # Days for forward return labels
ML_LABEL_THRESHOLD = 0.60       # Percentile threshold for positive labels
ML_ENSEMBLE_WEIGHTS = {"lr": 0.25, "rf": 0.35, "xgb": 0.40}
ML_MONTE_CARLO_SIMS = 5000     # Monte Carlo simulations for confidence intervals

# ── Synthesis Engine Settings ─────────────────────────────────────────
SYNTHESIS_BASE_WEIGHTS = {
    "fundamental": 0.35,
    "management": 0.25,
    "flows": 0.25,
    "technical": 0.15,
}

# ── Risk Parameters ────────────────────────────────────────────────────
VAR_CONFIDENCE = 0.95
CVAR_CONFIDENCE = 0.95
PCA_COMPONENTS = 3
VOL_REGIME_CLUSTERS = 3
ROLLING_VOL_WINDOW = 20

# ── Institutional Flow Thresholds ──────────────────────────────────────
INSTITUTIONAL_CHANGE_THRESHOLD = 0.0075  # 0.75%
MUTUAL_FUND_CHANGE_THRESHOLD = 0.0075

# ── LLM Settings ───────────────────────────────────────────────────────
# Provider priority: Gemini (if key present) → OpenAI → Dictionary fallback
LLM_PROVIDER = "gemini" if os.getenv("GEMINI_API_KEY", "") else "openai"
LLM_MODEL_OPENAI = "gpt-4o-mini"
LLM_MODEL_GEMINI = "gemini-2.0-flash"
LLM_MODEL = LLM_MODEL_GEMINI if LLM_PROVIDER == "gemini" else LLM_MODEL_OPENAI
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2000

# ── RAG Settings ───────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_RETRIEVAL = 5

# ── Stress Test Scenarios ──────────────────────────────────────────────
STRESS_SCENARIOS = {
    "COVID Crash (Feb-Mar 2020)": {"start": "2020-02-19", "end": "2020-03-23"},
    "2022 Rate Shock (Jan-Jun 2022)": {"start": "2022-01-03", "end": "2022-06-16"},
    "Tech Drawdown (Nov 2021-Jan 2023)": {"start": "2021-11-19", "end": "2023-01-06"},
    "Semi Selloff (Jul-Oct 2022)": {"start": "2022-07-01", "end": "2022-10-13"},
    "SVB Crisis (Mar 2023)": {"start": "2023-03-08", "end": "2023-03-15"},
}

# ── Default Portfolio ──────────────────────────────────────────────────
DEFAULT_PORTFOLIO = {
    "AAPL": {"shares": 50, "cost_basis": 145.00},
    "AMD": {"shares": 100, "cost_basis": 95.00},
    "META": {"shares": 30, "cost_basis": 280.00},
    "TSLA": {"shares": 40, "cost_basis": 200.00},
    "SMH": {"shares": 60, "cost_basis": 210.00},
}
