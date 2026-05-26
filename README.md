# INVICTUS

**Institutional-Grade Portfolio Intelligence Platform**

A multi-agent portfolio analytics system built on LangGraph that orchestrates 14 specialized agents across a parallel DAG pipeline to produce risk metrics, conviction signals, and AI-generated commentary for equity portfolios.

Live demo: [invictus-app.streamlit.app](https://invictus-7iskuf67l87iksp8vunmag.streamlit.app/)

---

## What It Does

Invictus takes a portfolio of equity holdings (tickers, shares, cost basis) and runs a full analytical pipeline:

1. **Portfolio Intelligence** — Risk metrics (VaR, CVaR, Sharpe, Sortino, max drawdown), PCA factor decomposition, volatility regime detection, stress testing against 5 historical scenarios (COVID crash, 2022 rate shock, etc.), portfolio Greeks, and P&L attribution.

2. **Conviction Intelligence** — Institutional flow analysis (insider transactions with stake-context scoring, fund accumulation trends, smart money concentration), 10-K/10-Q filing analysis via RAG, earnings call transcript analysis with credibility gating, and management outlook scoring across 6 dimensions.

3. **Bayesian Synthesis** — A conviction engine that combines fundamental, management, flow, and technical signals into a single outperformance probability per ticker, calibrated via Monte Carlo simulation.

4. **AI Commentary** — LLM-generated narrative that synthesizes all signals into actionable portfolio commentary, evaluated for numerical grounding and cross-agent consistency.

---

## Architecture

```
                    load_portfolio
                         │
           ┌─────────────┴─────────────┐
           │  STAGE 1 (6 parallel)      │
           │  risk · pca · vol_regime   │
           │  stress · greeks · pnl     │
           └─────────────┬─────────────┘
                    (fan-in barrier)
           ┌─────────────┴─────────────┐
           │  STAGE 2 (4 parallel)      │
           │  flows · 10k_rag          │
           │  filing_intel · earnings   │
           └─────────────┬─────────────┘
                    (fan-in barrier)
                         │
                 accumulation_model
                         │
               conviction_synthesis
                         │
               generate_commentary
                         │
              evaluate_commentary
```

Built on **LangGraph StateGraph** with fan-out/fan-in edges and barrier nodes for parallel synchronization. 7 stages, 15 nodes total. Stages 1-2 execute up to 10 agents in parallel; stages 3-7 run sequentially.

The shared state container (`PortfolioState`) is a Pydantic model that accumulates results across nodes — each agent reads what it needs and writes its outputs without coupling to other agents.

---

## Agents

| Agent | What It Computes |
|-------|-----------------|
| **Risk** | Annualized volatility, Sharpe, Sortino, VaR/CVaR (95%), max drawdown, beta, MCTR, correlation matrix |
| **PCA** | Principal component decomposition of portfolio returns — identifies hidden factor exposures |
| **Vol Regime** | K-means clustering on rolling volatility to detect low/normal/high regimes with transition probabilities |
| **Stress Test** | Portfolio replay against 5 historical scenarios (COVID crash, 2022 rate shock, tech drawdown, semi selloff, SVB crisis) with per-ticker attribution |
| **Greeks** | Delta, gamma, vega, theta — options-implied risk sensitivities using Black-Scholes |
| **P&L Attribution** | Return decomposition into market, sector, and idiosyncratic components |
| **Flow** | 3-bucket institutional flow scoring: insider intelligence (0.35), fund accumulation trend (0.40), capital concentration (0.25) |
| **Filing (RAG)** | 10-K/10-Q retrieval-augmented generation — TF-IDF vector retrieval with chunked SEC filings |
| **Filing Intel** | Analyst grades, guidance sentiment, estimate revisions from FMP |
| **Earnings Intel** | Earnings surprise scoring, management tone analysis, call transcript credibility gating |
| **ML Accumulation** | Bayesian signal model combining fundamental + technical features for accumulation probability |
| **Synthesis** | Weighted composite of 4 signal sources into a single conviction score per ticker with Monte Carlo confidence intervals |
| **Commentary** | LLM-generated portfolio narrative from all upstream signals |
| **Eval** | Numerical grounding checks, cross-agent consistency analysis, hallucination detection |

---

## Flow Agent — Scoring Methodology

The flow agent implements a non-trivial scoring model worth highlighting:

**Insider Intelligence (weight: 0.35)** — Materiality-based scoring where signal strength comes from what percentage of their stake an insider transacted, not raw dollar value. A CEO selling $340M of AAPL is noise if it's 1% of their 14% stake — routine 10b5-1 plan execution. Role-weighted (CEO/CFO = 3x, VP/Director = 2x) with 90-day exponential time decay.

**Fund Accumulation Trend (weight: 0.40)** — Institutional holders classified as smart money (hedge funds: Citadel, Renaissance, etc.), active (stock-picking funds), or passive (Vanguard, BlackRock index). Active fund overrides prevent misclassifying active arms of passive parents (e.g., Fidelity Contrafund is not passive). Score = `smart_trend * 0.50 + active_trend * 0.30 + breadth * 0.20`.

**Capital Concentration (weight: 0.25)** — Smart money presence as fraction of institutional base, centered at 15% = neutral. Confidence fade-in below 5% for mega-caps where passive dominance makes the metric unreliable.

---

## Evaluation & Observability

The platform includes a full evaluation and observability stack, accessible via the Developer Console (`?dev=invictus`):

**Evaluation modules** — Numerical grounding (verifies LLM commentary contains accurate numbers traceable to upstream agent outputs), cross-agent consistency analysis, answer stability testing, LLM cost breakdowns with prompt caching opportunity analysis, and a walk-forward backtest engine that replays historical conviction signals against forward returns to measure hit rates and calibration.

**Observability** — 6 telemetry collectors track agent execution latency, LLM token usage, ML prediction drift, conviction signal evolution, data fetch health, and session-level metrics. 3 analyzers (calibration, drift, hallucination) process collected telemetry into actionable diagnostics.

**Dev Console** (11 tabs) — Architecture visualization, agent performance, LLM quality, ML monitoring, conviction analytics, session analytics, data health, cost analysis, eval metrics, and backtest results.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph StateGraph (fan-out/fan-in DAG) |
| Frontend | Streamlit with custom CSS design system |
| LLM | Google Gemini 2.0 Flash (primary), OpenAI GPT-4o-mini (fallback) |
| RAG | LangChain + TF-IDF retrieval (scikit-learn) + OpenAI text-embedding-3-small |
| ML | scikit-learn, XGBoost, SciPy (Bayesian signal model) |
| Quant | NumPy, Pandas, ARCH (GARCH volatility modeling) |
| Market Data | yfinance (prices, institutional holders, insider transactions), FMP API (filings, analyst data) |
| Visualization | Plotly, Matplotlib, Seaborn |
| State Management | Pydantic v2 models with operator.add reducers |

---

## Project Structure

```
invictus/
├── agents/              # 14 agents + orchestrator + state schema
│   ├── orchestrator.py  # LangGraph StateGraph with 7 stages
│   ├── graph_state.py   # Pydantic PortfolioState container
│   ├── risk_agent.py    # VaR, CVaR, Sharpe, Sortino, drawdown
│   ├── flow_agent.py    # 3-bucket institutional flow scoring
│   ├── synthesis_agent.py # Bayesian conviction synthesis
│   └── ...              # 11 more agents
├── pages/
│   ├── landing/         # How It Works — architecture + methodology
│   ├── portfolio/       # 7 tabs: dashboard, risk, PCA, vol, stress, greeks, P&L
│   ├── conviction/      # 4 tabs: engine, flows, outlook, transcripts
│   ├── dev_analytics/   # 11 tabs: architecture through backtest
│   └── hypo_simulator.py # Hypothetical portfolio what-if analysis
├── observability/
│   ├── collectors/      # 6 telemetry collectors
│   └── analyzers/       # 3 diagnostic analyzers
├── evaluation/          # 5 evaluator modules + backtest tracker
├── backtest/            # Walk-forward backtest engine (4 modules)
├── design/              # Streamlit design system (tokens, components, charts)
├── data/
│   ├── demo/            # Cached demo data for Streamlit Cloud fallback
│   └── portfolio_loader.py # CSV/dict portfolio loading + price fetching
├── llm.py               # Unified LLM interface (Gemini + OpenAI)
├── config.py            # All constants, weights, thresholds
└── rag/                 # SEC filing retrieval (TF-IDF + chunking)
app.py                   # Streamlit entry point + pipeline orchestration
```

**104 Python files | ~21,000 lines of code**

---

## Running Locally

```bash
# Clone
git clone https://github.com/samirrc2/Invictus.git
cd Invictus

# Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# API keys (at minimum, one LLM key is needed for AI commentary)
cp .env.example .env
# Edit .env with your keys:
#   OPENAI_API_KEY=sk-...
#   GEMINI_API_KEY=AI...
#   FMP_API_KEY=...        (optional — for filing/earnings data)

# Run
streamlit run app.py
```

The app runs in **demo mode** by default with a pre-configured 5-stock portfolio (AAPL, AMD, META, TSLA, SMH). Upload a CSV with columns `Ticker, Shares, CostBasis` to analyze your own portfolio.

---

## Key Design Decisions

**Why LangGraph over simple sequential execution?** The fan-out/fan-in pattern lets risk, PCA, vol regime, stress, Greeks, and P&L run simultaneously in Stage 1, cutting pipeline latency roughly in half compared to sequential. The barrier nodes ensure downstream agents (synthesis, commentary) see all upstream results before executing.

**Why materiality-based insider scoring?** Raw dollar values are misleading for mega-caps. Tim Cook selling $105M of AAPL sounds alarming but is 15% of his 0.02% stake — routine compensation liquidation. The flow agent scores by `tx_pct_of_stake` to separate signal from noise.

**Why Bayesian synthesis over simple weighted averages?** The synthesis engine dynamically adjusts signal weights based on data quality (confidence gates) and cross-signal agreement. When fundamental and flow signals agree, the composite is stronger than either alone; when they conflict, the output is appropriately uncertain.

**Why a full evaluation harness?** LLM outputs are non-deterministic. The grounding evaluator catches when commentary claims "Sharpe improved to 1.4" but the risk agent computed 1.2. The consistency evaluator catches when the commentary contradicts the flow agent's verdict. Without this, the system would occasionally produce plausible-sounding but factually wrong analysis.

---

## Author

**Samir Chincholikar**

Senior Risk Analytics Engineer · Quantitative Finance · ML/AI

[GitHub](https://github.com/samirrc2) · [Email](mailto:samir.chincholikar@gmail.com)
