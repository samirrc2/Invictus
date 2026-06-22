# INVICTUS

**Institutional-Grade Equity Portfolio Intelligence Platform**

A multi-stage analytical pipeline built on LangGraph that orchestrates 15 specialized compute nodes across a parallel DAG — producing risk analytics, conviction signals, and AI-generated commentary for equity portfolios.

**[Live Demo →](https://invictus-7iskuf67l87iksp8vunmag.streamlit.app/)**

Upload a CSV or try the built-in demo portfolio (AAPL, AMD, META, TSLA, SMH) to see the full pipeline in action.

---

## What It Does

Invictus takes a portfolio of equity holdings (tickers, shares, cost basis) and runs a three-stage analytical pipeline:

**Portfolio Intelligence** — 7 risk dimensions computed in parallel: VaR/CVaR at 95%, Sharpe and Sortino ratios, max drawdown, beta, marginal contribution to risk (MCTR), PCA factor decomposition, K-Means volatility regime detection, stress testing against 5 historical scenarios, Black-Scholes Greeks, and P&L attribution decomposed into market, sector, and idiosyncratic components.

**Conviction Intelligence** — Per-ticker deep analysis across 4 signal sources: institutional flow scoring (insider transactions, fund accumulation trends, smart money concentration), fundamental analysis via yfinance and FMP, management outlook extraction from earnings transcripts and press releases (6 dimensions with credibility gating), and SEC 10-K retrieval-augmented analysis.

**Bayesian Synthesis** — A conviction engine that combines all upstream signals into a single outperformance probability per ticker, with dynamic signal weighting, cross-signal agreement detection, and Monte Carlo confidence intervals.

An AI commentary layer and numerical grounding evaluator sit on top — the LLM generates a portfolio narrative, and the eval harness verifies every number in it traces back to an upstream node output.

---

## Architecture

![Invictus System Architecture](assets/system_architecture.png)

The platform is organized as six layers — a Streamlit presentation tier, a LangGraph orchestration tier, and the analytics, LLM, and data layers beneath it — calling out to FMP, Yahoo Finance, and the Gemini/OpenAI APIs.

Built on **LangGraph StateGraph** with fan-out/fan-in edges and barrier nodes for parallel execution. 15 registered compute nodes + 2 synchronization barriers. Stages 1–2 run up to 10 nodes in parallel; stages 3–7 execute sequentially.

The shared state container (`PortfolioState`) is a Pydantic v2 model — each node reads what it needs and writes its outputs without coupling to other nodes.

---

## Pipeline Nodes

| Node | What It Computes |
|------|-----------------|
| **Risk** | Annualized vol, Sharpe, Sortino, VaR/CVaR (95%), max drawdown, beta, MCTR, correlation matrix |
| **PCA** | Principal component decomposition — identifies hidden factor exposures across holdings |
| **Vol Regime** | K-Means clustering on rolling volatility → Low / Medium / High regimes with transition history |
| **Stress Test** | Replay against 5 historical scenarios (COVID crash, 2022 rate shock, tech drawdown, semi selloff, SVB crisis) |
| **Greeks** | Delta, gamma, vega, theta — Black-Scholes options-implied risk sensitivities |
| **P&L Attribution** | Return decomposition into market, sector, and idiosyncratic components |
| **Flow** | 3-bucket scoring: insider intelligence (0.35), fund accumulation trend (0.40), capital concentration (0.25) |
| **Filing Intel** | Fundamental signals from yfinance financials + FMP analyst grades and estimate revisions |
| **Earnings Intel** | Earnings surprise scoring + management tone analysis via LLM with dictionary fallback |
| **10-K RAG** | TF-IDF retrieval over chunked SEC filings — extracts business drivers, risk factors, moat analysis |
| **Outlook** | Management outlook across 6 dimensions (guidance, capex, margins, competitive, demand, risk) with credibility gating |
| **ML Accumulation** | Bayesian signal model — sequential updating with log-linear Bayes factors over fundamental + technical features |
| **Synthesis** | Dynamic-weighted composite of 4 signal sources → single conviction probability with Monte Carlo CI |
| **Commentary** | LLM-generated portfolio narrative from all upstream signals |
| **Eval** | Numerical grounding checks + cross-node consistency + hallucination detection |

---

## Flow Scoring — Methodology Detail

The institutional flow node is the most methodologically complex module. Three scored sub-components:

**Insider Intelligence (0.35)** — Signal comes from what percentage of their stake an insider transacted, not raw dollar value. A CEO selling $340M of AAPL is noise if it's 1% of their stake. Role-weighted (CEO/CFO = 3×, VP/Director = 2×) with 90-day exponential time decay and materiality-based exit detection.

**Fund Accumulation Trend (0.40)** — Institutional holders classified as smart money (hedge funds), active (stock-picking), or passive (index). Active fund overrides handle active arms of passive parents (e.g., Fidelity Contrafund ≠ passive). Value-weighted breadth scoring replaces naive equal-weighting.

**Capital Concentration (0.25)** — Smart money as fraction of institutional base, centered at 15% = neutral. Confidence fade-in below 5% for mega-caps where passive dominance makes the metric unreliable. Smooth sigmoid replaces hard discontinuity at the 2% threshold.

---

## Evaluation & Observability

Accessible via the Developer Console (`?dev=invictus` URL parameter), the platform includes:

**Evaluation** — Numerical grounding evaluator (verifies LLM numbers trace to upstream outputs, target >85% grounding rate), cross-node consistency analysis, answer stability measurement (coefficient of variation across runs), LLM cost breakdowns, and a walk-forward backtest engine that replays conviction signals against actual forward returns.

**Observability** — 6 telemetry collectors (node latency, LLM tokens, ML drift, conviction signals, data health, sessions) writing to a local SQLite store. 3 diagnostic analyzers (calibration, drift, hallucination) process telemetry into alerts.

**Dev Console** — 11 tabs: Architecture, Node Performance, LLM Quality, ML Monitoring, Conviction Analytics, Conviction Intel, Session Analytics, Data Health, Cost Analysis, Eval Metrics, Backtest.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph StateGraph (fan-out/fan-in DAG) |
| Frontend | Streamlit with custom design system (tokens, components, charts) |
| LLM | Google Gemini 2.0 Flash (primary) → OpenAI GPT-4o-mini (fallback) |
| RAG | TF-IDF retrieval (scikit-learn) over chunked SEC filings |
| ML | Bayesian signal model (SciPy/NumPy) — no sklearn ensemble |
| Quant | NumPy, Pandas, scikit-learn (K-Means, PCA) |
| Market Data | yfinance (prices, holders, options) + FMP API (filings, transcripts, insiders, estimates) |
| Visualization | Plotly, Matplotlib, Seaborn |
| State | Pydantic v2 typed state container |
| Observability | SQLite + custom collectors/analyzers |

---

## Project Structure

```
Invictus/
├── app.py                              # Streamlit entry point — routing shell
├── requirements.txt
├── sample_portfolio.csv
├── LICENSE
├── README.md
├── TECHNICAL_ARCHITECTURE.md
├── .env.example
│
├── scripts/
│   ├── cache_flow_data.py
│   └── download_fmp_demo.py
│
└── invictus/
    ├── __init__.py
    ├── config.py                       # Constants, API keys, thresholds
    ├── llm.py                          # Centralized LLM gateway (Gemini → OpenAI fallback)
    ├── fmp_client.py                   # Shared FMP API client
    │
    ├── agents/                         # 15 compute nodes + orchestrator
    │   ├── orchestrator.py             # LangGraph StateGraph — 15 nodes, 2 barriers, 7 stages
    │   ├── graph_state.py              # Pydantic PortfolioState container
    │   ├── risk_agent.py               # VaR, CVaR, Sharpe, Sortino, drawdown, MCTR
    │   ├── pca_agent.py                # Principal component factor decomposition
    │   ├── vol_regime_agent.py         # K-Means volatility regime detection
    │   ├── stress_agent.py             # 5 historical scenario replay
    │   ├── greeks_agent.py             # Black-Scholes delta, gamma, vega, theta
    │   ├── pnl_agent.py               # Return decomposition (market, sector, idiosyncratic)
    │   ├── flow_agent.py               # 3-bucket institutional flow scoring
    │   ├── filing_agent.py             # Fundamental signals via yfinance + FMP
    │   ├── earnings_agent.py           # Earnings surprise + sentiment analysis
    │   ├── outlook_agent.py            # Management outlook — 6 dimensions + credibility
    │   ├── ml_agent.py                 # Bayesian accumulation signal model (v4)
    │   ├── synthesis_agent.py          # Bayesian conviction synthesis + Monte Carlo
    │   ├── commentary_agent.py         # LLM-generated portfolio narrative
    │   └── hypo_agent.py               # Hypothetical allocation risk engine
    │
    ├── data/
    │   ├── portfolio_loader.py         # CSV parsing, price fetch, state computation
    │   ├── smart_loader.py             # AI-powered CSV column detection
    │   └── demo/                       # Cached FMP data for Streamlit Cloud fallback
    │       ├── manifest.json
    │       ├── aapl/                   # 10 files: analyst_estimates, analyst_grades,
    │       ├── amd/                    #   earnings_calendar, earnings_surprises,
    │       ├── meta/                   #   flow_data, income_statement, insider_trading,
    │       ├── smh/                    #   press_releases, stock_news, transcripts
    │       └── tsla/
    │
    ├── pages/
    │   ├── hypo_simulator.py           # Allocation Engine UI
    │   │
    │   ├── landing/
    │   │   ├── __init__.py             # Landing page router
    │   │   ├── hero.py                 # Hero section + capability cards + topology
    │   │   ├── how_it_works.py         # Interactive screenshot walkthrough
    │   │   ├── _content.py             # Tab content data for How It Works
    │   │   └── _shared.py              # Shared landing page helpers
    │   │
    │   ├── portfolio/
    │   │   ├── __init__.py             # Portfolio Intelligence router
    │   │   ├── dashboard.py            # Overview — holdings, top movers, health
    │   │   ├── risk.py                 # Risk Analytics — VaR, Sharpe, drawdown, MCTR
    │   │   ├── pca.py                  # Factor Decomposition — PCA loadings
    │   │   ├── vol_regime.py           # Volatility Regimes — K-Means clustering
    │   │   ├── stress.py               # Stress Scenarios — 5 historical replays
    │   │   ├── greeks.py               # Sensitivity Analysis — Black-Scholes Greeks
    │   │   ├── attribution.py          # P&L Attribution — return decomposition
    │   │   └── _shared.py
    │   │
    │   ├── conviction/
    │   │   ├── __init__.py             # Conviction Intelligence router
    │   │   ├── engine.py               # Conviction Engine — composite probability
    │   │   ├── flows.py                # Capital Flows — insider + fund + concentration
    │   │   ├── outlook.py              # Management Outlook — 6 dimensions
    │   │   ├── transcript.py           # Transcript Analysis — credibility model
    │   │   ├── _engine_overview.py     # Engine sub-component: conviction cards
    │   │   ├── _engine_signals.py      # Engine sub-component: signal waterfall
    │   │   ├── _engine_confidence.py   # Engine sub-component: Monte Carlo CI
    │   │   └── _shared.py
    │   │
    │   └── dev_analytics/
    │       ├── __init__.py             # Dev Console router
    │       ├── error_log.py            # Structured error display with tracebacks
    │       ├── visitor_log.py          # IP/geolocation session tracking
    │       ├── architecture.py         # LangGraph topology visualization
    │       ├── agent_perf.py           # Node latency + throughput
    │       ├── llm_quality.py          # Grounding rate, cost, determinism
    │       ├── ml_monitoring.py        # Bayesian model drift + calibration
    │       ├── conviction_analytics.py # Signal quality + hit rates
    │       ├── conviction_intel.py     # Per-ticker conviction deep-dive
    │       ├── session.py              # Session analytics + pipeline history
    │       ├── data_health.py          # Data source availability + freshness
    │       ├── cost.py                 # LLM cost breakdown per node
    │       ├── eval_metrics.py         # Grounding + consistency evaluator results
    │       ├── backtest.py             # Walk-forward backtest UI
    │       └── _shared.py
    │
    ├── design/
    │   ├── __init__.py                 # Public exports: inject_styles, render_*, BRAND_*
    │   ├── tokens.py                   # Color palette, spacing, font sizes
    │   ├── styles.py                   # Global CSS injection
    │   ├── components.py               # Metric cards, section headers, badges
    │   ├── charts.py                   # Plotly/Matplotlib chart helpers
    │   ├── formatters.py               # Currency, percentage, delta formatting
    │   └── nav.py                      # Sidebar navigation rendering
    │
    ├── evaluation/
    │   ├── grounding_evaluator.py      # Numerical claim verification
    │   ├── consistency_evaluator.py    # Cross-run stability (CoV)
    │   ├── cost_analyzer.py            # Token usage + cost per node
    │   ├── backtest_tracker.py         # Conviction vs forward returns tracker
    │   └── eval_agent.py              # Eval orchestrator node
    │
    ├── backtest/
    │   ├── config.py                   # Backtest parameters + quality modes
    │   ├── data_loader.py              # Point-in-time historical data loader
    │   ├── runner.py                   # Walk-forward replay engine
    │   └── analyzer.py                 # Hit rates, IC, calibration, P&L curves
    │
    ├── observability/
    │   ├── store.py                    # SQLite WAL-mode connection manager
    │   ├── schema.py                   # Table definitions (agent_runs, llm_calls, visitor_log, …)
    │   ├── collectors/
    │   │   ├── agent_collector.py      # Node latency + success/error logging
    │   │   ├── llm_collector.py        # Token counts, model, prompt/response
    │   │   ├── ml_collector.py         # Bayesian model parameters + drift
    │   │   ├── conviction_collector.py # Signal scores + composite tracking
    │   │   ├── session_collector.py    # Pipeline start/complete + session metadata
    │   │   ├── data_collector.py       # Data source health + freshness
    │   │   └── visitor_collector.py    # IP geolocation + session tracking
    │   └── analyzers/
    │       ├── hallucination.py        # Grounding rate analysis
    │       ├── drift.py                # ML parameter drift detection
    │       └── calibration.py          # Conviction calibration curves
    │
    ├── analytics/
    │   └── tracker.py                  # Session ID generation
    │
    ├── rag/
    │   ├── rag_agent.py                # TF-IDF retrieval over chunked 10-K filings
    │   └── filings/
    │       ├── AAPL_10K.txt
    │       ├── AMD_10K.txt
    │       ├── META_10K.txt
    │       └── TSLA_10K.txt
    │
    └── static/
        ├── logo.png
        └── landing/                    # Screenshots for How It Works walkthrough
            ├── Overview/               (2 screenshots)
            ├── Risk Analytics/         (2 screenshots)
            ├── Factor Decomposition/   (2 screenshots)
            ├── Volatility Regimes/     (1 screenshot)
            ├── Stress Scenarios/       (2 screenshots)
            ├── P&L Attribution/        (2 screenshots)
            ├── Conviction Engine/      (2 screenshots)
            ├── Capital Flows/          (2 screenshots)
            ├── Management Outlook/     (2 screenshots)
            ├── Transcript Analysis/    (2 screenshots)
            └── Run Allocation Simulation/ (2 screenshots)
```

**106 Python files · ~22,800 lines of code**

---

## Running Locally

Requires **Python 3.10+**.

```bash
git clone https://github.com/samirrc2/Invictus.git
cd Invictus

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# API keys — at minimum, one LLM key is needed for AI features
cp .env.example .env
# OPENAI_API_KEY=sk-...       (LLM fallback)
# GEMINI_API_KEY=AI...        (primary LLM)
# FMP_API_KEY=...             (filings, transcripts, insider data)

streamlit run app.py
```

The app opens with a landing page explaining the system architecture and methodology. Click **Demo Mode** to load a pre-configured 5-stock portfolio and run the full pipeline, or upload a CSV with columns `Ticker, Shares, CostBasis`.

---

## Design Decisions

**LangGraph over sequential execution** — The fan-out/fan-in pattern runs 6 risk nodes simultaneously in Stage 1, then 4 conviction nodes in Stage 2. Barrier nodes ensure downstream nodes see complete upstream results.

**Materiality-based insider scoring** — Raw dollar values mislead on mega-caps. Tim Cook selling $105M of AAPL sounds alarming but is 15% of his 0.02% stake — routine 10b5-1 execution. The flow module scores by `tx_pct_of_stake` to separate signal from noise.

**Bayesian synthesis over weighted averages** — Signal weights adjust dynamically based on data quality (confidence gates) and cross-signal agreement. When fundamental and flow signals agree, the composite strengthens; when they conflict, the output is appropriately uncertain.

**Full evaluation harness** — LLM outputs are non-deterministic. The grounding evaluator catches when commentary claims "Sharpe improved to 1.4" but the risk node computed 1.2. Without this, the system would occasionally produce plausible but factually wrong analysis.

---

## Author

**Samir Chincholikar**

[GitHub](https://github.com/samirrc2) · [Email](mailto:samir.chincholikar@gmail.com)
