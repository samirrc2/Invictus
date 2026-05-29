# Invictus — Technical Architecture & Mathematical Framework

**Invictus Equity Portfolio Intelligence Platform**
A multi-stage analytical pipeline for institutional-grade portfolio analytics, conviction synthesis, and allocation simulation.

---

## System Overview

Invictus orchestrates 15 specialized compute nodes through a LangGraph StateGraph pipeline to produce portfolio intelligence across three domains: risk analytics, conviction intelligence, and allocation simulation. The platform processes real-time market data from yfinance and FMP, applies quantitative models, and synthesizes results through a conviction engine that outputs calibrated outperformance probabilities.

### Architecture

```
Data Layer            Compute Layer              Intelligence Layer        Presentation
─────────────       ──────────────────         ────────────────────      ────────────
Yahoo Finance  →    Risk Analytics          →  Portfolio Intelligence →  Streamlit UI
FMP API        →    PCA Factor Decomp       →
               →    Vol Regime Detection    →
               →    Stress Testing          →
               →    Greeks (Black-Scholes)  →
               →    P&L Attribution         →
               →    Flow Scoring            →  Conviction Intelligence
               →    Filing Intel            →
               →    Earnings Intel          →
               →    10-K RAG Retrieval      →
               →    Management Outlook      →
               →    ML Accumulation         →
               →    Conviction Synthesis    →  Allocation Engine
               →    Commentary (LLM)        →
               →    Eval Harness            →  Dev Analytics
```

### Orchestration

The LangGraph StateGraph executes nodes in a staged dependency graph with fan-out/fan-in synchronization:

```
Stage 0: load_portfolio
Stage 1: [compute_risk, run_pca, detect_vol_regime, run_stress_tests, compute_greeks, attribute_pnl]  (parallel)
          ↓ (fan-in barrier)
Stage 2: [analyze_flows, retrieve_10k_context, run_filing_intel, run_earnings_intel]  (parallel)
          ↓ (fan-in barrier)
Stage 3: run_accumulation_model  (depends on flows + filing + earnings)
Stage 4: run_conviction_synthesis  (depends on all intelligence)
Stage 5: generate_commentary
Stage 6: evaluate_commentary
```

15 registered compute nodes + 2 barrier nodes = 17 graph nodes total. Stages 1–2 execute up to 10 nodes in parallel; stages 3–6 run sequentially.

The shared state container (`PortfolioState`) is a Pydantic v2 model that accumulates results across nodes — each node reads what it needs and writes its outputs without coupling to other nodes.

---

## 1. Portfolio Risk Analytics

### 1.1 Return Computation

Daily log returns are computed as:

```
r_t = ln(P_t / P_{t-1})
```

Portfolio returns use a weighted linear combination:

```
R_p = Σ w_i · r_i
```

where `w_i` is the market-value weight of asset `i`, normalized such that `Σ w_i = 1`.

### 1.2 Annualized Volatility

```
σ_annual = σ_daily × √252
```

where `σ_daily` is the standard deviation of daily log returns and 252 is the number of trading days per year.

### 1.3 Value-at-Risk (VaR)

Three independent VaR estimation methods at the 95% confidence level:

**Historical VaR:**
```
VaR_95 = Percentile(R_p, 5%)
```
Non-parametric. Uses the actual 5th percentile of the historical return distribution.

**Parametric (Gaussian) VaR:**
```
VaR_95 = μ + z_0.05 · σ
```
where `z_0.05 = -1.645` is the standard normal quantile.

**Monte Carlo VaR:**
```
VaR_95 = Percentile(Bootstrap(R_p, n=10,000), 5%)
```
Bootstraps 10,000 samples from the empirical return distribution with replacement.

### 1.4 Conditional VaR (Expected Shortfall)

```
CVaR_95 = E[R | R ≤ VaR_95]
```

The average of all returns that fall below the VaR threshold — captures tail risk beyond VaR.

### 1.5 Risk-Adjusted Ratios

**Sharpe Ratio:**
```
S = (R_annual - R_f) / σ_annual
```

**Sortino Ratio:**
```
So = (R_annual - R_f) / σ_downside
```
where `σ_downside` uses only negative returns.

**Calmar Ratio:**
```
C = R_annual / |MaxDrawdown|
```

**Omega Ratio:**
```
Ω = Σ max(R_i - θ, 0) / Σ max(θ - R_i, 0)
```
where `θ = 0` is the threshold return.

### 1.6 Maximum Drawdown

```
Drawdown_t = (Cumulative_t - RunningMax_t) / RunningMax_t
MaxDrawdown = min(Drawdown_t) for all t
```

### 1.7 Marginal Contribution to Risk (MCTR)

```
MCTR_i = w_i · (Σ · w)_i / σ_p
```

where `Σ` is the annualized covariance matrix, `w` is the weight vector, and `σ_p = √(w' · Σ · w)`.

The percentage contribution of each position to total portfolio volatility.

### 1.8 Concentration (HHI)

```
HHI = Σ w_i²
```

Herfindahl-Hirschman Index. Ranges from `1/n` (perfectly diversified) to `1.0` (single position).

### 1.9 Distribution Statistics

**Jarque-Bera Test:**
```
JB = (n/6) · [S² + (K-3)²/4]
```

Tests whether portfolio returns are normally distributed. Rejection (p < 0.05) indicates fat tails or skewness — parametric VaR assumptions may be unreliable.

---

## 2. PCA Factor Decomposition

Principal Component Analysis on the standardized return matrix:

```
X_standardized = StandardScaler(Returns)
PCA(n_components) → eigenvalues, eigenvectors
```

**Explained Variance Ratio:**
```
EVR_k = λ_k / Σ λ_i
```

If PC1 explains >50% of variance, the portfolio is dominated by a single risk factor (e.g., "AI growth" theme). This is flagged as concentration risk.

**Factor Loadings Matrix:** Shows how each ticker loads on each principal component. High loadings on the same factor indicate correlated positions that amplify during drawdowns.

---

## 3. Volatility Regime Detection

### 3.1 Rolling Volatility

```
σ_rolling(t) = std(R_{t-20:t}) × √252
```

20-day rolling window, annualized.

### 3.2 K-Means Clustering

```
KMeans(n_clusters=3).fit(σ_rolling)
```

Clusters the rolling volatility series into three regimes:
- **Low** — stable, trending markets
- **Medium** — normal market conditions
- **High** — elevated risk, potential crisis

Cluster assignment determines the current regime, which feeds into the conviction synthesis engine's dynamic weighting (see Section 7.2).

---

## 4. Stress Testing

Historical scenario replay using actual factor shocks:

```
Stressed_Return_i = β_i · Scenario_Factor_Return
Stressed_Value_i = Current_Value_i × (1 + Stressed_Return_i)
Portfolio_Impact = Σ Stressed_Value_i - Σ Current_Value_i
```

Scenarios: COVID crash (2020), rate shock (2022), tech selloff, semiconductor correction, SVB crisis (2023).

Per-ticker impact identifies the most vulnerable and most resilient positions under each scenario.

---

## 5. Options Sensitivity (Greeks)

Black-Scholes Greeks computed for each ticker using ATM implied volatility:

**Delta:** `∂V/∂S` — price sensitivity to underlying movement

**Gamma:** `∂²V/∂S²` — delta's rate of change (convexity)

**Vega:** `∂V/∂σ` — sensitivity to implied volatility changes

**Theta:** `∂V/∂t` — time decay

**Implied Volatility** is extracted via Brent's root-finding method on the Black-Scholes formula:

```
C(S, K, T, r, σ) = S·N(d₁) - K·e^{-rT}·N(d₂)
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

Portfolio-level Greeks are market-value-weighted sums across all positions.

---

## 6. Bayesian Accumulation Signal Model (v4)

Replaced the earlier sklearn ensemble (LR + RF + XGBoost) with a mathematically transparent Bayesian framework where every parameter has an explicit financial rationale.

### 6.1 Mathematical Framework

Sequential Bayesian updating with log-linear Bayes factors:

```
Prior:      P(accumulation) = 0.5  (uninformative)
Update:     posterior_odds = prior_odds × ∏ BF_i(x_i)
Output:     P(accumulation | data) = posterior_odds / (1 + posterior_odds)
```

Each Bayes factor uses the log-linear form:

```
BF_i(x) = exp(κ_i · g_i(x))
```

where:
- `κ_i` = sensitivity parameter (how strongly this feature discriminates)
- `g_i(x)` = transformation mapping raw feature to signal space `[-1, +1]`

Properties:
- Conjugate to exponential family likelihoods
- Bounded: `BF ∈ [exp(-κ), exp(+κ)]` — no single feature can dominate
- Monotonically increasing in signal strength

### 6.2 κ Calibration

Each κ is calibrated so a "strong" signal (95th percentile) produces a Bayes factor of ~3:

```
κ × g_max ≈ ln(3) ≈ 1.1
```

Four κ tiers: `{0.5, 0.8, 1.1, 1.4}` corresponding to:
- 0.5 = suggestive (BF ≈ 1.5 at strong signal)
- 0.8 = moderate (BF ≈ 2.2)
- 1.1 = strong (BF ≈ 3.0)
- 1.4 = definitive (BF ≈ 4.0)

### 6.3 Feature Specification

19 features across 6 groups:

| Group | Features | κ Range |
|-------|----------|---------|
| Momentum (5) | 5d, 10d, 20d, 60d momentum; 20d relative strength | 0.5–1.1 |
| Technical (4) | RSI-14, MACD histogram, Bollinger %B, directional strength | 0.5–0.8 |
| Microstructure (3) | Return direction accumulation (RDA), price-activity divergence, recovery velocity | 0.5–1.1 |
| Flow (3) | Institutional conviction, insider alignment, capital participation | 1.1–1.4 |
| Fundamental (2) | Fundamental score, management score | 1.1 |
| Structure (2) | Distance from high, drawdown depth | 0.5–0.8 |

Flow features carry the highest κ values because institutional positioning is the most direct evidence of accumulation — the signal the model is designed to detect.

### 6.4 Output

```
P(accumulation | x₁...x₁₉) = σ(log_prior_odds + Σ κᵢ · gᵢ(xᵢ))
```

The posterior probability feeds into the synthesis engine as the "technical" signal channel.

---

## 7. Conviction Synthesis Engine

The core intelligence layer that synthesizes all upstream signals into a calibrated outperformance probability.

### 7.1 Signal Extraction

Four signal channels, each normalized to `[-1, +1]`:

```
S_fundamental = 0.40·conviction + 0.30·guidance + 0.30·risk_signal
S_management  = 0.70·confidence + 0.30·pressure_signal
S_flows       = flow_composite  (from flow node, already in [-1,1])
S_technical   = (bayesian_posterior - 0.5) × 2  → maps [0,1] → [-1,1]
```

Sub-signals with `[0,1]` range (risk, pressure) are transformed via `signal = -(x × 2 - 1)` to center at zero and invert polarity (high risk = bearish).

### 7.2 Dynamic Weighting

Base weights, adjusted by volatility regime:

| Signal | Base | Low Vol Adj | High Vol Adj |
|--------|------|-------------|--------------|
| Fundamental | 0.35 | -0.05 | +0.05 |
| Management | 0.25 | 0.00 | -0.10 |
| Flows | 0.25 | -0.05 | +0.05 |
| Technical | 0.15 | +0.10 | 0.00 |

In high-volatility regimes, management tone becomes less reliable (everyone sounds cautious) while flow and fundamental signals become more informative. In low-vol environments, technical/momentum signals gain weight.

Weights are further degraded by signal quality and always renormalized to sum to 1.0:

```
w_i_adjusted = (w_i_base + regime_adj_i) × q_i / Σ((w_j + adj_j) × q_j)
```

### 7.3 Signal Quality Assessment

Each signal scored for data quality `q_i ∈ [0, 1]`:

```
q_i = f(data_availability, source_confidence, recency)
```

Missing data → `q = 0.2`. Dictionary fallback → `q = 0.5`. Full LLM analysis → `q = 0.8`. ML with high CV → `q = 0.9`.

### 7.4 Signal Agreement Analysis

```
Agreement_Score = (bullish_count - bearish_count) / total_signals
```

Classification:
- `|agreement| > 0.6` → STRONG CONVERGENCE
- `|agreement| > 0.3` → MODERATE AGREEMENT
- `|agreement| > 0` → MIXED SIGNALS
- otherwise → SIGNAL DIVERGENCE

### 7.5 Composite Score

```
C = (Σ S_i × w_i) × M_agreement
```

where `M_agreement = 0.8 + 0.4 × agreement_score` — convergent signals amplify, divergent signals dampen.

### 7.6 Probability Mapping

Calibrated logistic transformation:

```
P(outperformance) = 1 / (1 + e^{-3C})
```

The factor `3` controls sensitivity — a composite score of `±0.5` maps to ~82%/18% probability. Calibrated to produce meaningful spread across the conviction range.

### 7.7 Confidence Adjustment

```
P_final = P × q_avg + 0.5 × (1 - q_avg)
```

When signal quality is low, the probability shrinks toward 50% (no information). With perfect quality, `P_final = P`.

### 7.8 Monte Carlo Confidence Intervals

5,000 simulations with noise proportional to signal uncertainty:

```
For each simulation k:
    For each signal i:
        S_i_perturbed = S_i + N(0, 0.3 × (1 - q_i))
        S_i_perturbed = clip(S_i_perturbed, -1, 1)
    C_k = Σ S_i_perturbed × w_i
    P_k = 1 / (1 + e^{-3·C_k})

CI_5  = Percentile(P_k, 5)
CI_95 = Percentile(P_k, 95)
CI_width = CI_95 - CI_5
```

Narrow CI (< 0.15) → high confidence. Wide CI (> 0.30) → signal uncertainty too high for reliable conviction.

---

## 8. Institutional Flow Scoring

Three scored sub-components per ticker:

### 8.1 Insider Intelligence (weight: 0.35)

Materiality-based scoring — signal strength = percentage of stake transacted, not raw dollar value:

```
materiality = transaction_value / estimated_holdings_value
signal = materiality × role_weight × time_decay
```

Role weights: CEO/CFO = 3×, VP/Director = 2×, Other = 1×. 90-day exponential time decay. Full position exits detected and flagged as high-materiality events.

### 8.2 Fund Accumulation Trend (weight: 0.40)

Institutional holders classified as:
- **Smart money** — Hedge funds (Citadel, Renaissance, etc.)
- **Active** — Stock-picking funds. Active fund overrides handle active arms of passive parents (Fidelity Contrafund ≠ Fidelity Index)
- **Passive** — Vanguard, BlackRock index funds

Score computation:
```
Score = smart_trend × 0.50 + active_trend × 0.30 + breadth × 0.20
```

Breadth uses value-weighted scoring (not equal-weighted) to prevent small funds from having outsized influence.

### 8.3 Capital Concentration (weight: 0.25)

Smart money presence as fraction of institutional base:

```
raw_score = (smart_money_pct - 15%) / 10%  → centered at 15% = neutral
score = sigmoid_smooth(raw_score)
```

Confidence fade-in below 5% for mega-caps where passive dominance makes the metric unreliable. Smooth sigmoid replaces hard discontinuity at the 2% threshold.

---

## 9. Management Outlook & Credibility

Two-stage analysis: WHAT management says × HOW they say it.

### 9.1 Outlook Scoring (6 Dimensions)

LLM analyzes earnings transcripts, press releases, and news to score 6 dimensions on `[-1, +1]`:

| Dimension | Weight | What It Captures |
|-----------|--------|-----------------|
| Demand Environment | 0.25 | Customer demand, order pipelines, revenue visibility |
| Strategic Confidence | 0.20 | Management's investment conviction, capex posture |
| Competitive Positioning | 0.15 | Market share trajectory, moat strength |
| Macro/Industry Outlook | 0.15 | Sector tailwinds/headwinds |
| Headwinds & Tailwinds | 0.15 | Explicit risk vs opportunity balance |
| Investment Thesis Clarity | 0.10 | Forward narrative coherence |

**Dictionary Fallback:** When LLM is unavailable, a curated keyword-based scorer produces meaningful (non-zero) dimension scores using financial vocabulary matching.

### 9.2 Credibility Gating (4 Linguistic Dimensions)

Forensic linguistics analysis — not WHAT they say, but HOW:

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Specificity | 0.30 | Concrete numbers/dates vs platitudes |
| Hedging Density | 0.25 | Assertive vs hedged language |
| Forward/Backward Ratio | 0.25 | Forward-looking vs retrospective |
| Dodge Detection | 0.20 | Evasion of analyst questions |

**Credibility Multiplier:**
```
raw_credibility = Σ dimension_score × weight  → [0, 1]
multiplier = 0.5 + 0.5 × raw_credibility      → [0.5, 1.0]
```

**Final Management Signal:**
```
management_score = outlook_score × credibility_multiplier
```

High outlook + low credibility = muted signal. High outlook + high credibility = amplified signal.

---

## 10. Allocation Engine

### 10.1 Hypothetical Portfolio Construction

Given existing portfolio weights `w` and new investment amounts `a`:

```
w_new_i = (w_old_i × V_total + a_i) / (V_total + Σ a_j)
```

### 10.2 Before/After Risk Comparison

The engine recomputes all risk metrics (Sharpe, volatility, drawdown, VaR, HHI, return) on the hypothetical portfolio and measures deltas:

```
Δ_metric = metric_after - metric_before
Impact_i = |Δ_i| × scale_factor_i
```

Scale factors normalize different metrics to comparable impact units.

### 10.3 Materiality Check

```
Material = (Σ a_i / V_total) > 2%
```

If the investment is less than 2% of portfolio value, the engine returns IMMATERIAL — the allocation is too small to meaningfully impact risk characteristics.

### 10.4 AI Portfolio Fit Analysis

For each metric with material change: direction (improvement/deterioration), magnitude (relative to scale), and contextual commentary. Top 3 improvements and top 3 risks selected by impact ranking.

---

## 11. LLM Integration

### 11.1 Centralized Gateway

All LLM calls route through `invictus/llm.py`:

```
Provider priority:  Gemini 2.0 Flash (if GEMINI_API_KEY present) → OpenAI GPT-4o-mini → raise
```

Three call modes: `call_llm_json()` (parsed dict), `call_llm_text()` (raw string), `call_llm_json_raw()` (JSON string).

### 11.2 Earnings Sentiment Analysis

LLM analyzes news context to extract:
- Management confidence score `[-1, +1]`
- Analyst pressure score `[0, 1]`
- Sentiment trend (Improving / Stable / Deteriorating)
- Tone drivers and analyst concerns

**Fallback:** Dictionary-based sentiment scoring using curated positive/negative keyword lists:
```
confidence = (pos_hits - neg_hits) / normalization_factor
pressure = pressure_hits / normalization_factor
```

### 11.3 AI Commentary Generation

LLM-generated portfolio narrative from structured data. Prompt explicitly instructs against hallucination.

### 11.4 Smart CSV Loader

LLM extracts portfolio holdings from arbitrary brokerage CSV formats. Fallback: pandas-based column name matching with common brokerage naming patterns.

---

## 12. Observability System

### 12.1 Architecture

```
Collectors (6)  →  SQLite Store  →  Analyzers (3)  →  Dev Console (11 tabs)
```

**Collectors:** Node execution latency, LLM token usage, ML prediction tracking, conviction score evolution, session events, data fetch health.

**Analyzers:** Hallucination detection (sentiment bias, determinism, fallback rates), drift analysis (prediction stability, conviction stability), calibration (node performance, data pipeline health).

### 12.2 LLM Quality Metrics

- **Fallback Rate** — percentage of LLM calls that fell through to dictionary-based fallback
- **Sentiment Bias** — mean sentiment across all calls (detects systematic positive/negative lean)
- **Determinism Score** — same prompt hash → same output? Measures output stability
- **Token Economics** — cost per call, cost per pipeline run, per-node consumption

### 12.3 ML Monitoring

- **Prediction Confidence Distribution** — ratio of high-confidence (>70%) vs low-confidence (40–60%) predictions
- **Ticker Stability** — same ticker across multiple runs → how much does the prediction vary?
- **Bayes Factor Analysis** — which features contributed most to the posterior for each ticker

---

## 13. Evaluation Framework

### 13.1 Numerical Grounding Evaluator

Cross-references numbers in LLM commentary against actual `PortfolioState` data. Extracts percentages, dollar amounts, decimals, and basis points from text, then matches against upstream node outputs. Target grounding rate: >85%.

### 13.2 Consistency Evaluator

Measures output stability across runs via coefficient of variation (CV) per ticker:
```
CV = σ(scores) / |μ(scores)|
```
CV < 0.1 = highly stable. CV > 0.3 = unstable.

### 13.3 Walk-Forward Backtest

For each evaluation date, runs a lightweight conviction mini-pipeline using ONLY data available as of that date:
1. Compute fundamental signals from point-in-time financials
2. Run Bayesian ML model on point-in-time price features
3. Synthesize into conviction probability
4. Compare against actual forward returns

LLM-dependent nodes are skipped (cost ~$50+ for 120 dates; historical transcripts not point-in-time accessible). Analyzes hit rates, calibration, and simulated P&L.

### 13.4 Cost Analyzer

Token economics, prompt caching opportunity analysis, cost-per-pipeline-run breakdown by node.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph StateGraph (fan-out/fan-in DAG with barrier nodes) |
| Frontend | Streamlit with custom design system (tokens, components, formatters, charts) |
| LLM | Google Gemini 2.0 Flash (primary) → OpenAI GPT-4o-mini (fallback) |
| RAG | TF-IDF retrieval (scikit-learn) over chunked SEC 10-K filings |
| ML | Bayesian signal model (NumPy/SciPy) — sequential updating with log-linear Bayes factors |
| Quant | NumPy, Pandas, scikit-learn (K-Means, PCA), SciPy (Jarque-Bera, Brent root-finding) |
| Market Data | yfinance (prices, holders, options) + FMP API (filings, transcripts, insiders, estimates) |
| Visualization | Plotly, Matplotlib, Seaborn |
| State | Pydantic v2 typed state container (`PortfolioState`) |
| Observability | SQLite + 6 custom collectors + 3 analyzers |
| Simulation | Monte Carlo (NumPy, 5,000 iterations for CI, 10,000 for VaR) |

---

## File Structure

```
app.py                              — Thin Streamlit routing shell
invictus/
  agents/                           — 15 compute nodes + orchestrator + state
    orchestrator.py                 — LangGraph StateGraph — 15 nodes, 2 barriers
    graph_state.py                  — Pydantic v2 PortfolioState container
    risk_agent.py                   — VaR, CVaR, Sharpe, Sortino, MCTR, drawdown
    pca_agent.py                    — PCA factor decomposition
    vol_regime_agent.py             — K-Means regime detection
    stress_agent.py                 — Historical scenario replay
    greeks_agent.py                 — Black-Scholes Greeks
    pnl_agent.py                    — P&L attribution
    flow_agent.py                   — 3-bucket institutional flow scoring
    filing_agent.py                 — Fundamental intelligence (yfinance + FMP)
    earnings_agent.py               — LLM sentiment analysis + dictionary fallback
    outlook_agent.py                — 6-dimension management outlook + credibility gate
    ml_agent.py                     — Bayesian accumulation signal model (v4)
    synthesis_agent.py              — Conviction synthesis + Monte Carlo CI
    commentary_agent.py             — LLM commentary generation
    hypo_agent.py                   — Allocation simulation engine
  pages/
    landing/                        — How It Works — architecture + methodology
    portfolio/                      — 7 sub-tabs: dashboard, risk, PCA, vol, stress, greeks, P&L
    conviction/                     — 4 sub-tabs: engine, flows, outlook, transcripts
    dev_analytics/                  — 11 sub-tabs: architecture through backtest
    hypo_simulator.py               — Allocation Engine UI
  observability/
    collectors/                     — 6 telemetry collectors
    analyzers/                      — 3 diagnostic analyzers
    store.py                        — SQLite interface
    schema.py                       — Table definitions
  evaluation/
    grounding_evaluator.py          — Numerical accuracy checks
    consistency_evaluator.py        — Cross-run stability analysis
    cost_analyzer.py                — Token economics
    backtest_tracker.py             — Conviction vs forward returns
  backtest/
    config.py                       — Backtest parameters
    data_loader.py                  — Point-in-time historical data
    runner.py                       — Walk-forward engine
    analyzer.py                     — Hit rate, calibration, P&L analysis
  design/                           — Design system (tokens, components, charts, nav)
  rag/                              — SEC 10-K retrieval (TF-IDF + chunking)
  data/
    demo/                           — Cached demo data for Streamlit Cloud fallback
    portfolio_loader.py             — CSV/dict loading + price fetching
    smart_loader.py                 — LLM-assisted CSV parsing
  llm.py                            — Centralized LLM gateway (Gemini → OpenAI)
  fmp_client.py                     — Shared FMP API client
  config.py                         — All constants, API keys, thresholds
```

**106 Python files · ~21,700 lines of code**

---

*Built by Samir Chincholikar. Platform architecture, pipeline design, conviction mathematics, and observability system designed for institutional portfolio intelligence workflows.*
