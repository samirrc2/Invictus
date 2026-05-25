# Invictus — Technical Architecture & Mathematical Framework

**Invictus Equity Portfolio Intelligence Platform**
A multi-agent AI system for institutional-grade portfolio analytics, conviction synthesis, and allocation simulation.

---

## System Overview

Invictus orchestrates 16 specialized agents through a LangGraph-based pipeline to produce portfolio intelligence across three domains: risk analytics, conviction intelligence, and allocation simulation. The platform processes real-time market data, applies quantitative models, and synthesizes results through a conviction engine that outputs calibrated outperformance probabilities.

### Architecture

```
Data Layer          Agent Layer              Intelligence Layer        Presentation
─────────────      ──────────────────       ────────────────────      ────────────
Yahoo Finance  →   Risk Analytics Agent  →  Portfolio Intelligence →  Streamlit UI
               →   PCA Factor Agent      →
               →   Vol Regime Agent      →
               →   Stress Test Agent     →
               →   Greeks Agent          →
               →   Flow Agent            →  Conviction Intelligence
               →   Filing Agent          →
               →   Earnings Agent (LLM)  →
               →   ML Accumulation Agent →
               →   Synthesis Agent       →  Allocation Engine
               →   Commentary Agent (LLM)→
               →   P&L Attribution Agent →
               →   Evaluation Agent      →  Dev Analytics
```

### Orchestration

The LangGraph orchestrator executes agents in a staged dependency graph:

```
Stage 0: load_portfolio
Stage 1: [compute_risk, run_pca, detect_vol_regime, run_stress_tests, compute_greeks]  (parallel)
Stage 2: [analyze_flows, run_filing_intel, run_earnings_intel]  (parallel)
Stage 3: run_accumulation_model  (depends on flows + filing + earnings)
Stage 4: run_conviction_synthesis  (depends on all intelligence)
Stage 5: attribute_pnl
Stage 6: generate_commentary
Stage 7: evaluate_commentary
Stage 8: produce_final_report
```

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

### 3.2 KMeans Clustering

```
KMeans(n_clusters=3).fit(σ_rolling)
```

Clusters the rolling volatility series into three regimes:
- **Low** — stable, trending markets
- **Medium** — normal market conditions
- **High** — elevated risk, potential crisis

Cluster assignment determines the current regime, which feeds into the conviction synthesis engine's dynamic weighting.

---

## 4. Stress Testing

Historical scenario replay using actual factor shocks:

```
Stressed_Return_i = β_i · Scenario_Factor_Return
Stressed_Value_i = Current_Value_i × (1 + Stressed_Return_i)
Portfolio_Impact = Σ Stressed_Value_i - Σ Current_Value_i
```

Scenarios include: COVID crash (2020), GFC (2008), rate shock (2022), tech selloff, semiconductor correction.

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

## 6. ML Accumulation Classifier

### 6.1 Feature Engineering

26 features across 7 groups:

| Group | Features |
|-------|----------|
| Momentum | 5d, 10d, 20d, 60d momentum; 20d, 60d relative strength |
| Technical | RSI-14, MACD histogram, Bollinger %B, ADX, ATR ratio, OBV trend |
| Flow | Institutional conviction, insider alignment, capital participation |
| Fundamental | Fundamental score, management score |
| Risk | 20d volatility, vol ratio, vol z-score |
| Microstructure | Volume z-score, price-volume divergence, recovery velocity |
| Structure | Distance from high, distance from low, drawdown depth |

### 6.2 Label Construction

Forward return classification with a 20-day horizon:

```
Label = 1  if  Forward_Return_20d > median(Forward_Returns)  else  0
```

Binary classification: accumulation (buying opportunity) vs. distribution.

### 6.3 Ensemble Model

Three models combined:

**Logistic Regression** — `C=0.1, max_iter=1000, class_weight='balanced'`
- Linear decision boundary, interpretable coefficients
- Regularized to prevent overfitting on noisy financial features

**Random Forest** — `n_estimators=200, max_depth=6, min_samples_leaf=10`
- Non-linear, handles feature interactions
- Depth-limited to avoid memorizing noise

**XGBoost** (optional) — `n_estimators=150, max_depth=4, learning_rate=0.05`
- Gradient-boosted ensemble, strongest individual model
- Trained separately due to VotingClassifier compatibility

**Ensemble Combination:**
```
P(accumulation) = 0.5 × P_LR + 0.5 × P_RF
```
If XGBoost available:
```
P(accumulation) = 0.35 × P_LR + 0.35 × P_RF + 0.30 × P_XGB
```

### 6.4 Validation

TimeSeriesSplit cross-validation (5 folds, no data leakage):
```
CV_Score = mean(accuracy across 5 temporal folds)
```

---

## 7. Conviction Synthesis Engine

The core intelligence layer that synthesizes all signals into a calibrated outperformance probability.

### 7.1 Signal Extraction

Four signal channels, each normalized to [-1, +1]:

```
S_fundamental = (fundamental_conviction + guidance_momentum - risk_deterioration) / 2
S_management  = management_confidence - 0.5 × analyst_pressure
S_flows       = flow_composite  (or 0.5·inst_conviction + 0.3·insider_alignment + 0.2·capital_participation)
S_technical   = (accumulation_prob - 0.5) × 2
```

### 7.2 Signal Quality Assessment

Each signal is scored for data quality `q_i ∈ [0, 1]`:

```
q_i = f(data_availability, source_confidence, recency)
```

Missing data → `q = 0.2`. Dictionary fallback → `q = 0.5`. Full LLM analysis → `q = 0.8`. ML with high CV → `q = 0.9`.

### 7.3 Dynamic Weighting

Weights are regime-conditional and quality-adjusted:

| Signal | Low Vol | Medium Vol | High Vol |
|--------|---------|------------|----------|
| Fundamental | 0.30 | 0.25 | 0.20 |
| Management | 0.20 | 0.20 | 0.15 |
| Flows | 0.25 | 0.30 | 0.35 |
| Technical | 0.25 | 0.25 | 0.30 |

In high-volatility regimes, flow and technical signals receive more weight (institutional behavior matters more in crisis). Weights are further adjusted by signal quality:

```
w_i_adjusted = w_i_base × q_i / Σ(w_j × q_j)
```

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

The factor `3` controls the sensitivity — a composite score of `±0.5` maps to approximately `82%/18%` probability. The steepness was calibrated to produce meaningful spread across the conviction range.

### 7.7 Confidence Adjustment

```
P_final = P × q_avg + 0.5 × (1 - q_avg)
```

When signal quality is low, the probability shrinks toward the neutral 50% baseline. With perfect quality (all signals at 1.0), `P_final = P`. This prevents overconfident predictions from sparse data.

### 7.8 Monte Carlo Confidence Intervals

5,000 simulations with noise injection proportional to signal uncertainty:

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

Narrow CI width (< 0.15) → high confidence in the estimate.
Wide CI width (> 0.30) → signal uncertainty is too high for reliable conviction.

---

## 8. Allocation Engine

### 8.1 Hypothetical Portfolio Construction

Given existing portfolio weights `w` and new investment amounts `a`:

```
w_new_i = (w_old_i × V_total + a_i) / (V_total + Σ a_j)
```

### 8.2 Before/After Risk Comparison

The engine recomputes all risk metrics (Sharpe, volatility, drawdown, VaR, HHI, return) on the hypothetical portfolio and measures deltas:

```
Δ_metric = metric_after - metric_before
```

Each delta is scored by impact magnitude and classified as improvement or risk:

```
Impact_i = |Δ_i| × scale_factor_i
```

Scale factors normalize different metrics to comparable impact units (e.g., 0.02 Sharpe change ≈ 1% volatility change in impact).

### 8.3 Materiality Check

```
Material = (Σ a_i / V_total) > 2%
```

If the investment is less than 2% of portfolio value, the engine returns IMMATERIAL — the allocation is too small to meaningfully impact risk characteristics.

### 8.4 AI Portfolio Fit Analysis

For each metric with material change:
- **Direction** — improvement or deterioration
- **Magnitude** — relative to the metric's scale
- **Commentary** — contextual explanation of what the change means for portfolio construction

Top 3 improvements and top 3 risks are selected by impact ranking.

---

## 9. LLM Integration

### 9.1 Earnings Sentiment Analysis

OpenAI GPT-4o analyzes news context to extract:
- Management confidence score `[-1, +1]`
- Analyst pressure score `[0, 1]`
- Sentiment trend (Improving / Stable / Deteriorating)
- Tone drivers and analyst concerns

**Fallback:** Dictionary-based sentiment scoring using curated positive/negative keyword lists when LLM is unavailable:

```
confidence = (pos_hits - neg_hits) / normalization_factor
pressure = pressure_hits / normalization_factor
```

### 9.2 AI Commentary Generation

Multi-style commentary (PM view, Risk Manager view) generated from structured portfolio data. The prompt explicitly instructs "Do not hallucinate data not provided" to minimize fabrication.

### 9.3 Smart CSV Loader

LLM extracts portfolio holdings from arbitrary brokerage CSV formats. Fallback: pandas-based column name matching with 12+ common brokerage naming patterns.

---

## 10. Observability System

### 10.1 Architecture

```
Collectors (6)  →  SQLite Store  →  Analyzers (3)  →  Dashboard (7 tabs)
```

**Collectors:** Agent execution, LLM calls, ML predictions, conviction scores, session events, data health.

**Analyzers:** Hallucination detection (sentiment bias, determinism, fallback rates), drift analysis (prediction stability, ensemble agreement, conviction stability), calibration (agent performance, session analytics, data pipeline health).

### 10.2 LLM Quality Metrics

- **Fallback Rate** — percentage of LLM calls that fell through to dictionary-based fallback
- **Sentiment Bias** — mean sentiment score across all calls (detects systematic positive/negative lean)
- **Determinism Score** — same prompt hash → same output? Measures output stability
- **Token Economics** — cost per call, cost per pipeline run, per-agent token consumption

### 10.3 ML Monitoring

- **Ensemble Agreement Rate** — how often LR and RF agree on direction
- **Prediction Confidence Distribution** — ratio of high-confidence (>70%) vs low-confidence (40-60%) predictions
- **Ticker Stability** — same ticker across multiple runs → how much does the prediction vary?

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph / Custom InvictusGraph |
| ML Models | scikit-learn (LR, RF, PCA, KMeans), XGBoost |
| LLM | OpenAI GPT-4o (with dictionary fallback) |
| Statistics | scipy (Jarque-Bera, Black-Scholes, Brent root-finding) |
| Numerical | NumPy, Pandas |
| Visualization | Plotly |
| Market Data | Yahoo Finance (yfinance) |
| UI Framework | Streamlit |
| Observability | SQLite + custom collectors/analyzers |
| Simulation | Monte Carlo (NumPy bootstrap) |

---

## File Structure

```
app.py                              — Thin routing shell (350 lines)
invictus/
  agents/                           — 13 specialized agents
    orchestrator.py                 — LangGraph-style staged executor
    risk_agent.py                   — VaR, CVaR, Sharpe, MCTR, drawdown
    pca_agent.py                    — Factor decomposition
    vol_regime_agent.py             — KMeans regime detection
    stress_agent.py                 — Historical scenario replay
    greeks_agent.py                 — Black-Scholes Greeks
    flow_agent.py                   — Institutional & insider flows
    filing_agent.py                 — Fundamental intelligence
    earnings_agent.py               — LLM sentiment analysis
    ml_agent.py                     — Ensemble accumulation classifier
    synthesis_agent.py              — Conviction synthesis + Monte Carlo
    commentary_agent.py             — LLM commentary generation
    hypo_agent.py                   — Allocation simulation engine
    pnl_agent.py                    — P&L attribution
  design/                           — Design system package (7 modules)
  pages/                            — Page modules (4 pages)
  observability/                    — AI observability system (14 files)
    collectors/                     — 6 data collection modules
    analyzers/                      — 3 analysis engines
  data/                             — Market data pipeline
  config.py                         — Configuration constants
```

**Total:** ~10,000 lines of Python across 50+ files.

---

## Key Differentiators

1. **Multi-agent orchestration** with dependency-aware staged execution — not a single monolithic model.
2. **Conviction synthesis** with mathematical rigor — dynamic weighting, signal agreement, Monte Carlo confidence intervals, calibrated probability mapping.
3. **AI observability** built-in — hallucination monitoring, sentiment bias detection, token economics, prediction drift tracking. Most AI platforms generate outputs but cannot measure quality.
4. **Graceful degradation** — every LLM-dependent path has a quantitative fallback. Platform functions fully without an API key.
5. **Institutional-grade risk analytics** — three VaR methods, MCTR, PCA factor decomposition, KMeans regime detection, historical stress testing. Not retail dashboard charts.

---

*Built by Samir Chincholikar. Platform architecture, agent design, conviction mathematics, and observability system designed for institutional portfolio intelligence workflows.*
