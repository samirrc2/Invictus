"""
Landing page content registry.

Single source of truth for all tab descriptions.
To add/edit content for any tab, only touch this file.

Images:
    Each sub-tab gets a FOLDER in invictus/static/landing/{SubTabName}/
    Drop PNG screenshots into the folder — they sort alphabetically.
    sections[0] maps to the 1st image, sections[1] to the 2nd, etc.

Structure:
    SECTIONS → list of heading groups
        Each heading → list of sub-tabs
            Each sub-tab → name, list of per-image sections
                Each section → title, bullets (matched to one screenshot)
"""

SECTIONS = [
    {
        "key": "arch",
        "title": "System Architecture",
        "subs": [],  # Special: rendered by hero.py, not bullet content
    },
    {
        "key": "pi",
        "title": "Portfolio Intelligence",
        "subs": [
            {
                "name": "Overview",
                "sections": [
                    {
                        "title": "Portfolio Snapshot & Exposure Summary",
                        "bullets": [
                            ("Total Value & Daily P&L", "Real-time aggregate portfolio valuation and today's dollar profit or loss across all positions. Positive values highlighted in green, negative in red — instantly see your mark-to-market NAV and whether you're up or down for the day."),
                            ("Total Return vs SPY Benchmark", "Your portfolio's cumulative return since inception displayed alongside the S&P 500 ETF (SPY) benchmark. This direct comparison tells you whether your stock-picking is generating alpha over the most common passive alternative."),
                            ("Health Indicators", "Five diagnostic cards — Risk Score, Diversification, Concentration, Sharpe Ratio, and Annualized Volatility — giving you an instant portfolio health check. Each card is color-coded by severity so problem areas stand out immediately."),
                            ("Allocation Bar", "Color-coded horizontal bar showing the weight distribution across all holdings. Wider segments mean higher concentration — if one segment dominates, you may be carrying unintended single-stock risk."),
                            ("Position-Level Detail Table", "Every holding displayed with shares, cost basis, current price, market value, portfolio weight, daily P&L, unrealized P&L ($ and %), annualized volatility, maximum drawdown, and beta relative to SPY. This is your complete risk and return profile per position."),
                        ],
                    },
                    {
                        "title": "Relative Performance (Normalized)",
                        "bullets": [
                            ("Normalized to 100", "All holdings are rebased to 100 at the start of the observation window so you can compare performance on an equal footing regardless of stock price. A value of 400 means that holding has returned 300% since the start."),
                            ("Per-Holding Trend Lines", "Each ticker gets its own color-coded line — track exactly when outperformance or underperformance started and whether it's accelerating or mean-reverting over the full price history window."),
                            ("SPY Benchmark Overlay", "The dotted SPY line gives you a passive benchmark reference. Any holding above this line is outperforming the market; any holding below is underperforming on a total-return basis."),
                            ("Time Horizon", "The chart covers the full price history window used by the pipeline — typically 12 months — giving you both short-term momentum and longer-term trend context in a single view."),
                        ],
                    },
                ],
            },
            {
                "name": "Risk Analytics",
                "sections": [
                    {
                        "title": "Risk Metrics & Downside Behavior",
                        "bullets": [
                            ("10-Card Risk Dashboard", "Ten metric cards displayed across two rows — Annualized Volatility, VaR (95%), CVaR (95%), Max Drawdown, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio, Annualized Return, and HHI concentration index. Each card shows the computed value with color-coded context."),
                            ("Drawdown Chart", "Time-series of rolling drawdown showing every peak-to-trough decline and how long recovery took. Clusters of deep drawdowns signal regime changes; long recovery periods signal structural portfolio problems."),
                            ("Return Distribution Histogram", "Histogram of daily portfolio returns with the normal distribution overlaid. Fat tails or skew tell you whether standard risk models underestimate your actual risk. Includes distribution statistics and the Jarque-Bera test for normality."),
                            ("Distribution Statistics", "Quantitative summary alongside the histogram — mean, standard deviation, skewness, kurtosis, and the Jarque-Bera p-value. If the p-value is low, your returns are significantly non-normal and tail risk is real."),
                        ],
                    },
                    {
                        "title": "Correlation Structure",
                        "bullets": [
                            ("Correlation Heatmap", "Full pairwise correlation matrix across all holdings, color-coded from deep red (high positive correlation) to blue (negative correlation). Clusters of high correlation mean your holdings move together — reducing the diversification you think you have."),
                            ("High-Correlation Pair Callout", "The system automatically identifies and flags the most highly correlated pair in your portfolio with an intelligence callout. If two of your holdings have a 0.92 correlation, you're effectively doubling down on the same bet."),
                            ("Diversification Insight", "The heatmap reveals your true diversification structure at a glance — a portfolio of 10 stocks with average pairwise correlation of 0.8 behaves more like a 3-stock portfolio in a drawdown. The visual pattern tells the story that no single number can."),
                        ],
                    },
                ],
            },
            {
                "name": "Factor Decomposition",
                "sections": [
                    {
                        "title": "Factor Summary & Variance Explained",
                        "bullets": [
                            ("Concentration Banner", "An intelligence callout at the top that flags whether your portfolio's factor exposure is concentrated or diversified, with an assessment label. If one factor dominates, you're making an implicit bet on that factor whether you intended to or not."),
                            ("Factor Interpretation Cards", "Up to four principal components displayed as cards, each with its explained variance percentage and a heuristic label — Market Beta, Idiosyncratic, Sector Rotation, or Style/Size — inferred from the loading patterns. Top loadings shown so you know which holdings drive each factor."),
                            ("Verbose Variance Explanation", "For the top three factors, a detailed paragraph explains what each component represents, how much variance it captures, and which holdings load most heavily on it. Written in plain English so the math translates into investment intuition."),
                            ("Scree Plot", "Bar-and-line chart showing individual and cumulative explained variance per component. The red cumulative line shows diminishing marginal information gain, with a dotted 90% threshold line — if the first 2 factors cross 90%, your portfolio is essentially a 2-factor bet."),
                        ],
                    },
                    {
                        "title": "Factor Loadings Matrix",
                        "bullets": [
                            ("Loading Heatmap", "A position-by-factor matrix showing how strongly each holding loads on each principal component. Strong positive loadings (green) mean the holding moves with the factor; strong negative loadings (red) mean it moves against it. Values near zero indicate minimal exposure."),
                            ("Cross-Position Comparison", "Reading across a row shows a single holding's exposure to every factor. Reading down a column shows which holdings drive a single factor. Opposite signs in the same column mean two holdings hedge each other on that dimension."),
                            ("Idiosyncratic Risk Detection", "Holdings with low loadings across all factors have significant stock-specific risk that isn't captured by any common factor. These are your purest single-stock bets — they won't be hedged by broad market moves or sector rotations."),
                        ],
                    },
                ],
            },
            {
                "name": "Volatility Regimes",
                "sections": [
                    {
                        "title": "Volatility Regime Detection",
                        "bullets": [
                            ("Current Regime Card", "Real-time classification of the current market environment showing the regime label (e.g., Low Vol, Medium Vol, High Vol), the current portfolio volatility level, and how many days the portfolio has been in this regime. Gives you immediate context for how to interpret today's risk numbers."),
                            ("Regime Statistics", "Per-regime summary cards showing the average volatility, percentage of total time spent in each regime, and typical duration in days. A regime where average vol is 2x the low-vol state tells you exactly how much your risk budget needs to flex when markets shift."),
                            ("Historical Regime Chart", "Rolling volatility time-series with each period color-coded by the HMM-detected regime. See patterns around macro events — earnings seasons, Fed meetings, geopolitical shocks — and build intuition for what triggers regime transitions in your specific portfolio."),
                        ],
                    },
                ],
            },
            {
                "name": "Stress Scenarios",
                "sections": [
                    {
                        "title": "Scenario Overview",
                        "bullets": [
                            ("Historical Crisis Replay", "Your current portfolio replayed through past market crises — 2008 GFC, 2020 COVID crash, 2022 rate shock, and other historical scenarios. Each scenario applies the actual historical factor shocks to your current holdings and weights."),
                            ("Scenario Summary Cards", "Cards displayed in rows of three, each showing the scenario name, projected portfolio return percentage, severity label (color-coded from mild to severe), dollar P&L impact, and the stressed portfolio value. Your vulnerability fingerprint at a glance."),
                            ("Severity Color Coding", "Scenarios are color-coded by impact severity — green for mild, amber for moderate, red for severe. You can instantly scan which historical events pose the greatest threat to your specific portfolio without reading a single number."),
                        ],
                    },
                    {
                        "title": "Scenario Drill-Down",
                        "bullets": [
                            ("Expandable Per-Scenario Detail", "Click into any scenario to see exactly how each position would be affected. Each drill-down opens with the most vulnerable and least impacted holdings highlighted in two columns — see your weak links and safe havens immediately."),
                            ("Most Vulnerable / Least Impacted", "Within each scenario, holdings are split into two groups — the positions that would suffer most and those that would hold up best. This tells you which positions are your stress-test weak links and which provide natural protection."),
                            ("Full Ticker Detail Table", "Complete per-position breakdown showing each holding's projected return, dollar loss, and contribution to overall portfolio stress impact. AMD might drop 45% in a rate shock while AAPL drops only 20% — this granularity drives informed hedging decisions."),
                        ],
                    },
                ],
            },
            {
                "name": "P&L Attribution",
                "sections": [
                    {
                        "title": "Return Decomposition & Cumulative Contribution",
                        "bullets": [
                            ("Attribution Summary Cards", "Three headline cards — Portfolio Return, Single-Stock Alpha, and Macro/Factor Beta — decomposing your total performance into what came from individual stock selection versus broad market and factor exposure."),
                            ("Cumulative Contribution Chart", "Multi-line chart showing each holding's cumulative contribution to portfolio return over time, with a dashed portfolio total line. See exactly when each position started contributing or detracting, and how contributions evolved as market conditions changed."),
                            ("Winner & Detractor Identification", "The chart makes it visually obvious which holdings pulled the portfolio up and which dragged it down. Lines above zero are net contributors; lines below are net detractors — regardless of whether the stock itself was profitable."),
                        ],
                    },
                    {
                        "title": "Alpha Efficiency & Rolling Attribution",
                        "bullets": [
                            ("Weight vs Contribution Scatter", "Scatter plot with each holding's portfolio weight on one axis and its return contribution on the other. A diagonal line marks the 'no alpha' boundary — holdings above the line contributed more than their weight would suggest (positive alpha); below the line, they underperformed their allocation."),
                            ("Rolling Attribution Windows", "Three time-window cards — 1-Week, 1-Month, and 3-Month — each showing the top three contributors for that period. See how attribution shifts over time: a stock that was your biggest winner last week might be a detractor over the trailing month."),
                            ("Alpha Efficiency Insight", "The scatter plot reveals whether your portfolio sizing matches your stock-picking skill. If your highest-conviction (largest weight) positions consistently sit below the diagonal, you're over-allocating to underperformers and under-allocating to your actual winners."),
                        ],
                    },
                ],
            },
        ],
    },
    {
        "key": "ci",
        "title": "Conviction Intelligence",
        "subs": [
            {
                "name": "Conviction Engine",
                "sections": [
                    {
                        "title": "Conviction Scores & Signal Overview",
                        "bullets": [
                            ("3-Signal Synthesis", "Each holding's conviction score is produced by fusing three intelligence layers — Capital Flows (institutional accumulation/distribution), Management Outlook (forward-looking sentiment from filings and news), and the Credibility Gate (linguistic analysis that attenuates the outlook signal). This multi-layer approach means no single data source can produce a false signal."),
                            ("Outperformance Probability", "Each holding gets a single probability score representing the likelihood it will outperform over the forward period. Displayed prominently on per-ticker overview cards alongside the conviction level (Strong Buy through Sell) and verdict."),
                            ("Per-Ticker Overview Cards", "Summary cards for each holding showing the probability percentage, conviction tier, verdict label, and the three individual layer scores (flow signal, management signal, credibility multiplier). Color-coded so you can scan your entire portfolio in seconds."),
                        ],
                    },
                    {
                        "title": "Per-Ticker Signal Breakdown",
                        "bullets": [
                            ("Metrics Row", "Four key metrics displayed per ticker — Outperformance Probability, Management Signal, Flow Signal, and overall Conviction score. These are the numbers that drive the final verdict, shown together so you can see where the signals agree and where they diverge."),
                            ("Signal Contribution Bar Chart", "Horizontal bar chart showing how each of the three signal layers contributed to the final conviction score. See exactly what's driving the recommendation — if it's all coming from one signal, the conviction is less robust than when all three agree."),
                            ("Signal Detail Table", "Per-ticker table with columns for Signal Layer, Score, Signal direction, and Key Findings. Full transparency — you can trace every conviction score back to the specific evidence that produced it."),
                            ("Source & Formula", "Each ticker's detail section ends with the data source attribution and the mathematical formula used to compute the final score. Nothing is a black box — every number is auditable."),
                        ],
                    },
                ],
            },
            {
                "name": "Capital Flows",
                "sections": [
                    {
                        "title": "Institutional Flow Summary",
                        "bullets": [
                            ("Flow Direction Scale", "Scores range from −1 (strong distribution / selling) to +1 (strong accumulation / buying). The scale is explained at the top of the page so you always know how to interpret the numbers."),
                            ("Per-Ticker Flow Cards", "Overview cards for each holding showing the composite flow score, accumulation/distribution label, insider intelligence score, and fund trend score. Four numbers that capture the full picture of who is buying and selling."),
                            ("Insider vs Fund Separation", "Institutional flow signal is split into two independent sub-signals — insider transactions (officers, directors buying/selling their own stock) and fund accumulation trends (hedge funds, mutual funds changing positions). These are distinct signals with different implications."),
                        ],
                    },
                    {
                        "title": "Per-Ticker Flow Detail",
                        "bullets": [
                            ("Ownership Snapshot Bar", "Visual bar showing the ownership distribution for each holding. Gives you immediate context on who holds the stock before diving into the transaction-level detail."),
                            ("Largest Holders", "Pill-style badges showing the top institutional holders by position size. Concentrated ownership among a few large holders creates overhang risk — if one decides to sell, the price impact can be severe."),
                            ("Insider Intelligence", "Dedicated section showing insider buy/sell transaction counts and a full transaction table with insider name, role, stake size, transaction direction, and date. Officer buying is one of the strongest signals in equity markets — they know their company best."),
                            ("Fund Accumulation Trend", "Active adding/reducing fund counts, smart money label, and a detailed holder table showing which institutions are building or cutting positions. When multiple unrelated funds simultaneously build positions, it's a stronger signal than any single fund's activity."),
                        ],
                    },
                ],
            },
            {
                "name": "Management Outlook",
                "sections": [
                    {
                        "title": "Management Outlook Overview",
                        "bullets": [
                            ("Outlook Score Scale", "Scores range from −1 (bearish management commentary) to +1 (bullish forward outlook), extracted from news, transcripts, and press releases. The scale and data sources are explained at the top of the page."),
                            ("Per-Ticker Outlook Cards", "Overview cards showing the raw outlook score, tone label (Strong Bullish, Mildly Bullish, Neutral, Mildly Bearish, Strong Bearish), a plain-English explanation of what the score means, and the data sources used to compute it."),
                            ("6 Qualitative Dimensions", "Each score is built from six independently scored dimensions — Demand Environment, Competitive Position, Strategic Confidence, Macro/Industry outlook, Headwinds & Tailwinds, and Thesis Clarity. The composite score fuses all six with credibility-weighted aggregation."),
                        ],
                    },
                    {
                        "title": "Per-Ticker Dimension Detail",
                        "bullets": [
                            ("Dimension Bar Chart", "Horizontal bar chart showing all six dimension scores for each ticker, color-coded (green for positive, red for negative, grey for neutral). The shape of the chart tells a story — a company scoring high on demand but low on competitive position sends a very different signal than one strong across all dimensions."),
                            ("Dimension Detail Table", "Full breakdown table with columns for Dimension, Weight, Score, Weighted Contribution, Signal direction, and Evidence. Every score is traceable to specific evidence extracted from the source material — nothing is a black box."),
                            ("Source & Formula", "Each ticker's detail ends with the data source attribution and the mathematical formula: outlook_score = Σ(weight × dim_score). Full auditability of how the composite score was computed."),
                            ("Earnings Context", "Quantitative anchor section at the bottom providing earnings-related context that grounds the qualitative outlook dimensions in hard financial data."),
                        ],
                    },
                ],
            },
            {
                "name": "Transcript Analysis",
                "sections": [
                    {
                        "title": "Credibility Gate Overview",
                        "bullets": [
                            ("Credibility Multiplier Scale", "The multiplier ranges from 0.50 (heavily discounted — vague, evasive management) to 1.00 (fully credible — specific, forward-looking communication). This layer gates the Management Outlook — it doesn't produce an independent signal; it attenuates Layer 2."),
                            ("Per-Ticker Credibility Cards", "Overview cards showing each holding's credibility multiplier, credibility label (High, Moderate, Low), a plain-English explanation of the gating effect, and the live formula: outlook_score × credibility = management_signal."),
                            ("Gating Logic", "High credibility means the management outlook signal passes through mostly unattenuated. Low credibility discounts the outlook — if management is vague and evasive, their bullish commentary is worth less. The multiplier makes this discount explicit and quantified."),
                        ],
                    },
                    {
                        "title": "Per-Ticker Credibility Detail",
                        "bullets": [
                            ("4 Linguistic Dimensions", "Each ticker's credibility is built from four scored dimensions — Hedging Density (less hedging = more credible), Specificity (concrete numbers vs vague platitudes), Forward/Backward Ratio (more forward-looking = higher confidence), and Dodge Detection (direct answers vs evasion)."),
                            ("Metrics Row & Dimension Table", "Three metric cards (Raw Credibility, Multiplier, Assessment) followed by a full detail table with Dimension, Weight, Score, Contribution, and the AI's reasoning for each score. Every credibility judgment is explained."),
                            ("Credibility Flags", "Two-column display of green flags (credibility strengths) and red flags (credibility concerns) extracted from the analysis. Specific, actionable observations — not generic labels."),
                            ("Gating Effect Formula", "A highlighted formula box showing the complete gating calculation: outlook_score × credibility_multiplier = management_signal. See the exact numerical impact of credibility on the final conviction input."),
                        ],
                    },
                ],
            },
        ],
    },
    {
        "key": "ae",
        "title": "Allocation Engine",
        "subs": [
            {
                "name": "Run Allocation Simulation",
                "sections": [
                    {
                        "title": "Current Portfolio & Simulation Setup",
                        "bullets": [
                            ("Portfolio Context Cards", "Five cards showing your current portfolio state before any simulation — Total Value, Number of Positions, Portfolio Volatility, HHI Concentration Index, and Concentration assessment. This is your baseline that the simulation will compare against."),
                            ("Conviction-Informed Candidates", "Each candidate ticker for allocation is displayed with its conviction card showing the engine's probability score and signal breakdown. You're not allocating blind — the conviction intelligence feeds directly into the simulation so you can see the signal quality for each candidate."),
                            ("Per-Stock Dollar Input", "Simple dollar-amount input for each candidate stock. Enter how much you want to hypothetically allocate to each position, then hit the Simulate button to see the full impact analysis on your portfolio."),
                        ],
                    },
                    {
                        "title": "Simulation Results & Portfolio Fit",
                        "bullets": [
                            ("Verdict Banner", "A prominent banner at the top of results — FAVORABLE, UNFAVORABLE, or MIXED — giving you the headline conclusion before you dive into the details. Color-coded green, red, or amber so the answer is instant."),
                            ("Weight Changes", "Expandable section showing exactly how portfolio weights shift with the proposed allocation. See the before and after for every position — both existing holdings (whose weights decrease as new capital is allocated) and the new candidates."),
                            ("Per-Stock Allocation Rationale", "Conviction-based cards for each stock in the proposed allocation, explaining why the allocation is or isn't supported by the intelligence signals. The rationale ties directly back to the conviction engine's analysis."),
                            ("Portfolio Fit Analysis", "Split into Pro and Con sections, each showing key risk metrics with before→after comparisons plus AI-generated commentary. Pro metrics highlight where the allocation improves your portfolio (e.g., better diversification); Con metrics flag where it hurts (e.g., higher concentration). The AI commentary explains the tradeoffs in plain English."),
                        ],
                    },
                ],
            },
        ],
    },
]
