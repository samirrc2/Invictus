"""
Landing Page — 5-section resume showcase.

Section 1: Hero + Orchestrator Topology + Pipeline Numbers
Section 2: Quantitative Methodology (Bayesian + Conviction + Allocation)
Section 3: Code Architecture (horizontal topology style)
Section 4: Evaluation & Reliability (metrics per layer)

Uses render_section_header from design system for consistent headings.
"""
import streamlit as st

from invictus.design import BRAND_BLUE
from invictus.design.components import render_section_header

# ── Design tokens (match design system) ──────────────────────────────
_B = BRAND_BLUE          # #1d4ed8
_BBG = "#eef2ff"
_BBR = "#c7d2fe"
_G = "#10b981"
_S50 = "#f8fafc"
_S200 = "#e2e8f0"
_S300 = "#cbd5e1"
_S400 = "#94a3b8"
_S500 = "#64748b"
_S700 = "#334155"
_S900 = "#0f172a"

# Font sizes matching gold standard (12px body, 13px headers)
_FS = "12px"   # body text
_FSS = "11px"  # small labels inside diagrams


def _sp(px: int = 16):
    st.markdown(f'<div style="height:{px}px;"></div>', unsafe_allow_html=True)


# ── Shared: node + arrow builders (12px body size) ───────────────────

def _pill(name, bg=_BBG, border=_BBR, color=_B):
    """Standard topology pill — 12px font, matches rest of app."""
    return (
        f'<div style="display:inline-flex;align-items:center;justify-content:center;'
        f'background:{bg};border:1px solid {border};border-radius:4px;'
        f'padding:5px 10px;font-size:{_FS};font-weight:600;color:{color};'
        f'white-space:nowrap;">{name}</div>')


def _pill_sm(name, bg=_BBG, border=_BBR, color=_B):
    """Smaller pill for inside parallel groups."""
    return (
        f'<div style="display:inline-flex;align-items:center;justify-content:center;'
        f'background:{bg};border:1px solid {border};border-radius:4px;'
        f'padding:4px 8px;font-size:{_FSS};font-weight:600;color:{color};'
        f'white-space:nowrap;">{name}</div>')


def _arrow():
    return (
        f'<div style="min-width:12px;padding:0 4px;text-align:center;'
        f'color:{_B};font-size:14px;flex-shrink:0;">→</div>')


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — HERO + TOPOLOGY + NUMBERS
# ══════════════════════════════════════════════════════════════════════

def _render_hero():
    st.markdown(
        f'<div style="text-align:center;padding:24px 0 8px 0;">'
        # Title with gradient accent
        f'<div style="font-size:36px;font-weight:900;color:{_S900};'
        f'letter-spacing:0.14em;margin-bottom:2px;">INVICTUS</div>'
        f'<div style="font-size:13px;font-weight:600;color:{_B};'
        f'letter-spacing:0.08em;text-transform:uppercase;'
        f'margin-bottom:10px;">Multi-Agent AI Platform for Equity Portfolio Intelligence</div>'
        # Gradient divider
        f'<div style="width:80px;height:3px;'
        f'background:linear-gradient(90deg,{_B},#60a5fa,{_B});'
        f'margin:0 auto 16px auto;border-radius:2px;"></div>'
        # Scenario hook — bold key phrases
        f'<div style="font-size:13px;color:{_S500};max-width:640px;'
        f'margin:0 auto;line-height:1.8;">'
        f'<span style="font-weight:700;color:{_S900};">$5,000 to invest.</span> '
        f'<span style="font-weight:700;color:{_S900};">Three stocks.</span> '
        f'Which one actually fits your portfolio? '
        f'Invictus runs '
        f'<span style="font-weight:700;color:{_B};">14 coordinated AI agents</span> '
        f'across '
        f'<span style="font-weight:700;color:{_B};">7 pipeline stages</span> '
        f'to give you a quantitative answer — not a gut feeling.</div>'
        f'</div>', unsafe_allow_html=True)


def _render_topology():
    render_section_header("Orchestrator Topology — 7 Stages")
    st.markdown(f'<div style="font-size:{_FS};color:{_S500};margin:-6px 0 8px 12px;">'
                f'End-to-end LangGraph pipeline from portfolio load to evaluated AI commentary.</div>',
                unsafe_allow_html=True)

    # Same pattern as Agent Data Flow — flex:1 groups, pills wrap inside
    def _topo_grp(label, items):
        boxes = "".join(_pill_sm(n) for n in items)
        return (
            f'<div style="flex:1;border:1px dashed {_BBR};border-radius:6px;'
            f'padding:6px 8px;display:flex;flex-direction:column;align-items:center;gap:4px;">'
            f'<div style="font-size:10px;font-weight:700;color:{_S400};'
            f'letter-spacing:0.06em;">{label}</div>'
            f'<div style="display:flex;gap:4px;flex-wrap:wrap;justify-content:center;">'
            f'{boxes}</div></div>')

    html = (
        f'<div style="border:1px solid {_S200};border-radius:8px;background:{_S50};'
        f'padding:14px 16px;">'
        f'<div style="display:flex;align-items:center;gap:0;width:100%;">'
        f'{_pill("Load")}{_arrow()}'
        f'{_topo_grp("PARALLEL — PORTFOLIO INTELLIGENCE", ["Risk Analytics", "Factor Decomp", "Vol Regimes", "Stress Test", "Greeks", "P&L Attribution"])}{_arrow()}'
        f'{_topo_grp("PARALLEL — CONVICTION INTELLIGENCE", ["Capital Flows", "10-K RAG", "Filing Signals", "Earnings Signals"])}{_arrow()}'
        f'{_topo_grp("SEQUENTIAL — SYNTHESIS", ["Bayesian Synthesis", "Conviction Engine", "AI Commentary"])}{_arrow()}'
        f'{_pill("Eval Harness", bg=_B, border=_B, color="#fff")}'
        f'</div></div>')

    st.markdown(html, unsafe_allow_html=True)


def _render_numbers():
    _sp(10)
    nums = [
        ("14", "Agents"), ("7", "Stages"), ("4", "Signal Sources"),
    ]
    pills = "".join(
        f'<div style="flex:1;border:1px solid {_S200};border-radius:4px;'
        f'background:{_S50};padding:5px 4px;text-align:center;">'
        f'<div style="font-size:14px;font-weight:900;color:{_B};'
        f'font-variant-numeric:tabular-nums;">{val}</div>'
        f'<div style="font-size:9px;font-weight:600;color:{_S500};'
        f'text-transform:uppercase;letter-spacing:0.04em;">{lab}</div>'
        f'</div>'
        for val, lab in nums)
    st.markdown(f'<div style="display:flex;gap:6px;">{pills}</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — QUANTITATIVE METHODOLOGY
# ══════════════════════════════════════════════════════════════════════

def _render_methodology():
    render_section_header("Quantitative Methodology")
    st.markdown(f'<div style="font-size:{_FS};color:{_S500};margin:-6px 0 8px 12px;">'
                f'Bayesian signal model, weighted conviction synthesis, and hypothetical allocation simulation.</div>',
                unsafe_allow_html=True)

    # ── Plain-English Summary ──
    st.markdown(
        f'<div style="border:1px solid {_S200};border-radius:8px;background:{_S50};'
        f'padding:16px 20px;margin-bottom:16px;">'

        # Two-sentence summary
        f'<div style="font-size:13px;color:{_S700};line-height:1.7;margin-bottom:14px;">'
        f'For each stock, Invictus collects 4 independent signals — fundamentals, management outlook, '
        f'institutional flows, and a Bayesian ML model — then combines them into a single '
        f'<span style="font-weight:700;color:{_B};">outperformance probability</span>. '
        f'When you consider adding a position, the allocation engine simulates how it would change '
        f'your portfolio\'s risk profile before you commit capital.</div>'

        # Visual flow: 4 signals → combine → probability → simulate → verdict
        f'<div style="display:flex;align-items:center;justify-content:center;gap:0;flex-wrap:wrap;">'
        # Signal sources
        f'<div style="display:flex;flex-direction:column;gap:3px;align-items:center;">'
        f'{_pill_sm("Fundamentals")}'
        f'{_pill_sm("Management")}'
        f'{_pill_sm("Inst. Flows")}'
        f'{_pill_sm("Bayesian ML")}'
        f'</div>'
        f'{_arrow()}'
        f'{_pill("Weighted Synthesis")}'
        f'{_arrow()}'
        f'<div style="text-align:center;padding:6px 14px;background:{_BBG};border:1.5px solid {_B};'
        f'border-radius:6px;">'
        f'<div style="font-size:14px;font-weight:800;color:{_B};">P = 71%</div>'
        f'<div style="font-size:10px;color:{_S400};font-weight:600;">OUTPERFORM</div></div>'
        f'{_arrow()}'
        f'{_pill("Allocation Sim")}'
        f'{_arrow()}'
        f'<div style="text-align:center;padding:6px 14px;background:{_G}11;border:1.5px solid {_G};'
        f'border-radius:6px;">'
        f'<div style="font-size:12px;font-weight:700;color:{_G};">FAVORABLE</div>'
        f'<div style="font-size:10px;color:{_S400};font-weight:600;">VERDICT</div></div>'
        f'</div>'

        f'</div>', unsafe_allow_html=True)

    # Pre-build pill strings for simulation flow (avoid backslash in f-string)
    sim_inputs = "".join(_pill(n) for n in [
        "Portfolio", "Conviction Scores", "$ Allocation"])
    sim_engine = _pill("Simulate Engine", bg=_B, border=_B, color="#fff")
    sim_arrow = _arrow()

    # ── Collapsible math detail — single expander, two-column layout inside ──
    with st.expander("Quantitative Methodology — Mathematical Detail", expanded=False):
        st.markdown(
            f'<div style="display:flex;gap:20px;align-items:stretch;">'

            # ── LEFT COLUMN: Conviction Intelligence ──
            f'<div style="flex:1;min-width:0;">'
            f'<div style="font-size:14px;font-weight:800;color:{_S900};margin-bottom:4px;">'
            f'Conviction Intelligence</div>'
            f'<div style="font-size:{_FSS};color:{_S500};margin-bottom:10px;">'
            f'How 4 signal sources combine into a single outperformance probability per stock.</div>'

            # Bayesian model
            f'<div style="background:{_BBG};border:1.5px solid {_B};'
            f'border-radius:6px;padding:12px 14px;margin-bottom:12px;">'
            f'<div style="font-size:{_FSS};font-weight:700;color:{_B};'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
            f'Bayesian Accumulation Model (ML Agent)</div>'
            f'<div style="font-family:monospace;font-size:{_FS};color:{_S700};line-height:1.9;">'
            f'BF<sub>i</sub>(x) = exp(κ<sub>i</sub> · g<sub>i</sub>(x))<br>'
            f'posterior_odds = prior_odds × ∏ BF<sub>i</sub>(x<sub>i</sub>)<br>'
            f'P(acc | data) = posterior_odds / (1 + posterior_odds)'
            f'</div>'
            f'<div style="font-size:{_FSS};color:{_S500};margin-top:6px;">'
            f'Sequential Bayesian updating with log-linear Bayes Factors. '
            f'Each feature produces bounded evidence BF ∈ [exp(−κ), exp(+κ)].</div>'
            f'</div>'

            # Synthesis
            f'<div style="background:{_S50};border:1px solid {_S200};'
            f'border-radius:6px;padding:10px 14px;margin-bottom:12px;">'
            f'<div style="font-size:{_FSS};font-weight:700;color:{_S400};'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
            f'Conviction Synthesis</div>'
            f'<div style="font-family:monospace;font-size:{_FS};color:{_S700};line-height:1.9;">'
            f'C = 0.35·fundamental + 0.25·management + 0.25·flows + 0.15·technical<br>'
            f'C<sub>adj</sub> = C × agreement_multiplier<br>'
            f'P<sub>outperform</sub> = σ(3 · C<sub>adj</sub>) = 1 / (1 + e<sup>−3·C<sub>adj</sub></sup>)'
            f'</div>'
            f'<div style="font-size:{_FSS};color:{_S500};margin-top:6px;">'
            f'Calibrated: C=±0.3 → P≈71%/29%, C=±0.6 → P≈86%/14%. '
            f'Dynamic weights shift with volatility regime.</div>'
            f'</div>'

            # 4 signal sources
            f'<div style="font-size:{_FS};color:{_S500};line-height:1.7;">'
            f'<span style="font-weight:700;color:{_B};">Fundamental (0.35):</span> '
            f'Filing intel — QoQ revenue, net income, operating income growth via yFinance [−1, +1]<br>'
            f'<span style="font-weight:700;color:{_B};">Management (0.25):</span> '
            f'6 outlook dims × credibility gate: mgmt_signal = outlook × (0.5 + 0.5 × raw_cred)<br>'
            f'<span style="font-weight:700;color:{_B};">Flows (0.25):</span> '
            f'13F filings, insider transactions, materiality-weighted, time-decayed [−1, +1]<br>'
            f'<span style="font-weight:700;color:{_B};">Technical (0.15):</span> '
            f'Bayesian ML accumulation model — posterior probability mapped to [−1, +1]'
            f'</div>'
            f'</div>'

            # ── DIVIDER ──
            f'<div style="width:2px;background:linear-gradient(180deg,'
            f'{_B} 0%,{_S300} 50%,{_B} 100%);'
            f'border-radius:1px;flex-shrink:0;"></div>'

            # ── RIGHT COLUMN: Allocation Engine ──
            f'<div style="flex:1;min-width:0;">'
            f'<div style="font-size:14px;font-weight:800;color:{_S900};margin-bottom:4px;">'
            f'Allocation Engine</div>'
            f'<div style="font-size:{_FSS};color:{_S500};margin-bottom:10px;">'
            f'Simulate adding a position and see how it changes your portfolio risk profile.</div>'

            # Risk Computation Model
            f'<div style="background:{_BBG};border:1.5px solid {_B};'
            f'border-radius:6px;padding:12px 14px;margin-bottom:12px;">'
            f'<div style="font-size:{_FSS};font-weight:700;color:{_B};'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
            f'Hypothetical Risk Model</div>'
            f'<div style="font-family:monospace;font-size:{_FS};color:{_S700};line-height:1.9;">'
            f'w<sub>hypo</sub> = rebalance(existing_weights, new_positions)<br>'
            f'σ<sub>hypo</sub> = std(r<sub>hypo</sub>) × √252<br>'
            f'VaR<sub>95</sub> = percentile(r<sub>hypo</sub>, 5%)<br>'
            f'Sharpe = (μ<sub>ann</sub> − r<sub>f</sub>) / σ<sub>hypo</sub>'
            f'</div>'
            f'<div style="font-size:{_FSS};color:{_S500};margin-top:6px;">'
            f'Recomputes all portfolio risk metrics with hypothetical weights. '
            f'Compares before vs after to quantify marginal impact.</div>'
            f'</div>'

            # Before → After
            f'<div style="background:{_S50};border:1px solid {_S200};'
            f'border-radius:6px;padding:10px 14px;margin-bottom:12px;">'
            f'<div style="font-size:{_FSS};font-weight:700;color:{_S400};'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
            f'Before → After Comparison</div>'
            f'<div style="font-family:monospace;font-size:{_FS};color:{_S700};line-height:1.9;">'
            f'delta = after_metrics − before_metrics<br>'
            f'verdict = evaluate(VaR, volatility, Sharpe, HHI, drawdown)<br>'
            f'pros = ai_commentary(improved_metrics)<br>'
            f'cons = ai_commentary(degraded_metrics)'
            f'</div>'
            f'<div style="font-size:{_FSS};color:{_S500};margin-top:6px;">'
            f'Each metric delta is ranked by magnitude. Material changes (>5% relative) '
            f'drive the verdict; immaterial changes are filtered out.</div>'
            f'</div>'

            # Detailed breakdown
            f'<div style="font-size:{_FS};color:{_S500};line-height:1.7;">'
            f'<span style="font-weight:700;color:{_B};">VaR (95%):</span> '
            f'Historical percentile — worst daily loss at 95% confidence<br>'
            f'<span style="font-weight:700;color:{_B};">Volatility:</span> '
            f'Annualized standard deviation of weighted portfolio returns<br>'
            f'<span style="font-weight:700;color:{_B};">Sharpe Ratio:</span> '
            f'Risk-adjusted return — excess return per unit of volatility<br>'
            f'<span style="font-weight:700;color:{_B};">HHI:</span> '
            f'Herfindahl concentration index — 0 (diversified) to 1 (single stock)<br>'
            f'<span style="font-weight:700;color:{_B};">Verdict:</span> '
            f'FAVORABLE / UNFAVORABLE / MIXED / NEUTRAL / IMMATERIAL'
            f'</div>'
            f'</div>'

            f'</div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — CODE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════

def _render_code_arch():
    render_section_header("Code Architecture")
    st.markdown(f'<div style="font-size:{_FS};color:{_S500};margin:-6px 0 8px 12px;">'
                f'Modular Python packages with centralized LLM calls, feature flags, and a thin Streamlit routing shell.</div>',
                unsafe_allow_html=True)

    # Module packages — full width flex
    mods = [
        ("agents/", "17"), ("pages/", "42"), ("evaluation/", "6"),
        ("observability/", "14"), ("backtest/", "5"), ("design/", "7"),
        ("rag/", "2"), ("data/", "3"), ("analytics/", "2"),
    ]
    mod_pills = "".join(
        f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;'
        f'background:{_BBG};border:1px solid {_BBR};border-radius:4px;'
        f'padding:6px 8px;">'
        f'<div style="font-size:{_FS};font-weight:700;color:{_B};">{name}</div>'
        f'<div style="font-size:10px;color:{_S400};">{cnt} modules</div>'
        f'</div>'
        for name, cnt in mods)

    st.markdown(
        f'<div style="border:1px solid {_S200};border-radius:8px;background:{_S50};'
        f'padding:14px 16px;margin-bottom:12px;">'
        f'<div style="font-size:{_FSS};font-weight:700;color:{_S400};'
        f'letter-spacing:0.06em;margin-bottom:8px;">MODULE PACKAGES — invictus/</div>'
        f'<div style="display:flex;gap:6px;">{mod_pills}</div>'
        f'<div style="display:flex;gap:10px;justify-content:center;margin-top:10px;">'
        f'{_pill("llm.py — centralized LLM calls")}'
        f'{_pill("config.py — feature flags")}'
        f'</div>'
        f'</div>', unsafe_allow_html=True)

    # Agent data flow — single horizontal row with labeled groups
    def _flow_grp(title, items):
        boxes = "".join(_pill_sm(i) for i in items)
        return (
            f'<div style="flex:1;border:1px dashed {_BBR};border-radius:6px;'
            f'padding:6px 8px;display:flex;flex-direction:column;align-items:center;gap:4px;">'
            f'<div style="font-size:10px;font-weight:700;color:{_S400};'
            f'letter-spacing:0.06em;">{title}</div>'
            f'<div style="display:flex;gap:4px;flex-wrap:wrap;justify-content:center;">'
            f'{boxes}</div></div>')

    st.markdown(
        f'<div style="border:1px solid {_S200};border-radius:8px;background:{_S50};'
        f'padding:12px 16px;margin-bottom:12px;">'
        f'<div style="font-size:{_FSS};font-weight:700;color:{_S400};'
        f'letter-spacing:0.06em;margin-bottom:8px;">AGENT DATA FLOW</div>'
        f'<div style="display:flex;align-items:center;gap:0;width:100%;">'
        f'{_flow_grp("DATA SOURCES", ["yFinance API", "FMP API", "13F Filings", "Transcripts"])}'
        f'{_arrow()}'
        f'{_flow_grp("ORCHESTRATOR", ["LangGraph", "7 Stages", "Parallel Exec"])}'
        f'{_arrow()}'
        f'{_flow_grp("INTELLIGENCE", ["Risk Analytics", "Conviction", "Allocation"])}'
        f'{_arrow()}'
        f'{_flow_grp("OUTPUT", ["Streamlit UI", "AI Commentary", "Eval Harness", "Backtest"])}'
        f'</div></div>', unsafe_allow_html=True)

    # Tech stack
    techs = [
        "Python", "LangGraph", "OpenAI", "Streamlit", "Plotly",
        "PCA", "KMeans", "Bayesian ML", "NLP", "yFinance", "FMP API",
    ]
    badges = "".join(
        f'<span style="display:inline-block;background:{_BBG};'
        f'border:1px solid {_BBR};border-radius:12px;'
        f'padding:4px 12px;font-size:{_FSS};font-weight:600;color:{_B};'
        f'margin:3px 4px;">{t}</span>'
        for t in techs)
    st.markdown(f'<div style="text-align:center;">{badges}</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATION & RELIABILITY
# ══════════════════════════════════════════════════════════════════════

def _render_eval():
    render_section_header("Evaluation & Reliability")
    st.markdown(f'<div style="font-size:{_FS};color:{_S500};margin:-6px 0 8px 12px;">'
                f'Every LLM output is grounded, calibrated, and tracked — not trusted blindly.</div>',
                unsafe_allow_html=True)

    def _eval_card(name, metrics):
        m_html = "".join(
            f'<div style="font-size:{_FSS};color:{_S500};line-height:1.5;'
            f'padding:1px 0;">• {m}</div>'
            for m in metrics)
        return (
            f'<div style="flex:1;border:1.5px solid {_B};border-radius:6px;'
            f'background:{_BBG};padding:10px 12px;">'
            f'<div style="font-size:{_FS};font-weight:800;color:{_B};'
            f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">'
            f'{name}</div>{m_html}</div>')

    # Row 1: Walk-Forward Backtest (full-width highlight card)
    st.markdown(
        f'<div style="border:1.5px solid {_B};border-radius:8px;background:{_BBG};'
        f'padding:14px 18px;margin-bottom:10px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div>'
        f'<div style="font-size:14px;font-weight:800;color:{_B};margin-bottom:4px;">'
        f'Walk-Forward Ex-Ante Backtest</div>'
        f'<div style="font-size:{_FS};color:{_S500};line-height:1.6;">'
        f'Replays conviction pipeline across 2024 using only point-in-time data — no look-ahead bias. '
        f'Computes fundamental + Bayesian ML signals monthly, measures against actual forward returns.</div>'
        f'</div>'
        f'<div style="display:flex;gap:12px;margin-left:20px;flex-shrink:0;">'
        f'<div style="text-align:center;padding:6px 14px;background:white;border-radius:6px;'
        f'border:1px solid {_S200};white-space:nowrap;min-width:72px;">'
        f'<div style="font-size:16px;font-weight:800;color:{_B};white-space:nowrap;">68–71%</div>'
        f'<div style="font-size:10px;color:{_S400};font-weight:600;">HIT RATE</div></div>'
        f'<div style="text-align:center;padding:6px 14px;background:white;border-radius:6px;'
        f'border:1px solid {_S200};white-space:nowrap;min-width:72px;">'
        f'<div style="font-size:16px;font-weight:800;color:{_B};">+0.331</div>'
        f'<div style="font-size:10px;color:{_S400};font-weight:600;">SPEARMAN ρ</div></div>'
        f'<div style="text-align:center;padding:6px 14px;background:white;border-radius:6px;'
        f'border:1px solid {_S200};white-space:nowrap;min-width:72px;">'
        f'<div style="font-size:16px;font-weight:800;color:{_B};">1.93</div>'
        f'<div style="font-size:10px;color:{_S400};font-weight:600;">SHARPE</div></div>'
        f'</div></div></div>', unsafe_allow_html=True)

    # Row 2: 4 eval pillars
    cards = [
        _eval_card("LLM Eval Harness", [
            "Numerical grounding rate",
            "Consistency across runs",
            "Cost per ticker analysis",
            "Determinism scoring (A-F)",
        ]),
        _eval_card("Conviction Calibration", [
            "Hit rate by horizon (5d-63d)",
            "Quintile return monotonicity",
            "Information coefficient (IC)",
            "Calibration curve vs 45° line",
        ]),
        _eval_card("Dev Console", [
            "11 analytics tabs with verdicts",
            "Red flag detection per tab",
            "Signal quality mode selector",
            "Cross-horizon verdict table",
        ]),
        _eval_card("Hallucination Guard", [
            "Grounding rate vs source",
            "Numerical consistency check",
            "Sentiment bias detection",
            "Per-claim verification",
        ]),
    ]

    all_cards = "".join(cards)
    st.markdown(
        f'<div style="border:1px solid {_S200};border-radius:8px;background:{_S50};'
        f'padding:12px 16px;margin-bottom:10px;">'
        f'<div style="font-size:{_FSS};font-weight:700;color:{_S400};'
        f'letter-spacing:0.06em;margin-bottom:8px;">'
        f'EVAL PIPELINE — METRICS AT EACH LAYER</div>'
        f'<div style="display:flex;gap:8px;width:100%;">'
        f'{all_cards}'
        f'</div></div>', unsafe_allow_html=True)

    # Observability stack — fill width
    collectors = "".join(
        _pill_sm(n) for n in ["Agent", "LLM", "ML", "Conviction", "Session", "Data"])
    analyzers = "".join(
        _pill_sm(n) for n in ["Hallucination", "Drift", "Calibration"])

    def _par_group(pills, label):
        return (
            f'<div style="flex:1;border:1px dashed {_BBR};border-radius:6px;padding:6px 10px;'
            f'display:flex;flex-direction:column;align-items:center;gap:4px;">'
            f'<div style="font-size:10px;font-weight:700;color:{_S400};'
            f'letter-spacing:0.06em;">{label}</div>'
            f'<div style="display:flex;gap:4px;flex-wrap:wrap;justify-content:center;">'
            f'{pills}</div></div>')

    st.markdown(
        f'<div style="border:1px solid {_S200};border-radius:8px;background:{_S50};'
        f'padding:12px 16px;">'
        f'<div style="font-size:{_FSS};font-weight:700;color:{_S400};'
        f'letter-spacing:0.06em;margin-bottom:8px;">'
        f'OBSERVABILITY STACK — 6 COLLECTORS → 3 ANALYZERS → SQLite STORE → VERDICT SUMMARIES</div>'
        f'<div style="display:flex;align-items:center;gap:0;width:100%;">'
        f'{_par_group(collectors, "COLLECTORS (6)")}{_arrow()}'
        f'{_par_group(analyzers, "ANALYZERS (3)")}{_arrow()}'
        f'{_pill("SQLite Store", bg=_B, border=_B, color="#fff")}{_arrow()}'
        f'{_pill("11 Verdict Tabs", bg=_B, border=_B, color="#fff")}'
        f'</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def render():
    _render_hero()
    _render_topology()
    _render_numbers()
    _render_methodology()
    _render_code_arch()
    _render_eval()
