"""
Invictus — AI Commentary Generation Agent
Generates portfolio commentary in multiple styles using templates.
Falls back to template-based generation when OpenAI API is not available.
"""
import numpy as np
from typing import Dict, Any, Optional

from invictus.agents.graph_state import PortfolioState
from invictus.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE


def _build_context(state: PortfolioState) -> Dict[str, str]:
    """Build context strings from state for commentary generation."""
    ctx = {}

    # Portfolio return
    pnl = state.pnl_attribution or {}
    port_ret = pnl.get("portfolio_return", 0)
    direction = "up" if port_ret >= 0 else "down"
    ctx["portfolio_move"] = f"Portfolio was {direction} {abs(port_ret):.2%} today."

    # Top/bottom contributors
    top = pnl.get("top_contributors", [])
    bottom = pnl.get("bottom_contributors", [])
    if top:
        ctx["top_drivers"] = ", ".join([f"{t['ticker']} ({t['contribution']:+.2%})" for t in top])
    if bottom:
        ctx["bottom_drivers"] = ", ".join([f"{t['ticker']} ({t['contribution']:+.2%})" for t in bottom])

    # Risk metrics
    rm = state.risk_metrics or {}
    if rm:
        ctx["volatility"] = f"Annualized vol: {rm.get('annualized_volatility', 0):.1%}"
        ctx["sharpe"] = f"Sharpe: {rm.get('sharpe_ratio', 0):.2f}"
        ctx["var"] = f"VaR 95%: {rm.get('var_95_historical', 0):.2%}"
        ctx["max_dd"] = f"Max drawdown: {rm.get('max_drawdown', 0):.1%}"

    # Vol regime
    ctx["vol_regime"] = pnl.get("vol_regime", "Unknown")

    # Sector attribution
    sector = pnl.get("sector_contributions")
    if sector is not None and hasattr(sector, "iterrows"):
        parts = []
        for _, row in sector.iterrows():
            parts.append(f"{row['Sector']}: {row['Contribution']:+.3%}")
        ctx["sector_breakdown"] = "; ".join(parts)

    return ctx


def _template_commentary(ctx: Dict[str, str], style: str) -> str:
    """Generate commentary from templates when LLM is unavailable."""
    port_move = ctx.get("portfolio_move", "Portfolio return data unavailable.")
    top = ctx.get("top_drivers", "N/A")
    bottom = ctx.get("bottom_drivers", "N/A")
    vol = ctx.get("volatility", "")
    sharpe = ctx.get("sharpe", "")
    var_str = ctx.get("var", "")
    regime = ctx.get("vol_regime", "Unknown")
    sectors = ctx.get("sector_breakdown", "")

    if style == "concise":
        return (
            f"{port_move} Top contributors: {top}. "
            f"Largest drags: {bottom}. "
            f"Regime: {regime} volatility. {vol}."
        )

    elif style == "risk_manager":
        return (
            f"RISK SUMMARY: {port_move}\n\n"
            f"Risk Metrics: {vol}. {sharpe}. {var_str}. Max drawdown: {ctx.get('max_dd', 'N/A')}.\n\n"
            f"Volatility Regime: Currently in {regime} volatility environment.\n\n"
            f"Key Exposures: Top risk contributors — {top}. "
            f"Underperformers — {bottom}.\n\n"
            f"Sector Attribution: {sectors}\n\n"
            f"Recommendation: {'Monitor elevated risk positions closely.' if regime == 'High' else 'Risk levels within normal parameters.'}"
        )

    elif style == "pm_investor":
        return (
            f"PORTFOLIO MANAGER BRIEF\n\n"
            f"{port_move}\n\n"
            f"Today's performance was primarily driven by {top}. "
            f"The largest detractors were {bottom}.\n\n"
            f"Sector breakdown: {sectors}\n\n"
            f"The portfolio is operating in a {regime.lower()} volatility regime with "
            f"{vol.lower()}. Risk-adjusted performance shows {sharpe}.\n\n"
            f"Key monitoring items: Names with outsized contribution should be reviewed "
            f"for position sizing relative to conviction level. "
            f"{'Consider reducing exposure given elevated volatility.' if regime == 'High' else 'Current positioning appears appropriate for the regime.'}"
        )

    else:  # detailed
        return (
            f"DETAILED PORTFOLIO COMMENTARY\n\n"
            f"Market Context & Portfolio Performance:\n"
            f"{port_move} The portfolio is operating in a {regime.lower()} volatility environment.\n\n"
            f"Attribution Analysis:\n"
            f"Top contributors to today's return: {top}.\n"
            f"Largest detractors: {bottom}.\n"
            f"Sector-level attribution: {sectors}\n\n"
            f"Risk Profile:\n"
            f"{vol}. {sharpe}. Historical {var_str}. {ctx.get('max_dd', '')}\n\n"
            f"Actionable Intelligence:\n"
            f"The most important names to monitor are the top contributors and detractors listed above. "
            f"Position sizing should be reviewed for any single-name contribution exceeding 50bps. "
            f"{'The elevated volatility regime warrants tighter stop-loss levels and reduced gross exposure.' if regime == 'High' else 'Risk parameters remain within acceptable bounds.'}"
        )


def _llm_commentary(ctx: Dict[str, str], style: str) -> Optional[str]:
    """Generate commentary using OpenAI API if available."""
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        style_instructions = {
            "concise": "Write a 2-3 sentence concise portfolio summary.",
            "detailed": "Write a detailed 4-paragraph portfolio commentary covering performance, attribution, risk, and actionable items.",
            "risk_manager": "Write from the perspective of a risk manager. Focus on risk metrics, exposures, regime, and protective actions.",
            "pm_investor": "Write from the perspective of a portfolio manager. Focus on alpha generation, positioning, and forward-looking views.",
        }

        prompt = f"""You are a senior portfolio analyst at a quantitative investment firm.
Generate portfolio commentary based on today's data.

Style: {style_instructions.get(style, style_instructions['detailed'])}

Data:
- {ctx.get('portfolio_move', '')}
- Top contributors: {ctx.get('top_drivers', 'N/A')}
- Largest drags: {ctx.get('bottom_drivers', 'N/A')}
- {ctx.get('volatility', '')}
- {ctx.get('sharpe', '')}
- {ctx.get('var', '')}
- Volatility regime: {ctx.get('vol_regime', 'Unknown')}
- Sector attribution: {ctx.get('sector_breakdown', '')}

Be specific with numbers. Do not hallucinate data not provided above."""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=800,
        )
        return response.choices[0].message.content

    except Exception as e:
        return None


def generate_commentary(state: PortfolioState) -> PortfolioState:
    """Generate commentary in multiple styles."""
    ctx = _build_context(state)

    styles = ["concise", "detailed", "risk_manager", "pm_investor"]
    commentary = {}
    source = "template"

    for style in styles:
        # Try LLM first, fall back to template
        llm_result = _llm_commentary(ctx, style)
        if llm_result:
            commentary[style] = llm_result
            source = "openai"
        else:
            commentary[style] = _template_commentary(ctx, style)

    state.commentary = commentary
    state.commentary = {
        **{s: c for s, c in commentary.items()},
        "_source": source,
        "_styles": styles,
    }

    return state
