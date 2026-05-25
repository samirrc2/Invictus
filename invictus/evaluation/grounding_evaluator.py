"""
invictus.evaluation.grounding_evaluator
========================================
Numerical grounding & factual accuracy evaluation.
Cross-references numbers in LLM output against actual PortfolioState data
to compute grounding rate and detect hallucinated numbers.

Target grounding rate: >85%.
"""
import re
from typing import Dict, Any, List, Optional, Tuple

from invictus.agents.graph_state import PortfolioState


# ── Number Extraction Patterns ──────────────────────────────────────
_PCT_PATTERN = re.compile(r'[+-]?\d+\.?\d*\s*%')               # 12.5%, -3.2%
_DOLLAR_PATTERN = re.compile(r'\$[+-]?\d[\d,]*\.?\d*')          # $1,234.56
_DECIMAL_PATTERN = re.compile(r'(?<!\w)[+-]?\d+\.\d{1,4}(?!\w)')  # 0.85, 1.234 (ratios/scores)
_BPS_PATTERN = re.compile(r'[+-]?\d+\.?\d*\s*(?:bps|basis points)', re.IGNORECASE)

# ── Metric Context Keywords ─────────────────────────────────────────
_METRIC_KEYWORDS = {
    "return": ["return", "performance", "gained", "lost", "up", "down", "moved"],
    "volatility": ["volatility", "vol", "annualized vol"],
    "var": ["var", "value at risk", "value-at-risk"],
    "sharpe": ["sharpe", "risk-adjusted"],
    "sortino": ["sortino"],
    "max_drawdown": ["drawdown", "max drawdown", "maximum drawdown"],
    "weight": ["weight", "allocation", "exposure", "position"],
    "contribution": ["contribution", "contributor", "detractor", "attribution"],
    "beta": ["beta"],
    "correlation": ["correlation", "correlated"],
    "cvar": ["cvar", "conditional var", "expected shortfall"],
}


def _extract_numbers(text: str) -> List[Dict[str, Any]]:
    """
    Extract all numerical values from text with their surrounding context.
    Returns list of {value, raw, type, context, position}.
    """
    extracted = []

    # Percentages
    for m in _PCT_PATTERN.finditer(text):
        raw = m.group()
        val = float(raw.replace('%', '').replace(' ', '').replace(',', ''))
        # Get surrounding context (±40 chars)
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        context = text[start:end].strip()
        extracted.append({
            "value": val / 100,  # normalize to decimal
            "raw": raw.strip(),
            "type": "percentage",
            "context": context,
            "position": m.start(),
        })

    # Dollar amounts
    for m in _DOLLAR_PATTERN.finditer(text):
        raw = m.group()
        val = float(raw.replace('$', '').replace(',', ''))
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        context = text[start:end].strip()
        extracted.append({
            "value": val,
            "raw": raw.strip(),
            "type": "dollar",
            "context": context,
            "position": m.start(),
        })

    # Basis points
    for m in _BPS_PATTERN.finditer(text):
        raw = m.group()
        val = float(re.search(r'[+-]?\d+\.?\d*', raw).group())
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        context = text[start:end].strip()
        extracted.append({
            "value": val / 10000,  # normalize to decimal
            "raw": raw.strip(),
            "type": "bps",
            "context": context,
            "position": m.start(),
        })

    # Deduplicate by position (percentages may overlap with decimals)
    seen_positions = set()
    deduped = []
    for item in sorted(extracted, key=lambda x: x["position"]):
        if item["position"] not in seen_positions:
            deduped.append(item)
            seen_positions.add(item["position"])

    return deduped


def _classify_metric(context: str) -> Optional[str]:
    """
    Classify what metric a number likely refers to based on surrounding text.
    """
    context_lower = context.lower()
    for metric, keywords in _METRIC_KEYWORDS.items():
        if any(kw in context_lower for kw in keywords):
            return metric
    return None


def _build_source_values(state: PortfolioState) -> Dict[str, List[float]]:
    """
    Extract all ground-truth numerical values from PortfolioState,
    organized by metric type.
    """
    sources: Dict[str, List[float]] = {
        "return": [],
        "volatility": [],
        "var": [],
        "sharpe": [],
        "sortino": [],
        "max_drawdown": [],
        "weight": [],
        "contribution": [],
        "beta": [],
        "correlation": [],
        "cvar": [],
        "other": [],
    }

    # Risk metrics
    rm = state.risk_metrics or {}
    if rm:
        if "annualized_volatility" in rm:
            sources["volatility"].append(rm["annualized_volatility"])
        if "var_95_historical" in rm:
            sources["var"].append(rm["var_95_historical"])
        if "var_95_parametric" in rm:
            sources["var"].append(rm["var_95_parametric"])
        if "var_95_monte_carlo" in rm:
            sources["var"].append(rm["var_95_monte_carlo"])
        if "cvar_95" in rm:
            sources["cvar"].append(rm["cvar_95"])
        if "sharpe_ratio" in rm:
            sources["sharpe"].append(rm["sharpe_ratio"])
        if "sortino_ratio" in rm:
            sources["sortino"].append(rm["sortino_ratio"])
        if "max_drawdown" in rm:
            sources["max_drawdown"].append(rm["max_drawdown"])
        if "beta" in rm:
            sources["beta"].append(rm["beta"])

    # P&L attribution
    pnl = state.pnl_attribution or {}
    if pnl:
        if "portfolio_return" in pnl:
            sources["return"].append(pnl["portfolio_return"])
        top = pnl.get("top_contributors", [])
        for t in top:
            if "contribution" in t:
                sources["contribution"].append(t["contribution"])
            if "return" in t:
                sources["return"].append(t["return"])
        bottom = pnl.get("bottom_contributors", [])
        for b in bottom:
            if "contribution" in b:
                sources["contribution"].append(b["contribution"])
            if "return" in b:
                sources["return"].append(b["return"])

    # Weights
    if state.weights:
        for w in state.weights.values():
            sources["weight"].append(w)

    # Portfolio-level values
    if state.daily_return_pct is not None:
        sources["return"].append(state.daily_return_pct / 100 if abs(state.daily_return_pct) > 1 else state.daily_return_pct)
    if state.total_value is not None:
        sources["other"].append(state.total_value)
    if state.total_unrealized_pnl is not None:
        sources["other"].append(state.total_unrealized_pnl)

    # Flatten for general matching
    sources["_all"] = []
    for vals in sources.values():
        if isinstance(vals, list):
            sources["_all"].extend(vals)

    return sources


def _match_number(
    extracted: Dict[str, Any],
    sources: Dict[str, List[float]],
    pct_tolerance: float = 0.005,    # ±0.5% for percentages
    dollar_tolerance: float = 1.0,    # ±$1 for dollar values
    ratio_tolerance: float = 0.05,    # ±0.05 for ratios/scores
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Try to match an extracted number against source values.
    Returns (is_grounded, matched_metric, matched_source_value).
    """
    value = extracted["value"]
    metric_hint = _classify_metric(extracted["context"])
    num_type = extracted["type"]

    # Determine tolerance based on type
    if num_type == "percentage" or num_type == "bps":
        tolerance = pct_tolerance
    elif num_type == "dollar":
        tolerance = dollar_tolerance
    else:
        tolerance = ratio_tolerance

    # First try: match within the hinted metric category
    if metric_hint and metric_hint in sources:
        for source_val in sources[metric_hint]:
            if abs(value - source_val) <= tolerance:
                return True, metric_hint, source_val

    # Second try: match across all source values
    for source_val in sources.get("_all", []):
        if num_type == "dollar":
            if abs(value - source_val) <= dollar_tolerance:
                return True, "matched_general", source_val
        else:
            if abs(value - source_val) <= tolerance:
                return True, "matched_general", source_val

    # Third try: percentage might be expressed as raw number (12.5 vs 0.125)
    if num_type == "percentage":
        raw_pct = value * 100  # convert back
        for source_val in sources.get("_all", []):
            if abs(raw_pct - source_val) <= 0.5:
                return True, "matched_raw_pct", source_val

    return False, None, None


def _check_direction_consistency(
    text: str,
    state: PortfolioState,
) -> List[Dict[str, str]]:
    """
    Check that directional language matches actual data.
    E.g., "portfolio gained" when portfolio_return is negative → inconsistency.
    """
    issues = []
    pnl = state.pnl_attribution or {}
    port_ret = pnl.get("portfolio_return", 0)
    text_lower = text.lower()

    # Portfolio direction
    positive_words = ["gained", "rallied", "surged", "climbed", "rose", "increased", "up"]
    negative_words = ["fell", "declined", "dropped", "lost", "decreased", "down", "slid"]

    has_positive = any(w in text_lower for w in positive_words)
    has_negative = any(w in text_lower for w in negative_words)

    # Exclude words used in attribution context (e.g., "detractor dragged down")
    attribution_context = ["contributor", "detractor", "drag", "driven by", "offset by"]
    in_attribution = any(w in text_lower for w in attribution_context)

    if port_ret >= 0 and has_negative and not has_positive and not in_attribution:
        issues.append({
            "type": "direction_mismatch",
            "detail": f"Commentary uses negative language but portfolio return was +{port_ret:.2%}",
            "severity": "high",
        })
    elif port_ret < 0 and has_positive and not has_negative and not in_attribution:
        issues.append({
            "type": "direction_mismatch",
            "detail": f"Commentary uses positive language but portfolio return was {port_ret:.2%}",
            "severity": "high",
        })

    # Check for magnitude misrepresentation
    if port_ret is not None and abs(port_ret) < 0.005:  # <0.5%
        dramatic_words = ["surged", "crashed", "soared", "plummeted", "spiked", "collapsed"]
        if any(w in text_lower for w in dramatic_words):
            issues.append({
                "type": "magnitude_mismatch",
                "detail": f"Dramatic language used for small move of {port_ret:+.2%}",
                "severity": "medium",
            })

    return issues


def evaluate_grounding(
    text: str,
    state: PortfolioState,
) -> Dict[str, Any]:
    """
    Full grounding evaluation of a piece of LLM-generated text.

    Returns:
        grounding_rate: float (0-1), percentage of numbers that match source data
        total_numbers: int
        grounded_numbers: int
        hallucinated: list of unmatched numbers with context
        direction_issues: list of directional inconsistencies
        details: per-number match results
    """
    # Extract numbers from text
    extracted = _extract_numbers(text)
    if not extracted:
        return {
            "grounding_rate": 1.0,  # no numbers to hallucinate
            "total_numbers": 0,
            "grounded_numbers": 0,
            "hallucinated": [],
            "direction_issues": [],
            "details": [],
            "status": "no_numbers",
        }

    # Build source values from state
    sources = _build_source_values(state)

    # Match each extracted number
    details = []
    grounded = 0
    hallucinated = []

    for item in extracted:
        is_grounded, matched_metric, matched_value = _match_number(item, sources)
        detail = {
            "raw": item["raw"],
            "value": item["value"],
            "type": item["type"],
            "context": item["context"],
            "is_grounded": is_grounded,
            "matched_metric": matched_metric,
            "matched_source": matched_value,
        }
        details.append(detail)

        if is_grounded:
            grounded += 1
        else:
            hallucinated.append({
                "raw": item["raw"],
                "context": item["context"],
                "classified_as": _classify_metric(item["context"]),
            })

    # Direction consistency check
    direction_issues = _check_direction_consistency(text, state)

    total = len(extracted)
    grounding_rate = grounded / max(total, 1)

    # Apply penalty for direction issues
    direction_penalty = len([i for i in direction_issues if i["severity"] == "high"]) * 0.1
    adjusted_rate = max(0, grounding_rate - direction_penalty)

    return {
        "grounding_rate": grounding_rate,
        "adjusted_grounding_rate": adjusted_rate,
        "total_numbers": total,
        "grounded_numbers": grounded,
        "hallucinated_count": len(hallucinated),
        "hallucinated": hallucinated[:10],  # cap for display
        "direction_issues": direction_issues,
        "details": details,
    }


def evaluate_all_commentary(state: PortfolioState) -> Dict[str, Any]:
    """
    Evaluate grounding across all commentary styles in state.
    """
    commentary = state.commentary or {}
    styles = commentary.get("_styles", [])
    if not styles:
        return {"status": "no_commentary", "results": {}}

    results = {}
    for style in styles:
        text = commentary.get(style, "")
        if text:
            results[style] = evaluate_grounding(text, state)

    # Aggregate
    total_numbers = sum(r["total_numbers"] for r in results.values())
    total_grounded = sum(r["grounded_numbers"] for r in results.values())
    all_hallucinated = []
    for r in results.values():
        all_hallucinated.extend(r["hallucinated"])
    all_direction_issues = []
    for r in results.values():
        all_direction_issues.extend(r["direction_issues"])

    return {
        "results": results,
        "aggregate": {
            "overall_grounding_rate": total_grounded / max(total_numbers, 1),
            "total_numbers_checked": total_numbers,
            "total_grounded": total_grounded,
            "total_hallucinated": len(all_hallucinated),
            "direction_issues": len(all_direction_issues),
            "best_style": max(results, key=lambda s: results[s]["grounding_rate"]) if results else "N/A",
            "worst_style": min(results, key=lambda s: results[s]["grounding_rate"]) if results else "N/A",
        },
    }
