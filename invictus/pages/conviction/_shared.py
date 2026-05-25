"""
Shared helpers for Conviction Intelligence sub-tabs.
Colors, score formatters, callout boxes.
"""
from invictus.design import (
    render_section_header, render_metric_card, render_commentary_box,
    apply_invictus_layout,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT, SLATE_500, SLATE_100,
)


def score_color(v: float, threshold: float = 0.1) -> str:
    """Return green/red/grey based on score magnitude."""
    if v > threshold:
        return SUCCESS_GREEN
    if v < -threshold:
        return DANGER_RED
    return "#94a3b8"


def conviction_color(level: str) -> str:
    return {
        "STRONG CONVICTION": SUCCESS_GREEN, "HIGH": "#10b981",
        "MODERATE POSITIVE": BRAND_BLUE, "NEUTRAL": "#94a3b8",
        "MODERATE NEGATIVE": "#f59e0b", "LOW / RISK": DANGER_RED,
    }.get(level, BRAND_BLUE)


def intelligence_callout(text: str) -> None:
    import streamlit as st
    st.markdown(
        f'<div style="border-left:3px solid {BRAND_BLUE};background:rgba(29,78,216,0.04);'
        f'padding:12px 16px;border-radius:0 6px 6px 0;margin:12px 0;'
        f'font-size:13px;color:#1e293b;line-height:1.7;font-style:italic;">{text}</div>',
        unsafe_allow_html=True,
    )


def ticker_section_header(ticker: str, label: str, score: float = None) -> None:
    """Render a per-ticker section header with optional score, matching flows gold standard."""
    import streamlit as st
    score_html = ""
    if score is not None:
        sc = score_color(score)
        score_html = f'<span style="color:{sc};font-size:16px;">{score:+.2f}</span>'
    st.markdown(
        f'<div style="font-size:14px;font-weight:800;color:{BRAND_BLUE};'
        f'background:#f8fafc;border-left:5px solid {BRAND_BLUE};'
        f'padding:10px 20px;margin:16px 0 8px 0;border-radius:0 8px 8px 0;'
        f'text-transform:uppercase;letter-spacing:1px;'
        f'display:flex;justify-content:space-between;align-items:center;">'
        f'{ticker} — {label}'
        f'{score_html}</div>',
        unsafe_allow_html=True,
    )


def sub_section_header(label: str, score_text: str = "", score_color_val: str = "") -> None:
    """Render a sub-section header inside an expander (e.g. Insider Intelligence +0.32)."""
    import streamlit as st
    score_html = ""
    if score_text:
        score_html = f'<span style="color:{score_color_val};float:right;font-size:13px;">{score_text}</span>'
    st.markdown(
        f'<div style="font-size:13px;font-weight:800;color:{BRAND_BLUE};letter-spacing:0.1em;'
        f'text-transform:uppercase;margin:10px 0 8px 0;border-bottom:2px solid {BRAND_BLUE};'
        f'padding-bottom:4px;">{label} {score_html}</div>',
        unsafe_allow_html=True,
    )


def flags_two_column(green_flags: list, red_flags: list,
                     green_label: str = "Strengths",
                     red_label: str = "Concerns") -> None:
    """Render green/red flag columns."""
    import streamlit as st
    if not green_flags and not red_flags:
        return
    fl1, fl2 = st.columns(2)
    with fl1:
        if green_flags:
            st.markdown(
                f'<div style="font-size:12px;font-weight:800;color:{SUCCESS_GREEN};'
                f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">'
                f'{green_label}</div>', unsafe_allow_html=True,
            )
            for item in green_flags:
                st.markdown(
                    f'<div style="border-left:2px solid {SUCCESS_GREEN};padding:3px 10px;'
                    f'margin-bottom:3px;font-size:12px;color:#334155;">{item}</div>',
                    unsafe_allow_html=True,
                )
    with fl2:
        if red_flags:
            st.markdown(
                f'<div style="font-size:12px;font-weight:800;color:{DANGER_RED};'
                f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">'
                f'{red_label}</div>', unsafe_allow_html=True,
            )
            for item in red_flags:
                st.markdown(
                    f'<div style="border-left:2px solid {DANGER_RED};padding:3px 10px;'
                    f'margin-bottom:3px;font-size:12px;color:#334155;">{item}</div>',
                    unsafe_allow_html=True,
                )


def formula_box(text: str) -> None:
    """Render a monospace formula callout."""
    import streamlit as st
    st.markdown(
        f'<div style="margin-top:12px;padding:10px 16px;background:rgba(29,78,216,0.04);'
        f'border-left:3px solid {BRAND_BLUE};border-radius:0 6px 6px 0;'
        f'font-family:monospace;font-size:12px;color:#1e293b;">{text}</div>',
        unsafe_allow_html=True,
    )


def source_line(source: str, formula: str) -> None:
    """Render a small grey source + formula line."""
    import streamlit as st
    st.markdown(
        f'<div style="font-size:12px;color:#64748b;margin-top:6px;font-family:monospace;">'
        f'Source: {source} | {formula}</div>',
        unsafe_allow_html=True,
    )


def analyst_ratings_banner(grades: dict) -> None:
    """
    Render a compact, institutional-grade analyst ratings strip.

    Expects grades dict with keys: strongBuy, buy, hold, sell, strongSell, consensus.
    Renders a horizontal bar showing the distribution + consensus label.
    """
    if not grades:
        return
    import streamlit as st

    sb  = grades.get("strongBuy", 0)
    b   = grades.get("buy", 0)
    h   = grades.get("hold", 0)
    s   = grades.get("sell", 0)
    ss  = grades.get("strongSell", 0)
    total = sb + b + h + s + ss
    consensus = grades.get("consensus", "N/A")

    if total == 0:
        return

    # Consensus color
    con_color = SUCCESS_GREEN if consensus in ("Buy", "Strong Buy") else \
                DANGER_RED if consensus in ("Sell", "Strong Sell") else "#94a3b8"

    # Build proportional bar segments
    segments = [
        (sb,  "Strong Buy",  "#047857"),   # dark green
        (b,   "Buy",         SUCCESS_GREEN),
        (h,   "Hold",        "#94a3b8"),
        (s,   "Sell",        "#f59e0b"),
        (ss,  "Strong Sell", DANGER_RED),
    ]
    bar_html = ""
    for count, label, color in segments:
        if count > 0:
            pct = count / total * 100
            bar_html += (
                f'<div style="width:{pct}%;background:{color};height:6px;'
                f'display:inline-block;" title="{label}: {count}"></div>'
            )

    # Build pill tags
    pills = []
    for count, label, color in segments:
        if count > 0:
            pills.append(
                f'<span style="font-size:12px;color:{color};font-weight:700;'
                f'margin-right:12px;white-space:nowrap;">{label}: {count}</span>'
            )

    st.markdown(
        f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:10px 16px;'
        f'background:#fafbfc;margin-bottom:12px;">'
        # Row 1: Consensus label + total analysts
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:6px;">'
        f'<span style="font-size:12px;font-weight:800;color:{con_color};'
        f'text-transform:uppercase;letter-spacing:0.08em;">'
        f'Analyst Consensus: {consensus}</span>'
        f'<span style="font-size:12px;color:#64748b;">{total} analysts</span>'
        f'</div>'
        # Row 2: Proportional bar
        f'<div style="width:100%;height:6px;background:#f1f5f9;border-radius:3px;'
        f'overflow:hidden;display:flex;margin-bottom:8px;">{bar_html}</div>'
        # Row 3: Count pills
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;">'
        f'{"".join(pills)}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
