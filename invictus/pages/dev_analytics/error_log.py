"""
invictus.pages.dev_analytics.error_log
=======================================
Dedicated Error Log — per-node failures with full tracebacks, line numbers,
module paths, input context, and timestamps.

Always shows errors from the most recent pipeline run. If no errors exist,
displays a green "All Clear" banner.
"""
import streamlit as st

from invictus.design import render_section_header, BRAND_BLUE, DANGER_RED, SUCCESS_GREEN
from invictus.pages.dev_analytics._shared import subtitle, conviction_card, health_color


# ── Stage Labels ───────────────────────────────────────────────────────

_STAGE_MAP = {
    "load_portfolio": "Stage 0 · Load",
    "compute_risk": "Stage 1 · Portfolio Intelligence",
    "run_pca": "Stage 1 · Portfolio Intelligence",
    "detect_vol_regime": "Stage 1 · Portfolio Intelligence",
    "run_stress_tests": "Stage 1 · Portfolio Intelligence",
    "compute_greeks": "Stage 1 · Portfolio Intelligence",
    "attribute_pnl": "Stage 1 · Portfolio Intelligence",
    "analyze_flows": "Stage 2 · Conviction Intelligence",
    "retrieve_10k_context": "Stage 2 · Conviction Intelligence",
    "run_filing_intel": "Stage 2 · Conviction Intelligence",
    "run_earnings_intel": "Stage 2 · Conviction Intelligence",
    "run_accumulation_model": "Stage 3 · ML Accumulation",
    "run_conviction_synthesis": "Stage 4 · Synthesis",
    "generate_commentary": "Stage 5 · Commentary",
    "evaluate_commentary": "Stage 6 · Evaluation",
    "FATAL": "Fatal · Top-Level Crash",
}


def _extract_origin(tb: str) -> str:
    """Pull the failing file + line from a traceback string.

    Looks for the last 'File "...", line N, in ...' before the exception.
    Returns something like: 'ml_agent.py:331 in _compute_features'
    """
    if not tb:
        return "Unknown"
    lines = tb.strip().splitlines()
    # Walk backwards to find the last File line (that's the actual crash site)
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith('File "'):
            try:
                # File "/path/to/ml_agent.py", line 331, in _compute_features
                parts = stripped.split('"')
                filepath = parts[1] if len(parts) >= 2 else ""
                filename = filepath.split("/")[-1] if "/" in filepath else filepath
                rest = stripped.split(", ")
                line_num = rest[1] if len(rest) >= 2 else ""
                func = rest[2] if len(rest) >= 3 else ""
                return f"{filename}:{line_num.replace('line ', '')} {func}"
            except (IndexError, ValueError):
                continue
    return "Unknown"


def _render_error_card(err: dict, idx: int, total: int):
    """Render one error card with full diagnostic detail."""
    node = err.get("node", "Unknown")
    error_type = err.get("error_type", "Exception")
    error_msg = err.get("error_message", "No message")
    tb = err.get("traceback", "")
    ts = err.get("timestamp", "")
    ctx = err.get("context", {})
    stage = _STAGE_MAP.get(node, "Unknown Stage")
    origin = _extract_origin(tb)

    is_fatal = node == "FATAL"
    badge_bg = "#7f1d1d" if is_fatal else "#991b1b"
    badge_label = "FATAL CRASH" if is_fatal else error_type

    # ── Header Card ──
    st.markdown(
        f'<div style="border:1px solid #fca5a5;border-radius:10px;overflow:hidden;'
        f'margin-bottom:2px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
        # Top bar
        f'<div style="background:#fef2f2;padding:14px 18px;border-bottom:1px solid #fecaca;">'
        # Row 1: badge + node + timestamp
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:8px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="background:{badge_bg};color:#fff;font-size:10px;font-weight:800;'
        f'padding:3px 10px;border-radius:4px;text-transform:uppercase;'
        f'letter-spacing:0.06em;">{badge_label}</span>'
        f'<span style="font-size:16px;font-weight:800;color:#0f172a;'
        f'font-family:\'SF Mono\',Monaco,Consolas,monospace;">{node}</span>'
        f'<span style="font-size:12px;color:#94a3b8;font-style:italic;">'
        f'{stage}</span>'
        f'</div>'
        f'<span style="font-size:11px;color:#94a3b8;'
        f'font-family:\'SF Mono\',Monaco,Consolas,monospace;">'
        f'{ts[:19].replace("T", " ") if ts else ""}</span>'
        f'</div>'
        # Row 2: error message
        f'<div style="font-size:14px;color:#dc2626;font-weight:700;'
        f'font-family:\'SF Mono\',Monaco,Consolas,monospace;'
        f'word-break:break-all;margin-bottom:6px;line-height:1.5;">'
        f'{error_msg[:400]}</div>'
        # Row 3: crash origin
        f'<div style="font-size:12px;color:#475569;">'
        f'<span style="font-weight:700;">Origin:</span> '
        f'<span style="font-family:\'SF Mono\',Monaco,Consolas,monospace;'
        f'color:#0f172a;">{origin}</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Expandable: Full Traceback ──
    with st.expander(f"Full Traceback — {node}", expanded=(idx == 0)):
        if tb:
            st.code(tb, language="python")
        else:
            st.caption("No traceback captured.")

        # Input context
        if ctx:
            st.markdown(
                '<div style="font-size:13px;font-weight:800;color:#0f172a;'
                'margin:12px 0 6px 0;">Input Context at Failure</div>',
                unsafe_allow_html=True,
            )
            ctx_lines = []
            for k, v in ctx.items():
                ctx_lines.append(f"{k}: {v}")
            st.code("\n".join(ctx_lines), language="yaml")

    st.markdown('<div style="margin-bottom:8px;"></div>', unsafe_allow_html=True)


def render_error_log():
    """Render the dedicated Error Log tab."""
    render_section_header("Pipeline Error Log")

    node_errors = st.session_state.get("node_errors", [])
    legacy_errors = st.session_state.get("pipeline_errors", [])
    has_errors = bool(node_errors) or bool(legacy_errors)

    if not has_errors:
        # ── All Clear Banner ──
        st.markdown(
            '<div style="background:#f0fdf4;border:1px solid #bbf7d0;'
            'border-radius:10px;padding:20px 24px;text-align:center;">'
            '<div style="font-size:15px;font-weight:800;color:#16a34a;'
            'margin-bottom:4px;">All Clear</div>'
            '<div style="font-size:13px;color:#64748b;">'
            'No errors captured from the last pipeline run. '
            'Errors will appear here automatically when a node fails — '
            'with full tracebacks, crash origin (file + line), and input data context.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Summary Strip ──
    if node_errors:
        n = len(node_errors)
        failed_nodes = [e.get("node", "?") for e in node_errors]
        error_types = [e.get("error_type", "?") for e in node_errors]

        subtitle(
            f'<span style="color:#ef4444;font-weight:800;">{n} node failure(s)</span> '
            f'from the last pipeline run. Full tracebacks with crash origin below.'
        )

        # Failed node pills
        pills = " ".join(
            f'<span style="background:#fef2f2;border:1px solid #fca5a5;color:#dc2626;'
            f'font-size:11px;font-weight:700;padding:3px 12px;border-radius:14px;'
            f'font-family:\'SF Mono\',Monaco,Consolas,monospace;">{nd}</span>'
            for nd in failed_nodes
        )
        st.markdown(
            f'<div style="margin-bottom:16px;">{pills}</div>',
            unsafe_allow_html=True,
        )

        # Summary cards
        c1, c2, c3 = st.columns(3)
        with c1:
            conviction_card("Failures", str(n), color=DANGER_RED)
        with c2:
            unique_types = list(set(error_types))
            conviction_card("Error Types", ", ".join(unique_types[:3]), color="#0f172a")
        with c3:
            unique_stages = list(set(_STAGE_MAP.get(nd, "?") for nd in failed_nodes))
            conviction_card("Stages Hit", str(len(unique_stages)), color=BRAND_BLUE)

        st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)

        # ── Per-Error Cards ──
        for i, err in enumerate(node_errors):
            _render_error_card(err, i, n)

    elif legacy_errors:
        # Fallback: legacy string errors (pre-upgrade runs)
        subtitle(
            f'<span style="color:#ef4444;font-weight:700;">{len(legacy_errors)} error(s)</span> '
            f'captured (legacy format — run pipeline again for full structured tracebacks).'
        )
        for i, err in enumerate(legacy_errors):
            with st.expander(
                f"Error {i + 1}: {err[:80].split(chr(10))[0]}",
                expanded=(i == 0),
            ):
                st.code(err, language="python")
