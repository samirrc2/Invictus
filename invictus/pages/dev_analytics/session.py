"""
invictus.pages.dev_analytics.session
=====================================
Session Analytics — usage overview, feature adoption, pipeline breakdown.
"""
import streamlit as st

from invictus.design import render_section_header, BRAND_BLUE, DANGER_RED, SUCCESS_GREEN
from invictus.pages.dev_analytics._shared import subtitle, conviction_card, health_color

# ── Node Error Display ─────────────────────────────────────────────────

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
    "FATAL": "Fatal · Top-Level",
}


def _render_node_error_card(err: dict, idx: int):
    """Render a single structured error card with full traceback."""
    node = err.get("node", "Unknown")
    error_type = err.get("error_type", "Exception")
    error_msg = err.get("error_message", "No message")
    tb = err.get("traceback", "")
    ts = err.get("timestamp", "")
    ctx = err.get("context", {})
    stage = _STAGE_MAP.get(node, "Unknown Stage")

    # Severity badge color
    is_fatal = node == "FATAL"
    badge_bg = "#7f1d1d" if is_fatal else "#991b1b"
    badge_text = "FATAL" if is_fatal else error_type

    # Header card
    st.markdown(
        f'<div style="border:1px solid #fca5a5;border-radius:8px;overflow:hidden;'
        f'margin-bottom:4px;">'
        f'<div style="background:#fef2f2;padding:12px 16px;border-bottom:1px solid #fecaca;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:6px;">'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="background:{badge_bg};color:#fff;font-size:10px;font-weight:800;'
        f'padding:2px 8px;border-radius:4px;text-transform:uppercase;letter-spacing:0.05em;">'
        f'{badge_text}</span>'
        f'<span style="font-size:15px;font-weight:800;color:#0f172a;font-family:monospace;">'
        f'{node}</span>'
        f'</div>'
        f'<span style="font-size:11px;color:#94a3b8;font-family:monospace;">{ts[:19] if ts else ""}</span>'
        f'</div>'
        f'<div style="font-size:12px;color:#64748b;margin-bottom:4px;">{stage}</div>'
        f'<div style="font-size:13px;color:#dc2626;font-weight:600;font-family:monospace;'
        f'word-break:break-all;">{error_msg[:300]}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Expandable details
    with st.expander(f"Full Traceback — {node}", expanded=(idx == 0)):
        if tb:
            st.code(tb, language="python")
        else:
            st.caption("No traceback captured.")

        if ctx:
            st.markdown(
                '<div style="font-size:12px;font-weight:700;color:#475569;'
                'margin:8px 0 4px 0;">Input Context at Failure</div>',
                unsafe_allow_html=True,
            )
            ctx_lines = []
            for k, v in ctx.items():
                ctx_lines.append(f"  {k}: {v}")
            st.code("\n".join(ctx_lines), language="yaml")


def render_session_analytics():
    # ── Per-Node Error Log (structured, with full tracebacks) ────
    node_errors = st.session_state.get("node_errors", [])
    legacy_errors = st.session_state.get("pipeline_errors", [])

    if node_errors:
        render_section_header("Pipeline Error Log")
        subtitle(
            f'<span style="color:#ef4444;font-weight:700;">{len(node_errors)} node failure(s)</span> '
            f'captured from the last pipeline run — per-node tracebacks, input context, and timestamps.'
        )

        # Summary strip: which nodes failed
        failed_nodes = [e.get("node", "?") for e in node_errors]
        pills = " ".join(
            f'<span style="background:#fef2f2;border:1px solid #fca5a5;color:#dc2626;'
            f'font-size:11px;font-weight:700;padding:2px 10px;border-radius:12px;'
            f'font-family:monospace;">{n}</span>'
            for n in failed_nodes
        )
        st.markdown(
            f'<div style="margin-bottom:12px;">{pills}</div>',
            unsafe_allow_html=True,
        )

        for i, err in enumerate(node_errors):
            _render_node_error_card(err, i)

        st.markdown('<div style="margin-bottom:16px;"></div>', unsafe_allow_html=True)

    elif legacy_errors:
        # Fallback: show legacy string errors if no structured data
        render_section_header("Pipeline Error Log")
        subtitle(
            f'<span style="color:#ef4444;font-weight:700;">{len(legacy_errors)} error(s)</span> '
            f'captured (legacy format — upgrade pipeline for full tracebacks).'
        )
        for i, err in enumerate(legacy_errors):
            with st.expander(f"Error {i + 1}: {err[:80].split(chr(10))[0]}", expanded=(i == 0)):
                st.code(err, language="python")
        st.markdown('<div style="margin-bottom:16px;"></div>', unsafe_allow_html=True)

    from invictus.observability.analyzers.calibration import analyze_session_analytics

    sa = analyze_session_analytics()
    if sa.get("status") == "no_data":
        if not errors:
            st.info("No session data collected yet.")
        return

    # ── Verdict Summary ───────────────────────────────────────────
    pcr = sa["pipeline_completion_rate"]
    demo_pct = sa["demo_mode_pct"]

    flags = []
    if pcr < 0.8:
        flags.append(f'Pipeline completion <b style="color:#ef4444;">{pcr:.0%}</b> — pipelines failing before finishing')
    if demo_pct > 0.8:
        flags.append(f'{demo_pct:.0%} of runs are demo mode — limited real data validation')
    avg_time = (sa.get("pipeline_avg_time_ms") or 0) / 1000
    if avg_time > 120:
        flags.append(f'Avg pipeline time <b>{avg_time:.0f}s</b> — may need optimization')

    if flags:
        flag_html = "".join(f'<div style="margin:2px 0;">⚠ {f}</div>' for f in flags)
        st.markdown(
            f'<div style="background:#fef2f211;border-left:3px solid #ef4444;padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:#ef4444;">Red Flags</b>{flag_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:#22c55e08;border-left:3px solid #22c55e;padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:#22c55e;">All Clear</b> — '
            f'{sa["unique_sessions"]} sessions, {pcr:.0%} completion rate, '
            f'avg pipeline time {avg_time:.1f}s.</div>',
            unsafe_allow_html=True,
        )

    render_section_header("Usage Overview")
    subtitle(
        'Platform usage metrics and feature adoption. '
        '<span style="color:#94a3b8;">Pipeline completion rate measures end-to-end reliability.</span>'
    )
    u1, u2, u3, u4 = st.columns(4)
    with u1: conviction_card("Sessions", str(sa["unique_sessions"]))
    with u2: conviction_card("Events", str(sa["total_events"]))
    with u3: conviction_card("Pipeline Runs", str(sa["pipeline_starts"]))
    with u4:
        pcr = sa["pipeline_completion_rate"]
        conviction_card("Completion", f"{pcr:.0%}", color=health_color(pcr))

    features = sa.get("feature_adoption", [])
    if features:
        render_section_header("Feature Adoption")
        subtitle('Most-used platform features ranked by usage count.')
        for f in features[:10]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:6px 12px;'
                f'border-bottom:1px solid #f1f5f9;">'
                f'<span style="font-size:13px;color:#334155;">{f["detail"]}</span>'
                f'<span style="font-size:13px;font-weight:700;color:{BRAND_BLUE};'
                f'font-variant-numeric:tabular-nums;">{f["uses"]}</span></div>',
                unsafe_allow_html=True,
            )

    render_section_header("Pipeline Breakdown")
    subtitle('Execution mode and timing statistics.')
    p1, p2, p3 = st.columns(3)
    with p1: conviction_card("Demo Mode %", f"{sa['demo_mode_pct']:.0%}")
    with p2: conviction_card("Avg Time", f"{(sa['pipeline_avg_time_ms'] or 0) / 1000:.1f}s")
    with p3: conviction_card("Completion", f"{sa['pipeline_completion_rate']:.0%}")
