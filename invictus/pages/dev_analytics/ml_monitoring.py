"""
invictus.pages.dev_analytics.ml_monitoring
===========================================
ML Monitoring — ensemble classifier health, prediction stability.
"""
import streamlit as st
import pandas as pd

from invictus.design import (
    render_section_header,
    BRAND_BLUE, DANGER_RED,
)
from invictus.pages.dev_analytics._shared import subtitle, conviction_card, health_color


def render_ml_monitoring():
    from invictus.observability.analyzers.drift import analyze_ml_drift

    ml = analyze_ml_drift()
    if ml.get("status") == "no_data":
        st.info("No ML prediction data collected yet.")
        return

    # ── Verdict Summary ───────────────────────────────────────────
    ar = ml["ensemble_agreement_rate"]
    hc = ml["high_confidence_pct"]
    stability = ml.get("ticker_stability", [])
    unstable = [t for t in stability if t.get("range_p", 0) > 0.3]

    flags = []
    if ar < 0.7:
        flags.append(f'Ensemble agreement <b style="color:{DANGER_RED};">{ar:.0%}</b> — models diverging significantly')
    if hc < 0.3:
        flags.append(f'Only <b style="color:{DANGER_RED};">{hc:.0%}</b> high-confidence predictions — model lacks conviction')
    if unstable:
        names = ", ".join(t["ticker"] for t in unstable[:3])
        flags.append(f'Unstable tickers (>30% range): <b style="color:{DANGER_RED};">{names}</b>')

    if flags:
        flag_html = "".join(f'<div style="margin:2px 0;">⚠ {f}</div>' for f in flags)
        st.markdown(
            f'<div style="background:#fef2f211;border-left:3px solid {DANGER_RED};padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:{DANGER_RED};">Red Flags</b>{flag_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:#22c55e08;border-left:3px solid #22c55e;padding:10px 14px;'
            f'border-radius:6px;font-size:12px;color:#475569;margin-bottom:12px;">'
            f'<b style="color:#22c55e;">All Clear</b> — '
            f'{ml["total_predictions"]} predictions, {ar:.0%} ensemble agreement, '
            f'{hc:.0%} high-confidence. Model stable.</div>',
            unsafe_allow_html=True,
        )

    render_section_header("ML Model Health")
    subtitle(
        'Bayesian accumulation model performance and prediction stability. '
        f'<span style="color:#22c55e;font-weight:700;">High agreement</span> = signals concur; '
        f'<span style="color:{DANGER_RED};font-weight:700;">low agreement</span> = model uncertainty.'
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1: conviction_card("Predictions", str(ml["total_predictions"]))
    with m2: conviction_card("Avg Probability", f"{ml['avg_probability']:.1%}")
    with m3: conviction_card("High Confidence", f"{ml['high_confidence_pct']:.0%}",
                              color=BRAND_BLUE, sub_label=">70% PROB")
    with m4:
        ar = ml["ensemble_agreement_rate"]
        conviction_card("Ensemble Agreement", f"{ar:.0%}",
                        color=health_color(ar, 0.85, 0.6),
                        sub_label="ALIGNED" if ar > 0.85 else "DIVERGENT")

    stability = ml.get("ticker_stability", [])
    if stability:
        render_section_header("Prediction Stability")
        subtitle(
            'Per-ticker prediction variance across multiple runs. '
            f'<span style="color:{DANGER_RED};font-weight:700;">High range</span> = unstable predictions — '
            'model may be overfitting to noise.'
        )
        st.dataframe(
            pd.DataFrame(stability).style.format({"avg_p": "{:.1%}", "range_p": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )
