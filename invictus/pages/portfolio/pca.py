"""
Factor Decomposition — PCA risk factor analysis.

Concentration metrics, scree plot, factor loadings heatmap, factor interpretation.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from invictus.pages.portfolio._shared import (
    render_section_header, render_concentration_banner,
    apply_invictus_layout, subtitle,
    BRAND_BLUE, SUCCESS_GREEN, DANGER_RED, DANGER_RED_ALT, SLATE_50,
)


def _interpret_factor(loadings_col):
    """Infer what a principal component likely represents based on loading pattern."""
    abs_loads = loadings_col.abs()
    top_tickers = abs_loads.nlargest(3).index.tolist()
    spread = abs_loads.max() - abs_loads.min()

    # If all loadings are similar sign/magnitude → market factor
    if abs_loads.std() < 0.15 and (loadings_col > 0).all():
        return "Market Beta", "Broad market directional exposure"
    # If highly concentrated in 1-2 names
    if abs_loads.iloc[0] > 0.6:
        return f"Idiosyncratic ({top_tickers[0]})", f"Dominated by {top_tickers[0]} specific risk"
    # If mixed signs → long/short or rotation
    pos = (loadings_col > 0.2).sum()
    neg = (loadings_col < -0.2).sum()
    if pos > 0 and neg > 0:
        return "Sector Rotation", "Long/short factor — positions move in opposite directions"
    return "Style/Size", f"Driven by {', '.join(top_tickers[:2])}"


def render():
    """Render the Factor Decomposition sub-tab."""
    if not st.session_state.pca_state:
        st.info("Load portfolio to view factor decomposition.")
        return

    pca = st.session_state.pca_state
    ev = pca["explained_variance"]
    loadings = pca["loadings"]

    # ── Concentration Banner ─────────────────────────────────────
    render_concentration_banner(pca["concentration"], pca["assessment"])

    cum_var = np.cumsum(ev)
    top_factor_var = ev[0] if len(ev) > 0 else 0

    # ── 1. Factor Summary (interpretation cards) ─────────────────
    render_section_header("Factor Summary")
    subtitle(
        'Inferred meaning of each principal component based on loading patterns. '
        '<span style="color:#94a3b8;">Labels are heuristic — verify against sector/style exposures.</span>'
    )
    n_show = min(len(ev), 4)
    interp_cols = st.columns(n_show)
    for i in range(n_show):
        col_name = loadings.columns[i] if i < len(loadings.columns) else f"PC{i+1}"
        factor_loads = loadings.iloc[:, i] if i < loadings.shape[1] else None
        with interp_cols[i]:
            if factor_loads is not None:
                label, desc = _interpret_factor(factor_loads)
                top_loaders = factor_loads.abs().nlargest(2)
                top_str = ", ".join(f"{t} ({factor_loads[t]:+.2f})" for t in top_loaders.index)
                _color = BRAND_BLUE if i == 0 else SUCCESS_GREEN if i == 1 else "#6366f1" if i == 2 else "#94a3b8"
                _var_pct = ev[i] if i < len(ev) else 0
                st.markdown(
                    f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;'
                    f'background:#fafbfc;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                    f'margin-bottom:4px;">'
                    f'<span style="font-size:16px;font-weight:800;color:#0f172a;">{col_name}</span>'
                    f'<span style="font-size:20px;font-weight:800;color:{_color};'
                    f'font-variant-numeric:tabular-nums;">{_var_pct:.0%}</span>'
                    f'</div>'
                    f'<div style="font-size:12px;font-weight:700;color:{_color};'
                    f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">'
                    f'{label}</div>'
                    f'<div style="font-size:12px;color:#64748b;margin-bottom:6px;'
                    f'font-style:italic;line-height:1.4;">{desc}</div>'
                    f'<div style="font-size:12px;color:#94a3b8;">Top: {top_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── 2. Variance Explained — verbose explanation (left) + scree (right)
    render_section_header("Variance Explained")
    subtitle(
        'How much of total portfolio variance each principal component captures. '
        '<span style="color:#94a3b8;">The red cumulative line shows diminishing marginal information gain.</span>'
    )
    ve_left, ve_right = st.columns(2)
    with ve_left:
        n_pc = min(len(ev), 3)
        _pc_colors = [BRAND_BLUE, SUCCESS_GREEN, "#6366f1"]
        for i in range(n_pc):
            col_name = loadings.columns[i] if i < len(loadings.columns) else f"PC{i+1}"
            factor_loads = loadings.iloc[:, i] if i < loadings.shape[1] else None
            _color = _pc_colors[i] if i < len(_pc_colors) else "#94a3b8"
            _var_pct = ev[i] if i < len(ev) else 0
            _cum_pct = cum_var[i] if i < len(cum_var) else 0

            _label, _desc = ("—", "—")
            _explain = ""
            if factor_loads is not None:
                _label, _desc = _interpret_factor(factor_loads)
                top_loaders = factor_loads.abs().nlargest(3)
                top_names = [f"{t} ({factor_loads[t]:+.2f})" for t in top_loaders.index]

                # Build verbose explanation paragraph
                if _label == "Market Beta":
                    _explain = (
                        f'{col_name} explains {_var_pct:.0%} of portfolio variance and represents '
                        f'broad market directional exposure. All positions load positively, meaning '
                        f'portfolio returns are primarily driven by overall market moves. '
                        f'Top loadings: {", ".join(top_names)}.'
                    )
                elif "Idiosyncratic" in _label:
                    _dominant = top_loaders.index[0]
                    _explain = (
                        f'{col_name} captures {_var_pct:.0%} of variance, dominated by '
                        f'{_dominant}-specific risk. This component moves independently of the '
                        f'broader market — {_dominant}\'s idiosyncratic behavior (earnings, news, '
                        f'sector shifts) drives this factor. '
                        f'Loadings: {", ".join(top_names)}.'
                    )
                elif _label == "Sector Rotation":
                    pos_tkrs = [t for t in factor_loads.index if factor_loads[t] > 0.2]
                    neg_tkrs = [t for t in factor_loads.index if factor_loads[t] < -0.2]
                    _explain = (
                        f'{col_name} explains {_var_pct:.0%} of variance as a long/short rotation '
                        f'factor. When {", ".join(pos_tkrs[:2]) if pos_tkrs else "some positions"} '
                        f'rise, {", ".join(neg_tkrs[:2]) if neg_tkrs else "others"} tend to fall. '
                        f'This captures sector or style rotation within the portfolio. '
                        f'Loadings: {", ".join(top_names)}.'
                    )
                else:
                    _explain = (
                        f'{col_name} captures {_var_pct:.0%} of variance, driven primarily by '
                        f'{", ".join(top_loaders.index[:2])}. This factor reflects style or size '
                        f'exposure differences between holdings. '
                        f'Loadings: {", ".join(top_names)}.'
                    )

            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:16px 18px;'
                f'background:#fafbfc;margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                f'margin-bottom:6px;">'
                f'<span style="font-size:12px;font-weight:700;color:#64748b;'
                f'text-transform:uppercase;letter-spacing:0.05em;">{_label}</span>'
                f'<span style="font-size:20px;font-weight:800;color:#0f172a;'
                f'font-variant-numeric:tabular-nums;">{_var_pct:.0%}</span>'
                f'</div>'
                f'<div style="font-size:13px;color:#334155;line-height:1.6;">{_explain}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with ve_right:
        with st.container(border=True):
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(ev))],
                y=ev, marker_color=BRAND_BLUE, name="Individual",
            ))
            fig_s.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(cum_var))],
                y=cum_var, mode="lines+markers", name="Cumulative",
                line=dict(color=DANGER_RED, width=2), marker=dict(size=6),
            ))
            fig_s.add_hline(y=0.90, line_dash="dot", line_color="#94a3b8",
                            annotation_text="90% threshold")
            apply_invictus_layout(fig_s, height=380, title="Scree Plot", showlegend=True)
            fig_s.update_layout(yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

    # ── 3. Factor Loadings Heatmap ───────────────────────────────
    render_section_header("Factor Loadings")
    subtitle(
        'Each cell shows how much a position loads onto a given factor. '
        f'<span style="color:{SUCCESS_GREEN};font-weight:700;">Strong positive</span> = moves with factor; '
        f'<span style="color:{DANGER_RED};font-weight:700;">strong negative</span> = moves against it. '
        '<span style="color:#94a3b8;">Values near zero indicate minimal exposure.</span>'
    )
    with st.container(border=True):
        fig_l = go.Figure(go.Heatmap(
            z=loadings.values,
            x=loadings.columns.tolist(), y=loadings.index.tolist(),
            colorscale=[[0, DANGER_RED_ALT], [0.5, SLATE_50], [1, SUCCESS_GREEN]],
            text=loadings.round(2).values, texttemplate="%{text}",
        ))
        apply_invictus_layout(fig_l, height=max(300, len(loadings) * 40),
                              title="Position × Factor Loading Matrix")
        fig_l.update_layout(margin=dict(t=30, b=10, l=80, r=10))
        st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})
