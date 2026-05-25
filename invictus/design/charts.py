"""
invictus.design.charts
======================
Plotly theming — consistent chart styling across the platform.
"""
from typing import Any, Optional

from invictus.design.tokens import (
    SLATE_100, SLATE_200, SLATE_500, SLATE_900,
)


def apply_invictus_layout(
    fig: Any,
    height: int = 450,
    hovermode: str = "closest",
    showlegend: bool = False,
    title: Optional[str] = None,
    margin: Optional[dict] = None,
) -> Any:
    """Apply Invictus chart layout to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, sans-serif",
        font_color=SLATE_900,
        hovermode=hovermode,
        height=height,
        title=dict(text=title, font=dict(size=14)) if title else dict(text=""),
        showlegend=showlegend,
        margin=margin or dict(t=20, b=20, l=20, r=20),
        xaxis=dict(
            showgrid=True, gridcolor=SLATE_100, linecolor=SLATE_200,
            tickfont=dict(size=11, color=SLATE_500),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=SLATE_100, linecolor=SLATE_200,
            tickfont=dict(size=11, color=SLATE_500),
        ),
        legend=dict(
            font=dict(size=11, color=SLATE_500),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=SLATE_200, borderwidth=1,
        ),
    )
    return fig
