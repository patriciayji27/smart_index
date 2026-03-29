"""Visualization functions for IV surfaces, smiles, and feature time series.

Two rendering backends:
  - Plotly (interactive): used in notebooks and the GitHub Pages site
  - Matplotlib (publication): used for static figures in reports

All Plotly figures use the project's dark theme to match the website palette.
All matplotlib figures call `apply_style()` from viz/style.py before rendering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Plotly is optional — fail gracefully if not installed
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL = True
except ImportError:
    _MPL = False

from smart_index.viz.style import apply_style, get_colors

# ── Dark theme tokens (match the website) ────────────────────────────────────
_BG      = "#0d0d0f"
_BG2     = "#13131a"
_BG3     = "#1a1a24"
_TEAL    = "#5be8dd"
_GOLD    = "#e8c97a"
_DIM     = "#7a7a88"
_BRIGHT  = "#f0f0f4"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG2,
    font=dict(family="DM Mono, monospace", size=11, color=_DIM),
    margin=dict(l=50, r=20, t=50, b=50),
)


def _require_plotly():
    if not _PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")


def _require_mpl():
    if not _MPL:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib")


# ---------------------------------------------------------------------------
# Plotly: 3D surface
# ---------------------------------------------------------------------------

def plot_surface_3d(
    surface: pd.DataFrame,
    title: str = "Implied Volatility Surface",
    date: str = "",
    height: int = 500,
) -> "go.Figure":
    """Interactive 3D surface plot from an IV surface DataFrame.

    Parameters
    ----------
    surface : DataFrame indexed by tenor (rows), columns = moneyness/delta.
              Values are implied vols (annualised, e.g. 0.18 for 18%).
    title : figure title
    date : optional date label added to title
    height : figure height in pixels
    """
    _require_plotly()

    tenors    = surface.index.values.astype(float)
    moneyness = surface.columns.values.astype(float)
    Z = surface.values * 100  # convert to percentage for display

    fig = go.Figure(data=[go.Surface(
        x=moneyness,
        y=tenors,
        z=Z,
        colorscale=[
            [0.0,  "#0d3d3a"],
            [0.3,  "#1a6b67"],
            [0.6,  _TEAL],
            [0.85, _GOLD],
            [1.0,  "#e86464"],
        ],
        showscale=False,
        contours=dict(z=dict(show=True, color="rgba(255,255,255,0.05)", width=1)),
        hovertemplate=(
            "Moneyness: %{x:.2f}<br>"
            "Tenor: %{y}d<br>"
            "IV: %{z:.1f}%<extra></extra>"
        ),
    )])

    full_title = f"{title} — {date}" if date else title

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=height,
        title=dict(text=full_title, font=dict(size=13, color=_BRIGHT)),
        scene=dict(
            xaxis=dict(title="Moneyness (delta)", color=_DIM, gridcolor=_BG3, backgroundcolor=_BG2),
            yaxis=dict(title="Tenor (days)",       color=_DIM, gridcolor=_BG3, backgroundcolor=_BG2),
            zaxis=dict(title="IV (%)",             color=_DIM, gridcolor=_BG3, backgroundcolor=_BG2),
            bgcolor=_BG2,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: smile slice
# ---------------------------------------------------------------------------

def plot_smile(
    surface: pd.DataFrame,
    tenors: list[int] | None = None,
    title: str = "Implied Volatility Smile",
    height: int = 350,
) -> "go.Figure":
    """Plot one or more smile slices (IV vs moneyness at fixed tenors).

    Parameters
    ----------
    surface : IV surface DataFrame (tenor rows × moneyness cols)
    tenors : list of tenors to overlay; defaults to [30, 60, 90]
    """
    _require_plotly()

    if tenors is None:
        tenors = [30, 60, 90]

    colors = [_TEAL, _GOLD, "#a8d8ea", "#e86464", _DIM]
    fig = go.Figure()

    idx = surface.index.values.astype(float)
    moneyness = surface.columns.values.astype(float)

    for i, t in enumerate(tenors):
        nearest = idx[np.argmin(np.abs(idx - t))]
        smile_row = surface.loc[nearest].values * 100
        fig.add_trace(go.Scatter(
            x=moneyness, y=smile_row,
            mode="lines+markers",
            name=f"{int(nearest)}d",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
            hovertemplate="Δ %{x:.2f}<br>IV: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=height,
        title=dict(text=title, font=dict(size=13, color=_BRIGHT)),
        xaxis=dict(title="Moneyness (delta)", color=_DIM, gridcolor=_BG3),
        yaxis=dict(title="IV (%)",            color=_DIM, gridcolor=_BG3),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_DIM)),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: term structure
# ---------------------------------------------------------------------------

def plot_term_structure(
    surface: pd.DataFrame,
    deltas: list[float] | None = None,
    title: str = "Term Structure of Implied Volatility",
    height: int = 350,
) -> "go.Figure":
    """Plot term structure (IV vs tenor) at fixed moneyness levels.

    Parameters
    ----------
    deltas : moneyness values to plot; defaults to [0.25, 0.50, 0.75]
    """
    _require_plotly()

    if deltas is None:
        deltas = [0.25, 0.50, 0.75]

    colors = [_TEAL, _BRIGHT, _GOLD]
    delta_labels = {0.25: "25Δ Put", 0.50: "ATM", 0.75: "25Δ Call"}
    fig = go.Figure()

    tenors = surface.index.values.astype(float)
    cols   = surface.columns.values.astype(float)

    for i, d in enumerate(deltas):
        nearest_col = cols[np.argmin(np.abs(cols - d))]
        ts = surface[nearest_col].values * 100
        label = delta_labels.get(d, f"Δ={d:.2f}")
        fig.add_trace(go.Scatter(
            x=tenors, y=ts,
            mode="lines+markers",
            name=label,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
            hovertemplate="%{x}d<br>IV: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=height,
        title=dict(text=title, font=dict(size=13, color=_BRIGHT)),
        xaxis=dict(title="Tenor (days)", color=_DIM, gridcolor=_BG3),
        yaxis=dict(title="IV (%)",       color=_DIM, gridcolor=_BG3),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_DIM)),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: feature time series
# ---------------------------------------------------------------------------

def plot_feature_panel(
    features: pd.DataFrame,
    cols: list[str] | None = None,
    title: str = "Surface Feature Panel",
    height_per_panel: int = 180,
) -> "go.Figure":
    """Stacked time series of surface features with shared x-axis.

    Parameters
    ----------
    cols : columns to plot; defaults to all numeric columns (up to 6)
    height_per_panel : height of each subplot in pixels
    """
    _require_plotly()

    if cols is None:
        cols = features.select_dtypes(include=[np.number]).columns.tolist()[:6]

    n = len(cols)
    colors = [_TEAL, _GOLD, "#a8d8ea", "#e86464", _DIM, _BRIGHT]

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=cols,
    )

    for i, col in enumerate(cols, start=1):
        if col not in features.columns:
            continue
        series = features[col].dropna()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines",
            name=col,
            line=dict(color=colors[(i - 1) % len(colors)], width=1.5),
            hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>",
        ), row=i, col=1)

    for i in range(1, n + 1):
        fig.update_yaxes(color=_DIM, gridcolor=_BG3, row=i, col=1)

    fig.update_xaxes(color=_DIM, gridcolor=_BG3)
    fig.update_annotations(font=dict(color=_DIM, size=10))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=height_per_panel * n,
        title=dict(text=title, font=dict(size=13, color=_BRIGHT)),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: regime timeline
# ---------------------------------------------------------------------------

_REGIME_COLORS: dict[str, str] = {
    "macro_driven":       "#e86464",
    "idiosyncratic":      _TEAL,
    "concentrated_name":  _GOLD,
    "calm":               "#b0b0c8",
    "elevated":           "#e8b450",
}


def plot_regime_timeline(
    regimes: pd.Series,
    price_series: pd.Series | None = None,
    title: str = "Regime Timeline",
    height: int = 300,
) -> "go.Figure":
    """Colour-coded regime timeline, optionally overlaid with a price series.

    Parameters
    ----------
    regimes : pd.Series of regime label strings, date-indexed
    price_series : optional price series to overlay (e.g. SPX level)
    """
    _require_plotly()

    rows = 2 if price_series is not None else 1
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=[0.3, 0.7] if rows == 2 else [1.0],
        vertical_spacing=0.02,
    )

    # Regime as coloured band (y=1 bar per regime)
    prev_label = None
    prev_date  = None
    shapes = []
    for date, label in regimes.items():
        if label != prev_label and prev_label is not None:
            shapes.append(dict(
                type="rect",
                x0=prev_date, x1=date,
                y0=0, y1=1, yref="paper",
                fillcolor=_REGIME_COLORS.get(prev_label, _DIM),
                opacity=0.18, line_width=0,
                layer="below",
            ))
        prev_label = label
        prev_date  = date

    # Final segment
    if prev_date is not None:
        shapes.append(dict(
            type="rect",
            x0=prev_date, x1=regimes.index[-1],
            y0=0, y1=1, yref="paper",
            fillcolor=_REGIME_COLORS.get(prev_label, _DIM),
            opacity=0.18, line_width=0,
            layer="below",
        ))

    # Dummy traces for legend
    seen: set[str] = set()
    for date, label in regimes.items():
        if label not in seen:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color=_REGIME_COLORS.get(label, _DIM), size=10, symbol="square"),
                name=label.replace("_", " ").title(),
                showlegend=True,
            ), row=1, col=1)
            seen.add(label)

    # Price overlay
    if price_series is not None:
        fig.add_trace(go.Scatter(
            x=price_series.index, y=price_series.values,
            mode="lines", name="Price",
            line=dict(color=_BRIGHT, width=1.5),
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
            showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(color=_DIM, gridcolor=_BG3, row=2, col=1)

    fig.update_xaxes(color=_DIM, gridcolor=_BG3)
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=height,
        title=dict(text=title, font=dict(size=13, color=_BRIGHT)),
        shapes=shapes,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_DIM), orientation="h"),
    )
    return fig


# ---------------------------------------------------------------------------
# Matplotlib: publication smile
# ---------------------------------------------------------------------------

def plot_smile_mpl(
    surface: pd.DataFrame,
    tenor: int = 30,
    ax: "plt.Axes | None" = None,
    label: str = "",
    color: str | None = None,
) -> "plt.Axes":
    """Single smile at a fixed tenor — matplotlib version for reports.

    Parameters
    ----------
    ax : optional existing axes to draw onto
    label : line label for legend
    color : matplotlib color string; defaults to project primary color
    """
    _require_mpl()
    apply_style()
    colors = get_colors()

    idx = surface.index.values.astype(float)
    nearest = idx[np.argmin(np.abs(idx - tenor))]
    smile = surface.loc[nearest].dropna()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    c = color or colors.get("primary", "#1f77b4")
    ax.plot(smile.index, smile.values * 100, marker="o", ms=4, lw=2, color=c, label=label or f"{int(nearest)}d")
    ax.set_xlabel("Moneyness (delta)")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title(f"Implied Volatility Smile — {int(nearest)}d Tenor")
    if label:
        ax.legend()
    return ax
