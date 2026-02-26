"""Volatility surface construction and interpolation.

Takes a raw option chain (with computed IVs) for a single date and builds
a regularised grid in (tenor × moneyness) space.

Moneyness can be expressed as:
  - delta (Black-Scholes delta, 0-1)
  - log-moneyness (ln(K/F))
  - strike ratio (K/S)

We default to delta-space because it normalises across spot levels and
is the convention on most vol desks.

Interpolation choices matter:
  - Linear interp is transparent but can produce kinks.
  - Cubic spline is smooth but can oscillate in sparse regions.
  - SVI (stochastic volatility inspired) is parametric and well-behaved
    but imposes model assumptions.

We start with linear, expose a `method` parameter, and add SVI later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, interp1d


def build_surface(
    chain: pd.DataFrame,
    tenor_grid: list[int] | None = None,
    moneyness_grid: list[float] | None = None,
    moneyness_col: str = "delta",
    iv_col: str = "iv",
    method: str = "linear",
) -> pd.DataFrame:
    """Construct an IV surface on a regular (tenor × moneyness) grid.

    Parameters
    ----------
    chain : DataFrame for a single date with columns:
        dte, `moneyness_col`, `iv_col` (plus option_type for filtering).
    tenor_grid : target tenors in calendar days.
    moneyness_grid : target moneyness values (delta or strike ratio).
    method : "linear", "cubic", or "nearest".

    Returns
    -------
    DataFrame indexed by tenor, columns = moneyness values, values = IV.
    """
    if tenor_grid is None:
        tenor_grid = [7, 14, 30, 60, 90, 120, 180, 365]
    if moneyness_grid is None:
        moneyness_grid = np.arange(0.10, 0.95, 0.05).round(2).tolist()

    # Filter to valid IVs
    df = chain.dropna(subset=[iv_col, moneyness_col, "dte"]).copy()
    df = df[df[iv_col] > 0]

    if len(df) < 10:
        raise ValueError(f"Too few valid IV observations ({len(df)}) to build surface.")

    points = df[["dte", moneyness_col]].values
    values = df[iv_col].values

    # Build meshgrid
    tenor_arr = np.array(tenor_grid)
    money_arr = np.array(moneyness_grid)
    grid_tenor, grid_money = np.meshgrid(tenor_arr, money_arr, indexing="ij")

    # Interpolate
    grid_iv = griddata(points, values, (grid_tenor, grid_money), method=method)

    surface = pd.DataFrame(
        grid_iv,
        index=pd.Index(tenor_grid, name="tenor"),
        columns=pd.Index(moneyness_grid, name="moneyness"),
    )
    return surface


def slice_smile(
    surface: pd.DataFrame,
    tenor: int,
) -> pd.Series:
    """Extract a single smile (moneyness → IV) at the nearest tenor."""
    idx = surface.index
    nearest = idx[np.argmin(np.abs(idx - tenor))]
    return surface.loc[nearest].dropna()


def slice_term_structure(
    surface: pd.DataFrame,
    moneyness: float = 0.50,
) -> pd.Series:
    """Extract term structure (tenor → IV) at fixed moneyness."""
    cols = surface.columns.astype(float)
    nearest_col = cols[np.argmin(np.abs(cols - moneyness))]
    return surface[nearest_col].dropna()


def surface_diagnostics(surface: pd.DataFrame) -> dict:
    """Quick health check on an interpolated surface.

    Returns dict with coverage stats, NaN counts, and basic arbitrage flags.
    """
    total = surface.size
    nans = surface.isna().sum().sum()
    negatives = (surface < 0).sum().sum()

    # Calendar spread arbitrage: total variance should be non-decreasing in tenor
    # (simplified check at ATM)
    atm_col = surface.columns[len(surface.columns) // 2]
    atm_ts = surface[atm_col].dropna()
    tenors = atm_ts.index.values
    total_var = (atm_ts.values ** 2) * (tenors / 365.0)
    cal_arb_violations = int((np.diff(total_var) < -1e-6).sum())

    return {
        "total_cells": total,
        "nan_cells": int(nans),
        "nan_pct": round(nans / total * 100, 1),
        "negative_iv_cells": int(negatives),
        "calendar_arb_violations": cal_arb_violations,
    }
