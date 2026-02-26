"""Extract tradeable features from the implied-vol surface.

Vocabulary
----------
- **Skew** : difference between OTM put IV and OTM call IV at a fixed tenor.
  Measured in vol points or as a ratio.  Steep negative skew → expensive
  downside protection.
- **Term slope** : IV at a far tenor minus IV at a near tenor for a fixed
  moneyness.  Positive slope (contango) is the normal state; inversion
  signals near-term stress.
- **Convexity (smile curvature)** : second derivative of the smile at ATM.
  High convexity → wings are expensive relative to ATM — tail hedging demand.
- **Wing richness** : ratio of far-OTM put IV to ATM IV at a fixed tenor.
  Captures crash-insurance premium beyond what skew alone measures.

All features are computed from an IV surface that has already been
interpolated onto a (tenor × moneyness) grid — see analytics/surface.py
for grid construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Skew
# ---------------------------------------------------------------------------

def skew_put_call(
    iv_surface: pd.DataFrame,
    tenor: int = 30,
    put_delta: float = 0.25,
    call_delta: float = 0.25,
) -> float:
    """25-delta put IV minus 25-delta call IV at a given tenor.

    Parameters
    ----------
    iv_surface : DataFrame indexed by tenor, columns are delta (or moneyness).
    tenor : target tenor in calendar days.
    put_delta, call_delta : delta levels (absolute).

    Returns
    -------
    Skew in vol points.  Positive ⇒ puts richer than calls (normal for equities).
    """
    row = _get_tenor_row(iv_surface, tenor)
    put_iv = _interp_delta(row, put_delta, side="put")
    call_iv = _interp_delta(row, call_delta, side="call")
    return put_iv - call_iv


def skew_ratio(
    iv_surface: pd.DataFrame,
    tenor: int = 30,
    put_delta: float = 0.25,
) -> float:
    """Ratio of 25d put IV to ATM IV — normalised skew measure."""
    row = _get_tenor_row(iv_surface, tenor)
    put_iv = _interp_delta(row, put_delta, side="put")
    atm_iv = _interp_delta(row, 0.50, side="call")
    if atm_iv == 0:
        return np.nan
    return put_iv / atm_iv


# ---------------------------------------------------------------------------
# Term structure slope
# ---------------------------------------------------------------------------

def term_slope(
    iv_surface: pd.DataFrame,
    delta: float = 0.50,
    near_tenor: int = 30,
    far_tenor: int = 90,
) -> float:
    """IV at far tenor minus IV at near tenor for a fixed delta.

    Positive → contango (normal).  Negative → backwardation (stress).
    """
    near = _interp_delta(_get_tenor_row(iv_surface, near_tenor), delta, side="call")
    far = _interp_delta(_get_tenor_row(iv_surface, far_tenor), delta, side="call")
    return far - near


# ---------------------------------------------------------------------------
# Convexity (smile curvature at ATM)
# ---------------------------------------------------------------------------

def smile_convexity(
    iv_surface: pd.DataFrame,
    tenor: int = 30,
    delta_width: float = 0.10,
) -> float:
    """Butterfly spread in vol space: (25d_put + 25d_call) / 2 - ATM.

    Approximates the second derivative of the smile.
    High → fat-tailed pricing.
    """
    row = _get_tenor_row(iv_surface, tenor)
    put_iv = _interp_delta(row, 0.50 - delta_width, side="put")
    call_iv = _interp_delta(row, 0.50 - delta_width, side="call")
    atm_iv = _interp_delta(row, 0.50, side="call")
    return (put_iv + call_iv) / 2 - atm_iv


# ---------------------------------------------------------------------------
# Wing richness
# ---------------------------------------------------------------------------

def wing_richness(
    iv_surface: pd.DataFrame,
    tenor: int = 30,
    wing_delta: float = 0.10,
) -> float:
    """10-delta put IV / ATM IV — measures crash-insurance premium."""
    row = _get_tenor_row(iv_surface, tenor)
    wing_iv = _interp_delta(row, wing_delta, side="put")
    atm_iv = _interp_delta(row, 0.50, side="call")
    if atm_iv == 0:
        return np.nan
    return wing_iv / atm_iv


# ---------------------------------------------------------------------------
# Time series: compute features across dates
# ---------------------------------------------------------------------------

def compute_daily_features(
    daily_surfaces: dict[str, pd.DataFrame],
    tenors: list[int] | None = None,
) -> pd.DataFrame:
    """Compute a panel of surface features from a dict of {date: surface}.

    Returns a DataFrame indexed by date with feature columns.
    """
    if tenors is None:
        tenors = [30, 90]

    records = []
    for date_str, surface in daily_surfaces.items():
        row = {"date": pd.Timestamp(date_str)}
        for t in tenors:
            try:
                row[f"skew_25d_{t}d"] = skew_put_call(surface, tenor=t)
                row[f"skew_ratio_{t}d"] = skew_ratio(surface, tenor=t)
                row[f"convexity_{t}d"] = smile_convexity(surface, tenor=t)
                row[f"wing_richness_{t}d"] = wing_richness(surface, tenor=t)
            except (KeyError, IndexError):
                pass  # Missing tenor slice — leave NaN
        if len(tenors) >= 2:
            try:
                row["term_slope_atm"] = term_slope(
                    surface, near_tenor=tenors[0], far_tenor=tenors[1]
                )
            except (KeyError, IndexError):
                pass
        records.append(row)

    return pd.DataFrame(records).set_index("date").sort_index()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_tenor_row(surface: pd.DataFrame, tenor: int) -> pd.Series:
    """Extract the row closest to `tenor` from a surface indexed by tenor."""
    idx = surface.index
    closest = idx[np.argmin(np.abs(idx - tenor))]
    return surface.loc[closest]


def _interp_delta(
    row: pd.Series, delta: float, side: str = "call"
) -> float:
    """Linearly interpolate IV at a target delta from a smile row.

    `row` should be a Series indexed by delta values (0-1 scale).
    For puts, delta is interpreted as |delta| on the put side.
    """
    # Simple linear interpolation — upgrade to cubic spline later
    deltas = np.array(row.index, dtype=float)
    ivs = row.values.astype(float)

    mask = np.isfinite(ivs)
    if mask.sum() < 2:
        return np.nan

    return float(np.interp(delta, deltas[mask], ivs[mask]))
