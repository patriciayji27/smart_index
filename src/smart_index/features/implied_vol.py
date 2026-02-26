"""Implied volatility extraction from option prices.

Uses Newton-Raphson on Black-Scholes for European options. For American-style
options on SPY/IWM you'd want a binomial or Bjerksund-Stensland adjustment —
but for SPX (European-exercise), BS inversion is standard practice.

Design note: We deliberately keep this simple and transparent rather than
relying on an external library. Understanding what the solver is doing
(and where it breaks) is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Black-Scholes analytical formulas
# ---------------------------------------------------------------------------

def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "C",
) -> float:
    """Black-Scholes price for European call/put.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry in years
    r : risk-free rate (annualised, continuous)
    sigma : volatility (annualised)
    option_type : "C" or "P"
    """
    if T <= 0 or sigma <= 0:
        # Intrinsic value only
        if option_type == "C":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (dPrice/dSigma)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


# ---------------------------------------------------------------------------
# Implied vol solver
# ---------------------------------------------------------------------------

def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    option_type: str = "C",
    method: str = "brentq",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float | None:
    """Solve for implied volatility given an option price.

    Parameters
    ----------
    price : observed option price (mid recommended)
    S, K, T, r : spot, strike, time-to-expiry (years), risk-free rate
    option_type : "C" or "P"
    method : "brentq" (robust) or "newton" (faster but can diverge)

    Returns
    -------
    Implied vol (annualised), or None if solver fails / price is nonsensical.
    """
    # Sanity checks
    intrinsic = max(S - K, 0) if option_type == "C" else max(K - S, 0)
    if price < intrinsic - 1e-6:
        return None  # Below intrinsic — bad quote or arb
    if T <= 0:
        return None

    if method == "brentq":
        return _solve_brentq(price, S, K, T, r, option_type, tol)
    elif method == "newton":
        return _solve_newton(price, S, K, T, r, option_type, tol, max_iter)
    else:
        raise ValueError(f"Unknown method: {method}")


def _solve_brentq(
    price: float, S: float, K: float, T: float, r: float,
    option_type: str, tol: float,
) -> float | None:
    """Brent's method — bounded, guaranteed convergence."""
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price

    try:
        return brentq(objective, 1e-4, 5.0, xtol=tol)
    except ValueError:
        return None  # No root in [0.01%, 500%] — likely bad data


def _solve_newton(
    price: float, S: float, K: float, T: float, r: float,
    option_type: str, tol: float, max_iter: int,
) -> float | None:
    """Newton-Raphson using vega as derivative. Faster but can fail."""
    sigma = 0.25  # initial guess
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, option_type) - price
        vega = bs_vega(S, K, T, r, sigma)
        if abs(vega) < 1e-12:
            return None  # Flat vega — can't improve
        sigma -= diff / vega
        if sigma <= 0:
            return None
        if abs(diff) < tol:
            return sigma
    return None  # Did not converge


# ---------------------------------------------------------------------------
# Vectorised computation over a chain DataFrame
# ---------------------------------------------------------------------------

def compute_iv_column(
    df: pd.DataFrame,
    price_col: str = "mid",
    r: float = 0.0,
    method: str = "brentq",
) -> pd.Series:
    """Add implied vol to an option chain DataFrame.

    Expects columns: underlying, strike, dte, option_type, and `price_col`.

    Returns
    -------
    pd.Series of implied vols (NaN where solver fails).
    """
    def _row_iv(row):
        T = row["dte"] / 365.0
        return implied_vol(
            price=row[price_col],
            S=row["underlying"],
            K=row["strike"],
            T=T,
            r=r,
            option_type=row["option_type"],
            method=method,
        )

    return df.apply(_row_iv, axis=1).astype(float)
