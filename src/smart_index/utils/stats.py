"""Statistical utilities used across the project."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling z-score (default ~3-month window)."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma


def rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling percentile rank within lookback window."""
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def realized_vol(
    returns: pd.Series,
    window: int = 21,
    annualize: int = 252,
) -> pd.Series:
    """Annualized rolling realized volatility (close-to-close)."""
    return returns.rolling(window).std() * np.sqrt(annualize)


def ewma_vol(
    returns: pd.Series,
    halflife: int = 21,
    annualize: int = 252,
) -> pd.Series:
    """EWMA volatility with specified halflife."""
    var = returns.ewm(halflife=halflife).var()
    return np.sqrt(var * annualize)


def vol_risk_premium(
    implied: pd.Series, realized: pd.Series
) -> pd.Series:
    """Implied minus realized vol — positive means vol sellers are compensated."""
    return implied - realized
