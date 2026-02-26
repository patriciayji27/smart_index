"""Date utilities: business-day logic, expiry dates, tenor math."""

from __future__ import annotations

import datetime as dt
from typing import Sequence

import numpy as np
import pandas as pd


def to_datetime(d: str | dt.date | dt.datetime | pd.Timestamp) -> pd.Timestamp:
    """Coerce various date types to pd.Timestamp."""
    return pd.Timestamp(d)


def business_days_between(start: str | pd.Timestamp, end: str | pd.Timestamp) -> int:
    """Count business days between two dates (exclusive of end)."""
    rng = pd.bdate_range(start=to_datetime(start), end=to_datetime(end))
    return max(len(rng) - 1, 0)


def calendar_days_to_expiry(
    as_of: str | pd.Timestamp, expiry: str | pd.Timestamp
) -> int:
    """Calendar days to expiry."""
    return (to_datetime(expiry) - to_datetime(as_of)).days


def annualization_factor(days: int, trading_days_per_year: int = 252) -> float:
    """Sqrt(T) annualization: days → fraction of year."""
    return np.sqrt(days / trading_days_per_year)


def third_friday(year: int, month: int) -> dt.date:
    """Monthly options expiry: third Friday of given month."""
    # First day of month
    first = dt.date(year, month, 1)
    # Days until Friday (weekday 4)
    offset = (4 - first.weekday()) % 7
    first_friday = first + dt.timedelta(days=offset)
    return first_friday + dt.timedelta(weeks=2)


def monthly_expiries(year: int) -> list[dt.date]:
    """All 12 monthly expiry dates for a given year."""
    return [third_friday(year, m) for m in range(1, 13)]


def nearest_tenor_bucket(
    dte: int, buckets: Sequence[int] = (7, 14, 30, 60, 90, 120, 180, 365)
) -> int:
    """Snap a DTE value to the nearest standard tenor bucket."""
    return min(buckets, key=lambda b: abs(b - dte))
