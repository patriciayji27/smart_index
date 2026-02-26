"""Data loaders — read raw files into standardised DataFrames.

Design principle: loaders are thin wrappers that handle file format differences
between data sources (CBOE, OptionMetrics, Yahoo, etc.) and return a common
schema. Cleaning logic lives in cleaning.py; feature engineering lives elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from smart_index.utils.io import resolve_data_path


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_parquet(category: str, *parts: str, **kwargs) -> pd.DataFrame:
    """Load a parquet file from a data category."""
    path = resolve_data_path(category, *parts)
    return pd.read_parquet(path, **kwargs)


def load_csv(category: str, *parts: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file from a data category."""
    path = resolve_data_path(category, *parts)
    return pd.read_csv(path, **kwargs)


# ---------------------------------------------------------------------------
# Option chain loader
# ---------------------------------------------------------------------------

# Minimal required columns after normalisation
OPTION_CHAIN_SCHEMA = [
    "date",           # observation date (pd.Timestamp)
    "expiry",         # expiry date (pd.Timestamp)
    "strike",         # strike price (float)
    "option_type",    # "C" or "P"
    "bid",            # bid price (float)
    "ask",            # ask price (float)
    "mid",            # (bid+ask)/2
    "volume",         # contracts traded (int)
    "open_interest",  # OI (int)
    "underlying",     # underlying price (float)
    "dte",            # calendar days to expiry (int)
]


def load_option_chain(
    source: Literal["cboe", "sample"] = "sample",
    ticker: str = "SPX",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load option chain data and normalise to common schema.

    Parameters
    ----------
    source : str
        Data source identifier — determines parsing logic.
    ticker : str
        Underlying ticker.
    start, end : str, optional
        Date filters (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame with columns matching OPTION_CHAIN_SCHEMA.
    """
    if source == "sample":
        df = _load_sample_chain(ticker)
    elif source == "cboe":
        df = _load_cboe_chain(ticker)
    else:
        raise ValueError(f"Unknown source: {source}")

    # Date filtering
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]

    return df.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)


def _load_sample_chain(ticker: str) -> pd.DataFrame:
    """Load from data/sample/ — for testing and reproducibility."""
    path = resolve_data_path("sample", f"{ticker.lower()}_chain_sample.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Sample data not found at {path}. "
            "See docs/methodology/data_sources.md for how to create sample data."
        )
    df = pd.read_parquet(path)
    return _normalise_columns(df)


def _load_cboe_chain(ticker: str) -> pd.DataFrame:
    """Load CBOE-format option chain data from data/raw/."""
    # TODO: implement CBOE-specific parsing
    # Expected: CSV files with CBOE's delayed-quote column names
    raise NotImplementedError("CBOE loader not yet implemented — see docs/methodology/data_sources.md")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort column name normalisation to OPTION_CHAIN_SCHEMA."""
    # Lowercase all column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Common renames (extend as you encounter new sources)
    rename_map = {
        "quote_date": "date",
        "trade_date": "date",
        "expiration": "expiry",
        "expiry_date": "expiry",
        "cp_flag": "option_type",
        "type": "option_type",
        "spot": "underlying",
        "underlying_price": "underlying",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure date types
    for col in ["date", "expiry"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Compute mid if missing
    if "mid" not in df.columns and {"bid", "ask"}.issubset(df.columns):
        df["mid"] = (df["bid"] + df["ask"]) / 2

    # Compute DTE if missing
    if "dte" not in df.columns and {"date", "expiry"}.issubset(df.columns):
        df["dte"] = (df["expiry"] - df["date"]).dt.days

    # Normalise option type
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].str.upper().str[0]  # "Call" → "C"

    return df
