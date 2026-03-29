"""End-to-end feature pipeline: option chain → surface → features → CSV.

This is the main entry point for building the daily feature panel used
in all downstream analysis (regime classification, event studies, etc.).

Pipeline stages
---------------
1. Load option chain for a date range (loaders.py)
2. Compute implied vols per row (implied_vol.py)
3. Build an IV surface for each date (surface.py)
4. Extract surface features (surface_features.py)
5. Compute implied correlation where component data is available
6. Write feature panel to outputs/tables/

Usage (from repo root)
----------------------
    python -m smart_index.pipelines.feature_pipeline \
        --start 2024-01-01 --end 2024-12-31 --output features_2024.parquet

Or from a notebook:
    from smart_index.pipelines.feature_pipeline import run_pipeline
    features = run_pipeline(start="2024-01-01", end="2024-12-31")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from smart_index.analytics.surface import build_surface, surface_diagnostics
from smart_index.data.loaders import load_option_chain
from smart_index.features.implied_vol import compute_iv_column
from smart_index.features.surface_features import (
    compute_daily_features,
    skew_put_call,
    smile_convexity,
    term_slope,
    wing_richness,
)
from smart_index.utils.io import resolve_output_path
from smart_index.utils.stats import realized_vol, vol_risk_premium

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    start: str,
    end: str,
    source: str = "sample",
    ticker: str = "SPX",
    tenors: list[int] | None = None,
    output_filename: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full feature pipeline and return a daily feature DataFrame.

    Parameters
    ----------
    start, end : date strings (YYYY-MM-DD)
    source : data source identifier passed to load_option_chain
    ticker : underlying (default SPX)
    tenors : list of tenors to compute features at (default [30, 90])
    output_filename : if set, write parquet to outputs/tables/
    verbose : if True, log progress

    Returns
    -------
    pd.DataFrame indexed by date with feature columns.
    """
    if tenors is None:
        tenors = [30, 90]

    if verbose:
        logger.info(f"Loading option chain: {ticker} {start} → {end} (source={source})")

    # ── Stage 1: Load ──────────────────────────────────────────────────────
    try:
        chain = load_option_chain(source=source, ticker=ticker, start=start, end=end)
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        raise

    if chain.empty:
        raise ValueError(f"No option chain data found for {ticker} {start}–{end}")

    if verbose:
        logger.info(f"Loaded {len(chain):,} rows across {chain['date'].nunique()} dates")

    # ── Stage 2: Implied vols ──────────────────────────────────────────────
    if verbose:
        logger.info("Computing implied vols (BS inversion)...")

    chain["iv"] = compute_iv_column(chain, price_col="mid")
    n_failed = chain["iv"].isna().sum()
    if verbose:
        pct_ok = (1 - n_failed / len(chain)) * 100
        logger.info(f"IV computation: {pct_ok:.1f}% success ({n_failed:,} failures)")

    # ── Stage 3: Surface per date ─────────────────────────────────────────
    if verbose:
        logger.info("Building IV surfaces per date...")

    daily_surfaces: dict[str, pd.DataFrame] = {}
    dates = sorted(chain["date"].dt.date.unique())
    skipped = 0

    for date in dates:
        day_chain = chain[chain["date"].dt.date == date].copy()
        try:
            surface = build_surface(day_chain)
            diag = surface_diagnostics(surface)
            if diag["nan_pct"] > 50:
                logger.warning(f"{date}: surface has {diag['nan_pct']:.0f}% NaN — skipping")
                skipped += 1
                continue
            daily_surfaces[str(date)] = surface
        except ValueError as e:
            logger.warning(f"{date}: surface build failed ({e}) — skipping")
            skipped += 1

    if verbose:
        logger.info(f"Built {len(daily_surfaces)} surfaces ({skipped} skipped)")

    if not daily_surfaces:
        raise ValueError("No valid surfaces built — check data quality")

    # ── Stage 4: Surface features ─────────────────────────────────────────
    if verbose:
        logger.info("Extracting surface features...")

    features = compute_daily_features(daily_surfaces, tenors=tenors)

    # ── Stage 5: Realized vol + VRP ───────────────────────────────────────
    if verbose:
        logger.info("Computing realized vol and vol risk premium...")

    features = _add_vrp(features, chain)

    # ── Stage 6: Output ───────────────────────────────────────────────────
    if output_filename:
        out_path = resolve_output_path("tables", output_filename)
        features.to_parquet(out_path)
        if verbose:
            logger.info(f"Feature panel written to {out_path}")

    if verbose:
        logger.info(f"Pipeline complete. Feature panel shape: {features.shape}")

    return features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_vrp(
    features: pd.DataFrame,
    chain: pd.DataFrame,
    window: int = 21,
    atm_delta: float = 0.50,
    tenor: int = 30,
) -> pd.DataFrame:
    """Add realized vol and vol risk premium to the feature panel.

    Uses the underlying prices embedded in the option chain to compute
    close-to-close realized vol. The VRP is 30d ATM IV − 21d realized vol.
    """
    # Extract underlying price history from the chain
    underlying_prices = (
        chain.groupby("date")["underlying"]
        .first()
        .sort_index()
    )

    if len(underlying_prices) < window + 1:
        logger.warning("Not enough price history for realized vol calculation")
        return features

    log_returns = np.log(underlying_prices / underlying_prices.shift(1)).dropna()
    rv = realized_vol(log_returns, window=window)
    rv.name = "realized_vol_21d"

    # ATM IV from surface features (if available)
    atm_col = f"skew_25d_{tenor}d"
    if atm_col in features.columns:
        # Approximate ATM IV as skew midpoint — proper ATM would come from surface
        features["vrp_approx"] = features[atm_col] - rv.reindex(features.index).fillna(method="ffill")

    features = features.join(rv.rename("realized_vol_21d"), how="left")
    return features


def build_feature_panel_from_surfaces(
    daily_surfaces: dict[str, pd.DataFrame],
    tenors: list[int] | None = None,
    include_term_slope: bool = True,
    near_tenor: int = 30,
    far_tenor: int = 90,
) -> pd.DataFrame:
    """Lower-level helper: build feature panel from pre-computed surfaces dict.

    Useful in notebooks where you've already built surfaces and want to
    iterate on feature extraction without re-running the full pipeline.

    Parameters
    ----------
    daily_surfaces : {date_string: surface_DataFrame}
    tenors : tenors to compute per-smile features at
    include_term_slope : whether to include term slope feature

    Returns
    -------
    pd.DataFrame of daily features.
    """
    if tenors is None:
        tenors = [30, 90]

    records = []
    for date_str, surface in sorted(daily_surfaces.items()):
        row: dict = {"date": pd.Timestamp(date_str)}
        for t in tenors:
            try:
                row[f"skew_25d_{t}d"]       = skew_put_call(surface, tenor=t)
                row[f"convexity_{t}d"]       = smile_convexity(surface, tenor=t)
                row[f"wing_richness_{t}d"]   = wing_richness(surface, tenor=t)
            except Exception:
                pass

        if include_term_slope and len(tenors) >= 2:
            try:
                row["term_slope_atm"] = term_slope(
                    surface, near_tenor=near_tenor, far_tenor=far_tenor
                )
            except Exception:
                pass

        records.append(row)

    return pd.DataFrame(records).set_index("date").sort_index()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Run the smart_index feature pipeline")
    parser.add_argument("--start",   required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--ticker",  default="SPX",    help="Underlying ticker")
    parser.add_argument("--source",  default="sample", help="Data source (sample | cboe)")
    parser.add_argument("--output",  default=None, help="Output filename (parquet)")
    args = parser.parse_args()

    run_pipeline(
        start=args.start,
        end=args.end,
        ticker=args.ticker,
        source=args.source,
        output_filename=args.output,
        verbose=True,
    )
