#!/usr/bin/env python3
"""CLI: build and save the daily feature panel.

Run from repo root:

    python scripts/run_daily_features.py --start 2024-01-01 --end 2024-12-31
    python scripts/run_daily_features.py --start 2024-01-01 --end 2024-12-31 --source cboe
    python scripts/run_daily_features.py --start 2024-01-01 --end 2024-12-31 --output my_features.parquet

The script writes a parquet file to outputs/tables/ (created if missing) and
prints a summary of the feature panel to stdout.

All heavy lifting is in smart_index.pipelines.feature_pipeline — this script
is just a thin CLI wrapper so you can run the pipeline without opening a notebook.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running directly
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from smart_index.pipelines.feature_pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the smart_index daily feature panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Use sample data (no external data needed):
  python scripts/run_daily_features.py --start 2024-01-01 --end 2024-06-30

  # Use CBOE data (requires data/raw/ to be populated):
  python scripts/run_daily_features.py \\
      --start 2022-01-01 --end 2024-12-31 \\
      --source cboe --output features_2022_2024.parquet
        """,
    )
    parser.add_argument("--start",  required=True, metavar="YYYY-MM-DD", help="Start date (inclusive)")
    parser.add_argument("--end",    required=True, metavar="YYYY-MM-DD", help="End date (inclusive)")
    parser.add_argument("--ticker", default="SPX",    help="Underlying ticker (default: SPX)")
    parser.add_argument("--source", default="sample", choices=["sample", "cboe"],
                        help="Data source (default: sample)")
    parser.add_argument("--output", default=None, metavar="FILENAME.parquet",
                        help="Output filename in outputs/tables/ (optional)")
    parser.add_argument("--tenors", default="30,90", metavar="30,60,90",
                        help="Comma-separated list of tenors in days (default: 30,90)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tenors = [int(t.strip()) for t in args.tenors.split(",")]

    print(f"\n  smart_index  ·  feature pipeline")
    print(f"  {'─' * 40}")
    print(f"  Ticker   : {args.ticker}")
    print(f"  Period   : {args.start} → {args.end}")
    print(f"  Source   : {args.source}")
    print(f"  Tenors   : {tenors}")
    print(f"  Output   : {args.output or '(not saved)'}")
    print()

    try:
        features = run_pipeline(
            start=args.start,
            end=args.end,
            ticker=args.ticker,
            source=args.source,
            tenors=tenors,
            output_filename=args.output,
            verbose=True,
        )
    except (FileNotFoundError, ValueError) as e:
        logging.error(str(e))
        return 1

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  Feature panel shape  : {features.shape[0]} dates × {features.shape[1]} features")
    print(f"  Date range           : {features.index[0].date()} → {features.index[-1].date()}")
    print(f"  Columns              : {', '.join(features.columns.tolist())}")
    print(f"  NaN pct              : {features.isna().mean().mean() * 100:.1f}%")
    print()
    print(features.describe().round(4).to_string())
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
