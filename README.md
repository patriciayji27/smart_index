# smart-index

**Volatility surface analytics, macro-vol linkages, and regime detection for equity index options.**

A research toolkit for dissecting the information embedded in SPX/VIX options markets — built as a learning project for quantitative trading internship preparation.

---

## What This Project Does

1. **Volatility Surface Construction & Feature Extraction**
   Ingest option chain data, build implied-vol surfaces, and extract tradeable features (skew, term structure slope, convexity, wing richness).

2. **Macro–Vol Linkage Analysis**
   Measure how macro state variables (rates, credit spreads, positioning data) map onto vol surface shape — and where those mappings break down.

3. **Event Studies on Vol Surfaces**
   Quantify how FOMC, CPI, NFP, and earnings events deform the surface pre/post-release, and whether the market's pre-event pricing is systematically biased.

4. **Regime Detection**
   Label market regimes (low-vol grind, vol expansion, crisis, mean-reversion) using both heuristic rules and statistical models, then study how surface features behave conditionally on regime.

---

## Project Structure

```
smart-index/
├── src/smart_index/      # Core library (data, features, analytics, viz)
├── notebooks/            # Exploratory analysis (numbered by topic)
├── config/               # YAML configs for paths, symbols, events
├── scripts/              # CLI entry points
├── docs/                 # Methodology notes, roadmap
├── tests/                # Unit tests
├── reports/              # Weekly/monthly research output
└── outputs/              # Figures, tables, logs (gitignored)
```

## Quick Start

```bash
# Clone
git clone https://github.com/<you>/smart-index.git
cd smart-index

# Install
pip install -e ".[dev]"

# Verify
make test
```

## Data

Raw option chain and macro data are **not** included in this repo (see `data/sample/` for small reproducibility examples). See `docs/methodology/data_sources.md` for how to obtain the data.

## Status

🚧 Active development — see `docs/roadmap.md` for current progress.

## License

MIT — see [LICENSE](LICENSE).
