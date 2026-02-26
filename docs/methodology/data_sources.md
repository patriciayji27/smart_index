# Data Sources

## Option Chain Data

| Source | Coverage | Cost | Notes |
|--------|----------|------|-------|
| **CBOE DataShop** | SPX/VIX historical chains | Paid | Gold standard; settlement prices |
| **OptionMetrics (via WRDS)** | Broad US equity options | Academic license | IvyDB; pre-computed greeks available |
| **Yahoo Finance (yfinance)** | Delayed snapshot chains | Free | Good for prototyping; no historical depth |
| **IBKR / TOS API** | Real-time chains | Brokerage account | Good for live updates; limited history |

**Recommendation**: Start with yfinance for prototyping, then move to CBOE/WRDS
for historical analysis.  The code is designed so that only the loader function
changes — all downstream code uses the normalised schema.

## Macro / Economic Data

| Series | Source | Frequency |
|--------|--------|-----------|
| Fed Funds Rate | FRED (DFF) | Daily |
| 2Y / 10Y Treasury | FRED (DGS2, DGS10) | Daily |
| IG / HY Credit Spreads | FRED (BAMLC0A0CM, BAMLH0A0HYM2) | Daily |
| VIX / VIX3M / VVIX | Yahoo / CBOE | Daily |
| CFTC Commitment of Traders | CFTC | Weekly |
| Gamma Exposure (GEX) estimates | SpotGamma / DIY | Daily (estimate) |

## Event Calendars

- **FOMC dates**: federalreserve.gov
- **CPI / NFP / PCE**: bls.gov, bea.gov
- **Options expiry**: derive from `third_friday()` or CBOE calendar
- **Custom events**: manually curated in `config/events.yaml`

## Sample Data

`data/sample/` contains small, anonymised datasets for unit tests and
reproducibility.  These are synthetic or heavily downsampled — never
commit proprietary data to the repo.
