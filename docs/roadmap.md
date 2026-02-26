# Roadmap

## Phase 1: Foundation (Weeks 1–2)
- [x] Repo structure, configs, packaging
- [ ] Data loaders: sample data + CBOE/OptionMetrics parser
- [ ] Implied vol solver (BS inversion) + unit tests
- [ ] Surface construction on a regular grid
- [ ] Basic surface feature extraction (skew, term slope)
- [ ] First notebook: visualise a single day's surface

## Phase 2: Feature Panel (Weeks 3–4)
- [ ] Build daily time series of surface features
- [ ] Realised vol computation (CC, Parkinson, Yang-Zhang)
- [ ] Vol risk premium (implied – realised) time series
- [ ] Macro data ingestion (FRED API: rates, credit, positioning)
- [ ] Correlation analysis: surface features × macro variables
- [ ] Blog post #1: "What the vol surface knows that VIX doesn't"

## Phase 3: Event Studies (Weeks 5–6)
- [ ] Event calendar construction (FOMC, CPI, NFP, OPEX)
- [ ] Pre/post event surface snapshots
- [ ] Implied vs realised move comparison
- [ ] Statistical tests for systematic bias
- [ ] Blog post #2: "Does the market overprice FOMC risk?"

## Phase 4: Regime Detection (Weeks 7–8)
- [ ] Heuristic regime labels (VIX level, term structure, skew)
- [ ] HMM on surface features
- [ ] Conditional feature behaviour by regime
- [ ] Blog post #3: "Reading regime shifts from the vol surface"

## Phase 5: Synthesis & Positioning (Weeks 9–10)
- [ ] Composite signal construction
- [ ] Simple P&L attribution (straddle, risk-reversal, calendar)
- [ ] Positioning data overlay (CFTC, GEX estimates)
- [ ] Final blog post: synthesis and forward-looking questions

## Ongoing
- Weekly research notes in `reports/weekly/`
- Refine tests and documentation as code evolves
