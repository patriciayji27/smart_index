# Project Overview

## Motivation

Equity index vol surfaces encode rich market expectations — about tail risk,
term structure of uncertainty, implied correlation, dealer positioning, and
sector dynamics — that go far beyond a single VIX number. This project
builds a toolkit to systematically extract, track, and analyse those signals.

## Research completed

### 1. The Dispersion Signal
The variance of a cap-weighted index decomposes into individual stock variances
and their cross-correlation terms. SPX options give the index variance; component
options give individual vols. Inverting this identity extracts market-implied
correlation (replicated from CBOE's COR3M methodology). The correlation *smile*
(COR10D − COR90D) turns out to be a leading indicator — it narrows 1–2 weeks
before VIX itself moves at regime transitions. The Mag7 concentration effect
(~33% of SPX by 2025) dominates the headline COR3M reading; separating
cap-weight vs equal-weight implied correlation isolates crowding from genuine
co-movement. Published: *The Dispersion Signal* (Substack, Mar 2026).

### 2. Dealer Gamma and the Feedback Loop
VIX exceeds subsequent 30-day realized vol by ~4 points on average. A
mechanical explanation: dealer gamma positioning creates a feedback loop
that actively suppresses SPX realized vol. Long GEX → dealers buy dips and
sell rips → dampens index moves specifically (not single-stock). This
endogenously widens the vol risk premium in long-gamma environments and
mechanically inflates implied correlation above realized. The vol risk premium
decomposes into a stable genuine component and a fragile mechanical one that
reverses when GEX flips. Examined three gamma-flip events in detail:
Volmageddon (Feb 2018), yen-carry unwind (Aug 2024), Liberation Day (Apr 2025).
Published: *Dealer Gamma and the Feedback Loop* (Substack, Mar 2026).

## In progress

### 3. The Tech Premium Signal
The rolling return differential between software/tech (XLK, IGV) and SPX
encodes real-rate sensitivity (long-duration DCF), risk appetite, and
Mag7 concentration effects simultaneously. At Mag7 weight > 30%, the tech
spread and COR3M are mechanically co-determined. Investigating whether spread
deterioration after sustained highs leads broad vol expansion by 4–6 weeks,
and how the spread behaves differently across the three correlation regimes.

## Planned

### 4. Event Studies
Systematic examination of pre-FOMC/CPI straddle break-evens vs realized
SPX moves (2020–2025), conditioned on regime. Preliminary hypothesis:
overpricing is concentrated in low-vol idiosyncratic regimes.

## Scope

- **Data**: SPX options (European exercise), macro from FRED, CBOE vol indices.
  Sector ETFs (XLK, IGV, SPY) for the tech spread research.
- **Frequency**: Daily-to-weekly signals. Not intraday microstructure.
- **Not a backtested strategy**: This is an analytics and research framework.
  Any P&L analysis is for intuition only, not live deployment.
