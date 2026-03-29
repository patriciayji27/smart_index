# Regime Classifier: Design Rationale

## Why the Current Version Is a Toy

The existing classifier does exactly one thing: checks if VIX is below 15, 22, or 30. The other
three sliders (term structure, skew, VRP) generate commentary text but don't change the classification.
The scatter plot shows random dots with no realistic correlation structure. This has two problems:

1. **Financially meaningless**: A VIX-only classifier tells you nothing you couldn't learn from
   glancing at a Bloomberg terminal. The whole point of this project is that the *shape* of
   the vol surface carries information VIX discards.

2. **Doesn't demonstrate the insight**: The core thesis — that identical VIX levels represent
   fundamentally different market states — requires the classifier to actually weight surface
   features, not just VIX.

## Design: Multi-Factor Composite With Feature Agreement

### Scoring Framework

Each of the 4 inputs maps to a normalised stress score (0–1) using sigmoid functions
calibrated to historical percentiles of SPX options data:

**VIX Level Score** (weight: 0.35)
- Calibrated to the long-run VIX distribution (mean ~19.5, median ~17.6)
- Score 0 at VIX ≈ 10 (5th percentile), score 1 at VIX ≈ 40 (98th percentile)
- Uses logistic function: score = 1 / (1 + exp(-k*(VIX - midpoint)))
- Midpoint = 20, steepness k = 0.18

**Term Structure Score** (weight: 0.25)
- VIX/VIX3M ratio: <0.85 = deep contango (score 0), >1.15 = deep backwardation (score 1)
- Midpoint at 1.0 (flat), steepness calibrated so 0.92 ≈ 0.3 (normal contango)
- This matters because backwardation without elevated VIX signals near-term event risk,
  while deep contango in elevated VIX suggests the market views stress as temporary

**Skew Score** (weight: 0.20)
- 25-delta put–call spread (expressed as negative: -8 = 8 vol pts of put richness)
- More negative = steeper = more put-hedging demand
- Score 0 at skew = -3 (very flat), score 1 at skew = -16 (extreme)
- CRITICAL NUANCE: In true crisis, skew can actually *flatten* because panic buying
  is indiscriminate across all strikes. The classifier flags this as a "divergence."

**VRP Score** (weight: 0.20)
- Implied minus realised volatility
- High positive VRP (>+6) = vol sellers well-compensated (low stress, score 0)
- Zero or negative VRP = realised catching up or exceeding implied (high stress, score 1)
- Negative VRP is rare (~5% of the time) and almost always coincides with drawdowns

### Composite Score → Regime Label

composite = 0.35 * vix_score + 0.25 * ts_score + 0.20 * skew_score + 0.20 * vrp_score

| Composite | Regime        | Description |
|-----------|---------------|-------------|
| < 0.18    | COMPRESSED    | Ultra-low vol. Carry-friendly but fragile. Short-vol strategies look attractive but snap risk is elevated. |
| 0.18–0.38 | NORMAL        | Balanced risk pricing. Standard market operation. |
| 0.38–0.58 | TRANSITIONAL  | Stress building. Hedging demand rising. Surface shape changing faster than VIX level suggests. |
| 0.58–0.78 | STRESSED      | Active risk-off. Protection is expensive. Watch for mean-reversion signals. |
| > 0.78    | CRISIS        | Extreme dislocation. Historical precedent suggests mean-reversion but timing is uncertain. |

### Feature Agreement (Conviction Score)

Calculate the standard deviation of the 4 individual scores:
- Low SD (< 0.12): Features aligned → high-conviction regime label
- Medium SD (0.12–0.22): Some disagreement → the ambiguity itself is informative
- High SD (> 0.22): Features contradicting each other → divergence signal

### Divergence Flags (The Real Value)

These are what make the classifier non-trivial:

| VIX  | Term Structure | Skew   | VRP   | Interpretation |
|------|---------------|--------|-------|----------------|
| Low  | Contango      | Steep  | High  | "Smart money hedging quietly" — someone buying protection in calm markets. Historically precedes volatility events by 2-4 weeks. |
| High | Contango      | Flat   | Pos   | "Temporary stress" — market believes the shock is contained. VIX elevated but term structure says it won't persist. |
| High | Backwardation | Steep  | Neg   | "Genuine crisis" — all features aligned on stress. Most likely state for sustained drawdown. |
| Mod  | Backwardation | Normal | Pos   | "Event-driven" — near-term uncertainty (FOMC, CPI) without broader macro stress. The vol crush post-event is the trade. |
| Low  | Contango      | Flat   | High  | "Complacency peak" — maximum carry, minimum protection. The fragility that precedes vol events (Jan 2018, Jan 2020). |
| High | Flat          | Flat   | Neg   | "Indiscriminate panic" — skew *collapses* because fear is across all strikes. Often marks capitulation bottoms. |

### Historical Calibration Points

These preset episodes let users see how the classifier labels known historical events:

| Episode                     | VIX  | VIX/VIX3M | Skew  | VRP  | Composite | Regime       |
|----------------------------|------|-----------|-------|------|-----------|--------------|
| 2017 Low-Vol Grind         | 10   | 0.82      | -5    | +6   | 0.05      | COMPRESSED   |
| Pre-Volmageddon (Jan 2018) | 11   | 0.78      | -4    | +7   | 0.04      | COMPRESSED   |
| Volmageddon (Feb 5 2018)   | 37   | 1.30      | -14   | -8   | 0.90      | CRISIS       |
| COVID Crash (Mar 2020)     | 66   | 1.35      | -5*   | -20  | 0.93      | CRISIS       |
| Meme Mania (Jan 2021)      | 24   | 0.95      | -4    | +2   | 0.37      | NORMAL       |
| Rate Shock (Sep 2022)      | 32   | 1.08      | -10   | -2   | 0.72      | STRESSED     |
| JPY Unwind (Aug 2024)      | 55   | 1.35      | -12   | -15  | 0.93      | CRISIS       |
| Now (Mar 2026)             | 27   | 1.01      | -12   | +1   | 0.56      | TRANSITIONAL |

*COVID skew flattened because panic was indiscriminate — puts AND calls got bid.
This is flagged as a "divergence" (high VIX + flat skew).

### What This Tells You That VIX Alone Doesn't

Consider two states with VIX = 25:

**State A**: VIX=25, Ratio=0.92, Skew=-7, VRP=+4
→ Composite: 0.38 (NORMAL-to-TRANSITIONAL border)
→ Term structure in contango says market views this as temporary
→ Normal skew, healthy VRP → standard vol expansion, not crisis
→ The vol is probably event-driven and will compress post-event

**State B**: VIX=25, Ratio=1.10, Skew=-13, VRP=-2
→ Composite: 0.64 (STRESSED)
→ Backwardation says near-term fear exceeds long-term
→ Steep skew = aggressive put-hedging
→ Negative VRP = realised vol already catching up
→ This is early-stage crisis — watch for acceleration

Same VIX. Completely different market states. The classifier makes this visible.
