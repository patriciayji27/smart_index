# Project Overview

## Motivation

Equity index vol surfaces encode a rich set of market expectations — about
tail risk, term structure of uncertainty, macro event pricing, and
positioning — that go far beyond a single VIX number.  This project builds
a toolkit to systematically extract, track, and analyse those signals.

## Core Questions

1. **What does the vol surface "know" that VIX doesn't?**
   VIX is a single number that aggregates the entire 30-day smile into a
   variance-swap price.  The shape of the smile (skew, convexity, wings)
   and the term structure carry information VIX discards.

2. **How do macro state variables map onto surface shape?**
   Do rate expectations, credit spreads, or positioning data predict
   changes in skew or term structure?  Where do these relationships
   break down — and is the breakdown itself tradeable?

3. **Does the market systematically misprice events?**
   Pre-FOMC, pre-CPI, pre-NFP: is the implied move consistently too
   large, too small, or correctly calibrated?  Does it depend on regime?

4. **Can regimes be detected in real time from surface features?**
   If skew steepening + term structure inversion + convexity rise
   consistently precede drawdowns, can we build a useful early-warning
   signal?

## Scope and Constraints

- **Data**: Primarily SPX options (European exercise, deep liquidity).
  SPY for short-dated / weekly studies.  Macro from FRED and public sources.
- **Horizon**: Focus on daily-to-weekly frequency signals, not
  intraday microstructure.
- **Not a backtested strategy**: This is an analytics and research
  framework, not a trading system.  Any P&L analysis is for intuition,
  not live deployment.
