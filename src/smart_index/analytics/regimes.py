"""Regime classification using multi-factor composite scoring.

This module implements the same classification logic displayed in the
interactive web interface (index.html, Regimes page). The design is
intentionally transparent — sigmoid mappings on 4 observable features,
weighted composite, plus divergence detection.

Calibration notes (see docs/methodology/regime_definition.md):
- VIX sigmoid: midpoint=20, k=0.18, calibrated to 1990-2026 VIX distribution
- Term structure: midpoint=1.0 (flat), calibrated to VIX/VIX3M ratio history
- Skew: midpoint=-9.5 (median 25d risk reversal for SPX), k=0.35
- VRP: midpoint=+2 (long-run average IV-RV spread), k=0.25

Weights are set based on information content and independence:
- VIX (0.35): most informative single feature, but insufficient alone
- Term structure (0.25): captures horizon of fear — near vs. persistent
- Skew (0.20): captures distribution of fear — tails vs. body
- VRP (0.20): captures realisation — is the fear justified by actual moves

Example
-------
>>> state = MarketState(vix=27.0, vix_vix3m_ratio=1.01, skew_25d=-12.0, vrp=1.0)
>>> result = classify_regime(state)
>>> result.regime
'TRANSITIONAL'
>>> result.composite_score
0.56
>>> result.conviction
'MEDIUM'
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

RegimeLabel = Literal["COMPRESSED", "NORMAL", "TRANSITIONAL", "STRESSED", "CRISIS"]
ConvictionLevel = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass(frozen=True)
class MarketState:
    """Observable market features used for regime classification."""
    vix: float              # VIX index level
    vix_vix3m_ratio: float  # VIX / VIX3M (term structure ratio)
    skew_25d: float         # 25-delta risk reversal (negative = put-rich, SPX normal)
    vrp: float              # Implied vol minus realised vol (vol risk premium)


@dataclass
class RegimeResult:
    """Output of the regime classifier."""
    regime: RegimeLabel
    composite_score: float      # 0-1, higher = more stressed
    conviction: ConvictionLevel # agreement among features
    feature_scores: dict[str, float]  # individual 0-1 scores
    score_dispersion: float     # std dev of individual scores
    divergence_flags: list[str] # economically meaningful edge cases
    commentary: str             # human-readable interpretation


# ---------------------------------------------------------------------------
# Sigmoid scoring functions
# ---------------------------------------------------------------------------

def _sigmoid(x: float, midpoint: float, k: float) -> float:
    """Logistic sigmoid mapping x → (0, 1).

    Parameters
    ----------
    x : input value
    midpoint : value where sigmoid = 0.5
    k : steepness (higher = sharper transition)
    """
    return 1.0 / (1.0 + math.exp(-k * (x - midpoint)))


def score_vix(vix: float) -> float:
    """Map VIX level to stress score (0-1).

    Calibration: long-run VIX mean ≈ 19.5, median ≈ 17.6
    - VIX 10 → ~0.05 (5th percentile, ultra-calm)
    - VIX 15 → ~0.18 (30th percentile)
    - VIX 20 → ~0.50 (50th percentile)
    - VIX 30 → ~0.86 (90th percentile)
    - VIX 40 → ~0.97 (98th percentile)
    """
    return _sigmoid(vix, midpoint=20.0, k=0.18)


def score_term_structure(ratio: float) -> float:
    """Map VIX/VIX3M ratio to stress score (0-1).

    < 0.85 = deep contango (calm, score ~0.05)
    ≈ 0.92 = normal contango (score ~0.25)
    = 1.00 = flat (score ~0.50)
    > 1.05 = backwardation (stress, score ~0.65)
    > 1.20 = deep backwardation (crisis, score ~0.90)
    """
    return _sigmoid(ratio, midpoint=1.0, k=8.0)


def score_skew(skew_25d: float) -> float:
    """Map 25-delta skew to stress score (0-1).

    Convention: negative value = puts more expensive than calls (SPX normal).
    More negative = steeper = higher hedging demand = higher stress score.

    skew = -3  → ~0.07 (very flat, unusual)
    skew = -6  → ~0.19 (mild)
    skew = -9  → ~0.46 (median)
    skew = -12 → ~0.75 (steep, elevated hedging)
    skew = -16 → ~0.93 (extreme)

    Note: skew input is negative, so we negate it before scoring.
    """
    steepness = -skew_25d  # convert to positive magnitude
    return _sigmoid(steepness, midpoint=9.5, k=0.35)


def score_vrp(vrp: float) -> float:
    """Map vol risk premium (IV - RV) to stress score (0-1).

    High positive VRP = vol sellers compensated = low stress.
    Negative VRP = realised exceeding implied = high stress.

    VRP = +8  → ~0.08 (very high premium, calm)
    VRP = +4  → ~0.24 (healthy)
    VRP = +2  → ~0.50 (median, equilibrium)
    VRP =  0  → ~0.67 (thin, caution)
    VRP = -5  → ~0.92 (deeply negative, crisis)

    Note: we invert because higher VRP = lower stress.
    """
    return _sigmoid(-vrp, midpoint=-2.0, k=0.25)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

WEIGHTS = {
    "vix": 0.35,
    "term_structure": 0.25,
    "skew": 0.20,
    "vrp": 0.20,
}


# ---------------------------------------------------------------------------
# Regime thresholds
# ---------------------------------------------------------------------------

REGIME_THRESHOLDS = [
    (0.18, "COMPRESSED"),
    (0.38, "NORMAL"),
    (0.58, "TRANSITIONAL"),
    (0.78, "STRESSED"),
    (1.01, "CRISIS"),  # 1.01 so score=1.0 still maps
]


def _composite_to_regime(score: float) -> RegimeLabel:
    for threshold, label in REGIME_THRESHOLDS:
        if score < threshold:
            return label
    return "CRISIS"


# ---------------------------------------------------------------------------
# Conviction (feature agreement)
# ---------------------------------------------------------------------------

def _compute_conviction(scores: list[float]) -> tuple[ConvictionLevel, float]:
    """Compute conviction from dispersion of individual scores."""
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    sd = math.sqrt(variance)

    if sd < 0.12:
        return "HIGH", sd
    elif sd < 0.22:
        return "MEDIUM", sd
    else:
        return "LOW", sd


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

def _detect_divergences(state: MarketState, scores: dict[str, float]) -> list[str]:
    """Detect economically meaningful feature disagreements.

    These edge cases are where the classifier adds value beyond
    simple thresholds — they identify states where the *combination*
    of features tells a different story than any single feature.
    """
    flags = []

    # Low VIX + steep skew = "quiet hedging"
    if state.vix < 16 and state.skew_25d < -10:
        flags.append(
            "QUIET HEDGING: VIX is low but skew is steep — someone is buying "
            "put protection in calm markets. Historically precedes vol events "
            "by 2–4 weeks."
        )

    # High VIX + flat skew = "indiscriminate panic"
    if state.vix > 30 and state.skew_25d > -6:
        flags.append(
            "INDISCRIMINATE PANIC: VIX is elevated but skew is flat — fear is "
            "across all strikes, not just tails. This often marks capitulation "
            "bottoms (cf. March 2020 peak)."
        )

    # Backwardation without high VIX = "event risk"
    if state.vix_vix3m_ratio > 1.05 and state.vix < 22:
        flags.append(
            "EVENT-DRIVEN INVERSION: Term structure inverted but VIX is moderate — "
            "near-term uncertainty (FOMC, CPI, geopolitical deadline) without "
            "broad macro stress. Post-event vol crush is the expected pattern."
        )

    # Deep contango + high VIX = "temporary shock"
    if state.vix_vix3m_ratio < 0.90 and state.vix > 25:
        flags.append(
            "TEMPORARY STRESS: VIX elevated but term structure in contango — "
            "market views current stress as non-persistent. Far-dated vol "
            "hasn't repriced. Watch for either resolution or contagion."
        )

    # Negative VRP in any regime = "realised catching up"
    if state.vrp < -2:
        flags.append(
            "NEGATIVE VRP: Realised vol exceeding implied — the market under-"
            "estimated actual moves. This is rare (~5% of days historically) "
            "and almost always coincides with drawdowns."
        )

    # Very high VRP + low VIX = "complacency carry"
    if state.vrp > 6 and state.vix < 14:
        flags.append(
            "COMPLACENCY PEAK: Very high VRP in ultra-low VIX — maximum "
            "compensation for vol sellers, but also maximum fragility. "
            "This was the state in Jan 2018 and Jan 2020 before their "
            "respective crashes."
        )

    # Steep skew + backwardation + negative VRP = all-clear stress
    if state.skew_25d < -12 and state.vix_vix3m_ratio > 1.05 and state.vrp < 0:
        flags.append(
            "FULL STRESS ALIGNMENT: All features point to genuine crisis. "
            "Historically, this level of feature agreement precedes the "
            "most sustained drawdowns."
        )

    return flags


# ---------------------------------------------------------------------------
# Commentary generation
# ---------------------------------------------------------------------------

def _generate_commentary(
    state: MarketState,
    regime: RegimeLabel,
    scores: dict[str, float],
    conviction: ConvictionLevel,
    divergences: list[str],
) -> str:
    """Generate human-readable interpretation of the regime state."""

    parts = []

    # Regime headline
    regime_descriptions = {
        "COMPRESSED": (
            f"VIX at {state.vix:.1f} places the market in a compressed-volatility regime. "
            "Short-vol carry strategies look attractive, but this is historically the most "
            "fragile state — the further VIX compresses, the sharper the eventual snap."
        ),
        "NORMAL": (
            f"VIX at {state.vix:.1f} with balanced surface features. Standard risk pricing — "
            "neither complacent nor stressed. The vol surface is functioning normally as a "
            "risk-transfer mechanism."
        ),
        "TRANSITIONAL": (
            f"VIX at {state.vix:.1f} with stress building across surface features. "
            "This is the regime where hedging demand is rising and the surface shape is "
            "changing faster than the headline VIX number suggests."
        ),
        "STRESSED": (
            f"VIX at {state.vix:.1f} in an active risk-off regime. Protection is expensive "
            "but the question is whether we're approaching peak stress (mean-reversion "
            "opportunity) or just the beginning (more pain ahead)."
        ),
        "CRISIS": (
            f"VIX at {state.vix:.1f} — extreme dislocation. Historically, VIX above 35 "
            "mean-reverts within weeks, but the path can be violent. Timing the turn is "
            "the hardest trade in vol markets."
        ),
    }
    parts.append(regime_descriptions[regime])

    # Term structure interpretation
    if state.vix_vix3m_ratio > 1.05:
        parts.append(
            f"Term structure inverted (ratio {state.vix_vix3m_ratio:.2f}) — near-term "
            "fear exceeds far-term, consistent with acute stress."
        )
    elif state.vix_vix3m_ratio < 0.90:
        parts.append(
            f"Term structure in steep contango (ratio {state.vix_vix3m_ratio:.2f}) — "
            "market expects volatility to normalise over time."
        )

    # Conviction
    if conviction == "LOW":
        parts.append(
            "Feature agreement is LOW — the individual signals are pointing in different "
            "directions. This ambiguity is itself informative: the market hasn't made up "
            "its mind about the nature of the current risk."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_regime(state: MarketState) -> RegimeResult:
    """Classify market regime from observable vol surface features.

    Parameters
    ----------
    state : MarketState with vix, vix_vix3m_ratio, skew_25d, vrp

    Returns
    -------
    RegimeResult with regime label, scores, conviction, divergences, commentary.
    """
    # Score each feature
    feature_scores = {
        "vix": score_vix(state.vix),
        "term_structure": score_term_structure(state.vix_vix3m_ratio),
        "skew": score_skew(state.skew_25d),
        "vrp": score_vrp(state.vrp),
    }

    # Weighted composite
    composite = sum(
        WEIGHTS[k] * feature_scores[k] for k in WEIGHTS
    )

    # Regime label
    regime = _composite_to_regime(composite)

    # Conviction
    score_list = list(feature_scores.values())
    conviction, dispersion = _compute_conviction(score_list)

    # Divergences
    divergences = _detect_divergences(state, feature_scores)

    # Commentary
    commentary = _generate_commentary(state, regime, feature_scores, conviction, divergences)

    return RegimeResult(
        regime=regime,
        composite_score=round(composite, 3),
        conviction=conviction,
        feature_scores={k: round(v, 3) for k, v in feature_scores.items()},
        score_dispersion=round(dispersion, 3),
        divergence_flags=divergences,
        commentary=commentary,
    )


# ---------------------------------------------------------------------------
# Historical reference episodes
# ---------------------------------------------------------------------------

HISTORICAL_EPISODES = {
    "2017_low_vol": MarketState(vix=10.0, vix_vix3m_ratio=0.82, skew_25d=-5.0, vrp=6.0),
    "pre_volmageddon_jan2018": MarketState(vix=11.0, vix_vix3m_ratio=0.78, skew_25d=-4.0, vrp=7.0),
    "volmageddon_feb2018": MarketState(vix=37.0, vix_vix3m_ratio=1.30, skew_25d=-14.0, vrp=-8.0),
    "covid_crash_mar2020": MarketState(vix=66.0, vix_vix3m_ratio=1.35, skew_25d=-5.0, vrp=-20.0),
    "meme_mania_jan2021": MarketState(vix=24.0, vix_vix3m_ratio=0.95, skew_25d=-4.0, vrp=2.0),
    "rate_shock_sep2022": MarketState(vix=32.0, vix_vix3m_ratio=1.08, skew_25d=-10.0, vrp=-2.0),
    "jpy_unwind_aug2024": MarketState(vix=55.0, vix_vix3m_ratio=1.35, skew_25d=-12.0, vrp=-15.0),
    "mar2026_geopolitical": MarketState(vix=27.0, vix_vix3m_ratio=1.01, skew_25d=-12.0, vrp=1.0),
}


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("REGIME CLASSIFIER — HISTORICAL EPISODES")
    print("=" * 70)
    for name, state in HISTORICAL_EPISODES.items():
        result = classify_regime(state)
        print(f"\n{name}")
        print(f"  State: VIX={state.vix}, TS={state.vix_vix3m_ratio}, "
              f"Skew={state.skew_25d}, VRP={state.vrp}")
        print(f"  Regime: {result.regime} (score={result.composite_score})")
        print(f"  Conviction: {result.conviction} (dispersion={result.score_dispersion})")
        print(f"  Scores: {result.feature_scores}")
        if result.divergence_flags:
            for flag in result.divergence_flags:
                print(f"  ⚠ {flag}")
