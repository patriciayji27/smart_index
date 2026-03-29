"""Regime classification from vol surface and correlation features.

Two layers:
1. Heuristic classifier — transparent, rule-based, matches the interactive
   classifier on the project website. Useful as a baseline and for sanity
   checking any statistical model.

2. GaussianHMM skeleton — data-driven state discovery over the same feature
   set. Planned for Phase 7; scaffolded here so the interface is stable.

The three regimes from the dispersion signal research:
  MACRO_DRIVEN       — high COR3M, flat corr smile, VIX > 22
  IDIOSYNCRATIC      — low COR3M, steep corr smile, wide VRP
  CONCENTRATED_NAME  — VIX rising while COR3M falling (vol localized)
  CALM               — low VIX, normal structure
  ELEVATED           — elevated VIX without clear correlation signal

Feature columns expected by both classifiers:
  vix                 float   CBOE VIX level
  vix_vix3m_ratio     float   VIX / VIX3M (term structure)
  skew_25d            float   25d put − 25d call IV, 30d tenor
  iv_rv_spread        float   30d implied vol − 30d realized vol
  cor3m               float   COR3M implied correlation (0–100 scale)
  corr_smile_width    float   COR10D − COR90D
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# Regime labels
RegimeLabel = Literal[
    "macro_driven",
    "idiosyncratic",
    "concentrated_name",
    "calm",
    "elevated",
]

REGIME_DESCRIPTIONS: dict[str, str] = {
    "macro_driven": (
        "High implied correlation + elevated VIX. Factor exposure (rates, dollar, "
        "credit) dominates stock-specific stories. Flat correlation smile confirms "
        "broad co-movement. Index hedges expensive; dispersion strategies face headwinds."
    ),
    "idiosyncratic": (
        "Low implied correlation + steep correlation smile. Stocks trade on individual "
        "fundamentals. SPX puts cheap relative to component vol. VRP wide — market "
        "hedging something that hasn't happened yet. Best period for dispersion carry."
    ),
    "concentrated_name": (
        "VIX rising while COR3M is low or falling. Vol localized to one or two "
        "mega-caps rather than the broad market. Equal-weight vs cap-weight correlation "
        "spread widens. Alarming on VIX screen; correlation signal reveals it's not systemic."
    ),
    "calm": (
        "Low VIX, normal term structure. Dealer gamma likely elevated; mechanical vol "
        "suppression at work. VRP wide because implied stays bid while realized is "
        "compressed. GEX flip point well above current spot."
    ),
    "elevated": (
        "Elevated VIX with moderate implied correlation — uncertainty without clear "
        "macro driver. Watch correlation smile narrowing as a leading indicator of "
        "regime transition toward macro_driven."
    ),
}


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

@dataclass
class HeuristicThresholds:
    """Tunable thresholds for the heuristic classifier.

    These defaults replicate the interactive classifier on the website.
    Adjust to your data's percentile distributions if using a long history.
    """
    # Macro-driven: both COR3M and VIX elevated
    cor3m_macro:    float = 48.0
    vix_macro:      float = 22.0

    # Idiosyncratic: low COR3M, steep smile, calm VIX
    cor3m_idio:     float = 30.0
    vix_idio:       float = 22.0
    smile_steep:    float = 18.0   # COR10D - COR90D threshold

    # Concentrated-name: VIX elevated but COR3M low
    vix_conc:       float = 20.0
    cor3m_conc:     float = 35.0

    # Calm: low VIX
    vix_calm:       float = 17.0


def classify_heuristic(
    vix: float,
    cor3m: float,
    corr_smile_width: float,
    vix_vix3m_ratio: float = 0.95,
    iv_rv_spread: float = 3.0,
    thresholds: HeuristicThresholds | None = None,
) -> RegimeLabel:
    """Classify a single observation into one of five regimes.

    Parameters
    ----------
    vix : CBOE VIX level
    cor3m : COR3M implied correlation (0–100 scale)
    corr_smile_width : COR10D − COR90D (correlation smile steepness)
    vix_vix3m_ratio : VIX / VIX3M — term structure signal
    iv_rv_spread : 30d implied vol − 30d realized vol
    thresholds : optional custom thresholds

    Returns
    -------
    Regime label string.
    """
    t = thresholds or HeuristicThresholds()

    # Order matters — most specific checks first
    if cor3m > t.cor3m_macro and vix > t.vix_macro:
        return "macro_driven"

    if cor3m < t.cor3m_idio and vix < t.vix_idio and corr_smile_width > t.smile_steep:
        return "idiosyncratic"

    if vix > t.vix_conc and cor3m < t.cor3m_conc:
        return "concentrated_name"

    if vix < t.vix_calm:
        return "calm"

    return "elevated"


def classify_series(
    features: pd.DataFrame,
    thresholds: HeuristicThresholds | None = None,
) -> pd.Series:
    """Apply heuristic classifier to a DataFrame of daily features.

    Parameters
    ----------
    features : DataFrame with columns matching the feature set described
               in the module docstring. Missing columns are filled with
               neutral defaults (classifier degrades gracefully).

    Returns
    -------
    pd.Series of regime labels, indexed like `features`.
    """
    def _safe(col: str, default: float) -> pd.Series:
        return features[col] if col in features.columns else pd.Series(default, index=features.index)

    vix   = _safe("vix", 18.0)
    cor3m = _safe("cor3m", 35.0)
    smile = _safe("corr_smile_width", 20.0)
    ratio = _safe("vix_vix3m_ratio", 0.95)
    vrp   = _safe("iv_rv_spread", 3.0)

    labels = []
    for i in range(len(features)):
        labels.append(classify_heuristic(
            vix=float(vix.iloc[i]),
            cor3m=float(cor3m.iloc[i]),
            corr_smile_width=float(smile.iloc[i]),
            vix_vix3m_ratio=float(ratio.iloc[i]),
            iv_rv_spread=float(vrp.iloc[i]),
            thresholds=thresholds,
        ))

    return pd.Series(labels, index=features.index, name="regime")


# ---------------------------------------------------------------------------
# Regime summary statistics
# ---------------------------------------------------------------------------

def regime_summary(
    features: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Compute conditional feature statistics by regime.

    Useful for validating that the regimes are empirically distinct
    and for presenting regime characteristics in research output.

    Returns a DataFrame indexed by regime with mean/std/count for each feature.
    """
    joined = features.join(regimes.rename("regime"), how="inner")
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    records = []
    for regime_label, group in joined.groupby("regime"):
        row: dict = {"regime": regime_label, "count": len(group)}
        for col in numeric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"]  = group[col].std()
        records.append(row)

    return pd.DataFrame(records).set_index("regime")


def regime_transitions(regimes: pd.Series) -> pd.DataFrame:
    """Compute regime transition matrix (empirical probabilities).

    Row = current regime, column = next regime, value = transition probability.
    Useful for checking whether regimes are persistent (diagonal should dominate)
    and for understanding which transitions are most common.
    """
    labels = sorted(regimes.dropna().unique())
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)

    prev = None
    for curr in regimes.dropna():
        if prev is not None:
            matrix.loc[prev, curr] += 1
        prev = curr

    # Normalise rows to get probabilities
    row_sums = matrix.sum(axis=1)
    return matrix.div(row_sums, axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# HMM classifier (Phase 7 — scaffolded, not yet fitted)
# ---------------------------------------------------------------------------

class HMMRegimeClassifier:
    """Gaussian HMM over vol surface features — Phase 7 placeholder.

    Once `hmmlearn` is available and a feature panel is built, this class
    will fit a GaussianHMM and map hidden states to the three economic regimes
    via post-hoc labeling (match states to heuristic labels by majority vote).

    The interface is designed to match `classify_series()` so downstream
    code can swap classifiers without changes.
    """

    def __init__(self, n_states: int = 4, random_state: int = 42) -> None:
        self.n_states = n_states
        self.random_state = random_state
        self._model = None
        self._state_labels: dict[int, RegimeLabel] = {}
        self._feature_cols: list[str] = []

    def fit(self, features: pd.DataFrame, heuristic_labels: pd.Series) -> "HMMRegimeClassifier":
        """Fit the HMM and map hidden states to economic regime labels.

        Parameters
        ----------
        features : daily feature panel (normalised — pass z-scores)
        heuristic_labels : heuristic regime labels for post-hoc state assignment
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                "hmmlearn is required for HMMRegimeClassifier. "
                "Install with: pip install hmmlearn"
            )

        self._feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        X = features[self._feature_cols].dropna().values

        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
        )
        self._model.fit(X)

        # Assign economic labels to hidden states via majority vote
        aligned = features[self._feature_cols].dropna()
        hidden_states = self._model.predict(aligned.values)
        joined = pd.Series(hidden_states, index=aligned.index).rename("state")
        heuristic_aligned = heuristic_labels.reindex(aligned.index)

        for state_id in range(self.n_states):
            mask = joined == state_id
            if mask.sum() > 0:
                majority = heuristic_aligned[mask].mode()
                self._state_labels[state_id] = majority.iloc[0] if len(majority) > 0 else "elevated"

        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict regime labels for a feature panel."""
        if self._model is None:
            raise RuntimeError("Call .fit() before .predict()")

        X = features[self._feature_cols].dropna().values
        states = self._model.predict(X)
        labels = [self._state_labels.get(s, "elevated") for s in states]
        return pd.Series(labels, index=features[self._feature_cols].dropna().index, name="regime_hmm")

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
