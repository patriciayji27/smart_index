"""Plot styling — apply consistent theme across all figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

from smart_index.utils.io import load_config


def apply_style() -> None:
    """Apply project-wide matplotlib style from config/plotting.yaml."""
    cfg = load_config("plotting")["style"]
    plt.style.use(cfg.get("matplotlib_style", "seaborn-v0_8-whitegrid"))
    plt.rcParams.update({
        "figure.figsize": cfg.get("figsize_default", [10, 6]),
        "figure.dpi": cfg.get("dpi", 150),
        "font.family": cfg.get("font_family", "sans-serif"),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def get_colors() -> dict:
    """Return colour palette from config."""
    return load_config("plotting")["colors"]


def get_regime_colors() -> dict[str, str]:
    """Return regime → colour mapping."""
    return load_config("plotting")["colors"]["regime"]
