"""I/O utilities: config loading, path resolution, data persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/smart_index/utils/io.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file by name (without extension).

    >>> cfg = load_config("symbols")
    >>> cfg["index"]["primary"]
    'SPX'
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_data_path(category: str, *parts: str) -> Path:
    """Resolve a data path from paths.yaml config.

    >>> resolve_data_path("raw", "spx_options", "2025-01.parquet")
    PosixPath('.../data/raw/spx_options/2025-01.parquet')
    """
    paths_cfg = load_config("paths")
    base = paths_cfg["data"].get(category)
    if base is None:
        raise KeyError(f"Unknown data category: {category}")
    return PROJECT_ROOT / base / Path(*parts)


def resolve_output_path(category: str, *parts: str) -> Path:
    """Resolve an output path, creating directories as needed."""
    paths_cfg = load_config("paths")
    base = paths_cfg["outputs"].get(category)
    if base is None:
        raise KeyError(f"Unknown output category: {category}")
    out = PROJECT_ROOT / base / Path(*parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist; return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
