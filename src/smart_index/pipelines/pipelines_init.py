"""End-to-end pipelines: data → features → analytics → output."""
from smart_index.pipelines.feature_pipeline import (
    run_pipeline,
    build_feature_panel_from_surfaces,
)
__all__ = ["run_pipeline", "build_feature_panel_from_surfaces"]
