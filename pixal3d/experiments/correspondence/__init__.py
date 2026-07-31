"""Multi-view correspondence experiments."""

from .metrics import (
    branch_norms,
    feature_cosine_error,
    feature_l2_drift,
    weight_diagnostics,
)
from .projection import (
    MultiviewProjection,
    oracle_reliability,
    project_points_for_views,
    sample_oracle_masks,
)

__all__ = [
    "branch_norms",
    "feature_cosine_error",
    "feature_l2_drift",
    "weight_diagnostics",
    "MultiviewProjection",
    "oracle_reliability",
    "project_points_for_views",
    "sample_oracle_masks",
]
