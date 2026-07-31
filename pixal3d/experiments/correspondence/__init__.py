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
from .warps import (
    ControlledWarp,
    affine_forward_field,
    affine_inverse_grid,
    invert_forward_field,
    normalized_to_pixel,
    pixel_to_normalized,
    smooth_random_forward_field,
    warp_affine,
    warp_with_forward_field,
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
    "ControlledWarp",
    "affine_forward_field",
    "affine_inverse_grid",
    "invert_forward_field",
    "normalized_to_pixel",
    "pixel_to_normalized",
    "smooth_random_forward_field",
    "warp_affine",
    "warp_with_forward_field",
]
