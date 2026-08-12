"""STC-v2++ motion-aligned noise deformation components."""

from .noise_deformation import (
    DeformationNoiseOutput,
    NoiseDeformationHead,
    backward_warp,
    build_deformed_clip_noise,
    cosine_feature_matching_loss,
    edge_aware_smoothness_loss,
    make_backward_sampling_grid,
    normalize_lineage_channels,
    variance_preserving_fusion,
)

__all__ = [
    "DeformationNoiseOutput",
    "NoiseDeformationHead",
    "backward_warp",
    "build_deformed_clip_noise",
    "cosine_feature_matching_loss",
    "edge_aware_smoothness_loss",
    "make_backward_sampling_grid",
    "normalize_lineage_channels",
    "variance_preserving_fusion",
]
