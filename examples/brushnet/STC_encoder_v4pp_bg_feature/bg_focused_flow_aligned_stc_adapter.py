"""BG-focused extension of the V4 flow-aligned RGB-STC adapter.

V4 aligns every spatial location before temporal attention.  V4++ retains the
same predicted bidirectional flow and confidence computation, but applies the
alignment residual only where the internal degraded-background mask is one.
The high-quality ROI therefore follows the exact raw V2 feature path.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

try:
    from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        FlowAlignedRGBSTCAdapter,
    )
except ModuleNotFoundError:  # Imported as examples.brushnet.STC_encoder_v4pp_bg_feature.
    from ..STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        FlowAlignedRGBSTCAdapter,
    )


class BGFocusedFlowAlignedRGBSTCAdapter(FlowAlignedRGBSTCAdapter):
    """V4 with hard BG gating on the local alignment residual."""

    def encode(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Tuple[int, int] = None,
    ):
        batch, frames, height, width = self.stc_adapter._validate_inputs(
            rgb_sequence, bg_mask_sequence
        )
        if output_size is None:
            output_size = (height // 8, width // 8)
        output_size = (int(output_size[0]), int(output_size[1]))
        if output_size[0] < 1 or output_size[1] < 1:
            raise ValueError("output_size must contain positive dimensions")

        pixel_condition = self.build_pixel_condition(
            rgb_sequence, bg_mask_sequence
        ).flatten(0, 1)
        spatial_flat = self.stc_adapter.spatial_encoder(pixel_condition)
        if spatial_flat.shape[-2:] != output_size:
            spatial_flat = F.interpolate(
                spatial_flat,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        _, channels, encoded_height, encoded_width = spatial_flat.shape
        spatial_features = spatial_flat.reshape(
            batch, frames, channels, encoded_height, encoded_width
        )

        flow_forward, flow_backward = self._decode_bidirectional_flow(
            spatial_features
        )
        fully_aligned, confidence = self._align_spatial_features(
            spatial_features, flow_forward, flow_backward
        )

        feature_bg_mask = F.interpolate(
            bg_mask_sequence.flatten(0, 1).to(
                device=spatial_features.device, dtype=spatial_features.dtype
            ),
            size=(encoded_height, encoded_width),
            mode="nearest",
        ).reshape(batch, frames, 1, encoded_height, encoded_width)
        feature_bg_mask = (feature_bg_mask >= 0.5).to(spatial_features.dtype)

        # Frame zero has no predecessor.  For t>0, keep the V4 residual only
        # on degraded BG; ROI remains bit-identical to the raw V2 feature.
        aligned_current = spatial_features[:, 1:] + feature_bg_mask[:, 1:] * (
            fully_aligned[:, 1:] - spatial_features[:, 1:]
        )
        aligned_spatial = torch.cat(
            (spatial_features[:, :1], aligned_current), dim=1
        )

        tokens = aligned_spatial.permute(0, 3, 4, 1, 2).reshape(
            batch * encoded_height * encoded_width, frames, channels
        )
        for temporal_block in self.stc_adapter.temporal_blocks:
            tokens = temporal_block(tokens)
        features = tokens.reshape(
            batch, encoded_height, encoded_width, frames, channels
        ).permute(0, 3, 4, 1, 2)
        features_flat = features.flatten(0, 1)
        features_flat = self.stc_adapter.output_act(
            self.stc_adapter.output_norm(features_flat)
        )
        delta = self.stc_adapter.zero_conv(features_flat).reshape(
            batch, frames, 4, encoded_height, encoded_width
        )
        return (
            delta,
            features,
            spatial_features,
            aligned_spatial,
            flow_forward,
            flow_backward,
            confidence,
        )
