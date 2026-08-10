"""RGB-STC condition adapter with a train-only bidirectional flow head.

The deployable restoration branch is exactly the RGB-STC v2 adapter. During
training, the same spatio-temporal feature tensor is also decoded into adjacent
bidirectional optical flow. Clean-video RAFT flow supervises this auxiliary
head, so gradients teach the RGB-STC representation about motion without
requiring optical flow at inference.

Mask convention after dataset loading:

* ``M_BG == 1``: strongly degraded background that must be restored.
* ``M_BG == 0``: high-quality ROI that must be preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
try:
    from STC_encoder_v2_rgb.rgb_stc_adapter import RGBSTCConditionAdapter
except ModuleNotFoundError:  # Imported as examples.brushnet.STC_encoder_v3_rgb_flow.
    from ..STC_encoder_v2_rgb.rgb_stc_adapter import RGBSTCConditionAdapter


def _flow_bounds(value: Sequence[float]) -> Tuple[float, float]:
    if isinstance(value, (int, float)):
        result = (float(value), float(value))
    else:
        result = tuple(float(item) for item in value)
    if len(result) != 2 or any(item <= 0.0 for item in result):
        raise ValueError("flow_max_displacement must be positive (x,y)")
    return result


class SharedDirectionalFlowHead(nn.Module):
    """Decode query-to-reference flow from an ordered adjacent feature pair.

    The returned field is defined on ``query`` coordinates and samples
    ``reference``. The same weights serve both temporal directions.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3 * channels, 2 * channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        # Zero flow is a stable starting point. On the first optimization step,
        # L_flow updates this final layer; later steps propagate into RGB-STC.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        reference: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if reference.ndim != 4 or reference.shape != query.shape:
            raise ValueError("reference and query must match [N,C,H,W]")
        ordered_pair = torch.cat(
            (reference, query, query - reference), dim=1
        )
        return self.network(ordered_pair)


@dataclass
class RGBSTCFlowOutput:
    delta_bg: torch.Tensor
    features: torch.Tensor
    latent_bg_mask: torch.Tensor
    predicted_flow_forward: Optional[torch.Tensor]
    predicted_flow_backward: Optional[torch.Tensor]


class RGBSTCFlowAdapter(ModelMixin, ConfigMixin):
    """One checkpointable trainable model containing RGB-STC and flow head."""

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        hidden_channels: int = 64,
        num_heads: int = 2,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        downsample_factor: int = 8,
        output_channels: int = 4,
        condition_mode: str = "full_rgb_bg_mask",
        flow_max_displacement: Tuple[float, float] = (8.0, 8.0),
    ):
        super().__init__()
        bounds = _flow_bounds(flow_max_displacement)
        self.stc_adapter = RGBSTCConditionAdapter(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            downsample_factor=downsample_factor,
            output_channels=output_channels,
            condition_mode=condition_mode,
        )
        self.flow_head = SharedDirectionalFlowHead(int(hidden_channels))
        # ConfigMixin serializes the original argument, so retain a canonical
        # runtime value for tensor construction and backward-compatible loads.
        self._flow_max_displacement = bounds

    @property
    def zero_conv(self):
        return self.stc_adapter.zero_conv

    def build_pixel_condition(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
    ) -> torch.Tensor:
        return self.stc_adapter.build_pixel_condition(
            rgb_sequence, bg_mask_sequence
        )

    def _decode_bidirectional_flow(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 5:
            raise ValueError("features must have shape [B,T,C,H,W]")
        batch, frames, channels, height, width = features.shape
        pairs = frames - 1
        if pairs <= 0:
            empty = features.new_zeros(batch, 0, 2, height, width)
            return empty, empty

        previous = features[:, :-1].reshape(-1, channels, height, width)
        current = features[:, 1:].reshape(-1, channels, height, width)
        bounds = features.new_tensor(self._flow_max_displacement).reshape(
            1, 2, 1, 1
        )

        # Backward is defined on frame t+1 and samples frame t.
        backward = self.flow_head(previous, current).tanh() * bounds
        # Forward is defined on frame t and samples frame t+1.
        forward = self.flow_head(current, previous).tanh() * bounds
        backward = backward.reshape(batch, pairs, 2, height, width)
        forward = forward.reshape(batch, pairs, 2, height, width)
        return forward, backward

    def forward(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        predict_flow: bool = True,
        return_dict: bool = True,
    ):
        stc_output = self.stc_adapter(
            rgb_sequence,
            bg_mask_sequence,
            output_size=output_size,
            return_dict=True,
        )
        flow_forward = flow_backward = None
        if predict_flow:
            # Deliberately no detach: L_flow -> head -> features -> RGB-STC.
            flow_forward, flow_backward = self._decode_bidirectional_flow(
                stc_output.features
            )
        result = RGBSTCFlowOutput(
            delta_bg=stc_output.delta_bg,
            features=stc_output.features,
            latent_bg_mask=stc_output.latent_bg_mask,
            predicted_flow_forward=flow_forward,
            predicted_flow_backward=flow_backward,
        )
        if return_dict:
            return result
        return (
            result.delta_bg,
            result.features,
            result.latent_bg_mask,
            result.predicted_flow_forward,
            result.predicted_flow_backward,
        )


def augment_brushnet_condition(
    model: RGBSTCFlowAdapter,
    base_condition_latents: torch.Tensor,
    rgb_sequence: torch.Tensor,
    bg_mask_sequence: torch.Tensor,
    injection_scale: float = 1.0,
    predict_flow: bool = True,
):
    """Inject only the BG-gated latent delta; keep BrushNet mask channel intact."""
    if base_condition_latents.ndim != 4 or base_condition_latents.shape[1] != 4:
        raise ValueError(
            "base_condition_latents must have shape [B*T,4,h,w]"
        )
    batch, frames = rgb_sequence.shape[:2]
    if base_condition_latents.shape[0] != batch * frames:
        raise ValueError("base_condition_latents batch must equal B*T")

    output = model(
        rgb_sequence,
        bg_mask_sequence,
        output_size=base_condition_latents.shape[-2:],
        predict_flow=predict_flow,
        return_dict=True,
    )
    base_sequence = base_condition_latents.reshape(
        batch, frames, 4, *base_condition_latents.shape[-2:]
    )
    delta = output.delta_bg.to(dtype=base_sequence.dtype)
    latent_bg_mask = output.latent_bg_mask.to(dtype=base_sequence.dtype)
    augmented = base_sequence + float(injection_scale) * delta
    brushnet_condition = torch.cat(
        (augmented, latent_bg_mask), dim=2
    ).flatten(0, 1)
    return brushnet_condition, output, augmented
