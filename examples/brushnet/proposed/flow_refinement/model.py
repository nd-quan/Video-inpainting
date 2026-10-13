"""Image-conditioned 1/4 matching followed by 1/2 fine flow correction.

All public flow tensors are [N,2,H,W], dx/dy in pixels of their own grid.
Input RGB is [N,3,H,W] in [-1,1].  A zero residual exactly preserves the
full-resolution frozen RAFT estimate; only predicted residuals are upsampled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_flow(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must be [N,2,H,W]")
    old_h, old_w = flow.shape[-2:]
    new_h, new_w = map(int, size)
    if min(old_h, old_w, new_h, new_w) < 1:
        raise ValueError("flow dimensions must be positive")
    if (old_h, old_w) == (new_h, new_w):
        return flow
    scaled = F.interpolate(flow.float(), size=(new_h, new_w), mode="bilinear", align_corners=False)
    return torch.stack((scaled[:, 0] * new_w / old_w,
                        scaled[:, 1] * new_h / old_h), dim=1)


def sample_with_flow(source: torch.Tensor, flow: torch.Tensor, dx: int = 0, dy: int = 0):
    """Sample at p+flow(p)+(dx,dy); return samples and an in-bounds mask."""
    if source.ndim != 4 or flow.ndim != 4 or source.shape[0] != flow.shape[0] or source.shape[-2:] != flow.shape[-2:]:
        raise ValueError("source and flow must share batch and spatial dimensions")
    if flow.shape[1] != 2:
        raise ValueError("flow must have two dx/dy channels")
    h, w = flow.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h, device=flow.device, dtype=flow.dtype),
                          torch.arange(w, device=flow.device, dtype=flow.dtype), indexing="ij")
    sx = x[None] + flow[:, 0] + dx
    sy = y[None] + flow[:, 1] + dy
    valid = ((sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1))[:, None]
    grid = torch.stack((2 * (sx + .5) / w - 1, 2 * (sy + .5) / h - 1), dim=-1)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped, valid


def warp_with_flow(source: torch.Tensor, flow: torch.Tensor):
    return sample_with_flow(source, flow)


def normalized_flow_cue(flow: torch.Tensor) -> torch.Tensor:
    """Bound motion input scale without changing geometric flow units."""
    h, w = flow.shape[-2:]
    return torch.stack((flow[:, 0] / max(w, 1), flow[:, 1] / max(h, 1)), dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.GroupNorm(8, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels), nn.SiLU(),
        )
        self.skip = (nn.Identity() if stride == 1 and in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class SharedPyramid(nn.Module):
    def __init__(self, half_channels: int, quarter_channels: int):
        super().__init__()
        self.half_branch = nn.Sequential(ConvBlock(3, half_channels, stride=2), ConvBlock(half_channels, half_channels))
        self.quarter_branch = nn.Sequential(ConvBlock(half_channels, quarter_channels, stride=2),
                                     ConvBlock(quarter_channels, quarter_channels))

    def forward(self, rgb: torch.Tensor):
        half = self.half_branch(rgb)
        return half, self.quarter_branch(half)


def local_matching_cost(target: torch.Tensor, source: torch.Tensor, flow: torch.Tensor,
                        radius: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """L1 costs around p+F(p), with explicit per-candidate geometric validity.

    Looping over offsets avoids materializing [N,C,(2r+1)^2,H,W] all at once.
    There is no Python loop over spatial pixels. Invalid candidates receive
    a high cost and a separate validity channel.
    """
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if target.shape != source.shape or flow.shape != (target.shape[0], 2, *target.shape[-2:]):
        raise ValueError("matching feature/flow shape mismatch")
    target = F.normalize(target.float(), dim=1, eps=1e-6)
    source = F.normalize(source.float(), dim=1, eps=1e-6)
    costs, masks = [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            candidate, valid = sample_with_flow(source, flow.float(), dx, dy)
            cost = (target - candidate).abs().mean(dim=1, keepdim=True)
            costs.append(torch.where(valid, cost, torch.ones_like(cost) * 2.0))
            masks.append(valid.to(cost.dtype))
    return torch.cat(costs, dim=1), torch.cat(masks, dim=1)


@dataclass(frozen=True)
class RefinerConfig:
    half_channels: int = 32
    quarter_channels: int = 64
    match_channels: int = 32
    search_radius: int = 4
    quarter_max_displacement: float = 8.0
    half_max_displacement: float = 4.0
    quarter_only: bool = False

    def validate(self):
        if self.half_channels % 8 or self.quarter_channels % 8:
            raise ValueError("feature channels must be divisible by 8")
        if self.match_channels < 1 or self.search_radius < 0:
            raise ValueError("matching channels and radius must be valid")
        if self.quarter_max_displacement <= 0 or self.half_max_displacement <= 0:
            raise ValueError("maximum residual displacements must be positive")


class FlowRefinementBlock(nn.Module):
    """The same parameters refine forward and backward flow by swapping frames."""

    def __init__(self, config: RefinerConfig = None):
        super().__init__()
        self.config = config or RefinerConfig()
        self.config.validate()
        c = self.config
        self.encoder = SharedPyramid(c.half_channels, c.quarter_channels)
        self.project = nn.Conv2d(c.quarter_channels, c.match_channels, 1)
        cost_channels = (2*c.search_radius + 1)**2
        self.quarter_head = nn.Sequential(
            ConvBlock(c.quarter_channels*3 + cost_channels*2 + 3, 96),
            ConvBlock(96, 64), nn.Conv2d(64, 2, 3, padding=1),
        )
        self.half_head = nn.Sequential(
            ConvBlock(c.half_channels*3 + 3, 64),
            ConvBlock(64, 32), nn.Conv2d(32, 2, 3, padding=1),
        )
        nn.init.zeros_(self.quarter_head[-1].weight)
        nn.init.zeros_(self.quarter_head[-1].bias)
        nn.init.zeros_(self.half_head[-1].weight)
        nn.init.zeros_(self.half_head[-1].bias)

    def config_dict(self) -> Dict:
        return asdict(self.config)

    def _refine_direction(self, target_half, source_half, target_quarter, source_quarter,
                          coarse_rgb: torch.Tensor):
        h2, w2 = target_half.shape[-2:]
        h4, w4 = target_quarter.shape[-2:]
        coarse_4 = resize_flow(coarse_rgb, (h4, w4))
        match_target = self.project(target_quarter)
        match_source = self.project(source_quarter)
        costs, validity = local_matching_cost(match_target, match_source, coarse_4,
                                               self.config.search_radius)
        warped_4, valid_4 = warp_with_flow(source_quarter, coarse_4)
        cues_4 = torch.cat((target_quarter.float(), warped_4.float(),
                            (target_quarter-warped_4).float(),
                            costs, validity, normalized_flow_cue(coarse_4.float()),
                            valid_4.float()), dim=1)
        delta_4 = torch.tanh(self.quarter_head(cues_4)) * self.config.quarter_max_displacement
        if self.config.quarter_only:
            delta_2 = torch.zeros_like(resize_flow(delta_4, (h2, w2)))
        else:
            coarse_2 = resize_flow(coarse_rgb, (h2, w2))
            delta_4_to_2 = resize_flow(delta_4, (h2, w2))
            init_2 = coarse_2 + delta_4_to_2
            warped_2, valid_2 = warp_with_flow(source_half, init_2)
            cues_2 = torch.cat((target_half.float(), warped_2.float(),
                                (target_half-warped_2).float(),
                                normalized_flow_cue(init_2.float()), valid_2.float()), dim=1)
            delta_2 = torch.tanh(self.half_head(cues_2)) * self.config.half_max_displacement
        delta_sum_2 = resize_flow(delta_4, (h2, w2)) + delta_2
        delta_rgb = resize_flow(delta_sum_2, coarse_rgb.shape[-2:])
        # Crucially: the base full-resolution field itself is never resampled.
        refined = coarse_rgb.float() + delta_rgb
        return refined, delta_rgb

    def forward(self, image0: torch.Tensor, image1: torch.Tensor,
                coarse_forward: torch.Tensor, coarse_backward: torch.Tensor,
                return_residuals: bool = False):
        if image0.ndim != 4 or image0.shape != image1.shape or image0.shape[1] != 3:
            raise ValueError("image0/image1 must be matching [N,3,H,W]")
        n, _, h, w = image0.shape
        if h % 4 or w % 4:
            raise ValueError("image resolution must be divisible by four")
        expected = (n, 2, h, w)
        if coarse_forward.shape != expected or coarse_backward.shape != expected:
            raise ValueError(f"forward/backward coarse flow must be {expected}")
        if image0.device != image1.device or image0.device != coarse_forward.device or image0.device != coarse_backward.device:
            raise ValueError("images and flows must be on the same device")
        half0, quarter0 = self.encoder(image0.float())
        half1, quarter1 = self.encoder(image1.float())
        forward, delta_f = self._refine_direction(half0, half1, quarter0, quarter1,
                                                   coarse_forward.detach())
        backward, delta_b = self._refine_direction(half1, half0, quarter1, quarter0,
                                                    coarse_backward.detach())
        return (forward, backward, delta_f, delta_b) if return_residuals else (forward, backward)
