"""Common geometry for direct and rescaled clean-latent guidance.

Flow is a destination-to-source sampling displacement [dx,dy]. RAFT forward
lives on frame i and samples i+1; backward lives on i+1 and samples i.
All geometry runs in FP32. Confidence is measured in RGB coordinates once,
so changing the feature-warp scale does not change the confidence ablation.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GuidanceConfig:
    warp_mode: str = "direct"
    upscale_factor: int = 4
    use_confidence_mask: bool = True
    fb_alpha: float = 0.01
    fb_beta: float = 0.5
    direction: str = "alternating"

    def __post_init__(self):
        if self.warp_mode not in ("direct", "dgaf"):
            raise ValueError("warp_mode must be direct or dgaf")
        if type(self.upscale_factor) is not int or self.upscale_factor < 2:
            raise ValueError("upscale_factor must be an integer >= 2")
        if self.direction not in ("alternating", "bidirectional", "previous", "next"):
            raise ValueError("Unknown guidance direction")
        if any(not math.isfinite(v) or v < 0 for v in (self.fb_alpha, self.fb_beta)):
            raise ValueError("FB thresholds must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)


def resize_flow(flow, size):
    """Resize [N,2,H,W] and scale displacement independently in x and y."""
    height, width = flow.shape[-2:]
    result = F.interpolate(flow.float(), size=size, mode="bilinear", align_corners=False)
    return result * result.new_tensor([size[1] / width, size[0] / height])[None, :, None, None]


def sample_with_flow(source, flow):
    """Bilinear backward sampling; reject out-of-bounds and nonfinite flow."""
    if source.ndim != 4 or flow.shape != (source.shape[0], 2, *source.shape[-2:]):
        raise ValueError("source [N,C,H,W] and flow [N,2,H,W] must share their grid")
    height, width = source.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=torch.float32),
        torch.arange(width, device=source.device, dtype=torch.float32), indexing="ij",
    )
    coordinates = torch.stack((x, y), dim=0)[None] + flow.float()
    valid = torch.isfinite(coordinates).all(dim=1, keepdim=True)
    valid = valid & (coordinates[:, :1] >= 0) & (coordinates[:, :1] <= width - 1)
    valid = valid & (coordinates[:, 1:] >= 0) & (coordinates[:, 1:] <= height - 1)
    coordinates = torch.nan_to_num(coordinates, nan=-1, posinf=-1, neginf=-1)
    grid = 2 * (coordinates + 0.5) / coordinates.new_tensor([width, height])[None, :, None, None] - 1
    sampled = F.grid_sample(source.float(), grid.permute(0, 2, 3, 1),
                            mode="bilinear", padding_mode="zeros", align_corners=False)
    return torch.where(valid, sampled, 0), valid.float()


def warp_clean(source, flow_rgb, config):
    """Latent -> optional nearest upscale -> warp -> nearest downscale."""
    size = source.shape[-2:]
    factor = config.upscale_factor if config.warp_mode == "dgaf" else 1
    working_size = (size[0] * factor, size[1] * factor)
    expanded = F.interpolate(source.float(), size=working_size, mode="nearest")
    warped, valid = sample_with_flow(expanded, resize_flow(flow_rgb, working_size))
    return (F.interpolate(warped, size=size, mode="nearest"),
            F.interpolate(valid, size=size, mode="nearest"))


@torch.no_grad()
def consistency_mask(flow, reverse, config):
    reverse_aligned, valid = sample_with_flow(reverse, flow)
    error = (flow.float() + reverse_aligned).square().sum(dim=1, keepdim=True)
    magnitude = flow.float().square().sum(dim=1, keepdim=True) + reverse_aligned.square().sum(dim=1, keepdim=True)
    # Standard forward/backward consistency is an occlusion proxy, not a
    # ground-truth visibility classifier. beta is in squared RGB pixels.
    return valid * (error <= config.fb_alpha * magnitude + config.fb_beta).float()


class ClipWarper:
    """Explicit clip-local geometry. No hidden cross-clip/model global state."""
    def __init__(self, forward_rgb, backward_rgb, bg_mask, config):
        self.config = config
        self.bg = bg_mask.detach().float()
        if self.bg.ndim != 5 or self.bg.shape[2] != 1:
            raise ValueError("BG mask must be [B,T,1,h,w]")
        b, t = self.bg.shape[:2]
        if forward_rgb.shape != backward_rgb.shape or forward_rgb.shape[:3] != (b, t - 1, 2):
            raise ValueError("Flow must contain B clips of T-1 adjacent pairs")
        if not torch.isfinite(forward_rgb).all() or not torch.isfinite(backward_rgb).all():
            raise ValueError("Nonfinite RAFT flow")
        self.forward = forward_rgb.detach().float().flatten(0, 1)
        self.backward = backward_rgb.detach().float().flatten(0, 1)
        self.previous_conf = self.next_conf = None
        if config.use_confidence_mask and t > 1:
            self.previous_conf = consistency_mask(self.backward, self.forward, config)
            self.next_conf = consistency_mask(self.forward, self.backward, config)

    @torch.no_grad()
    def guidance(self, clean, step_index):
        """Return [B,T,5,h,w]: four gated latent channels + validity channel.

        clean is the prediction cached from the preceding diffusion step.
        Every frame reads the same snapshot, so frames can run in one batch.
        Source ROI is allowed; only the destination is gated to degraded BG.
        """
        b, t, _, h, w = self.bg.shape
        output = self.bg.new_zeros(b, t, 4, h, w)
        support = self.bg.new_zeros(b, t, 1, h, w)
        if clean is None or t == 1:
            return torch.cat((output, support), dim=2)
        if clean.shape != output.shape or not torch.isfinite(clean).all():
            raise ValueError("Predicted clean latent must be finite [B,T,4,h,w]")
        direction = self.config.direction
        if direction == "alternating":
            direction = "previous" if step_index % 2 else "next"
        for use_previous in (True, False):
            if direction == "previous" and not use_previous or direction == "next" and use_previous:
                continue
            source = clean[:, :-1] if use_previous else clean[:, 1:]
            flow = self.backward if use_previous else self.forward
            conf = self.previous_conf if use_previous else self.next_conf
            warped, valid = warp_clean(source.flatten(0, 1), flow, self.config)
            if conf is not None:
                valid = valid * F.interpolate(conf, size=(h, w), mode="nearest")
            target = slice(1, None) if use_previous else slice(None, -1)
            weight = valid.reshape(b, t - 1, 1, h, w) * self.bg[:, target]
            output[:, target] += warped.reshape(b, t - 1, 4, h, w) * weight
            support[:, target] += weight
        output = output / support.clamp_min(1)
        return torch.cat((output, support.clamp_max(1)), dim=2)


def prediction_to_clean(sample, prediction, alpha, prediction_type):
    """Use the actual scheduler prediction convention, in FP32."""
    sample, prediction = sample.float(), prediction.float()
    alpha = torch.as_tensor(alpha, device=sample.device, dtype=torch.float32)
    if prediction_type == "epsilon":
        return (sample - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
    if prediction_type == "v_prediction":
        return alpha.sqrt() * sample - (1 - alpha).sqrt() * prediction
    if prediction_type == "sample":
        return prediction
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


def rollout_target(sample, clean_gt, alpha, prediction_type):
    """Residual target after a detached model-generated DDIM transition.

    Such a state is not the originally sampled q(z_t|z0). Reusing the initial
    noise as its target would be wrong; derive its effective residual instead.
    """
    alpha = torch.as_tensor(alpha, device=sample.device, dtype=torch.float32)
    effective_noise = (sample.float() - alpha.sqrt() * clean_gt.float()) / (1 - alpha).sqrt().clamp_min(1e-8)
    if prediction_type == "epsilon":
        return effective_noise
    if prediction_type == "v_prediction":
        return alpha.sqrt() * effective_noise - (1 - alpha).sqrt() * clean_gt.float()
    if prediction_type == "sample":
        return clean_gt.float()
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


class SequenceWarper(ClipWarper):
    """Same-step sampling from the immediately preceding frame in the sweep."""
    @torch.no_grad()
    def frame_guidance(self, source_clean, target, source):
        b, t, _, h, w = self.bg.shape
        if source_clean is None:
            return self.bg.new_zeros(b, 5, h, w)
        if not 0 <= target < t or not 0 <= source < t or abs(target-source) != 1:
            raise ValueError("Guidance requires adjacent frames in the same sequence")
        previous = source < target
        pair = min(source, target)
        flow = (self.backward if previous else self.forward).reshape(b, t-1, 2, *self.forward.shape[-2:])[:, pair]
        confidence = self.previous_conf if previous else self.next_conf
        warped, valid = warp_clean(source_clean, flow, self.config)
        if confidence is not None:
            confidence = confidence.reshape(b, t-1, 1, *confidence.shape[-2:])[:, pair]
            valid *= F.interpolate(confidence, size=(h, w), mode="nearest")
        weight = valid * self.bg[:, target]
        return torch.cat((warped * weight, weight), dim=1)
