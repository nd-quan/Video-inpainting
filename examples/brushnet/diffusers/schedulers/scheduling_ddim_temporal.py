from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .scheduling_ddim import DDIMScheduler as BaseDDIMScheduler
from .scheduling_ddim import DDIMSchedulerOutput


def _decoder_sample(decoder_output):
    if hasattr(decoder_output, "sample"):
        return decoder_output.sample
    if isinstance(decoder_output, (tuple, list)) and decoder_output:
        return decoder_output[0]
    if torch.is_tensor(decoder_output):
        return decoder_output
    raise TypeError("The decoder must return a tensor, a tuple, or an object with a `sample` attribute.")


def decode_with_chunks(
    decoder: Callable,
    latents: torch.FloatTensor,
    scaling_factor: float,
    chunk_size: int = 1,
) -> torch.FloatTensor:
    """Decode a latent clip while preserving its autograd graph."""
    if scaling_factor <= 0:
        raise ValueError(f"`scaling_factor` must be positive, but got {scaling_factor}.")

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = latents.shape[0]

    decoded = []
    for start in range(0, latents.shape[0], chunk_size):
        latent_chunk = latents[start : start + chunk_size] / float(scaling_factor)
        decoded.append(_decoder_sample(decoder(latent_chunk)))
    return torch.cat(decoded, dim=0)


def resize_flow(flow: torch.FloatTensor, size: Tuple[int, int]) -> torch.FloatTensor:
    """Resize flow while preserving horizontal and vertical pixel displacement."""
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"`flow` must have shape [B, 2, H, W], but got {tuple(flow.shape)}.")

    target_height, target_width = size
    source_height, source_width = flow.shape[-2:]
    if (source_height, source_width) == (target_height, target_width):
        return flow

    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    resized[:, 0] *= float(target_width) / float(source_width)
    resized[:, 1] *= float(target_height) / float(source_height)
    return resized


def backward_warp(
    source: torch.FloatTensor,
    flow_current_to_source: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Warp `source` into current-frame coordinates.

    `flow_current_to_source[p]` maps current pixel p to its sampling position
    in `source`. Flow values are measured in pixels.
    """
    if source.ndim != 4:
        raise ValueError(f"`source` must have shape [B, C, H, W], but got {tuple(source.shape)}.")
    if flow_current_to_source.ndim != 4 or flow_current_to_source.shape[1] != 2:
        raise ValueError(
            "`flow_current_to_source` must have shape [B, 2, H, W], "
            f"but got {tuple(flow_current_to_source.shape)}."
        )
    if source.shape[0] != flow_current_to_source.shape[0]:
        raise ValueError(
            "Source and flow batch sizes must match, "
            f"but got {source.shape[0]} and {flow_current_to_source.shape[0]}."
        )

    height, width = source.shape[-2:]
    flow = resize_flow(flow_current_to_source, (height, width)).to(device=source.device, dtype=source.dtype)

    y, x = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    sample_x = x.unsqueeze(0) + flow[:, 0]
    sample_y = y.unsqueeze(0) + flow[:, 1]

    in_bounds = (
        (sample_x >= 0)
        & (sample_x <= width - 1)
        & (sample_y >= 0)
        & (sample_y <= height - 1)
    ).unsqueeze(1)

    grid_x = 2.0 * sample_x / float(width - 1) - 1.0 if width > 1 else torch.zeros_like(sample_x)
    grid_y = 2.0 * sample_y / float(height - 1) - 1.0 if height > 1 else torch.zeros_like(sample_y)
    grid = torch.stack((grid_x, grid_y), dim=-1)

    warped = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped, in_bounds.to(dtype=source.dtype)


def build_stable_bg_mask(
    bg_masks: torch.FloatTensor,
    flow_backward: torch.FloatTensor,
    visibility: Optional[torch.FloatTensor] = None,
    threshold: Optional[float] = 0.5,
) -> torch.FloatTensor:
    """
    Select pixels that are BG in two adjacent aligned frames.

    `bg_masks` uses BG=1 and ROI=0. `flow_backward[k]` maps frame k+1 to k.
    """
    if bg_masks.ndim != 4 or bg_masks.shape[1] != 1:
        raise ValueError(f"`bg_masks` must have shape [N, 1, H, W], but got {tuple(bg_masks.shape)}.")
    if flow_backward.ndim != 4 or flow_backward.shape[1] != 2:
        raise ValueError(
            f"`flow_backward` must have shape [N-1, 2, H, W], but got {tuple(flow_backward.shape)}."
        )
    if flow_backward.shape[0] != bg_masks.shape[0] - 1:
        raise ValueError(
            "`flow_backward` must contain N-1 flows, "
            f"but got {flow_backward.shape[0]} flows for {bg_masks.shape[0]} masks."
        )

    previous_bg = bg_masks[:-1].float()
    flow = flow_backward.to(device=previous_bg.device, dtype=previous_bg.dtype)
    warped_previous_bg, in_bounds = backward_warp(previous_bg, flow)
    if threshold is not None:
        warped_previous_bg = (warped_previous_bg > float(threshold)).to(previous_bg.dtype)

    stable_bg = bg_masks[1:].float() * warped_previous_bg * in_bounds
    if visibility is not None:
        if visibility.ndim != 4 or visibility.shape[1] != 1:
            raise ValueError(
                f"`visibility` must have shape [N-1, 1, H, W], but got {tuple(visibility.shape)}."
            )
        if visibility.shape[0] != stable_bg.shape[0]:
            raise ValueError(
                "Visibility and stable-BG pair counts must match, "
                f"but got {visibility.shape[0]} and {stable_bg.shape[0]}."
            )
        visibility = F.interpolate(
            visibility.float(),
            size=stable_bg.shape[-2:],
            mode="nearest",
        ).to(device=stable_bg.device)
        stable_bg = stable_bg * visibility
    return stable_bg


def bg_temporal_loss(
    predicted_images: torch.FloatTensor,
    flow_backward: torch.FloatTensor,
    stable_bg: torch.FloatTensor,
    charbonnier_eps: float = 1e-3,
    detach_previous: bool = False,
    loss_type: str = "l2",
) -> torch.FloatTensor:
    """Compute robust adjacent-frame temporal loss only on stable BG."""
    if predicted_images.ndim != 4:
        raise ValueError(
            f"`predicted_images` must have shape [N, C, H, W], but got {tuple(predicted_images.shape)}."
        )
    if predicted_images.shape[0] < 2:
        return predicted_images.sum() * 0.0

    pair_count = predicted_images.shape[0] - 1
    if flow_backward.shape[0] != pair_count:
        raise ValueError(
            f"Expected {pair_count} backward flows for {predicted_images.shape[0]} frames, "
            f"but got {flow_backward.shape[0]}."
        )
    if stable_bg.ndim != 4 or stable_bg.shape[1] != 1 or stable_bg.shape[0] != pair_count:
        raise ValueError(
            f"`stable_bg` must have shape [{pair_count}, 1, H, W], but got {tuple(stable_bg.shape)}."
        )

    images = predicted_images.float()
    previous_images = images[:-1].detach() if detach_previous else images[:-1]
    current_images = images[1:]
    flow = flow_backward.to(device=images.device, dtype=images.dtype)
    warped_previous, in_bounds = backward_warp(previous_images, flow)

    pair_mask = F.interpolate(
        stable_bg.float(),
        size=images.shape[-2:],
        mode="nearest",
    ).to(device=images.device)
    pair_mask = pair_mask * in_bounds

    difference = current_images - warped_previous

    if loss_type == "l1":
        robust_error = difference.abs()
    elif loss_type == "l2":
        robust_error = difference.square()
    elif loss_type == "charbonnier":
        robust_error = torch.sqrt(difference.square() + float(charbonnier_eps) ** 2)
    else:
        raise ValueError(f"Unknown temporal loss type: {loss_type}")


    # robust_error = torch.sqrt(difference.square() + float(charbonnier_eps) ** 2)
    numerator = (robust_error * pair_mask).sum()
    denominator = (pair_mask.sum() * images.shape[1]).clamp_min(1.0)
    return numerator / denominator


class TemporalDDIMScheduler(BaseDDIMScheduler):
    """DDIM scheduler with optional training-free stable-BG temporal guidance."""

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        super().set_timesteps(num_inference_steps, device=device)
        self.temporal_step_count = 0
        self.last_temporal_loss = None
        self.last_temporal_raw_grad_norm = None
        self.last_temporal_masked_grad_norm = None
        self.last_temporal_grad_norm = None
        self.last_temporal_update_norm = None
        self.last_temporal_active_frames = 0
        self.last_temporal_skipped_reason = None
        self.temporal_guidance_calls = 0
        self.temporal_guidance_applied_steps = 0
        self.temporal_guidance_skipped_steps = 0

    def set_temporal_guidance(
        self,
        *,
        decoder: Callable,
        flow_backward: torch.FloatTensor,
        stable_bg: torch.FloatTensor,
        bg_masks: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 1e-3,
        start_step: int = 15,
        end_step: Optional[int] = 40,
        every_n_steps: int = 1,
        decode_chunk_size: int = 1,
        vae_scaling_factor: float = 0.18215,
        charbonnier_eps: float = 1e-3,
        loss_scale: float = 1024.0,
        normalize_grad: bool = True,
        detach_previous: bool = False,
        enabled: bool = True,
        loss_type: str = "l2",
    ):
        """Attach fixed clip-level temporal conditions to this scheduler."""
        if not callable(decoder):
            raise TypeError("`decoder` must be callable.")
        if flow_backward.ndim != 4 or flow_backward.shape[1] != 2:
            raise ValueError(
                f"`flow_backward` must have shape [N-1, 2, H, W], but got {tuple(flow_backward.shape)}."
            )
        if stable_bg.ndim != 4 or stable_bg.shape[1] != 1:
            raise ValueError(f"`stable_bg` must have shape [N-1, 1, H, W], but got {tuple(stable_bg.shape)}.")
        if stable_bg.shape[0] != flow_backward.shape[0]:
            raise ValueError(
                "Flow and stable-BG pair counts must match, "
                f"but got {flow_backward.shape[0]} and {stable_bg.shape[0]}."
            )
        if bg_masks is not None:
            if bg_masks.ndim != 4 or bg_masks.shape[1] != 1:
                raise ValueError(f"`bg_masks` must have shape [N, 1, H, W], but got {tuple(bg_masks.shape)}.")
            if bg_masks.shape[0] != flow_backward.shape[0] + 1:
                raise ValueError(
                    "`bg_masks` must contain one mask per frame, "
                    f"but got {bg_masks.shape[0]} masks for {flow_backward.shape[0] + 1} frames."
                )
        if guidance_scale < 0:
            raise ValueError(f"`guidance_scale` must be non-negative, but got {guidance_scale}.")
        if start_step < 0:
            raise ValueError(f"`start_step` must be non-negative, but got {start_step}.")
        if end_step is not None and end_step <= start_step:
            raise ValueError(f"`end_step` must be greater than `start_step`, but got [{start_step}, {end_step}).")
        if every_n_steps <= 0:
            raise ValueError(f"`every_n_steps` must be positive, but got {every_n_steps}.")
        if decode_chunk_size <= 0:
            raise ValueError(f"`decode_chunk_size` must be positive, but got {decode_chunk_size}.")
        if vae_scaling_factor <= 0:
            raise ValueError(f"`vae_scaling_factor` must be positive, but got {vae_scaling_factor}.")
        if loss_scale <= 0:
            raise ValueError(f"`loss_scale` must be positive, but got {loss_scale}.")

        self.temporal_decoder = decoder
        self.temporal_flow_backward = flow_backward.detach()
        self.temporal_stable_bg = stable_bg.detach()
        self.temporal_bg_masks = None if bg_masks is None else bg_masks.detach()
        self.temporal_guidance_scale = float(guidance_scale)
        self.temporal_start_step = int(start_step)
        self.temporal_end_step = None if end_step is None else int(end_step)
        self.temporal_every_n_steps = int(every_n_steps)
        self.temporal_decode_chunk_size = int(decode_chunk_size)
        self.temporal_vae_scaling_factor = float(vae_scaling_factor)
        self.temporal_charbonnier_eps = float(charbonnier_eps)
        self.temporal_loss_scale = float(loss_scale)
        self.temporal_normalize_grad = bool(normalize_grad)
        self.temporal_detach_previous = bool(detach_previous)
        self.temporal_guidance_enabled = bool(enabled)
        self.temporal_step_count = 0
        self.temporal_loss_type = loss_type

    def _skip_temporal_guidance(
        self,
        reason: str,
        pred_original_sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        self.last_temporal_skipped_reason = reason
        self.temporal_guidance_skipped_steps = int(
            getattr(self, "temporal_guidance_skipped_steps", 0)
        ) + 1
        return pred_original_sample

    def clear_temporal_guidance(self):
        """Disable guidance and release clip-specific tensors."""
        self.temporal_guidance_enabled = False
        self.temporal_decoder = None
        self.temporal_flow_backward = None
        self.temporal_stable_bg = None
        self.temporal_bg_masks = None
        self.temporal_step_count = 0

    def _should_run_temporal_guidance(self, batch_size: int) -> bool:
        if not bool(getattr(self, "temporal_guidance_enabled", False)) or batch_size < 2:
            return False

        step_count = int(getattr(self, "temporal_step_count", 0))
        start_step = int(getattr(self, "temporal_start_step", 0))
        end_step = getattr(self, "temporal_end_step", None)
        every_n_steps = int(getattr(self, "temporal_every_n_steps", 1))
        in_window = step_count >= start_step and (end_step is None or step_count < int(end_step))
        return in_window and step_count % every_n_steps == 0

    def _frame_bg_mask(
        self,
        stable_bg: torch.FloatTensor,
        frame_count: int,
        size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.FloatTensor:
        bg_masks = getattr(self, "temporal_bg_masks", None)
        if bg_masks is not None:
            if bg_masks.shape[0] != frame_count:
                raise ValueError(
                    f"Temporal BG mask batch {bg_masks.shape[0]} does not match latent batch {frame_count}."
                )
            return F.interpolate(
                bg_masks.to(device=device, dtype=dtype),
                size=size,
                mode="nearest",
            )

        pair_masks = F.interpolate(
            stable_bg.to(device=device, dtype=dtype),
            size=size,
            mode="nearest",
        )
        frame_masks = torch.zeros(
            (frame_count, 1, size[0], size[1]),
            device=device,
            dtype=dtype,
        )
        frame_masks[:-1] = torch.maximum(frame_masks[:-1], pair_masks)
        frame_masks[1:] = torch.maximum(frame_masks[1:], pair_masks)
        return frame_masks

    def _apply_temporal_guidance(
        self,
        pred_original_sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        self.temporal_guidance_calls = int(getattr(self, "temporal_guidance_calls", 0)) + 1
        decoder = getattr(self, "temporal_decoder", None)
        flow_backward = getattr(self, "temporal_flow_backward", None)
        stable_bg = getattr(self, "temporal_stable_bg", None)
        if decoder is None or flow_backward is None or stable_bg is None:
            raise RuntimeError(
                "Temporal guidance is enabled but its decoder/flow/stable-BG data is missing. "
                "Call `set_temporal_guidance(...)` before sampling."
            )

        frame_count = pred_original_sample.shape[0]
        expected_pairs = frame_count - 1
        if flow_backward.shape[0] != expected_pairs or stable_bg.shape[0] != expected_pairs:
            raise ValueError(
                "Temporal tensors do not match the current clip: "
                f"frames={frame_count}, flows={flow_backward.shape[0]}, stable_masks={stable_bg.shape[0]}."
            )

        with torch.enable_grad():
            z0 = pred_original_sample.detach().clone().requires_grad_(True)
            predicted_images = decode_with_chunks(
                decoder=decoder,
                latents=z0,
                scaling_factor=float(getattr(self, "temporal_vae_scaling_factor", 0.18215)),
                chunk_size=int(getattr(self, "temporal_decode_chunk_size", 1)),
            )
            if not bool(torch.isfinite(predicted_images).all()):
                return self._skip_temporal_guidance(
                    "non-finite VAE prediction",
                    pred_original_sample,
                )

            temporal_loss = bg_temporal_loss(
                predicted_images=predicted_images,
                flow_backward=flow_backward,
                stable_bg=stable_bg,
                charbonnier_eps=float(getattr(self, "temporal_charbonnier_eps", 1e-3)),
                detach_previous=bool(getattr(self, "temporal_detach_previous", False)),
                loss_type=getattr(self, "temporal_loss_type", "l2")
            )
            if not bool(torch.isfinite(temporal_loss)):
                return self._skip_temporal_guidance(
                    "non-finite temporal loss",
                    pred_original_sample,
                )

            # z0 and the VAE run in FP16 during inference. Scaling the loss keeps
            # small gradients representable until they are converted to FP32.
            loss_scale = float(getattr(self, "temporal_loss_scale", 1024.0))
            scaled_temporal_grad = torch.autograd.grad(
                temporal_loss * loss_scale,
                z0,
                retain_graph=False,
                create_graph=False,
            )[0]
            temporal_grad = scaled_temporal_grad.float() / loss_scale

        frame_bg_mask = self._frame_bg_mask(
            stable_bg=stable_bg,
            frame_count=frame_count,
            size=z0.shape[-2:],
            device=z0.device,
            dtype=z0.dtype,
        )
        raw_grad_norm = torch.linalg.vector_norm(
            temporal_grad.reshape(frame_count, -1),
            ord=2,
            dim=1,
        )
        self.last_temporal_raw_grad_norm = float(raw_grad_norm.mean().detach().cpu())

        temporal_grad = temporal_grad * frame_bg_mask.float()
        if not bool(torch.isfinite(temporal_grad).all()):
            return self._skip_temporal_guidance(
                "non-finite temporal gradient",
                pred_original_sample,
            )

        grad_norm = torch.linalg.vector_norm(
            temporal_grad.reshape(frame_count, -1),
            ord=2,
            dim=1,
        )
        self.last_temporal_masked_grad_norm = float(grad_norm.mean().detach().cpu())
        active_frames = grad_norm > 1e-9
        if not bool(active_frames.any()):
            self.last_temporal_loss = float(temporal_loss.detach().cpu())
            self.last_temporal_grad_norm = 0.0
            self.last_temporal_update_norm = 0.0
            self.last_temporal_active_frames = 0
            if bool((raw_grad_norm > 1e-9).any()):
                reason = "BG latent mask removed all temporal gradients"
            else:
                reason = "zero raw temporal gradient after loss scaling"
            return self._skip_temporal_guidance(reason, pred_original_sample)

        if bool(getattr(self, "temporal_normalize_grad", True)):
            # Keep the normalization in FP32. Casting a small norm to FP16 can
            # underflow it to zero and turn an otherwise valid gradient into NaN.
            safe_norm = grad_norm.clamp_min(1e-9)
            temporal_grad = temporal_grad / safe_norm.view(-1, 1, 1, 1)
            temporal_grad = temporal_grad * active_frames.to(
                device=z0.device,
                dtype=temporal_grad.dtype,
            ).view(-1, 1, 1, 1)

        update = -float(getattr(self, "temporal_guidance_scale", 1e-3)) * temporal_grad
        if not bool(torch.isfinite(update).all()):
            return self._skip_temporal_guidance(
                "non-finite temporal update",
                pred_original_sample,
            )

        guided_sample_fp32 = pred_original_sample.float() + update
        if not bool(torch.isfinite(guided_sample_fp32).all()):
            return self._skip_temporal_guidance(
                "non-finite guided predicted-x0",
                pred_original_sample,
            )
        guided_sample = guided_sample_fp32.to(dtype=pred_original_sample.dtype)

        update_norm = torch.linalg.vector_norm(
            update.reshape(frame_count, -1),
            ord=2,
            dim=1,
        )
        self.last_temporal_loss = float(temporal_loss.detach().cpu())
        self.last_temporal_grad_norm = float(grad_norm.mean().detach().cpu())
        self.last_temporal_update_norm = float(update_norm.mean().detach().cpu())
        self.last_temporal_active_frames = int(active_frames.sum().detach().cpu())
        self.last_temporal_skipped_reason = None
        self.temporal_guidance_applied_steps = int(
            getattr(self, "temporal_guidance_applied_steps", 0)
        ) + 1
        return guided_sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """Run one DDIM step and optionally replace its predicted-x0 term."""
        base_output = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            variance_noise=variance_noise,
            return_dict=True,
        )

        pred_original_sample = base_output.pred_original_sample
        prev_sample = base_output.prev_sample
        should_run = self._should_run_temporal_guidance(pred_original_sample.shape[0])

        if should_run:
            if use_clipped_model_output:
                raise ValueError(
                    "Temporal predicted-x0 guidance does not support `use_clipped_model_output=True`; "
                    "keep its default value `False`."
                )

            guided_original_sample = self._apply_temporal_guidance(pred_original_sample)
            timestep_value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
            prev_timestep = timestep_value - self.config.num_train_timesteps // self.num_inference_steps
            if prev_timestep >= 0:
                alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
            else:
                alpha_prod_t_prev = self.final_alpha_cumprod
            alpha_prod_t_prev = alpha_prod_t_prev.to(device=prev_sample.device, dtype=torch.float32)

            # This correction is equivalent to replacing pred_original_sample
            # before the DDIM equation while keeping its noise direction fixed.
            correction = alpha_prod_t_prev.sqrt() * (
                guided_original_sample.float() - pred_original_sample.float()
            )
            corrected_prev_sample = prev_sample.float() + correction
            if bool(torch.isfinite(corrected_prev_sample).all()):
                prev_sample = corrected_prev_sample.to(dtype=prev_sample.dtype)
            else:
                self.last_temporal_skipped_reason = "non-finite corrected DDIM sample"
            pred_original_sample = guided_original_sample

        self.temporal_step_count = int(getattr(self, "temporal_step_count", 0)) + 1

        if not return_dict:
            return (prev_sample,)
        return DDIMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )


# Convenient aliases for experiment scripts.
CustomDDIMScheduler = TemporalDDIMScheduler
DDIMTemporalScheduler = TemporalDDIMScheduler
