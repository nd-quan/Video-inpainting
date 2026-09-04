"""DDIM scheduler that composes CGE spatial and latent temporal guidance.

The base :class:`CustomDDIMScheduler` retains the existing finite-difference
CGE update.  This subclass subsequently applies a latent predicted-x0 temporal
correction.  Consequently the DDIM transition is

``x_(t-1) = x_(t-1)^base + g_CGE + sqrt(alpha_bar_(t-1)) * Delta x0_temp``.

The temporal term is deliberately latent-only: CGE already needs a VAE decode
to evaluate its image-space codec residual, so a second RGB temporal decode
would needlessly duplicate the most expensive differentiable operation.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .scheduling_ddim_CGE import CustomDDIMScheduler, DDIMSchedulerOutput
from .scheduling_ddim_temporal import bg_temporal_loss


class CGETemporalDDIMScheduler(CustomDDIMScheduler):
    """CGE DDIM plus optional normalized pair-BG latent temporal guidance."""

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        super().set_timesteps(num_inference_steps, device=device)
        self.temporal_step_count = 0
        self.temporal_guidance_calls = 0
        self.temporal_guidance_applied_steps = 0
        self.temporal_guidance_skipped_steps = 0
        self.last_temporal_loss = None
        self.last_temporal_update_norm = None
        self.last_temporal_skipped_reason = None

    def set_temporal_guidance(
        self,
        *,
        flow_backward: torch.FloatTensor,
        stable_bg: torch.FloatTensor,
        guidance_scale: float = 1e-4,
        start_step: int = 25,
        end_step: Optional[int] = 35,
        every_n_steps: int = 1,
        loss_type: str = "l2",
        detach_previous: bool = True,
        normalize_grad: bool = True,
        enabled: bool = True,
    ) -> None:
        """Attach fixed pairwise conditions for latent temporal guidance."""
        if flow_backward.ndim != 4 or flow_backward.shape[1] != 2:
            raise ValueError("flow_backward must have shape [N-1,2,H,W]")
        if stable_bg.ndim != 4 or stable_bg.shape[1] != 1:
            raise ValueError("stable_bg must have shape [N-1,1,H,W]")
        if stable_bg.shape[0] != flow_backward.shape[0]:
            raise ValueError("flow_backward and stable_bg pair counts must match")
        if guidance_scale < 0.0:
            raise ValueError("temporal guidance_scale must be non-negative")
        if start_step < 0 or (end_step is not None and end_step <= start_step):
            raise ValueError("invalid temporal guidance window")
        if every_n_steps < 1:
            raise ValueError("temporal every_n_steps must be positive")
        if loss_type not in {"l1", "l2", "charbonnier"}:
            raise ValueError("unsupported temporal loss_type")

        self.temporal_flow_backward = flow_backward.detach()
        self.temporal_stable_bg = stable_bg.detach()
        self.temporal_guidance_scale = float(guidance_scale)
        self.temporal_start_step = int(start_step)
        self.temporal_end_step = None if end_step is None else int(end_step)
        self.temporal_every_n_steps = int(every_n_steps)
        self.temporal_loss_type = str(loss_type)
        self.temporal_detach_previous = bool(detach_previous)
        self.temporal_normalize_grad = bool(normalize_grad)
        self.temporal_guidance_enabled = bool(enabled)
        self.temporal_step_count = 0

    def clear_temporal_guidance(self) -> None:
        self.temporal_guidance_enabled = False
        self.temporal_flow_backward = None
        self.temporal_stable_bg = None
        self.temporal_step_count = 0

    def _should_run_temporal_guidance(self, frame_count: int) -> bool:
        if not bool(getattr(self, "temporal_guidance_enabled", False)) or frame_count < 2:
            return False
        step = int(getattr(self, "temporal_step_count", 0))
        start = int(getattr(self, "temporal_start_step", 0))
        end = getattr(self, "temporal_end_step", None)
        cadence = int(getattr(self, "temporal_every_n_steps", 1))
        return step >= start and (end is None or step < int(end)) and step % cadence == 0

    @staticmethod
    def _frame_pair_bg_mask(
        stable_bg: torch.Tensor,
        frame_count: int,
        size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pair = F.interpolate(
            stable_bg.to(device=device, dtype=dtype), size=size, mode="nearest"
        )
        frame = torch.zeros((frame_count, 1, *size), device=device, dtype=dtype)
        frame[:-1] = torch.maximum(frame[:-1], pair)
        frame[1:] = torch.maximum(frame[1:], pair)
        return frame

    def _apply_latent_temporal_guidance(
        self, pred_original_sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.temporal_guidance_calls += 1
        flow_backward = getattr(self, "temporal_flow_backward", None)
        stable_bg = getattr(self, "temporal_stable_bg", None)
        if flow_backward is None or stable_bg is None:
            raise RuntimeError("Temporal flow/stable-BG conditions are not configured")

        frame_count = int(pred_original_sample.shape[0])
        if flow_backward.shape[0] != frame_count - 1:
            raise ValueError("Temporal flow pair count does not match latent batch")
        with torch.enable_grad():
            z0 = pred_original_sample.detach().clone().requires_grad_(True)
            loss = bg_temporal_loss(
                predicted_images=z0,
                flow_backward=flow_backward,
                stable_bg=stable_bg,
                detach_previous=bool(getattr(self, "temporal_detach_previous", True)),
                loss_type=getattr(self, "temporal_loss_type", "l2"),
            )
            if not bool(torch.isfinite(loss)):
                self.last_temporal_skipped_reason = "non-finite latent temporal loss"
                self.temporal_guidance_skipped_steps += 1
                return pred_original_sample
            grad = torch.autograd.grad(loss, z0, retain_graph=False, create_graph=False)[0]

        frame_bg = self._frame_pair_bg_mask(
            stable_bg,
            frame_count,
            z0.shape[-2:],
            z0.device,
            z0.dtype,
        )
        grad = grad.float() * frame_bg.float()
        norms = torch.linalg.vector_norm(grad.reshape(frame_count, -1), ord=2, dim=1)
        active = norms > 1e-9
        if not bool(active.any()) or not bool(torch.isfinite(grad).all()):
            self.last_temporal_loss = float(loss.detach().cpu())
            self.last_temporal_update_norm = 0.0
            self.last_temporal_skipped_reason = "empty or non-finite latent temporal gradient"
            self.temporal_guidance_skipped_steps += 1
            return pred_original_sample
        if bool(getattr(self, "temporal_normalize_grad", True)):
            grad = grad / norms.clamp_min(1e-9).view(-1, 1, 1, 1)
            grad = grad * active.to(device=z0.device, dtype=grad.dtype).view(-1, 1, 1, 1)

        update = -float(getattr(self, "temporal_guidance_scale", 1e-4)) * grad
        guided = pred_original_sample.float() + update
        if not bool(torch.isfinite(guided).all()):
            self.last_temporal_skipped_reason = "non-finite latent temporal update"
            self.temporal_guidance_skipped_steps += 1
            return pred_original_sample
        self.last_temporal_loss = float(loss.detach().cpu())
        self.last_temporal_update_norm = float(
            torch.linalg.vector_norm(update.reshape(frame_count, -1), ord=2, dim=1)
            .mean()
            .detach()
            .cpu()
        )
        self.last_temporal_skipped_reason = None
        self.temporal_guidance_applied_steps += 1
        return guided.to(dtype=pred_original_sample.dtype)

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
        """Apply base DDIM/CGE then the latent predicted-x0 temporal term."""
        output = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            variance_noise=variance_noise,
            return_dict=True,
        )
        pred_original_sample = output.pred_original_sample
        prev_sample = output.prev_sample
        if self._should_run_temporal_guidance(pred_original_sample.shape[0]):
            if use_clipped_model_output:
                raise ValueError("Temporal guidance requires use_clipped_model_output=False")
            guided_x0 = self._apply_latent_temporal_guidance(pred_original_sample)
            timestep_value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
            previous_timestep = timestep_value - self.config.num_train_timesteps // self.num_inference_steps
            alpha_previous = (
                self.alphas_cumprod[previous_timestep]
                if previous_timestep >= 0
                else self.final_alpha_cumprod
            ).to(device=prev_sample.device, dtype=torch.float32)
            correction = alpha_previous.sqrt() * (
                guided_x0.float() - pred_original_sample.float()
            )
            corrected = prev_sample.float() + correction
            if bool(torch.isfinite(corrected).all()):
                output.prev_sample = corrected.to(dtype=prev_sample.dtype)
                output.pred_original_sample = guided_x0
            else:
                self.last_temporal_skipped_reason = "non-finite CGE+temporal DDIM correction"
                self.temporal_guidance_skipped_steps += 1
        self.temporal_step_count = int(getattr(self, "temporal_step_count", 0)) + 1
        if return_dict:
            return output
        return (output.prev_sample,)

