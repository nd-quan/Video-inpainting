"""Experimental CGE -> equivalent clean latent -> temporal DDIM transition.

For eta=0 and fixed predicted noise, z_s^C = a*x0 + b + delta_C is
represented by x0_C = x0 + delta_C/a. Temporal guidance is evaluated at
x0_C, and only a*(guided_x0_C - x0_C) is added to the CGE-guided z_s.
The original CGE and parallel combined schedulers remain unchanged.
"""
import torch

from .scheduling_ddim_CGE import CustomDDIMScheduler
from .scheduling_ddim_cge_temporal import CGETemporalDDIMScheduler


class CGEThenTemporalDDIMScheduler(CGETemporalDDIMScheduler):
    """Sequential guidance with a frozen DDIM noise direction; no extra UNet."""

    def set_timesteps(self, num_inference_steps, device=None):
        super().set_timesteps(num_inference_steps, device=device)
        self.sequential_step_reports = []

    def step(self, model_output, timestep, sample, eta=0.0,
             use_clipped_model_output=False, generator=None,
             variance_noise=None, return_dict=True):
        if eta != 0.0:
            raise ValueError("CGE-then-temporal experiment requires eta=0")
        if use_clipped_model_output:
            raise ValueError("CGE-then-temporal requires use_clipped_model_output=False")
        if getattr(self, "use_pgd_bg_residual", False):
            raise ValueError("This experiment supports direct CGE only, not PGD")
        if not getattr(self, "direct_cge_guidance", True):
            raise ValueError("This experiment requires direct_cge_guidance=True")

        # Capture this step's direct CGE correction without changing shared code.
        # The parent calls cond_fn only when its CGE window/cadence is active.
        original_cond_fn = getattr(self, "cond_fn", None)
        corrections = []

        def capture_cge(*args, **kwargs):
            delta = original_cond_fn(*args, **kwargs)
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError("Non-finite CGE correction")
            corrections.append(delta.detach().float())
            return delta

        self.cond_fn = capture_cge if original_cond_fn is not None else None
        try:
            # Bypass the parallel temporal step, retaining its CGE implementation.
            output = CustomDDIMScheduler.step(
                self, model_output=model_output, timestep=timestep, sample=sample,
                eta=eta, use_clipped_model_output=use_clipped_model_output,
                generator=generator, variance_noise=variance_noise, return_dict=True,
            )
        finally:
            self.cond_fn = original_cond_fn
        if len(corrections) > 1:
            raise RuntimeError("Expected at most one CGE correction per DDIM step")

        timestep_value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        previous_timestep = timestep_value - self.config.num_train_timesteps // self.num_inference_steps
        alpha_previous = (
            self.alphas_cumprod[previous_timestep]
            if previous_timestep >= 0 else self.final_alpha_cumprod
        ).to(device=sample.device, dtype=torch.float32)
        a = alpha_previous.sqrt()
        if not bool(torch.isfinite(a)) or not bool(a > 0):
            raise FloatingPointError("Invalid sqrt(alpha_previous)")
        x0 = output.pred_original_sample
        delta_cge = corrections[0] if corrections else torch.zeros_like(x0, dtype=torch.float32)
        x0_cge = x0.detach().float() + delta_cge / a
        if not bool(torch.isfinite(x0_cge).all()):
            raise FloatingPointError("Non-finite equivalent clean latent after CGE")

        temporal_requested = self._should_run_temporal_guidance(x0.shape[0])
        applied_before = self.temporal_guidance_applied_steps
        temporal_delta = torch.zeros_like(x0_cge)
        output.pred_original_sample = x0_cge.to(x0.dtype)
        if temporal_requested:
            # This method creates a fresh detached gradient leaf at x0_cge.
            guided_x0 = self._apply_latent_temporal_guidance(x0_cge)
            # The CGE correction is already in output.prev_sample: do not add it twice.
            temporal_delta = a * (guided_x0.float() - x0_cge)
            corrected = output.prev_sample.float() + temporal_delta
            if not bool(torch.isfinite(corrected).all()):
                raise FloatingPointError("Non-finite sequential CGE/TGE output")
            output.prev_sample = corrected.to(output.prev_sample.dtype)
            output.pred_original_sample = guided_x0.to(x0.dtype)

        def mean_norm(value):
            return float(torch.linalg.vector_norm(value.float().flatten(1), dim=1).mean().detach().cpu())

        report = {
            "step_index": self.temporal_step_count, "timestep": timestep_value,
            "previous_timestep": previous_timestep, "sqrt_alpha_previous": float(a.cpu()),
            "cge_applied": bool(corrections), "cge_delta_norm": mean_norm(delta_cge),
            "equivalent_clean_delta_norm": mean_norm(delta_cge / a),
            "temporal_requested": temporal_requested,
            "temporal_applied": self.temporal_guidance_applied_steps > applied_before,
            "temporal_input": "equivalent_clean_after_cge",
            "temporal_prev_delta_norm": mean_norm(temporal_delta),
            "temporal_loss": self.last_temporal_loss if temporal_requested else None,
            "temporal_skip_reason": self.last_temporal_skipped_reason if temporal_requested else None,
        }
        self.sequential_step_reports.append(report)
        if corrections or temporal_requested:
            print("[CGE->TGE] step={} cge={} cge_norm={:.6g} clean_shift={:.6g} tge={} loss={}".format(
                report["step_index"], report["cge_applied"], report["cge_delta_norm"],
                report["equivalent_clean_delta_norm"], report["temporal_applied"], report["temporal_loss"]),
                flush=True)
        self.temporal_step_count += 1
        return output if return_dict else (output.prev_sample,)
