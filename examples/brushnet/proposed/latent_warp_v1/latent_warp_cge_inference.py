"""CGE -> temporal guidance for the 14-input latent-warp BrushNet.

Same two-pass latent-warp denoising as ``latent_warp_inference.denoise_clip``,
but each DDIM step is run by ``LatentWarpCGEThenTemporalScheduler``:

1. DDIM step with the (frozen) predicted noise.
2. CGE (codec-guided) correction on the DDIM sample, only at ``cge_steps``.
3. Latent temporal guidance evaluated at the CGE-equivalent clean latent
   ``x0 + delta_cge / sqrt(alpha_prev)`` and added on top (no double counting).

This is the production ``cge_then_tg`` order (inference/restoration_engine.py)
with one deliberate difference: the temporal gradient is masked with each
frame's own BG mask, exactly as the no-CGE latent-warp evaluator does through
``TemporalDDIMScheduler(bg_masks=...)``.  With CGE disabled the guidance is
therefore the same as ``denoise_clip``.  Existing files are not modified.
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from latent_warp_utils import build_latent_warp_condition, predict_x0
from diffusers.schedulers.scheduling_ddim_temporal import build_stable_bg_mask
from diffusers.schedulers.scheduling_ddim_cge_then_temporal import (
    CGEThenTemporalDDIMScheduler,
)

CGE_CODEC_MODES = ("bg_only", "dual_region")


class LatentWarpCGEThenTemporalScheduler(CGEThenTemporalDDIMScheduler):
    """Sequential CGE -> latent TG with per-frame BG masks for the TG gradient."""

    def set_temporal_guidance(self, *, bg_masks=None, **kwargs):
        super().set_temporal_guidance(**kwargs)
        if bg_masks is not None:
            if bg_masks.ndim != 4 or bg_masks.shape[1] != 1:
                raise ValueError("bg_masks must have shape [N,1,H,W]")
            if bg_masks.shape[0] != kwargs["flow_backward"].shape[0] + 1:
                raise ValueError("bg_masks needs one mask per frame")
            bg_masks = bg_masks.detach()
        self.temporal_bg_masks = bg_masks

    def clear_temporal_guidance(self):
        super().clear_temporal_guidance()
        self.temporal_bg_masks = None

    def _frame_pair_bg_mask(self, stable_bg, frame_count, size, device, dtype):
        bg_masks = getattr(self, "temporal_bg_masks", None)
        if bg_masks is None:
            return CGEThenTemporalDDIMScheduler._frame_pair_bg_mask(
                stable_bg, frame_count, size, device, dtype
            )
        if bg_masks.shape[0] != frame_count:
            raise ValueError("Temporal BG mask batch does not match latent batch")
        return F.interpolate(
            bg_masks.to(device=device, dtype=dtype), size=size, mode="nearest"
        )


def cge_window(cge_steps):
    """Sorted, equally spaced loop indices -> (start, end_exclusive, cadence)."""
    if (
        not cge_steps
        or any(type(i) is not int or i < 0 for i in cge_steps)
        or list(cge_steps) != sorted(set(cge_steps))
    ):
        raise ValueError("cge_steps must be sorted unique nonnegative integers")
    cadence = cge_steps[1] - cge_steps[0] if len(cge_steps) > 1 else 1
    if list(range(cge_steps[0], cge_steps[-1] + 1, cadence)) != list(cge_steps):
        raise ValueError("cge_steps must be equally spaced")
    return cge_steps[0], cge_steps[-1] + 1, cadence


def build_codec(config):
    """VCM-RS operator in a subprocess that uses ``codec_python``.

    ``bg_only`` (production fast CGE): QP``cge_bg_quality`` background branch
    plus a direct ROI fidelity term.  ``dual_region``: both ROI and BG branches,
    hard-merged, compared over the full image.
    """
    from vcmrs_codec_adapter import VCMRSBackgroundOnlyCodec, VCMRSDualRegionCodec

    regional = VCMRSDualRegionCodec(
        vcmrs_root=Path(config["codec_root"]),
        python_executable=Path(config["codec_python"]),
        roi_quality=config["cge_roi_quality"],
        bg_quality=config["cge_bg_quality"],
        frame_rate=config["cge_frame_rate"],
        configuration=config["cge_configuration"],
        intra_period=config["cge_intra_period"],
        nn_intra_qp_offset=config["cge_nn_intra_qp_offset"],
        profile=config["cge_profile"],
        max_parallel=config["cge_max_parallel"],
        timeout_seconds=config["cge_timeout_seconds"],
        # "cpu": hide GPUs from the codec subprocess (its detectron2 RoI model
        # otherwise competes with BrushNet for the same GPU memory).
        cuda_visible_devices="" if config.get("cge_codec_device", "cuda") == "cpu" else None,
    )
    if config["cge_codec_mode"] == "bg_only":
        return VCMRSBackgroundOnlyCodec(regional)
    return regional


def make_scheduler(pretrained_model, config, codec, vae):
    """Scheduler with CGE attached (or temporal-only when CGE is disabled)."""
    from diffusers.schedulers.scheduling_ddim_CGE import cond_fn

    scheduler = LatentWarpCGEThenTemporalScheduler.from_pretrained(
        pretrained_model, subfolder="scheduler"
    )
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("This checkpoint was trained for epsilon prediction")
    scheduler.register_to_config(clip_sample=False, thresholding=False)
    scheduler.direct_cge_guidance = True
    scheduler.use_pgd_bg_residual = False
    scheduler.per_frame_cge = False
    scheduler.decode_chunk_size = 1
    scheduler.cge_max_evals = -1
    scheduler.vae_scaling_factor = float(vae.config.scaling_factor)
    scheduler.cond_fn = None
    if config["cge_enabled"] and config["cge_guidance_scale"] > 0:
        start, end, cadence = cge_window(config["cge_steps"])
        scheduler.cond_fn = cond_fn
        scheduler.cge_codec = codec
        scheduler.guidance_scale_cge = float(config["cge_guidance_scale"])
        scheduler.cge_scale_schedule = config["cge_scale_schedule"]
        scheduler.cge_start_step, scheduler.cge_end_step = start, end
        scheduler.cge_every_n_steps = cadence
    return scheduler


def expected_cge_steps(config):
    if not config["cge_enabled"] or config["cge_guidance_scale"] <= 0:
        return []
    return list(config["cge_steps"])


@torch.no_grad()
def denoise_clip_cge(
    predict_noise,
    scheduler,
    latents,
    base_condition,
    bg_masks,
    flow_backward,
    valid_backward,
    observed,
    decoder,
    config,
):
    """``denoise_clip`` with CGE -> TG.

    ``observed``: decoded degraded input frames in [-1,1], [N,3,H,W] (the CGE
    observation). ``bg_masks``: BG=1, [N,1,H,W] at the same resolution.
    CGE and TG compute their own gradients with ``torch.enable_grad``.
    """
    if base_condition.shape[1] != 5:
        raise ValueError("Expected decoded-input latent (4) + BG mask (1)")
    frames = latents.shape[0]
    if frames < 2:
        raise ValueError("Oracle inference requires at least two consecutive frames")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("CGE/TG need autograd; do not run under inference_mode")
    scheduler.set_timesteps(config["num_inference_steps"], device=latents.device)
    use_cge = scheduler.cond_fn is not None
    if use_cge:
        scheduler.x_lr = observed.to(latents.device, torch.float32)
        scheduler.mask = (1 - bg_masks).to(latents.device, torch.float32)  # ROI=1
        scheduler.decoder = decoder
        for i in range(frames):
            scheduler.cge_codec.prepare_region_mask(scheduler.mask[i], scheduler.x_lr[i])
    temporal_enabled = (
        config.get("temporal_guidance_enabled", True)
        and config["temporal_guidance_scale"] > 0
    )
    stable = build_stable_bg_mask(bg_masks, flow_backward, visibility=valid_backward)
    scheduler.set_temporal_guidance(
        flow_backward=flow_backward,
        stable_bg=stable,
        bg_masks=bg_masks,
        guidance_scale=config["temporal_guidance_scale"],
        start_step=config["temporal_start"],
        end_step=config["temporal_end"],
        every_n_steps=config["temporal_every_n_steps"],
        loss_type="l2",
        normalize_grad=True,
        detach_previous=False,
        enabled=temporal_enabled,
    )
    zero_warp = torch.zeros(
        (frames, 5, *latents.shape[-2:]),
        device=latents.device,
        dtype=base_condition.dtype,
    )
    reports = []
    try:
        for index, timestep in enumerate(scheduler.timesteps):
            model_input = scheduler.scale_model_input(latents, timestep)
            if config["latent_warp_enabled"]:
                # Same two-pass procedure as training and denoise_clip.
                base_noise = predict_noise(
                    model_input, timestep, torch.cat([base_condition, zero_warp], 1)
                )
                x0 = predict_x0(latents, base_noise, timestep.expand(frames), scheduler)
                warped, fraction, stats = build_latent_warp_condition(
                    x0,
                    bg_masks,
                    flow_backward.unsqueeze(0),
                    valid_backward.unsqueeze(0),
                    1,
                    frames,
                    x0_clamp=config["latent_warp_x0_clamp"] or None,
                )
                extra = torch.cat([warped, fraction], 1).to(base_condition.dtype)
            else:
                extra, stats = zero_warp, {}
            noise = predict_noise(
                model_input, timestep, torch.cat([base_condition, extra], 1)
            )
            latents = scheduler.step(noise, timestep, latents, eta=0.0).prev_sample
            if not torch.isfinite(latents).all():
                raise FloatingPointError(
                    "Non-finite denoised latents at step {}".format(index)
                )
            step = dict(scheduler.sequential_step_reports[-1])
            step.update(
                step_index=index,
                warp=stats,
                temporal_update_norm=scheduler.last_temporal_update_norm
                if step["temporal_applied"]
                else None,
                cge_scale_effective=getattr(
                    scheduler, "last_cge_guidance_scale_effective", None
                )
                if step["cge_applied"]
                else None,
            )
            reports.append(step)
    finally:
        scheduler.clear_temporal_guidance()
        scheduler.x_lr = scheduler.mask = scheduler.decoder = None
    applied_cge = [r["step_index"] for r in reports if r["cge_applied"]]
    if applied_cge != expected_cge_steps(config):
        raise RuntimeError(
            "CGE audit failed: applied {} expected {}".format(
                applied_cge, expected_cge_steps(config)
            )
        )
    return latents, dict(
        steps=reports,
        cge_applied_steps=applied_cge,
        cge_codec_evals=int(getattr(scheduler, "cge_codec_eval_count", 0)),
        temporal_calls=scheduler.temporal_guidance_calls,
        temporal_applied=scheduler.temporal_guidance_applied_steps,
        temporal_skipped=scheduler.temporal_guidance_skipped_steps,
        stable_bg_fraction=float(stable.mean()),
        temporal_guidance_enabled=temporal_enabled,
        latent_warp_enabled=config["latent_warp_enabled"],
        guidance_order="CGE -> equivalent clean latent -> latent TG",
    )


def isolate_cge_environment(codec_workers):
    """Explicit config owns codec behaviour; drop unrelated shell CGE flags."""
    for key in list(os.environ):
        if key.startswith(("CGE_", "TEMPORAL_", "TEACHER_FLOW_")):
            del os.environ[key]
    os.environ["CGE_CODEC_WORKERS"] = str(codec_workers)
