#!/usr/bin/env python3
"""Isolated four-frame probe/control; never edits or resumes the full evaluation."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "temporal_control"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    output = EXPERIMENT / "diagnostics" / args.mode
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_latent_tg0p1_v7student_512/run_config.json").read_text())
    for key in list(os.environ):
        if key.startswith(("CGE_", "TEMPORAL_", "TEACHER_FLOW_")):
            del os.environ[key]
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "LONG_TEST_ROOT": reference["dataset"],
        "CHECKPOINT_DIR": reference["checkpoint"], "V7_RAFT_STUDENT_PATH": reference["flow_checkpoint"],
        "OUTPUT_ROOT": str(output), "SEQUENCES": "Traffic", "MAX_IMAGES": "4",
        "DISCOVER_HIERARCHICAL_SEQUENCES": "1", "PREFLIGHT_ONLY": "0", "FORCE_REGENERATE": "0",
        "GUIDANCE_MODE": "combined" if args.mode == "probe" else "temporal",
        "TEMPORAL_GUIDANCE_SPACE": "latent", "TEMPORAL_FLOW_BACKEND": "v7_student",
        "TEMPORAL_BG_MASK_MODE": "pair_intersection", "TEMPORAL_GUIDANCE_SCALE": "0.1",
        "TEMPORAL_CLIP_SIZE": "4", "TEMPORAL_CLIP_STRIDE": "4",
        "TEMPORAL_START_STEP": "25", "TEMPORAL_END_STEP": "40", "TEMPORAL_EVERY_N_STEPS": "1",
        "TEMPORAL_DETACH_PREVIOUS": "1", "NUM_INFERENCE_STEPS": "50",
        "MAIN_NOISE_SEED": "1234", "SHARED_NOISE_SEED": "6789",
        "SHARED_BG_NOISE_STRENGTH": "1", "VARIANCE_PRESERVING_SHARED_NOISE": "1",
        "CGE_GUIDANCE_SCALE": "0.0001", "CGE_START_STEP": "25", "CGE_END_STEP": "40",
        "CGE_CODEC_MODE": "bg_only", "CGE_SCALE_SCHEDULE": "fixed",
        "CGE_VCMRS_BG_QUALITY": "52", "CGE_VCMRS_PROFILE": "train_match",
        "CGE_VCMRS_CONFIGURATION": "RandomAccess", "CGE_VCMRS_INTRA_PERIOD": "64",
        "CGE_VCMRS_NN_INTRA_QP_OFFSET": "-5", "CGE_CODEC_WORKERS": "1",
        "CGE_VCMRS_MAX_PARALLEL": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    })
    sys.path[:0] = [str(ROOT / "examples/brushnet"), str(ROOT / "examples/brushnet/proposed")]
    if args.mode == "temporal_control":
        runpy.run_path(str(ROOT / reference["evaluator"]), run_name="__main__")
        return

    import torch
    from diffusers.schedulers.scheduling_ddim_CGE import CustomDDIMScheduler
    import diffusers.schedulers.scheduling_ddim_cge_then_temporal as sequential
    from diffusers.schedulers.scheduling_ddim_temporal import bg_temporal_loss

    OriginalScheduler = sequential.CGEThenTemporalDDIMScheduler

    def frame_norm(x):
        return torch.linalg.vector_norm(x.float().flatten(1), dim=1)

    def values(x):
        return x.detach().float().cpu().tolist()

    def masked_temporal_gradient(scheduler, x):
        with torch.enable_grad():
            leaf = x.detach().float().requires_grad_(True)
            loss = bg_temporal_loss(leaf, scheduler.temporal_flow_backward,
                                    scheduler.temporal_stable_bg,
                                    detach_previous=scheduler.temporal_detach_previous,
                                    loss_type=scheduler.temporal_loss_type)
            raw = torch.autograd.grad(loss, leaf)[0]
        mask = scheduler._frame_pair_bg_mask(scheduler.temporal_stable_bg, len(x),
                                             x.shape[-2:], x.device, torch.float32)
        masked = raw * mask
        norm = frame_norm(masked)
        normalized = masked / norm.clamp_min(1e-9).view(-1, 1, 1, 1)
        normalized *= (norm > 1e-9).view(-1, 1, 1, 1)
        return float(loss.detach().cpu()), masked, normalized

    class AuditScheduler(OriginalScheduler):
        def _apply_latent_temporal_guidance(self, x0_cge):
            if self.temporal_step_count == 25:
                old_loss, old_grad, old_direction = masked_temporal_gradient(self, self.audit_x0)
                new_loss, new_grad, new_direction = masked_temporal_gradient(self, x0_cge)
                self.audit.update({
                    "temporal_loss_before_cge": old_loss, "temporal_loss_after_cge": new_loss,
                    "temporal_gradient_norm_before": values(frame_norm(old_grad)),
                    "temporal_gradient_norm_after": values(frame_norm(new_grad)),
                    "temporal_direction_cosine": values((old_direction * new_direction).flatten(1).sum(1)),
                    "temporal_direction_difference_norm": values(frame_norm(new_direction - old_direction)),
                })
                self.audit_old_direction = old_direction.detach()
                self.audit_new_direction = new_direction.detach()
                self.audit_x0_cge = x0_cge.detach()
            return super()._apply_latent_temporal_guidance(x0_cge)

        def step(self, model_output, timestep, sample, **kwargs):
            if self.temporal_step_count != 25:
                return super().step(model_output, timestep, sample, **kwargs)
            callback = self.cond_fn
            self.cond_fn = None
            try:
                baseline = CustomDDIMScheduler.step(self, model_output, timestep, sample, return_dict=True)
            finally:
                self.cond_fn = callback
            self.audit = {"step_index": 25, "timestep": int(timestep), "latent_dtype": str(sample.dtype),
                          "shape": list(sample.shape)}

            def capture(x0, **kw):
                self.audit_x0 = x0.detach().clone()
                delta = callback(x0=x0, **kw)
                self.audit_cge_delta = delta.detach().clone()
                return delta

            self.cond_fn = capture
            try:
                result = super().step(model_output, timestep, sample, **dict(kwargs, return_dict=True))
            finally:
                self.cond_fn = callback
            a = float(self.sequential_step_reports[-1]["sqrt_alpha_previous"])
            prev_base = baseline.prev_sample
            prev_cge = prev_base + self.audit_cge_delta
            delta_cge_realized = prev_cge.float() - prev_base.float()
            delta_temporal = -a * 0.1 * self.audit_new_direction
            prev_parallel = (prev_cge.float() - a * 0.1 * self.audit_old_direction).to(prev_cge.dtype)
            self.audit.update({
                "sqrt_alpha_previous": a,
                "x0_norm": values(frame_norm(self.audit_x0)),
                "cge_nominal_delta_norm": values(frame_norm(self.audit_cge_delta)),
                "cge_realized_delta_norm": values(frame_norm(delta_cge_realized)),
                "cge_fp16_unchanged_coordinate_fraction": values((prev_cge == prev_base).float().flatten(1).mean(1)),
                "clean_shift_relative_norm": values(frame_norm(self.audit_x0_cge - self.audit_x0.float()) / frame_norm(self.audit_x0)),
                "temporal_nominal_delta_norm": values(frame_norm(delta_temporal)),
                "temporal_realized_delta_norm": values(frame_norm(result.prev_sample.float() - prev_cge.float())),
                "temporal_fp16_unchanged_coordinate_fraction": values((result.prev_sample == prev_cge).float().flatten(1).mean(1)),
                "sequential_vs_parallel_float_delta_norm": values(frame_norm(-a * 0.1 * (self.audit_new_direction - self.audit_old_direction))),
                "sequential_vs_parallel_fp16_delta_norm": values(frame_norm(result.prev_sample.float() - prev_parallel.float())),
                "sequential_vs_parallel_fp16_changed_coordinate_fraction": values((result.prev_sample != prev_parallel).float().flatten(1).mean(1)),
                "cge_vs_temporal_correction_cosine": values(torch.nn.functional.cosine_similarity(
                    self.audit_cge_delta.float().flatten(1), delta_temporal.flatten(1), dim=1)),
            })
            state = {"x0_before_cge": self.audit_x0.cpu(), "x0_after_cge": self.audit_x0_cge.cpu(),
                     "cge_delta": self.audit_cge_delta.cpu(), "flow": self.temporal_flow_backward.cpu(),
                     "stable_bg": self.temporal_stable_bg.cpu(), "prev_base": prev_base.cpu(),
                     "temporal_direction_before": self.audit_old_direction.cpu(),
                     "temporal_direction_after": self.audit_new_direction.cpu()}
            torch.save(state, output / "step25_state.pt")
            (output / "step25_probe.json").write_text(json.dumps(self.audit, indent=2) + "\n")
            print("AUDIT COMPLETE " + json.dumps(self.audit, indent=2), flush=True)
            raise SystemExit(0)

    # Patch only this diagnostic process; the four evaluation workers are untouched.
    sequential.CGEThenTemporalDDIMScheduler = AuditScheduler
    runpy.run_path(str(ROOT / "examples/brushnet/test_brushnet_fixedBG_cge_then_temporal_equivalent_clean.py"), run_name="__main__")


if __name__ == "__main__":
    main()
