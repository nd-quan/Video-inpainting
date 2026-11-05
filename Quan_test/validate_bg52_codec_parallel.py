#!/usr/bin/env python3
"""Compare serial/P4 codec reconstructions, CGE and subsequent TGE on one frozen clip."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512"
OUTPUT = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em3_tg0p1_v7student_codecP4_512"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    for key in list(os.environ):
        if key.startswith(("CGE_", "TEMPORAL_", "TEACHER_FLOW_")):
            del os.environ[key]
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                       "CGE_GUIDANCE_SCALE": "0.001", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    sys.path[:0] = [str(ROOT / "examples/brushnet"), str(ROOT / "examples/brushnet/proposed")]
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    import diffusers.schedulers.scheduling_ddim_CGE as cge
    from diffusers.schedulers.scheduling_ddim_cge_then_temporal import CGEThenTemporalDDIMScheduler
    from vcmrs_codec_adapter import VCMRSDualRegionCodec, VCMRSBackgroundOnlyCodec

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = json.loads((BASE / "run_config.json").read_text())
    state_path = BASE / "diagnostics/probe/step25_state.pt"
    state = torch.load(state_path, map_location="cpu")
    probe = json.loads((BASE / "diagnostics/probe/step25_probe.json").read_text())
    x0 = state["x0_before_cge"].to("cuda")
    assert list(x0.shape) == [4, 4, 64, 64]
    resize = transforms.Resize((512, 512))
    image_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(), transforms.Lambda(lambda x: (x > 0.5).float())])
    frames, masks = [], []
    for frame_id in (117, 118, 119, 120):
        relative = Path("Class_A/Traffic") / ("{:06d}.png".format(frame_id))
        image = cv2.imread(str(Path(config["dataset"]) / "input" / relative))[:, :, ::-1]
        mask = (cv2.imread(str(Path(config["dataset"]) / "mask" / relative)).sum(-1) > 255)
        image = resize(Image.fromarray(image.astype(np.uint8)).convert("RGB"))
        mask = resize(Image.fromarray(np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)).convert("RGB"))
        frames.append(image_transform(image))
        masks.append(mask_transform(mask.convert("L")))
    x_lr = torch.stack(frames).to("cuda", dtype=torch.float32) * 2 - 1
    roi = torch.stack(masks).to("cuda", dtype=torch.float32)
    model = ROOT / "examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5"
    vae = AutoencoderKL.from_pretrained(str(model), subfolder="vae", torch_dtype=torch.float16).to("cuda").eval()
    vae.requires_grad_(False)
    out = OUTPUT / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    variants = {}
    timings = {}
    original_roundtrip = cge._codec_roundtrip01
    for workers in (1, 4):
        os.environ["CGE_CODEC_WORKERS"] = str(workers)
        codec = VCMRSBackgroundOnlyCodec(VCMRSDualRegionCodec(
            vcmrs_root=Path("/home/cilab/ndquan/vcm-rs"),
            python_executable=Path("/home/cilab/ndquan/envs/vcm/bin/python"),
            bg_quality=config["cge_bg_quality"], roi_quality=20, profile=config["cge_profile"],
            configuration=config["cge_configuration"], intra_period=config["cge_intra_period"],
            nn_intra_qp_offset=config["cge_nn_intra_qp_offset"], frame_rate=config["cge_frame_rate"],
            max_parallel=workers))
        for index in range(4):
            codec.prepare_region_mask(roi[index], x_lr[index])
        captured = {}
        lock = threading.Lock()
        active, peak = 0, 0

        def capture(*positional, **keyword):
            nonlocal active, peak
            # The original function receives codec, image, ROI mask, call key positionally.
            key = keyword.get("call_key", positional[3] if len(positional) > 3 else None)
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                value = original_roundtrip(*positional, **keyword)
                cpu = value.detach().cpu()
                with lock:
                    if key in captured:
                        raise RuntimeError("Duplicate codec call key")
                    captured[key] = cpu
                return value
            finally:
                with lock:
                    active -= 1

        cge._codec_roundtrip01 = capture
        options = SimpleNamespace(cge_codec=codec, guidance_scale_cge=0.001,
                                  cge_scale_schedule="fixed", per_frame_cge=False,
                                  decode_chunk_size=1, vae_scaling_factor=float(vae.config.scaling_factor),
                                  cge_codec_eval_count=0)
        print("START codec_workers={}".format(workers), flush=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            delta = cge.cond_fn(x0.clone(), torch.tensor(probe["timestep"], device="cuda"),
                                x_lr, roi, vae.decode, args=options)
        finally:
            cge._codec_roundtrip01 = original_roundtrip
        torch.cuda.synchronize()
        timings[str(workers)] = time.perf_counter() - started
        scheduler = CGEThenTemporalDDIMScheduler(clip_sample=False)
        scheduler.set_timesteps(50)
        scheduler.set_temporal_guidance(flow_backward=state["flow"].to("cuda"),
            stable_bg=state["stable_bg"].to("cuda"), guidance_scale=0.1, start_step=25,
            end_step=40, detach_previous=True, loss_type="l2", enabled=True)
        a = float(probe["sqrt_alpha_previous"])
        clean = x0.float() + delta.float() / a
        guided = scheduler._apply_latent_temporal_guidance(clean)
        temporal_delta = a * (guided.float() - clean)
        previous_cge = state["prev_base"].to("cuda") + delta
        previous_combined = (previous_cge.float() + temporal_delta).to(previous_cge.dtype)
        variants[workers] = {"codec_reconstructions": captured, "cge_delta": delta.detach().cpu(),
                            "equivalent_clean": clean.detach().cpu(), "temporal_delta": temporal_delta.detach().cpu(),
                            "prev_sample": previous_combined.detach().cpu(),
                            "temporal_loss": scheduler.last_temporal_loss, "peak_codec_calls": peak}
        assert len(captured) == 12
        assert scheduler.temporal_guidance_applied_steps == 1
        print("DONE workers={} seconds={:.3f} peak={} codec_calls={}".format(workers, timings[str(workers)], peak, len(captured)), flush=True)

    serial, parallel = variants[1], variants[4]
    assert serial["codec_reconstructions"].keys() == parallel["codec_reconstructions"].keys()
    recon_diff = max(float((serial["codec_reconstructions"][k].float() - parallel["codec_reconstructions"][k].float()).abs().max())
                     for k in serial["codec_reconstructions"])
    differences = {k: float((serial[k].float() - parallel[k].float()).abs().max())
                   for k in ("cge_delta", "equivalent_clean", "temporal_delta", "prev_sample")}
    passed = recon_diff == 0 and all(v <= 1e-6 for v in differences.values()) and parallel["peak_codec_calls"] == 4
    sources = [ROOT / "examples/brushnet/vcmrs_codec_adapter.py",
               ROOT / "examples/brushnet/diffusers/schedulers/scheduling_ddim_CGE.py",
               ROOT / "examples/brushnet/diffusers/schedulers/scheduling_ddim_cge_then_temporal.py",
               ROOT / "examples/brushnet/diffusers/schedulers/scheduling_ddim_cge_temporal.py",
               ROOT / "examples/brushnet/inference_experiments/test_brushnet_fixedBG_cge_then_temporal_equivalent_clean_timed.py"]
    report = {"passed": passed, "scope": "Same frozen first four Traffic frames at step 25; real VAE and real train_match codec. This validates one guided transition, not a full-rollout speed benchmark.",
              "frame_ids": [117, 118, 119, 120], "state": str(state_path), "cge_scale": 0.001, "tge_scale": 0.1,
              "codec_calls_per_variant": 12, "peak_calls": {str(k): v["peak_codec_calls"] for k, v in variants.items()},
              "codec_reconstruction_max_abs_difference": recon_diff, "max_abs_differences": differences,
              "temporal_losses": {str(k): v["temporal_loss"] for k, v in variants.items()},
              "cge_seconds": timings, "cge_speedup": timings["1"] / timings["4"],
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (out / "codec_parallel_equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Parallel codec equivalence validation failed; full launch blocked")


if __name__ == "__main__":
    main()
