#!/usr/bin/env python
"""Oracle latent-warp evaluation with CGE -> latent temporal guidance.

Same data, GT RAFT flow cache, clip windows, seeds, noise, ROI handling and
image metrics as ``infer_latent_warp_v1.py`` (imported, not modified). Each
DDIM step applies codec-guided enhancement (VCM-RS, run by ``codec_python``
in a subprocess) at ``cge_steps``, then latent temporal guidance at the
CGE-equivalent clean latent.  The main process runs in the guided_diff env;
the codec env is never activated.

--preflight is CPU-only: it checks data, checkpoint, RAFT and codec paths.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import infer_latent_warp_v1 as base  # sets sys.path and the HF cache defaults

HERE = base.HERE
DEFAULT_CONFIG = HERE / "configs/inference_test_oracle_cge.json"
CGE_KEYS = {
    "codec_root": str,
    "codec_python": str,
    "cge_codec_mode": str,
    "cge_steps": list,
    "cge_guidance_scale": (int, float),
    "cge_scale_schedule": str,
    "cge_bg_quality": int,
    "cge_roi_quality": int,
    "cge_profile": str,
    "cge_configuration": str,
    "cge_intra_period": int,
    "cge_nn_intra_qp_offset": int,
    "cge_frame_rate": int,
    "cge_codec_workers": int,
    "cge_max_parallel": int,
    "cge_timeout_seconds": (int, float),
}


def read_config(path):
    """Validate the base keys with the original reader, then the CGE keys."""
    from latent_warp_cge_inference import CGE_CODEC_MODES, cge_window

    config = json.loads(Path(path).read_text())
    missing = set(CGE_KEYS) - set(config)
    if missing:
        raise ValueError("Missing CGE keys: " + str(sorted(missing)))
    if type(config.get("cge_enabled")) is not bool:
        raise ValueError("cge_enabled must be boolean")
    base_part = {k: v for k, v in config.items() if k not in CGE_KEYS and k != "cge_codec_device"}
    base_part["cge_enabled"] = False  # the original reader only accepts false
    with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
        json.dump(base_part, handle)
        handle.flush()
        validated = base.read_config(handle.name)
    validated["cge_enabled"] = config["cge_enabled"]
    validated["cge_codec_device"] = config.get("cge_codec_device", "cuda")
    if validated["cge_codec_device"] not in ("cuda", "cpu"):
        raise ValueError("cge_codec_device must be cuda or cpu")
    for key, kind in CGE_KEYS.items():
        if not isinstance(config[key], kind) or isinstance(config[key], bool):
            raise ValueError("{} has the wrong type".format(key))
        validated[key] = config[key]
    cge_window(validated["cge_steps"])
    if validated["cge_steps"][-1] >= validated["num_inference_steps"]:
        raise ValueError("cge_steps must be inside the denoising loop")
    if validated["cge_codec_mode"] not in CGE_CODEC_MODES:
        raise ValueError("cge_codec_mode must be one of " + str(CGE_CODEC_MODES))
    if validated["cge_scale_schedule"] not in ("fixed", "noise_level"):
        raise ValueError("cge_scale_schedule must be fixed or noise_level")
    if validated["cge_profile"] not in ("train_match", "vtm_only"):
        raise ValueError("cge_profile must be train_match or vtm_only")
    if validated["cge_configuration"] not in ("RandomAccess", "AllIntra"):
        raise ValueError("cge_configuration must be RandomAccess or AllIntra")
    if not 0 <= validated["cge_bg_quality"] <= 63 or not 0 <= validated["cge_roi_quality"] <= 63:
        raise ValueError("CGE QPs must be in [0, 63]")
    scale = validated["cge_guidance_scale"]
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("cge_guidance_scale must be finite and nonnegative")
    for key in ("cge_codec_workers", "cge_max_parallel", "cge_intra_period", "cge_frame_rate"):
        if validated[key] < 1:
            raise ValueError(key + " must be positive")
    if validated["cge_timeout_seconds"] <= 0:
        raise ValueError("cge_timeout_seconds must be positive")
    return validated


def check_codec(config, run_import=True):
    root, python = Path(config["codec_root"]), Path(config["codec_python"])
    if not (root / "vcmrs/encoder.py").is_file():
        raise FileNotFoundError(root / "vcmrs/encoder.py")
    if not python.is_file():
        raise FileNotFoundError(python)
    if not (root / "output").is_dir():
        raise FileNotFoundError("VCM-RS needs a writable folder: " + str(root / "output"))
    if run_import:
        env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES="")
        subprocess.run(
            [str(python), "-c", "import vcmrs.encoder"],
            cwd=root, env=env, check=True, capture_output=True, timeout=300,
        )


def preflight(config, sequences, require_checkpoint=True):
    report = base.preflight(config, sequences, require_checkpoint)
    if config["cge_enabled"]:
        check_codec(config)
    cge = dict(
        cge=config["cge_enabled"] and config["cge_guidance_scale"] > 0,
        cge_steps=config["cge_steps"],
        cge_guidance_scale=config["cge_guidance_scale"],
        cge_codec_mode=config["cge_codec_mode"],
        cge_bg_quality=config["cge_bg_quality"],
        codec_root=config["codec_root"],
        codec_python=config["codec_python"],
        guidance_order="CGE -> equivalent clean latent -> latent TG",
    )
    print(json.dumps(cge, indent=2), flush=True)
    report.update(cge)
    return report


def validate_resume_output(output, config, sequences):
    """As in the original evaluator, but reads the CGE-extended config."""
    from PIL import Image

    saved = read_config(Path(output) / "resolved_inference_config.json")
    different = [key for key in config if config[key] != saved[key]]
    if different:
        raise ValueError("Resume configuration differs: " + str(different))
    count = 0
    for sequence in sequences:
        folder = Path(output) / sequence["name"]
        actual = {p.stem for p in folder.glob("*.png")}
        if actual - set(sequence["stems"]):
            raise ValueError("Unexpected existing frames in " + str(folder))
        for stem in actual:
            path = folder / (stem + ".png")
            try:
                with Image.open(path) as image:
                    image.verify()
            except OSError:
                backup = path.with_name(path.name + ".invalid-before-resume")
                if backup.exists():
                    raise ValueError("Existing corrupt-frame backup: " + str(backup))
                path.replace(backup)
                continue
            count += 1
    return count


def run_inference(config, sequences, device, max_clips=0, resume=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from latent_warp_inference import InferenceModels
    from latent_warp_cge_inference import (
        build_codec,
        denoise_clip_cge,
        isolate_cge_environment,
        make_scheduler,
    )
    from shared_bg_noise_training import (
        make_sequence_shared_background_noise,
        mix_shared_background_noise,
    )

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not resume:
        raise FileExistsError("Use a new output_dir for every evaluation: " + str(output))
    if resume:
        existing = validate_resume_output(output, config, sequences)
        print("Resume: preserving {} existing PNGs".format(existing), flush=True)
    base.save_json(output / "resolved_inference_config.json", config)
    isolate_cge_environment(config["cge_codec_workers"])
    models = scheduler = None
    metrics = []
    processed = 0
    skipped = 0
    with (output / "guidance_audit.jsonl").open("a" if resume else "w") as audit:
        for sequence in sequences:
            seen = set()
            for stems in base.clip_windows(
                sequence["stems"], config["clip_length"], config["clip_stride"]
            ):
                if max_clips and processed >= max_clips:
                    break
                owned = [stem for stem in stems if stem not in seen]
                if resume and all(
                    (output / sequence["name"] / (stem + ".png")).is_file()
                    for stem in owned
                ):
                    seen.update(stems)
                    skipped += 1
                    continue
                if models is None:
                    models = InferenceModels(config, device)
                    codec = build_codec(config) if config["cge_enabled"] else None
                    scheduler = make_scheduler(
                        config["pretrained_model"], config, codec, models.vae
                    )
                pairs = [
                    base.load_pair(config, sequence, a, b)
                    for a, b in zip(stems[:-1], stems[1:])
                ]
                flow = torch.stack([p[0] for p in pairs]).to(device)
                valid = torch.stack([p[1] for p in pairs]).to(device)
                original = []
                masks_pil = []
                for stem in stems:
                    with Image.open(sequence["files"]["inputs"][stem]) as image:
                        original.append(image.convert("RGB").copy())
                    with Image.open(sequence["files"]["masks"][stem]) as mask:
                        masks_pil.append(mask.convert("L").copy())
                size = config["resolution"]
                images = torch.stack(
                    [
                        base.load_rgb(sequence["files"]["inputs"][s], size) * 2 - 1
                        for s in stems
                    ]
                )
                bg = torch.stack(
                    [
                        torch.from_numpy(
                            (
                                np.asarray(m.resize((size, size), Image.Resampling.NEAREST))
                                < 128
                            ).copy()
                        ).float()[None]
                        for m in masks_pil
                    ]
                ).to(device)
                generator = torch.Generator(device=device).manual_seed(
                    base.stable_seed(config["seed"], sequence["name"] + ":" + stems[0])
                )
                condition = models.prepare(images, bg, original, masks_pil, generator)
                template = torch.zeros(
                    (1, len(stems), 4, size // 8, size // 8),
                    device=device,
                    dtype=models.dtype,
                )
                shared = make_sequence_shared_background_noise(
                    template, [sequence["name"]], config["seed"]
                )
                independent = torch.stack(
                    [
                        torch.randn(
                            (4, size // 8, size // 8),
                            generator=torch.Generator(device="cpu").manual_seed(
                                base.stable_seed(config["seed"], sequence["name"] + ":" + s)
                            ),
                        ).to(device, models.dtype)
                        for s in stems
                    ]
                )[None]
                masks = F.interpolate(bg, size=(size // 8, size // 8), mode="nearest")[None]
                z = mix_shared_background_noise(
                    independent, shared, masks, config["shared_bg_noise_strength"], True
                )[0]
                z *= scheduler.init_noise_sigma
                latent, report = denoise_clip_cge(
                    models.predict_noise,
                    scheduler,
                    z,
                    condition,
                    bg,
                    flow,
                    valid,
                    images.to(device),
                    models.vae.decode,
                    config,
                )
                decoded = models.decode(latent).cpu()
                for j, stem in enumerate(stems):
                    if stem in seen:
                        continue
                    seen.add(stem)
                    destination = output / sequence["name"] / (stem + ".png")
                    if resume and destination.is_file():
                        continue
                    generated = Image.fromarray(
                        (decoded[j].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
                    )
                    target_size = (
                        original[j].size if config["restore_native_size"] else (size, size)
                    )
                    generated = generated.resize(target_size, Image.Resampling.BILINEAR)
                    input_image = original[j].resize(target_size, Image.Resampling.BILINEAR)
                    roi = (
                        np.asarray(masks_pil[j].resize(target_size, Image.Resampling.NEAREST))
                        >= 128
                    )
                    if config["preserve_roi"]:
                        generated = Image.fromarray(
                            np.where(
                                roi[..., None],
                                np.asarray(input_image),
                                np.asarray(generated),
                            ).astype("uint8")
                        )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.stem + ".tmp.png")
                    generated.save(temporary)
                    temporary.replace(destination)
                    with Image.open(sequence["files"]["gt"][stem]) as gt:
                        target = (
                            np.asarray(
                                gt.convert("RGB").resize(target_size, Image.Resampling.BILINEAR),
                                dtype="float32",
                            )
                            / 255
                        )
                    row = base.measure_frame(
                        np.asarray(generated, dtype="float32") / 255, target, roi
                    )
                    row.update(sequence=sequence["name"], frame=stem)
                    metrics.append(row)
                report.update(
                    sequence=sequence["name"],
                    frames=stems,
                    cge=bool(report["cge_applied_steps"]),
                    flow_source="gt",
                )
                audit.write(json.dumps(report, allow_nan=False) + "\n")
                audit.flush()
                processed += 1
                print(
                    "Inference clip {}: {} {}..{}; CGE steps {} ({} codec evals); "
                    "temporal {}/{} applied".format(
                        processed,
                        sequence["name"],
                        stems[0],
                        stems[-1],
                        report["cge_applied_steps"],
                        report["cge_codec_evals"],
                        report["temporal_applied"],
                        report["temporal_calls"],
                    ),
                    flush=True,
                )
            if max_clips and processed >= max_clips:
                break
    if resume and not max_clips:
        metrics = base.measure_saved_outputs(config, sequences)
    base.save_json(output / "frame_metrics.json", metrics)
    summary = {
        key: float(np.mean([m[key] for m in metrics if m[key] is not None]))
        for key in ("psnr_full", "psnr_bg", "psnr_roi", "ssim_full")
        if any(m[key] is not None for m in metrics)
    }
    use_cge = config["cge_enabled"] and config["cge_guidance_scale"] > 0
    summary.update(
        frames=len(metrics),
        clips=processed + skipped,
        resumed=resume,
        newly_computed_clips=processed,
        reused_clips=skipped,
        oracle_gt_flow=True,
        cge=use_cge,
        cge_steps=config["cge_steps"] if use_cge else [],
        cge_guidance_scale=config["cge_guidance_scale"] if use_cge else 0.0,
        cge_codec_mode=config["cge_codec_mode"],
        cge_bg_quality=config["cge_bg_quality"],
        cge_codec_device=config["cge_codec_device"],
        guidance_order="CGE -> equivalent clean latent -> latent TG",
        checkpoint=config["checkpoint_dir"],
        overlap_policy="first window owns each frame",
        complete_evaluation=not max_clips,
        latent_warp_enabled=config["latent_warp_enabled"],
        temporal_guidance_enabled=config["temporal_guidance_enabled"],
        effective_temporal_guidance_scale=(
            config["temporal_guidance_scale"] if config["temporal_guidance_enabled"] else 0.0
        ),
    )
    base.save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    base.add_ablation_arguments(parser)
    cge = parser.add_mutually_exclusive_group()
    cge.add_argument("--cge", dest="cge_enabled", action="store_true")
    cge.add_argument("--no-cge", dest="cge_enabled", action="store_false")
    parser.set_defaults(cge_enabled=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--sequences", nargs="+", help="e.g. test_1/BasketballPass")
    parser.add_argument("--device", default="cuda:0", help="Local device within the allocation")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--prepare-flows-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Preserve existing PNGs and compute only missing first-owned frames")
    parser.add_argument("--max-clips", type=int, default=0,
                        help="Smoke inference only; 0 evaluates all clips")
    args = parser.parse_args(argv)
    if args.max_clips < 0:
        parser.error("max-clips must be nonnegative")
    config = read_config(args.config)
    base.apply_ablation_overrides(config, args)
    if args.cge_enabled is not None:
        config["cge_enabled"] = args.cge_enabled
    if args.checkpoint_dir:
        config["checkpoint_dir"] = str(Path(args.checkpoint_dir).resolve())
    if args.output_dir:
        config["output_dir"] = str(Path(args.output_dir).resolve())
    sequences = base.discover(config, args.sequences)
    preflight(config, sequences, require_checkpoint=not args.prepare_flows_only)
    if args.preflight:
        return
    if (
        not args.prepare_flows_only
        and not args.resume
        and Path(config["output_dir"]).exists()
        and any(Path(config["output_dir"]).iterdir())
    ):
        raise FileExistsError("Use a new output_dir: " + config["output_dir"])
    import torch

    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No GPU access; submit through job run -g 1")
    flow_sequences = sequences
    if args.max_clips and not args.prepare_flows_only:
        remaining = args.max_clips
        flow_sequences = []
        for sequence in sequences:
            windows = list(
                base.clip_windows(sequence["stems"], config["clip_length"], config["clip_stride"])
            )[:remaining]
            selected = sorted({s for window in windows for s in window}, key=int)
            flow_sequences.append(dict(sequence, stems=selected))
            remaining -= len(windows)
            if remaining <= 0:
                break
    base.prepare_flows(config, flow_sequences, device)
    if not args.prepare_flows_only:
        run_inference(config, sequences, device, args.max_clips, resume=args.resume)


if __name__ == "__main__":
    main()
