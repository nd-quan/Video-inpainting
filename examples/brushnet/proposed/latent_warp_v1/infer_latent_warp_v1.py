#!/usr/bin/env python
"""Oracle evaluation on test/<suite>/<sequence>/{inputs,masks,gt}.

--preflight is CPU-only and does not load diffusion/RAFT weights.
Normal execution prepares GT RAFT flows, then runs 14-channel latent warp +
latent temporal guidance. It never imports the codec/CGE implementation.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BRUSHNET = HERE.parents[1]
REPO = HERE.parents[3]
sys.path[:0] = [str(HERE), str(BRUSHNET)]
os.environ.setdefault("HF_HOME", str(REPO / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(REPO / "hf_cache/hub"))
DEFAULT_CONFIG = HERE / "configs/inference_test_oracle.json"


def read_config(path):
    config = json.loads(Path(path).read_text())
    expected = set(json.loads(DEFAULT_CONFIG.read_text()))
    optional = {"temporal_guidance_enabled", "raft_student_checkpoint"}
    missing = expected - optional - set(config)
    unknown = set(config) - expected - optional
    if missing or unknown:
        raise ValueError("Config keys differ: " + str(sorted(missing | unknown)))
    # Older saved configurations enable temporal guidance through their scale.
    config.setdefault("temporal_guidance_enabled", True)
    if config["cge_enabled"] is not False:
        raise ValueError("CGE must remain false: codec integration is unavailable")
    if config["flow_source"] not in ("gt", "student") or config["temporal_guidance_space"] != "latent":
        raise ValueError(
            "Expected GT or student flow and latent temporal guidance"
        )
    for key in (
        "resolution",
        "clip_length",
        "clip_stride",
        "num_inference_steps",
        "model_batch_size",
        "raft_iterations",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if config["resolution"] < 128 or config["resolution"] % 8:
        raise ValueError("resolution must be >=128 and divisible by 8")
    if (
        config["clip_length"] < 2
        or not 1 <= config["clip_stride"] < config["clip_length"]
    ):
        raise ValueError(
            "Require 1 <= clip_stride < clip_length, with clip_length >= 2"
        )
    if (
        not 0
        <= config["temporal_start"]
        < config["temporal_end"]
        <= config["num_inference_steps"]
    ):
        raise ValueError("Invalid temporal window [start, end)")
    if (
        type(config["temporal_every_n_steps"]) is not int
        or config["temporal_every_n_steps"] < 1
    ):
        raise ValueError("temporal_every_n_steps must be a positive integer")
    for key in ("temporal_guidance_scale", "latent_warp_x0_clamp", "fusion_scale"):
        if (
            not isinstance(config[key], (float, int))
            or not math.isfinite(config[key])
            or config[key] < 0
        ):
            raise ValueError(key + " must be finite and nonnegative")
    if not 0 <= config["shared_bg_noise_strength"] <= 1 or config["guidance_scale"] < 1:
        raise ValueError(
            "Invalid shared noise strength or classifier-free guidance scale"
        )
    if config["mixed_precision"] not in ("no", "fp16"):
        raise ValueError("mixed_precision must be no or fp16")
    for key in (
        "latent_warp_enabled",
        "temporal_guidance_enabled",
        "preserve_roi",
        "restore_native_size",
    ):
        if type(config[key]) is not bool:
            raise ValueError(key + " must be boolean")
    return config


def add_ablation_arguments(parser):
    temporal = parser.add_mutually_exclusive_group()
    temporal.add_argument(
        "--temporal-guidance", dest="temporal_guidance_enabled", action="store_true"
    )
    temporal.add_argument(
        "--no-temporal-guidance", dest="temporal_guidance_enabled", action="store_false"
    )
    warp = parser.add_mutually_exclusive_group()
    warp.add_argument("--latent-warp", dest="latent_warp_enabled", action="store_true")
    warp.add_argument(
        "--no-latent-warp", dest="latent_warp_enabled", action="store_false"
    )
    parser.set_defaults(temporal_guidance_enabled=None, latent_warp_enabled=None)


def apply_ablation_overrides(config, args):
    for key in ("temporal_guidance_enabled", "latent_warp_enabled"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".{}.tmp".format(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def discover(config, names=None):
    root = Path(config["dataset_root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    sequences = []
    for suite in sorted(root.iterdir()):
        if not suite.is_dir():
            continue
        for seq in sorted(suite.iterdir()):
            if seq.name == "videos" or not (seq / "inputs").is_dir():
                continue
            name = suite.name + "/" + seq.name
            if names and name not in names:
                continue
            by_kind = {}
            for kind in ("inputs", "masks", "gt"):
                files = {p.stem: p for p in (seq / kind).glob("*.png")}
                if not files or any(not stem.isdigit() for stem in files):
                    raise ValueError(
                        "Missing or nonnumeric PNG frames: " + str(seq / kind)
                    )
                by_kind[kind] = files
            stems = sorted(by_kind["inputs"], key=int)
            if any(set(by_kind[k]) != set(stems) for k in ("masks", "gt")):
                raise ValueError("GT/input/mask frame names differ: " + name)
            if len({int(s) for s in stems}) != len(stems):
                raise ValueError("Ambiguous numeric frame names: " + name)
            sequences.append(dict(name=name, stems=stems, files=by_kind))
    if names and set(names) != {s["name"] for s in sequences}:
        raise ValueError("Unknown sequence selection: " + str(names))
    if not sequences:
        raise ValueError("No test sequences found")
    return sequences


def clip_windows(stems, length, stride):
    """Cover every frame, with context overlap; never cross an index gap."""
    runs = []
    for stem in stems:
        if not runs or int(stem) != int(runs[-1][-1]) + 1:
            runs.append([])
        runs[-1].append(stem)
    for run in runs:
        if len(run) < 2:
            raise ValueError("Isolated frame has no temporal context: " + run[0])
        starts = list(range(0, max(1, len(run) - length + 1), stride))
        tail = max(0, len(run) - length)
        if tail not in starts:
            starts.append(tail)
        for start in starts:
            yield run[start : start + length]


def preflight(config, sequences, require_checkpoint=True):
    from PIL import Image

    count = pairs = clips = 0
    for sequence in sequences:
        windows = list(
            clip_windows(
                sequence["stems"], config["clip_length"], config["clip_stride"]
            )
        )
        clips += len(windows)
        pairs += sum(
            int(b) == int(a) + 1
            for a, b in zip(sequence["stems"][:-1], sequence["stems"][1:])
        )
        for stem in sequence["stems"]:
            with Image.open(sequence["files"]["inputs"][stem]) as image:
                size = image.size
            for kind in ("masks", "gt"):
                with Image.open(sequence["files"][kind][stem]) as image:
                    if image.size != size:
                        raise ValueError(
                            "GT/input/mask dimensions differ: "
                            + sequence["name"]
                            + "/"
                            + stem
                        )
            count += 1
    if require_checkpoint:
        checkpoint = Path(config["checkpoint_dir"])
        for name in (
            "brushnet/config.json",
            "brushnet/diffusion_pytorch_model.safetensors",
            "ipadapter/model.safetensors",
            "ipadapter/fusion_module.safetensors",
        ):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        model_config = json.loads((checkpoint / "brushnet/config.json").read_text())
        if (
            model_config["conditioning_channels"] != 10
            or model_config["in_channels"] != 4
        ):
            raise ValueError("Not a latent-warp 14-input BrushNet checkpoint")
        for name in (
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "vae/config.json",
            "text_encoder/config.json",
            "tokenizer/vocab.json",
        ):
            if not (Path(config["pretrained_model"]) / name).is_file():
                raise FileNotFoundError(Path(config["pretrained_model"]) / name)
    if not Path(config["raft_weights"]).is_file():
        raise FileNotFoundError(config["raft_weights"])
    if not (Path(config["propainter_root"]) / "RAFT/raft.py").is_file():
        raise FileNotFoundError(Path(config["propainter_root"]) / "RAFT/raft.py")
    if config["flow_source"] == "student":
        for file in ("config.json", "pytorch_model.bin"):
            path = student_component(config) / file
            if not path.is_file():
                raise FileNotFoundError(path)
    report = dict(
        sequences=len(sequences),
        frames=count,
        adjacent_gt_pairs=pairs,
        clips=clips,
        oracle=config["flow_source"] == "gt",
        flow_source=config["flow_source"],
        cge=False,
        temporal_space="latent",
        temporal_window=[config["temporal_start"], config["temporal_end"]],
        temporal_every_n_steps=config["temporal_every_n_steps"],
        temporal_guidance_scale=config["temporal_guidance_scale"],
        temporal_guidance_enabled=config["temporal_guidance_enabled"],
        latent_warp_enabled=config["latent_warp_enabled"],
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def load_rgb(path, resolution):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (resolution, resolution), Image.Resampling.BILINEAR
        )
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


@lru_cache(maxsize=128)
def file_digest(path, size, mtime_ns):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def student_component(config):
    return Path(config["raft_student_checkpoint"]) / "raft_student"


def load_student(config, device):
    sys.path.insert(0, str(HERE.parent / "STC_encoder_v7_raft_flow_distillation"))
    from raft_student import RAFTStudentFlowPredictor

    return RAFTStudentFlowPredictor.from_pretrained(
        student_component(config),
        propainter_root=config["propainter_root"],
        raft_checkpoint=config["raft_weights"],
    ).eval().requires_grad_(False).to(device)


def pair_signature(sequence, a, b, config):
    sources = []
    for stem in (a, b):
        path = sequence["files"]["inputs" if config["flow_source"] == "student" else "gt"][stem]
        stat = path.stat()
        sources.append(file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    weights = config["raft_weights"]
    if weights:
        path = Path(weights).resolve()
        stat = path.stat()
        weights = file_digest(str(path), stat.st_size, stat.st_mtime_ns)
    else:
        raise ValueError("An original RAFT checkpoint is required")
    payload = dict(
        gt=sources,
        resolution=config["resolution"],
        weights=weights,
        iters=config["raft_iterations"],
        fb_alpha=0.01,
        fb_beta=0.5,
        backend="original_propainter_raft_large",
        version=2,
    )
    if config["flow_source"] == "student":
        payload["inputs"] = payload.pop("gt")
        payload["backend"] = "v7_raft_student"
        for name in ("config.json", "pytorch_model.bin"):
            path = student_component(config) / name
            stat = path.stat()
            payload[name] = file_digest(str(path), stat.st_size, stat.st_mtime_ns)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def pair_path(config, sequence, a, b):
    return Path(config["flow_cache_dir"]) / sequence["name"] / (a + "_" + b + ".npz")


def load_pair(config, sequence, a, b):
    import numpy as np
    import torch

    path = pair_path(config, sequence, a, b)
    with np.load(path, allow_pickle=False) as data:
        if str(data["signature"].item()) != pair_signature(sequence, a, b, config):
            raise ValueError(
                "Stale GT flow cache; remove this pair and regenerate: " + str(path)
            )
        size = config["resolution"]
        arrays = []
        for key, channels in (("teacher_b", 2), ("valid_b", 1)):
            value = data[key].astype("float32")
            if value.shape != (channels, size, size) or not np.isfinite(value).all():
                raise ValueError("Invalid flow data: " + str(path) + ":" + key)
            arrays.append(torch.from_numpy(value))
    return tuple(arrays)


def fb_valid(flow, opposite):
    from diffusers.schedulers.scheduling_ddim_temporal import backward_warp

    warped, bounds = backward_warp(opposite, flow)
    return (
        (flow + warped).square().sum(1, keepdim=True)
        <= 0.01 * (flow.square() + warped.square()).sum(1, keepdim=True) + 0.5
    ).float() * bounds


def prepare_flows(config, sequences, device):
    import numpy as np
    import torch
    from original_raft_flow import load_original_raft

    student = config["flow_source"] == "student"
    model = None
    written = existing = 0
    for sequence in sequences:
        for a, b in zip(sequence["stems"][:-1], sequence["stems"][1:]):
            if int(b) != int(a) + 1:
                continue
            path = pair_path(config, sequence, a, b)
            if path.exists():
                load_pair(config, sequence, a, b)
                existing += 1
                continue
            if model is None:
                model = load_student(config, device) if student else load_original_raft(config, device)
            frames = torch.stack(
                [
                    load_rgb(sequence["files"]["inputs" if student else "gt"][s], config["resolution"])
                    for s in (a, b)
                ]
            ).to(device)
            with torch.no_grad():
                if student:
                    forward, backward = model.predict_bidirectional(
                        frames[:1] * 2 - 1, frames[1:] * 2 - 1,
                        return_all=False, pair_batch_size=1,
                    )
                    forward, backward = forward.float(), backward.float()
                else:
                    _, flows = model(
                        frames * 2 - 1,
                        frames.flip(0) * 2 - 1,
                        iters=config["raft_iterations"],
                        test_mode=True,
                    )
                    flows = flows.float()
                    forward, backward = flows[:1], flows[1:]
                vf, vb = fb_valid(forward, backward), fb_valid(backward, forward)
            arrays = dict(
                teacher_f=forward[0].cpu().numpy(), teacher_b=backward[0].cpu().numpy()
            )
            if any(
                not np.isfinite(v).all() or np.abs(v).max() > 60000
                for v in arrays.values()
            ):
                raise FloatingPointError("Invalid RAFT flow: " + str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + ".{}.tmp.npz".format(os.getpid()))
            np.savez_compressed(
                tmp,
                **{k: v.astype("float16") for k, v in arrays.items()},
                valid_f=vf[0].cpu().numpy().astype("uint8"),
                valid_b=vb[0].cpu().numpy().astype("uint8"),
                signature=np.asarray(pair_signature(sequence, a, b, config))
            )
            tmp.replace(path)
            written += 1
            if written == 1 or written % 25 == 0:
                print(
                    "GT flow: {} new, {} reused; {} {}->{}".format(
                        written, existing, sequence["name"], a, b
                    ),
                    flush=True,
                )
    del model
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    save_json(
        Path(config["flow_cache_dir"]) / "metadata.json",
        dict(
            dataset_root=config["dataset_root"],
            source="decoded inputs" if student else "clean GT (oracle)",
            backend="V7 RAFT student" if student else "original ProPainter RAFT-Large",
            raft_student_checkpoint=config.get("raft_student_checkpoint"),
            raft_weights=config["raft_weights"],
            resolution=config["resolution"],
            iterations=config["raft_iterations"],
            convention="teacher_b: frame k+1 -> k; RGB-pixel displacements; FB consistency alpha=.01 beta=.5",
        ),
    )
    print("Flow ready: {} new, {} reused".format(written, existing), flush=True)


def stable_seed(base, key):
    return int.from_bytes(
        hashlib.sha256((str(base) + ":" + key).encode()).digest()[:8], "little"
    ) % (2**63 - 1)


def measure_frame(prediction, target, roi):
    import numpy as np
    from skimage.metrics import structural_similarity

    error = (prediction - target) ** 2

    def psnr(mask):
        if not mask.any():
            return None
        mse = float(error[mask].mean())
        return min(100.0, -10 * math.log10(max(mse, 1e-10)))

    return dict(
        psnr_full=psnr(np.ones(roi.shape, dtype=bool)),
        psnr_bg=psnr(~roi),
        psnr_roi=psnr(roi),
        ssim_full=float(
            structural_similarity(target, prediction, data_range=1.0, channel_axis=2)
        ),
    )


def validate_resume_output(output, config, sequences):
    """Refuse mixed settings and preserve every valid, already-saved PNG."""
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


def measure_saved_outputs(config, sequences):
    import numpy as np
    from PIL import Image

    rows = []
    for sequence in sequences:
        for stem in sequence["stems"]:
            path = Path(config["output_dir"]) / sequence["name"] / (stem + ".png")
            with Image.open(path) as image:
                prediction = np.asarray(image.convert("RGB"), dtype="float32") / 255
                size = image.size
            with Image.open(sequence["files"]["gt"][stem]) as image:
                target = (
                    np.asarray(
                        image.convert("RGB").resize(size, Image.Resampling.BILINEAR),
                        dtype="float32",
                    )
                    / 255
                )
            with Image.open(sequence["files"]["masks"][stem]) as image:
                roi = (
                    np.asarray(
                        image.convert("L").resize(size, Image.Resampling.NEAREST)
                    )
                    >= 128
                )
            row = measure_frame(prediction, target, roi)
            row.update(sequence=sequence["name"], frame=stem)
            rows.append(row)
    return rows


def run_inference(config, sequences, device, max_clips=0, resume=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from latent_warp_inference import InferenceModels, denoise_clip
    from diffusers.schedulers.scheduling_ddim_temporal import TemporalDDIMScheduler
    from shared_bg_noise_training import (
        make_sequence_shared_background_noise,
        mix_shared_background_noise,
    )

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not resume:
        raise FileExistsError(
            "Use a new output_dir for every evaluation: " + str(output)
        )
    if resume:
        existing = validate_resume_output(output, config, sequences)
        print("Resume: preserving {} existing PNGs".format(existing), flush=True)
    save_json(output / "resolved_inference_config.json", config)
    models = scheduler = None
    metrics = []
    processed = 0
    skipped = 0
    with (output / "guidance_audit.jsonl").open("a" if resume else "w") as audit:
        for sequence in sequences:
            seen = set()
            for stems in clip_windows(
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
                    scheduler = TemporalDDIMScheduler.from_pretrained(
                        config["pretrained_model"], subfolder="scheduler"
                    )
                    if scheduler.config.prediction_type != "epsilon":
                        raise ValueError(
                            "This checkpoint was trained for epsilon prediction"
                        )
                    scheduler.register_to_config(clip_sample=False, thresholding=False)
                pairs = [
                    load_pair(config, sequence, a, b)
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
                        load_rgb(sequence["files"]["inputs"][s], size) * 2 - 1
                        for s in stems
                    ]
                )
                bg = torch.stack(
                    [
                        torch.from_numpy(
                            (
                                np.asarray(
                                    m.resize((size, size), Image.Resampling.NEAREST)
                                )
                                < 128
                            ).copy()
                        ).float()[None]
                        for m in masks_pil
                    ]
                ).to(device)
                generator = torch.Generator(device=device).manual_seed(
                    stable_seed(config["seed"], sequence["name"] + ":" + stems[0])
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
                                stable_seed(config["seed"], sequence["name"] + ":" + s)
                            ),
                        ).to(device, models.dtype)
                        for s in stems
                    ]
                )[None]
                masks = F.interpolate(bg, size=(size // 8, size // 8), mode="nearest")[
                    None
                ]
                z = mix_shared_background_noise(
                    independent, shared, masks, config["shared_bg_noise_strength"], True
                )[0]
                z *= scheduler.init_noise_sigma
                latent, report = denoise_clip(
                    models.predict_noise,
                    scheduler,
                    z,
                    condition,
                    bg,
                    flow,
                    valid,
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
                        (decoded[j].permute(1, 2, 0).numpy() * 255)
                        .round()
                        .astype("uint8")
                    )
                    target_size = (
                        original[j].size
                        if config["restore_native_size"]
                        else (size, size)
                    )
                    generated = generated.resize(target_size, Image.Resampling.BILINEAR)
                    input_image = original[j].resize(
                        target_size, Image.Resampling.BILINEAR
                    )
                    roi = (
                        np.asarray(
                            masks_pil[j].resize(target_size, Image.Resampling.NEAREST)
                        )
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
                                gt.convert("RGB").resize(
                                    target_size, Image.Resampling.BILINEAR
                                ),
                                dtype="float32",
                            )
                            / 255
                        )
                    row = measure_frame(
                        np.asarray(generated, dtype="float32") / 255, target, roi
                    )
                    row.update(sequence=sequence["name"], frame=stem)
                    metrics.append(row)
                report.update(
                    sequence=sequence["name"], frames=stems, cge=False, flow_source=config["flow_source"]
                )
                audit.write(json.dumps(report, allow_nan=False) + "\n")
                audit.flush()
                processed += 1
                print(
                    "Inference clip {}: {} {}..{}; temporal {}/{} applied".format(
                        processed,
                        sequence["name"],
                        stems[0],
                        stems[-1],
                        report["temporal_applied"],
                        report["temporal_calls"],
                    ),
                    flush=True,
                )
            if max_clips and processed >= max_clips:
                break
    if resume and not max_clips:
        metrics = measure_saved_outputs(config, sequences)
    save_json(output / "frame_metrics.json", metrics)
    summary = {
        key: float(np.mean([m[key] for m in metrics if m[key] is not None]))
        for key in ("psnr_full", "psnr_bg", "psnr_roi", "ssim_full")
        if any(m[key] is not None for m in metrics)
    }
    summary.update(
        frames=len(metrics),
        clips=processed + skipped,
        resumed=resume,
        newly_computed_clips=processed,
        reused_clips=skipped,
        oracle_gt_flow=config["flow_source"] == "gt",
        raft_student_checkpoint=config.get("raft_student_checkpoint"),
        cge=False,
        checkpoint=config["checkpoint_dir"],
        overlap_policy="first window owns each frame",
        complete_evaluation=not max_clips,
        latent_warp_enabled=config["latent_warp_enabled"],
        temporal_guidance_enabled=config["temporal_guidance_enabled"],
        effective_temporal_guidance_scale=(
            config["temporal_guidance_scale"]
            if config["temporal_guidance_enabled"]
            else 0.0
        ),
    )
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_ablation_arguments(parser)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--sequences", nargs="+", help="e.g. test_1/BasketballPass")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Local device within the scheduler allocation",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--prepare-flows-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Preserve existing PNGs and compute only missing first-owned frames",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Smoke inference only; 0 evaluates all clips",
    )
    args = parser.parse_args(argv)
    if args.max_clips < 0:
        parser.error("max-clips must be nonnegative")
    config = read_config(args.config)
    apply_ablation_overrides(config, args)
    if args.checkpoint_dir:
        config["checkpoint_dir"] = str(Path(args.checkpoint_dir).resolve())
    if args.output_dir:
        config["output_dir"] = str(Path(args.output_dir).resolve())
    sequences = discover(config, args.sequences)
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
    # Prepare only the sequence prefix required by a one-clip smoke test.
    flow_sequences = sequences
    if args.max_clips and not args.prepare_flows_only:
        remaining = args.max_clips
        flow_sequences = []
        for sequence in sequences:
            windows = list(
                clip_windows(
                    sequence["stems"], config["clip_length"], config["clip_stride"]
                )
            )[:remaining]
            selected = sorted({s for window in windows for s in window}, key=int)
            flow_sequences.append(dict(sequence, stems=selected))
            remaining -= len(windows)
            if remaining <= 0:
                break
    prepare_flows(config, flow_sequences, device)
    if not args.prepare_flows_only:
        run_inference(config, sequences, device, args.max_clips, resume=args.resume)


if __name__ == "__main__":
    main()
