#!/usr/bin/env python
"""Train only an RGB-STC condition adapter on the frozen V8 checkpoint.

This is deliberately a phase-1 ablation.  The V8 BrushNet, U-Net, IP-Adapter,
FGBG fusion, VAE, text encoder, and image encoder are all frozen.  Shared BG
noise and the standard diffusion target remain identical to the baseline; no
flow predictor and no auxiliary reconstruction penalty are used.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

# Running a script below this new subfolder would otherwise import a pip
# diffusers instead of this repository's BrushNet-enabled vendored package.
THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from safetensors import safe_open
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPVisionModelWithProjection,
)

from diffusers import AutoencoderKL, BrushNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from ip_adapter.ip_adapter import ImageProjModel
from shared_bg_noise_training import (
    HierarchicalV8ClipDataset,
    collate_shared_noise_clips,
    make_sequence_shared_background_noise,
    sample_clip_timesteps,
    sample_shared_background_noise,
)
from STC_encoder_v2_rgb.frozen_v8 import (
    build_frozen_v8_context,
    frozen_v8_predict,
    install_and_load_ip_adapter,
    load_fusion_module,
)
from STC_encoder_v2_rgb.rgb_stc_adapter import (
    RGBSTCConditionAdapter,
    augment_brushnet_condition,
)


logger = get_logger(__name__)

REPO_ROOT = BRUSHNET_DIR.parents[1]
DEFAULT_BASE_MODEL = (
    BRUSHNET_DIR
    / "base_model"
    / "stable-diffusion-v1-5"
    / "stable-diffusion-v1-5"
)
DEFAULT_DATASET = Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow")
DEFAULT_BASELINE = (
    REPO_ROOT
    / "experiments"
    / "train_sharedNoise_sameBG_0.9"
    / "checkpoint-2000"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments" / "train_rgb_stc_v2_sharedNoise_0.9"
)
DEFAULT_IMAGE_ENCODER = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class EpochRandomSampler(Sampler[int]):
    """Epoch-keyed shuffle so a resumed epoch can reproduce its permutation."""

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + self.epoch
        )
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self):
        return len(self.data_source)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Frozen-V8 training of a pixel-space RGB-STC adapter."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default=str(DEFAULT_BASE_MODEL),
    )
    parser.add_argument(
        "--baseline_checkpoint",
        default=str(DEFAULT_BASELINE),
        help="V8 checkpoint containing brushnet/ and ipadapter/ explicit files.",
    )
    parser.add_argument(
        "--image_encoder_name_or_path", default=DEFAULT_IMAGE_ENCODER
    )
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET))
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--clip_stride", type=int, default=6)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--dataloader_pin_memory", action="store_true")

    parser.add_argument("--shared_bg_noise_strength", type=float, default=0.9)
    parser.add_argument("--shared_bg_mask_threshold", type=float, default=0.5)
    parser.add_argument(
        "--sequence_shared_noise_refresh",
        choices=("epoch", "run"),
        default="epoch",
    )
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--variance_preserving_shared_noise",
        dest="variance_preserving_shared_noise",
        action="store_true",
    )
    noise_group.add_argument(
        "--linear_shared_bg_noise",
        dest="variance_preserving_shared_noise",
        action="store_false",
    )
    parser.set_defaults(variance_preserving_shared_noise=True)
    parser.add_argument("--independent_frame_timesteps", action="store_true")

    parser.add_argument(
        "--condition_mode",
        choices=("full_rgb_bg_mask", "videocomposer_roi_masked"),
        default="full_rgb_bg_mask",
    )
    parser.add_argument("--stc_hidden_channels", type=int, default=64)
    parser.add_argument("--stc_num_heads", type=int, default=2)
    parser.add_argument("--stc_num_layers", type=int, default=1)
    parser.add_argument("--stc_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--stc_dropout", type=float, default=0.0)
    parser.add_argument("--stc_injection_scale", type=float, default=1.0)
    parser.add_argument("--fusion_scale", type=float, default=1.0)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="A RGB-STC checkpoint path or 'latest'; never pass the V8 baseline here.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--report_to", default="tensorboard")
    parser.add_argument("--tracker_project_name", default="train_rgb_stc_v2")
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Validate paths, checkpoint schema, mask convention, and one clip, then exit.",
    )

    args = parser.parse_args(input_args)
    args.pretrained_model_name_or_path = os.path.abspath(
        os.path.expanduser(args.pretrained_model_name_or_path)
    )
    args.baseline_checkpoint = os.path.abspath(
        os.path.expanduser(args.baseline_checkpoint)
    )
    args.dataset_root = os.path.abspath(os.path.expanduser(args.dataset_root))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    if args.resolution <= 0 or args.resolution % 8:
        parser.error("--resolution must be positive and divisible by 8")
    if args.clip_length < 2 or args.clip_stride < 1:
        parser.error("--clip_length must be >=2 and --clip_stride must be >=1")
    if args.train_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("batch size and gradient accumulation must be positive")
    if args.max_train_steps < 1 or args.checkpointing_steps < 1:
        parser.error("training and checkpoint step counts must be positive")
    if not 0.0 <= args.shared_bg_noise_strength <= 1.0:
        parser.error("--shared_bg_noise_strength must be in [0,1]")
    if not 0.0 < args.shared_bg_mask_threshold <= 1.0:
        parser.error("--shared_bg_mask_threshold must be in (0,1]")
    if not math.isfinite(args.stc_injection_scale):
        parser.error("--stc_injection_scale must be finite")
    return args


def baseline_paths(args) -> Dict[str, Path]:
    root = Path(args.baseline_checkpoint)
    return {
        "root": root,
        "brushnet": root / "brushnet",
        "brushnet_config": root / "brushnet" / "config.json",
        "brushnet_weights": root
        / "brushnet"
        / "diffusion_pytorch_model.safetensors",
        "ip_adapter": root / "ipadapter" / "model.safetensors",
        "fusion": root / "ipadapter" / "fusion_module.safetensors",
    }


def _safe_shapes(path: Path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {
            key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()
        }


def validate_baseline_schema(args) -> Dict[str, object]:
    paths = baseline_paths(args)
    required = [
        Path(args.pretrained_model_name_or_path),
        Path(args.dataset_root) / args.train_split / "GT",
        Path(args.dataset_root) / args.train_split / "input",
        Path(args.dataset_root) / args.train_split / "mask",
        paths["brushnet_config"],
        paths["brushnet_weights"],
        paths["ip_adapter"],
        paths["fusion"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

    with paths["brushnet_config"].open("r", encoding="utf-8") as handle:
        brushnet_config = json.load(handle)
    expected_config = {
        "in_channels": 4,
        "conditioning_channels": 5,
        "cross_attention_dim": 768,
    }
    mismatches = {
        key: (brushnet_config.get(key), expected)
        for key, expected in expected_config.items()
        if brushnet_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Unexpected BrushNet config: {mismatches}")

    brush_shapes = _safe_shapes(paths["brushnet_weights"])
    if brush_shapes.get("conv_in_condition.weight") != (320, 9, 3, 3):
        raise ValueError(
            "BrushNet conv_in_condition must accept noisy(4)+condition(5) channels"
        )

    ip_shapes = _safe_shapes(paths["ip_adapter"])
    image_keys = [key for key in ip_shapes if key.startswith("image_proj_model.")]
    processor_keys = [key for key in ip_shapes if key.startswith("unet.")]
    if len(ip_shapes) != 100 or len(image_keys) != 4 or len(processor_keys) != 96:
        raise ValueError(
            "Expected the exact V8 IP schema: 100 tensors = 4 image projection + 96 processors"
        )
    if ip_shapes.get("image_proj_model.proj.weight") != (3072, 1024):
        raise ValueError("IP image projection is not 1024 -> 4x768")

    fusion_shapes = _safe_shapes(paths["fusion"])
    if len(fusion_shapes) != 10 or fusion_shapes.get("out_proj.weight") != (
        1024,
        2048,
    ):
        raise ValueError("Unexpected FGBG fusion checkpoint schema")
    return {
        "brushnet_tensors": len(brush_shapes),
        "ip_adapter_tensors": len(ip_shapes),
        "fusion_tensors": len(fusion_shapes),
        "brushnet_condition_channels": 5,
    }


def make_dataset(args, tokenizer):
    return HierarchicalV8ClipDataset(
        dataset_root=args.dataset_root,
        split=args.train_split,
        tokenizer=tokenizer,
        clip_image_processor=CLIPImageProcessor(),
        clip_length=args.clip_length,
        stride=args.clip_stride,
        resolution=args.resolution,
    )


def run_preflight(args):
    schema = validate_baseline_schema(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    dataset = make_dataset(args, tokenizer)
    sample = dataset[0]
    mask = sample["masks"]
    unique_mask = sorted(float(value) for value in torch.unique(mask))
    if any(value not in (0.0, 1.0) for value in unique_mask):
        raise ValueError(f"Dataset M_BG is not binary: {unique_mask}")
    if sample["conditioning_pixel_values"].shape != (
        args.clip_length,
        3,
        args.resolution,
        args.resolution,
    ):
        raise ValueError("Unexpected RGB sequence layout")
    if sample["masks"].shape != (
        args.clip_length,
        1,
        args.resolution,
        args.resolution,
    ):
        raise ValueError("Unexpected mask sequence layout")

    # Verify source polarity independently of the loader implementation. Raw
    # PNG is 0=BG/255=ROI, while every downstream module expects M_BG=1 and
    # M_ROI=0. A removed or duplicated 1-M must stop the run at preflight.
    _, relative_paths, _ = dataset.clips[0]
    resampling = getattr(Image, "Resampling", Image)
    raw_mask_values = set()
    raw_to_internal_mismatches = 0
    for frame_index, relative_path in enumerate(relative_paths):
        raw_path = dataset.roots["mask"] / relative_path
        with Image.open(raw_path) as raw_image:
            raw_array = np.asarray(
                raw_image.convert("L").resize(
                    (args.resolution, args.resolution),
                    resample=resampling.NEAREST,
                ),
                dtype=np.uint8,
            ).copy()
        raw_mask_values.update(int(value) for value in np.unique(raw_array))
        raw_roi = torch.from_numpy((raw_array >= 128).astype(np.float32))[None]
        expected_internal_bg = 1.0 - raw_roi
        raw_to_internal_mismatches += int(
            torch.count_nonzero(
                sample["masks"][frame_index] - expected_internal_bg
            )
        )
    if raw_to_internal_mismatches:
        raise ValueError(
            "Raw mask polarity failed: expected internal M_BG = 1 - raw_M_ROI, "
            f"mismatched pixels={raw_to_internal_mismatches}"
        )

    report = {
        "status": "ok",
        "baseline_checkpoint": args.baseline_checkpoint,
        "dataset_root": args.dataset_root,
        "clip_count": len(dataset),
        "source_frame_count": dataset.frame_count,
        "covered_frame_count": dataset.covered_frame_count,
        "branch_count": dataset.branch_count,
        "sample_video": sample["video"],
        "sample_frame_ids": sample["frame_ids"].tolist(),
        "rgb_sequence_shape": list(sample["conditioning_pixel_values"].shape),
        "mask_sequence_shape": list(sample["masks"].shape),
        "mask_values": unique_mask,
        "raw_mask_values": sorted(raw_mask_values),
        "raw_to_internal_mask_mismatches": raw_to_internal_mismatches,
        "degraded_bg_ratio": float(mask.mean()),
        "condition_mode": args.condition_mode,
        "mask_semantics": "1=degraded_BG_restore, 0=HQ_ROI_preserve",
        "loss": "global diffusion objective only",
        "shared_noise": {
            "rho": args.shared_bg_noise_strength,
            "variance_preserving": args.variance_preserving_shared_noise,
            "refresh": args.sequence_shared_noise_refresh,
        },
        **schema,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def resolve_resume_checkpoint(args) -> Optional[Path]:
    value = args.resume_from_checkpoint
    if not value:
        return None
    output = Path(args.output_dir)
    if value == "latest":
        pointer = output / "latest.json"
        if pointer.is_file():
            try:
                with pointer.open("r", encoding="utf-8") as handle:
                    checkpoint_name = str(json.load(handle)["checkpoint"])
                if re.fullmatch(r"checkpoint-\d+", checkpoint_name):
                    candidate = output / checkpoint_name
                    if _is_complete_training_checkpoint(candidate):
                        return candidate.resolve()
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        candidates = []
        if output.is_dir():
            for path in output.iterdir():
                match = re.fullmatch(r"checkpoint-(\d+)", path.name)
                if path.is_dir() and match and _is_complete_training_checkpoint(path):
                    candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError(
                f"No complete RGB-STC checkpoints below {output}"
            )
        return max(candidates, key=lambda item: item[0])[1]
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = output / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    if not _is_complete_training_checkpoint(path):
        raise ValueError(f"Incomplete RGB-STC training checkpoint: {path}")
    return path


def _is_complete_training_checkpoint(path: Path) -> bool:
    adapter_dir = path / "stc_adapter"
    state_dir = path / "accelerator_state"
    adapter_weights = (
        adapter_dir / "diffusion_pytorch_model.safetensors"
    ).is_file() or (adapter_dir / "diffusion_pytorch_model.bin").is_file()
    state_model = (state_dir / "model.safetensors").is_file() or (
        state_dir / "pytorch_model.bin"
    ).is_file()
    random_state = state_dir.is_dir() and any(
        state_dir.glob("random_states_*.pkl")
    )
    return all(
        (
            path.is_dir(),
            (path / "metadata.json").is_file(),
            (adapter_dir / "config.json").is_file(),
            adapter_weights,
            state_model,
            (state_dir / "optimizer.bin").is_file(),
            (state_dir / "scheduler.bin").is_file(),
            random_state,
        )
    )


def read_resume_metadata(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"RGB-STC checkpoint has no metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_dump(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _prune_checkpoints(output_dir: Path, limit: int):
    if limit is None or limit <= 0:
        return
    checkpoints = []
    for path in output_dir.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    for _, path in checkpoints[: max(0, len(checkpoints) - limit)]:
        shutil.rmtree(path)
        logger.info("Removed old RGB-STC checkpoint %s", path)


def save_training_checkpoint(
    accelerator,
    adapter,
    args,
    global_step: int,
    epoch: int,
    next_batch_index: int,
):
    checkpoint = Path(args.output_dir) / f"checkpoint-{global_step}"
    state_dir = checkpoint / "accelerator_state"
    if accelerator.is_main_process:
        checkpoint.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(state_dir))
    # All ranks must finish their model/optimizer/RNG writes before rank zero
    # publishes a checkpoint as resumable.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(adapter).save_pretrained(
            checkpoint / "stc_adapter", safe_serialization=True
        )
        metadata = {
            "format_version": 1,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "next_batch_index": int(next_batch_index),
            "num_processes": int(accelerator.num_processes),
            "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
            "baseline_checkpoint": args.baseline_checkpoint,
            "image_encoder_name_or_path": args.image_encoder_name_or_path,
            "dataset_root": args.dataset_root,
            "train_split": args.train_split,
            "mask_semantics": "M_BG=1 degraded background; M_BG=0 HQ ROI",
            "loss": "diffusion_mse_only",
            "condition_mode": args.condition_mode,
            "stc_hidden_channels": args.stc_hidden_channels,
            "stc_num_heads": args.stc_num_heads,
            "stc_num_layers": args.stc_num_layers,
            "stc_mlp_ratio": args.stc_mlp_ratio,
            "stc_dropout": args.stc_dropout,
            "resolution": args.resolution,
            "clip_length": args.clip_length,
            "clip_stride": args.clip_stride,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "shared_bg_noise_strength": args.shared_bg_noise_strength,
            "shared_bg_mask_threshold": args.shared_bg_mask_threshold,
            "variance_preserving_shared_noise": args.variance_preserving_shared_noise,
            "sequence_shared_noise_refresh": args.sequence_shared_noise_refresh,
            "independent_frame_timesteps": args.independent_frame_timesteps,
            "stc_injection_scale": args.stc_injection_scale,
            "fusion_scale": args.fusion_scale,
            "seed": args.seed,
            "mixed_precision": args.mixed_precision,
            "gradient_checkpointing": args.gradient_checkpointing,
            "allow_tf32": args.allow_tf32,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler,
            "lr_warmup_steps": args.lr_warmup_steps,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_weight_decay": args.adam_weight_decay,
            "adam_epsilon": args.adam_epsilon,
            "use_8bit_adam": args.use_8bit_adam,
            "max_grad_norm": args.max_grad_norm,
            "max_train_steps": args.max_train_steps,
        }
        _json_dump(checkpoint / "metadata.json", metadata)
        _json_dump(
            Path(args.output_dir) / "latest.json",
            {"checkpoint": checkpoint.name, "global_step": int(global_step)},
        )
        _prune_checkpoints(
            Path(args.output_dir), int(args.checkpoints_total_limit)
        )
    accelerator.wait_for_everyone()


def main(args):
    validate_baseline_schema(args)
    resume_path = resolve_resume_checkpoint(args)
    resume_metadata = read_resume_metadata(resume_path)

    output_path = Path(args.output_dir).resolve()
    existing_stage_outputs = []
    if output_path.is_dir():
        existing_stage_outputs = [
            path
            for path in output_path.iterdir()
            if re.fullmatch(r"checkpoint-\d+", path.name)
            or path.name == "stc_adapter"
        ]
    if resume_path is None:
        if existing_stage_outputs:
            raise ValueError(
                "Output directory already contains RGB-STC results. Use "
                "--resume_from_checkpoint latest or choose a new --output_dir: "
                f"{args.output_dir}"
            )
    else:
        resume_step = int(resume_metadata["global_step"])
        if resume_path.parent.resolve() != output_path:
            if existing_stage_outputs:
                raise ValueError(
                    "External checkpoint resume requires an empty/new output "
                    f"directory, but results already exist below {output_path}"
                )
        else:
            newer_complete = []
            for path in existing_stage_outputs:
                match = re.fullmatch(r"checkpoint-(\d+)", path.name)
                if (
                    match
                    and int(match.group(1)) > resume_step
                    and _is_complete_training_checkpoint(path)
                ):
                    newer_complete.append(path.name)
            if newer_complete:
                raise ValueError(
                    "Refusing to rewind an output lineage past newer complete "
                    f"checkpoints: {sorted(newer_complete)}"
                )

    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=str(Path(args.output_dir) / args.logging_dir),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Respect an explicit --mixed_precision no instead of allowing an
        # Accelerate config/environment default to override it through None.
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,
    )
    # Model/data ordering is controlled separately; stochastic VAE/noise/time
    # streams should be distinct across DDP ranks.
    set_seed(args.seed, device_specific=True)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if resume_metadata:
        if Path(str(resume_metadata["baseline_checkpoint"])).resolve() != Path(
            args.baseline_checkpoint
        ).resolve():
            raise ValueError("Resume checkpoint was trained against a different V8 baseline")
        if int(resume_metadata["num_processes"]) != accelerator.num_processes:
            raise ValueError("Exact mid-epoch resume requires the same number of processes")
        resume_contract = {
            "pretrained_model_name_or_path": str(
                Path(args.pretrained_model_name_or_path).resolve()
            ),
            "image_encoder_name_or_path": args.image_encoder_name_or_path,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "train_split": args.train_split,
            "condition_mode": args.condition_mode,
            "stc_hidden_channels": args.stc_hidden_channels,
            "stc_num_heads": args.stc_num_heads,
            "stc_num_layers": args.stc_num_layers,
            "stc_mlp_ratio": args.stc_mlp_ratio,
            "stc_dropout": args.stc_dropout,
            "resolution": args.resolution,
            "clip_length": args.clip_length,
            "clip_stride": args.clip_stride,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "shared_bg_noise_strength": args.shared_bg_noise_strength,
            "shared_bg_mask_threshold": args.shared_bg_mask_threshold,
            "variance_preserving_shared_noise": args.variance_preserving_shared_noise,
            "sequence_shared_noise_refresh": args.sequence_shared_noise_refresh,
            "independent_frame_timesteps": args.independent_frame_timesteps,
            "stc_injection_scale": args.stc_injection_scale,
            "fusion_scale": args.fusion_scale,
            "seed": args.seed,
            "mixed_precision": args.mixed_precision,
            "gradient_checkpointing": args.gradient_checkpointing,
            "allow_tf32": args.allow_tf32,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler,
            "lr_warmup_steps": args.lr_warmup_steps,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_weight_decay": args.adam_weight_decay,
            "adam_epsilon": args.adam_epsilon,
            "use_8bit_adam": args.use_8bit_adam,
            "max_grad_norm": args.max_grad_norm,
            "max_train_steps": args.max_train_steps,
        }
        for key, current_value in resume_contract.items():
            saved_value = resume_metadata.get(key)
            if key in {"dataset_root", "pretrained_model_name_or_path"} and saved_value is not None:
                saved_value = str(Path(str(saved_value)).resolve())
            if saved_value != current_value:
                raise ValueError(
                    f"Resume contract mismatch for {key}: "
                    f"checkpoint={saved_value!r}, current={current_value!r}"
                )

    paths = baseline_paths(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    brushnet = BrushNetModel.from_pretrained(str(paths["brushnet"]))
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder_name_or_path
    )
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    ip_report = install_and_load_ip_adapter(
        unet, image_proj_model, paths["ip_adapter"]
    )
    fusion_module = load_fusion_module(
        paths["fusion"], embed_dim=image_encoder.config.projection_dim
    )

    if resume_path is not None:
        adapter = RGBSTCConditionAdapter.from_pretrained(
            resume_path / "stc_adapter"
        )
        if adapter.config.condition_mode != args.condition_mode:
            raise ValueError("Resume checkpoint condition_mode mismatch")
    else:
        adapter = RGBSTCConditionAdapter(
            hidden_channels=args.stc_hidden_channels,
            num_heads=args.stc_num_heads,
            num_layers=args.stc_num_layers,
            mlp_ratio=args.stc_mlp_ratio,
            dropout=args.stc_dropout,
            condition_mode=args.condition_mode,
        )

    for module in (
        vae,
        text_encoder,
        unet,
        brushnet,
        image_encoder,
        image_proj_model,
        fusion_module,
    ):
        module.requires_grad_(False)
    adapter.requires_grad_(True)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        brushnet.enable_gradient_checkpointing()

    dataset = make_dataset(args, tokenizer)
    sampler = EpochRandomSampler(dataset, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        collate_fn=collate_shared_noise_clips,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )

    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ImportError("Install bitsandbytes to use --use_8bit_adam") from error
        optimizer_class = bnb.optim.AdamW8bit
    optimizer = optimizer_class(
        adapter.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    adapter, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        adapter, optimizer, dataloader, lr_scheduler
    )
    # Accelerator shards the dataloader across ranks. Epoch math must use the
    # prepared per-rank length; otherwise multi-GPU runs stop before reaching
    # max_train_steps.
    updates_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    estimated_train_epochs = math.ceil(
        max(0, args.max_train_steps) / updates_per_epoch
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # Match the original V8 mixed-precision contract. Only inference-only
    # encoders were permanently cast; BrushNet, U-Net/IP, projection, and
    # fusion retained FP32 weights and ran under autocast.
    for module in (vae, text_encoder, image_encoder):
        module.to(accelerator.device, dtype=weight_dtype)
    for module in (unet, brushnet, image_proj_model, fusion_module):
        module.to(accelerator.device, dtype=torch.float32)
    vae.eval()
    text_encoder.eval()
    image_encoder.eval()
    image_proj_model.eval()
    fusion_module.eval()
    # Gradient checkpointing is active only in training mode. Parameters remain
    # frozen, but the graph must pass through these modules back to RGB-STC.
    unet.train()
    brushnet.train()

    frozen_modules = {
        "vae": vae,
        "text_encoder": text_encoder,
        "unet": unet,
        "brushnet": brushnet,
        "image_encoder": image_encoder,
        "image_proj_model": image_proj_model,
        "fusion_module": fusion_module,
    }
    accidentally_trainable = [
        f"{module_name}.{parameter_name}"
        for module_name, module in frozen_modules.items()
        for parameter_name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if accidentally_trainable:
        raise RuntimeError(
            "Frozen V8 parameters became trainable: "
            + ", ".join(accidentally_trainable[:5])
        )
    trainable_count = sum(
        parameter.numel()
        for parameter in accelerator.unwrap_model(adapter).parameters()
        if parameter.requires_grad
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    adapter_parameter_ids = {
        id(parameter)
        for parameter in adapter.parameters()
        if parameter.requires_grad
    }
    if optimizer_parameter_ids != adapter_parameter_ids:
        raise RuntimeError("Optimizer must contain exactly the RGB-STC parameters")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process and args.report_to != "none":
        tracker_config = {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        accelerator.init_trackers(
            args.tracker_project_name, config=tracker_config
        )

    global_step = int(resume_metadata.get("global_step", 0))
    first_epoch = int(resume_metadata.get("epoch", 0))
    resume_batch_index = int(resume_metadata.get("next_batch_index", 0))
    if resume_path is not None:
        accelerator.load_state(str(resume_path / "accelerator_state"))
        logger.info("Resumed RGB-STC state from %s at step %d", resume_path, global_step)

    total_clip_batch = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** RGB-STC phase-1 training *****")
    logger.info("Baseline checkpoint: %s", args.baseline_checkpoint)
    logger.info("IP mapping: %s", ip_report)
    logger.info("Dataset clips: %d; covered frames: %d/%d", len(dataset), dataset.covered_frame_count, dataset.frame_count)
    logger.info("Clip layout: T=%d stride=%d effective clips/update=%d", args.clip_length, args.clip_stride, total_clip_batch)
    logger.info("Mask: M_BG=1 degraded BG; M_BG=0 HQ ROI")
    logger.info("Condition: full degraded RGB + M_BG sequence -> BG-gated delta_z")
    logger.info("Loss: standard diffusion objective only (no flow, no L_bg)")
    logger.info("Trainable RGB-STC parameters: %s", f"{trainable_count:,}")
    logger.info(
        "Estimated epochs without AMP-skipped updates: %d",
        estimated_train_epochs,
    )

    progress = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="RGB-STC steps",
    )
    optimizer.zero_grad(set_to_none=True)
    fresh_zero_identity_checked = resume_path is not None
    stop_training = global_step >= args.max_train_steps

    epoch = first_epoch
    while not stop_training:
        sampler.set_epoch(epoch)
        # Newer Accelerate's DataLoaderShard calls set_epoch(self.iteration)
        # inside __iter__. Set the prepared loader as well, otherwise a resumed
        # epoch can silently fall back to the epoch-0 shuffle before skipping.
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)
        adapter.train()
        for batch_index, batch in enumerate(dataloader):
            if epoch == first_epoch and batch_index < resume_batch_index:
                continue
            with accelerator.accumulate(adapter):
                num_clips = int(batch["clip_batch_size"])
                num_frames = int(batch["num_frames"])
                if num_frames != args.clip_length:
                    raise ValueError(
                        f"Batch T={num_frames} differs from configured T={args.clip_length}"
                    )
                rgb_sequence = batch["conditioning_pixel_values"].to(
                    accelerator.device, non_blocking=True
                ).reshape(
                    num_clips,
                    num_frames,
                    3,
                    args.resolution,
                    args.resolution,
                )
                bg_mask_sequence = batch["masks"].to(
                    accelerator.device, non_blocking=True
                ).reshape(
                    num_clips,
                    num_frames,
                    1,
                    args.resolution,
                    args.resolution,
                )

                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(
                            accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    base_condition_latents = vae.encode(
                        batch["conditioning_pixel_values"].to(
                            accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                    ).latent_dist.sample()
                    base_condition_latents = (
                        base_condition_latents * vae.config.scaling_factor
                    )
                    latent_bg_mask = F.interpolate(
                        batch["masks"].to(
                            accelerator.device, dtype=latents.dtype, non_blocking=True
                        ),
                        size=latents.shape[-2:],
                        mode="nearest",
                    )
                    latent_bg_mask = (
                        latent_bg_mask >= args.shared_bg_mask_threshold
                    ).to(latents.dtype)
                    latent_clips = latents.reshape(
                        num_clips, num_frames, *latents.shape[1:]
                    )
                    mask_clips = latent_bg_mask.reshape(
                        num_clips,
                        num_frames,
                        1,
                        *latent_bg_mask.shape[-2:],
                    )
                    refresh_index = (
                        epoch
                        if args.sequence_shared_noise_refresh == "epoch"
                        else 0
                    )
                    sequence_noise = make_sequence_shared_background_noise(
                        latent_template=latent_clips,
                        sequence_keys=batch["videos"],
                        base_seed=args.seed,
                        refresh_index=refresh_index,
                    )
                    noise = sample_shared_background_noise(
                        latent_template=latent_clips,
                        bg_mask=mask_clips,
                        strength=args.shared_bg_noise_strength,
                        variance_preserving=args.variance_preserving_shared_noise,
                        shared_bg_noise=sequence_noise,
                    ).reshape_as(latents)
                    timesteps = sample_clip_timesteps(
                        num_clips=num_clips,
                        clip_length=num_frames,
                        num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                        device=latents.device,
                        share_across_clip=not args.independent_frame_timesteps,
                    )
                    noisy_latents = noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )
                    brushnet_text, unet_context = build_frozen_v8_context(
                        batch=batch,
                        image_encoder=image_encoder,
                        fusion_module=fusion_module,
                        image_proj_model=image_proj_model,
                        text_encoder=text_encoder,
                        fusion_scale=args.fusion_scale,
                        device=accelerator.device,
                        dtype=weight_dtype,
                        autocast_context=accelerator.autocast,
                    )

                with accelerator.autocast():
                    brushnet_condition, stc_output, augmented_condition = (
                        augment_brushnet_condition(
                            adapter=adapter,
                            base_condition_latents=base_condition_latents,
                            rgb_sequence=rgb_sequence,
                            bg_mask_sequence=bg_mask_sequence,
                            injection_scale=args.stc_injection_scale,
                        )
                    )
                    if not fresh_zero_identity_checked:
                        if torch.count_nonzero(stc_output.delta_bg).item() != 0:
                            raise RuntimeError(
                                "Fresh ZeroConv must be an exact zero correction"
                            )
                        if not torch.equal(
                            augmented_condition.flatten(0, 1),
                            base_condition_latents,
                        ):
                            raise RuntimeError(
                                "Fresh RGB-STC must exactly reproduce baseline condition"
                            )
                        fresh_zero_identity_checked = True
                    model_prediction = frozen_v8_predict(
                        brushnet=brushnet,
                        unet=unet,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        brushnet_condition=brushnet_condition,
                        brushnet_text_hidden_states=brushnet_text,
                        unet_hidden_states=unet_context,
                    )
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(
                            latents, noise, timesteps
                        )
                    else:
                        raise ValueError(
                            "Unknown prediction type: "
                            f"{noise_scheduler.config.prediction_type}"
                        )
                    loss = F.mse_loss(
                        model_prediction.float(), target.float(), reduction="mean"
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        adapter.parameters(), args.max_grad_norm
                    )
                # Accelerate 0.21 does not set its overflow flag on the first
                # GradScaler overflow because it has no previous cached scale.
                # Compare the scaler values ourselves so a skipped AMP update
                # never advances the LR schedule or global_step.
                scale_before = None
                if accelerator.sync_gradients and accelerator.scaler is not None:
                    scale_before = float(accelerator.scaler.get_scale())
                optimizer.step()
                if scale_before is not None:
                    scale_after = float(accelerator.scaler.get_scale())
                    optimizer_step_was_skipped = scale_after < scale_before
                else:
                    optimizer_step_was_skipped = (
                        accelerator.optimizer_step_was_skipped
                    )
                if not optimizer_step_was_skipped:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients and optimizer_step_was_skipped:
                logger.warning(
                    "AMP skipped an RGB-STC optimizer update; global_step is unchanged"
                )
            if accelerator.sync_gradients and not optimizer_step_was_skipped:
                global_step += 1
                progress.update(1)
                if global_step % args.logging_steps == 0:
                    metric_values = torch.stack(
                        (
                            loss.detach().float(),
                            stc_output.delta_bg.detach().float().abs().mean(),
                            stc_output.latent_bg_mask.detach().float().mean(),
                        )
                    )
                    metric_values = accelerator.gather(
                        metric_values.unsqueeze(0)
                    ).mean(dim=0)
                    logs = {
                        "train/loss": float(metric_values[0]),
                        "train/lr": float(lr_scheduler.get_last_lr()[0]),
                        "train/delta_abs_mean": float(metric_values[1]),
                        "train/degraded_bg_ratio": float(metric_values[2]),
                    }
                    if args.report_to != "none":
                        accelerator.log(logs, step=global_step)
                    progress.set_postfix(
                        loss=f"{logs['train/loss']:.4f}",
                        delta=f"{logs['train/delta_abs_mean']:.5f}",
                    )
                if global_step % args.checkpointing_steps == 0:
                    save_training_checkpoint(
                        accelerator,
                        adapter,
                        args,
                        global_step=global_step,
                        epoch=epoch,
                        next_batch_index=batch_index + 1,
                    )
                if global_step >= args.max_train_steps:
                    stop_training = True
                    break
        resume_batch_index = 0
        epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = Path(args.output_dir) / "stc_adapter"
        accelerator.unwrap_model(adapter).save_pretrained(
            final_dir, safe_serialization=True
        )
        _json_dump(
            Path(args.output_dir) / "final_metadata.json",
            {
                "global_step": int(global_step),
                "baseline_checkpoint": args.baseline_checkpoint,
                "mask_semantics": "M_BG=1 degraded background; M_BG=0 HQ ROI",
                "condition_mode": args.condition_mode,
                "loss": "diffusion_mse_only",
                "stc_injection_scale": args.stc_injection_scale,
            },
        )
    accelerator.wait_for_everyone()
    if args.report_to != "none":
        accelerator.end_training()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.preflight_only:
        run_preflight(parsed)
    else:
        main(parsed)
