#!/usr/bin/env python
"""Train RGB-STC with diffusion restoration and clean-teacher flow losses.

Only ``RGBSTCFlowAdapter`` is trainable. The checkpoint-2000 BrushNet, U-Net,
IP-Adapter, FGBG fusion, VAE, text encoder, and image encoder stay frozen.
Shared-noise sampling is unchanged from V8. Optical flow is an auxiliary
training target and is never injected into diffusion or required at inference.
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
from torch.utils.data import DataLoader
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
from STC_encoder_v2_rgb.rgb_stc_adapter import RGBSTCConditionAdapter
from STC_encoder_v2_rgb.train_rgb_stc_shared_noise import (
    EpochRandomSampler,
    baseline_paths,
    validate_baseline_schema,
)
from STC_encoder_v3_rgb_flow.flow_supervision import compute_teacher_flow_loss
from STC_encoder_v3_rgb_flow.rgb_stc_flow_adapter import (
    RGBSTCFlowAdapter,
    augment_brushnet_condition,
)
from STC_encoder_v3_rgb_flow.teacher_flow_data import (
    TeacherFlowV8ClipDataset,
    collate_teacher_flow_clips,
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
DEFAULT_TEACHER = DEFAULT_DATASET / "teacher_flows_512x512"
DEFAULT_BASELINE = (
    REPO_ROOT
    / "experiments"
    / "train_sharedNoise_sameBG_0.9"
    / "checkpoint-2000"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments" / "train_rgb_stc_v3_flow_sharedNoise_0.9"
)
DEFAULT_IMAGE_ENCODER = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# Backward-compatible hooks used by the V4 flow-aligned wrapper. Defaults keep
# every V3 training/checkpoint behavior unchanged.
EXPERIMENT_NAME = "rgb_stc_v3_diffusion_plus_flow"
FLOW_INFERENCE_DEPENDENCY = False
INFERENCE_COMPONENT = "stc_adapter"
TRAINING_LOG_TITLE = "RGB-STC v3: L_diff + L_flow"


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained_model_name_or_path", default=str(DEFAULT_BASE_MODEL)
    )
    parser.add_argument("--baseline_checkpoint", default=str(DEFAULT_BASELINE))
    parser.add_argument(
        "--image_encoder_name_or_path", default=DEFAULT_IMAGE_ENCODER
    )
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET))
    parser.add_argument("--teacher_flow_root", default=str(DEFAULT_TEACHER))
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
    parser.add_argument(
        "--flow_max_displacement",
        type=float,
        nargs=2,
        metavar=("MAX_DX", "MAX_DY"),
        default=(8.0, 8.0),
        help="Maximum predicted displacement in latent pixels.",
    )
    parser.add_argument("--flow_loss_weight", type=float, default=0.01)
    parser.add_argument("--flow_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument(
        "--flow_region",
        choices=("bg", "all"),
        default="bg",
        help="Default bg supervises flow only where restoration is required.",
    )
    parser.add_argument(
        "--init_stc_adapter",
        default=None,
        help="Optional v2 stc_adapter folder/checkpoint for warm start.",
    )

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
    parser.add_argument(
        "--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16"
    )
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="A v3 checkpoint path or 'latest'; never the V8 baseline.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--report_to", default="tensorboard")
    parser.add_argument("--tracker_project_name", default="train_rgb_stc_v3_flow")
    parser.add_argument("--preflight_only", action="store_true")

    args = parser.parse_args(input_args)
    for name in (
        "pretrained_model_name_or_path",
        "baseline_checkpoint",
        "dataset_root",
        "teacher_flow_root",
        "output_dir",
    ):
        setattr(args, name, os.path.abspath(os.path.expanduser(getattr(args, name))))
    if args.init_stc_adapter:
        args.init_stc_adapter = os.path.abspath(
            os.path.expanduser(args.init_stc_adapter)
        )

    if args.resolution <= 0 or args.resolution % 8:
        parser.error("--resolution must be positive and divisible by 8")
    if args.clip_length < 2 or args.clip_stride < 1:
        parser.error("--clip_length must be >=2 and --clip_stride >=1")
    if args.train_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("batch size and gradient accumulation must be positive")
    if args.max_train_steps < 1 or args.checkpointing_steps < 1:
        parser.error("training/checkpoint step counts must be positive")
    if not 0.0 <= args.shared_bg_noise_strength <= 1.0:
        parser.error("--shared_bg_noise_strength must be in [0,1]")
    if not 0.0 < args.shared_bg_mask_threshold <= 1.0:
        parser.error("--shared_bg_mask_threshold must be in (0,1]")
    if args.flow_loss_weight <= 0.0 or not math.isfinite(args.flow_loss_weight):
        parser.error("--flow_loss_weight must be finite and positive")
    if args.flow_charbonnier_eps <= 0.0:
        parser.error("--flow_charbonnier_eps must be positive")
    if any(value <= 0.0 for value in args.flow_max_displacement):
        parser.error("--flow_max_displacement values must be positive")
    if args.resume_from_checkpoint and args.init_stc_adapter:
        parser.error("Resume and v2 warm-start are mutually exclusive")
    return args


def make_dataset(args, tokenizer):
    return TeacherFlowV8ClipDataset(
        dataset_root=args.dataset_root,
        split=args.train_split,
        tokenizer=tokenizer,
        clip_image_processor=CLIPImageProcessor(),
        clip_length=args.clip_length,
        stride=args.clip_stride,
        resolution=args.resolution,
        teacher_flow_root=args.teacher_flow_root,
    )


def resolve_stc_component(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    nested = path / "stc_adapter"
    if nested.is_dir():
        path = nested
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"No STC config below {path}")
    if not any(
        (path / name).is_file()
        for name in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")
    ):
        raise FileNotFoundError(f"No STC weights below {path}")
    return path


def run_preflight(args):
    schema = validate_baseline_schema(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )
    dataset = make_dataset(args, tokenizer)
    sample = dataset[0]
    mask = sample["masks"]
    mask_values = sorted(float(value) for value in torch.unique(mask))
    if any(value not in (0.0, 1.0) for value in mask_values):
        raise ValueError(f"Internal M_BG must be binary: {mask_values}")

    # Independently verify raw 0=BG/255=ROI -> internal 1=BG/0=ROI.
    _, relative_paths, _ = dataset.clips[0]
    resampling = getattr(Image, "Resampling", Image)
    raw_values = set()
    mismatches = 0
    for frame_index, relative_path in enumerate(relative_paths):
        with Image.open(dataset.roots["mask"] / relative_path) as image:
            raw = np.asarray(
                image.convert("L").resize(
                    (args.resolution, args.resolution), resample=resampling.NEAREST
                ),
                dtype=np.uint8,
            ).copy()
        raw_values.update(int(value) for value in np.unique(raw))
        expected_bg = 1.0 - torch.from_numpy((raw >= 128).astype(np.float32))[None]
        mismatches += int(torch.count_nonzero(mask[frame_index] - expected_bg))
    if mismatches:
        raise ValueError(f"Raw/internal mask polarity mismatch: {mismatches} pixels")

    pairs = args.clip_length - 1
    latent_size = (args.resolution // 8, args.resolution // 8)
    zero_flow = torch.zeros(1, pairs, 2, *latent_size)
    flow_report = compute_teacher_flow_loss(
        zero_flow,
        zero_flow.clone(),
        sample["teacher_flow_forward"].unsqueeze(0),
        sample["teacher_flow_backward"].unsqueeze(0),
        mask.unsqueeze(0),
        sample["teacher_valid_forward"].unsqueeze(0),
        sample["teacher_valid_backward"].unsqueeze(0),
        region=args.flow_region,
        charbonnier_eps=args.flow_charbonnier_eps,
    )
    init_component = None
    if args.init_stc_adapter:
        init_component = str(resolve_stc_component(args.init_stc_adapter))

    report = {
        "status": "ok",
        "baseline_checkpoint": args.baseline_checkpoint,
        "dataset_root": args.dataset_root,
        "teacher_flow_root": args.teacher_flow_root,
        "teacher_metadata": dataset.teacher_metadata,
        "required_teacher_pairs": dataset.required_teacher_pair_count,
        "dropped_cross_segment_clips": dataset.dropped_cross_segment_clips,
        "clip_count": len(dataset),
        "source_frames": dataset.frame_count,
        "covered_frames": dataset.covered_frame_count,
        "sample_video": sample["video"],
        "sample_frame_ids": sample["frame_ids"].tolist(),
        "rgb_shape": list(sample["conditioning_pixel_values"].shape),
        "internal_mask_shape": list(mask.shape),
        "internal_mask_values": mask_values,
        "raw_mask_values": sorted(raw_values),
        "raw_to_internal_mismatches": mismatches,
        "mask_semantics": "raw:0=degraded_BG,1=HQ_ROI; internal:M_BG=1",
        "teacher_flow_shape": list(sample["teacher_flow_forward"].shape),
        "flow_prediction_shape": list(zero_flow.shape),
        "flow_region": args.flow_region,
        "zero_flow_teacher_loss": float(flow_report.loss),
        "teacher_valid_forward_ratio": float(flow_report.valid_forward_ratio),
        "teacher_valid_backward_ratio": float(flow_report.valid_backward_ratio),
        "loss": f"L_diff + {args.flow_loss_weight} * L_flow",
        "init_stc_adapter": init_component,
        "shared_noise": {
            "rho": args.shared_bg_noise_strength,
            "variance_preserving": args.variance_preserving_shared_noise,
            "refresh": args.sequence_shared_noise_refresh,
        },
        **schema,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _model_weights_exist(path: Path) -> bool:
    return any(
        (path / name).is_file()
        for name in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")
    )


def _is_complete_checkpoint(path: Path) -> bool:
    model_dir = path / "stc_flow_model"
    state_dir = path / "accelerator_state"
    state_model = any(
        (state_dir / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    random_state = state_dir.is_dir() and any(state_dir.glob("random_states_*.pkl"))
    return all(
        (
            path.is_dir(),
            (path / "metadata.json").is_file(),
            (model_dir / "config.json").is_file(),
            _model_weights_exist(model_dir),
            state_model,
            (state_dir / "optimizer.bin").is_file(),
            (state_dir / "scheduler.bin").is_file(),
            random_state,
        )
    )


def resolve_resume_checkpoint(args) -> Optional[Path]:
    value = args.resume_from_checkpoint
    if not value:
        return None
    output = Path(args.output_dir)
    if value == "latest":
        candidates = []
        if output.is_dir():
            for path in output.iterdir():
                match = re.fullmatch(r"checkpoint-(\d+)", path.name)
                if match and _is_complete_checkpoint(path):
                    candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError(f"No complete v3 checkpoints below {output}")
        return max(candidates, key=lambda item: item[0])[1].resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = output / path
    path = path.resolve()
    if not _is_complete_checkpoint(path):
        raise ValueError(f"Incomplete v3 training checkpoint: {path}")
    return path


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
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
        logger.info("Removed old v3 checkpoint %s", path)


def checkpoint_metadata(args, accelerator, global_step, epoch, next_batch_index):
    return {
        "format_version": 1,
        "experiment": EXPERIMENT_NAME,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "num_processes": int(accelerator.num_processes),
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "baseline_checkpoint": args.baseline_checkpoint,
        "image_encoder_name_or_path": args.image_encoder_name_or_path,
        "dataset_root": args.dataset_root,
        "teacher_flow_root": args.teacher_flow_root,
        "train_split": args.train_split,
        "mask_semantics": "raw BG=0/ROI=1; internal M_BG=1/M_ROI=0",
        "loss": "L_diff + flow_loss_weight * bidirectional_teacher_Charbonnier",
        "flow_loss_weight": args.flow_loss_weight,
        "flow_charbonnier_eps": args.flow_charbonnier_eps,
        "flow_region": args.flow_region,
        "flow_max_displacement": list(args.flow_max_displacement),
        "flow_inference_dependency": FLOW_INFERENCE_DEPENDENCY,
        "inference_component": INFERENCE_COMPONENT,
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
        "init_stc_adapter": args.init_stc_adapter,
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


def save_training_checkpoint(
    accelerator, model, args, global_step, epoch, next_batch_index
):
    checkpoint = Path(args.output_dir) / f"checkpoint-{global_step}"
    state_dir = checkpoint / "accelerator_state"
    if accelerator.is_main_process:
        checkpoint.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(state_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        bare = accelerator.unwrap_model(model)
        bare.save_pretrained(
            checkpoint / "stc_flow_model", safe_serialization=True
        )
        # This smaller component is the only model needed for inference.
        bare.stc_adapter.save_pretrained(
            checkpoint / "stc_adapter", safe_serialization=True
        )
        _json_dump(
            checkpoint / "metadata.json",
            checkpoint_metadata(
                args, accelerator, global_step, epoch, next_batch_index
            ),
        )
        _json_dump(
            Path(args.output_dir) / "latest.json",
            {"checkpoint": checkpoint.name, "global_step": int(global_step)},
        )
        _prune_checkpoints(Path(args.output_dir), args.checkpoints_total_limit)
    accelerator.wait_for_everyone()


def _resume_contract(args) -> Dict:
    return {
        "experiment": EXPERIMENT_NAME,
        "pretrained_model_name_or_path": str(
            Path(args.pretrained_model_name_or_path).resolve()
        ),
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
        "image_encoder_name_or_path": args.image_encoder_name_or_path,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "teacher_flow_root": str(Path(args.teacher_flow_root).resolve()),
        "train_split": args.train_split,
        "condition_mode": args.condition_mode,
        "stc_hidden_channels": args.stc_hidden_channels,
        "stc_num_heads": args.stc_num_heads,
        "stc_num_layers": args.stc_num_layers,
        "stc_mlp_ratio": args.stc_mlp_ratio,
        "stc_dropout": args.stc_dropout,
        "flow_max_displacement": list(args.flow_max_displacement),
        "flow_loss_weight": args.flow_loss_weight,
        "flow_charbonnier_eps": args.flow_charbonnier_eps,
        "flow_region": args.flow_region,
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
    }


def main(args):
    validate_baseline_schema(args)
    resume_path = resolve_resume_checkpoint(args)
    resume_metadata = (
        _read_json(resume_path / "metadata.json") if resume_path else {}
    )

    output_path = Path(args.output_dir).resolve()
    existing_results = []
    if output_path.is_dir():
        existing_results = [
            path
            for path in output_path.iterdir()
            if re.fullmatch(r"checkpoint-\d+", path.name)
            or path.name in {"stc_flow_model", "stc_adapter"}
        ]
    if resume_path is None and existing_results:
        raise ValueError(
            "Output already contains v3 results; use --resume_from_checkpoint "
            f"latest or a new --output_dir: {args.output_dir}"
        )

    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=str(Path(args.output_dir) / args.logging_dir),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,
    )
    set_seed(args.seed, device_specific=True)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if resume_metadata:
        if int(resume_metadata.get("num_processes", -1)) != accelerator.num_processes:
            raise ValueError("Exact resume requires the same number of processes")
        for key, current in _resume_contract(args).items():
            saved = resume_metadata.get(key)
            if key in {
                "pretrained_model_name_or_path",
                "baseline_checkpoint",
                "dataset_root",
                "teacher_flow_root",
            } and saved is not None:
                saved = str(Path(str(saved)).resolve())
            if saved != current:
                raise ValueError(
                    f"Resume contract mismatch {key}: saved={saved!r}, current={current!r}"
                )

    paths = baseline_paths(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
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

    if resume_path:
        model = RGBSTCFlowAdapter.from_pretrained(
            resume_path / "stc_flow_model"
        )
    else:
        model = RGBSTCFlowAdapter(
            hidden_channels=args.stc_hidden_channels,
            num_heads=args.stc_num_heads,
            num_layers=args.stc_num_layers,
            mlp_ratio=args.stc_mlp_ratio,
            dropout=args.stc_dropout,
            condition_mode=args.condition_mode,
            flow_max_displacement=tuple(args.flow_max_displacement),
        )
        if args.init_stc_adapter:
            component = resolve_stc_component(args.init_stc_adapter)
            source = RGBSTCConditionAdapter.from_pretrained(component)
            if source.config.condition_mode != args.condition_mode:
                raise ValueError("Warm-start condition_mode mismatch")
            model.stc_adapter.load_state_dict(source.state_dict(), strict=True)
            logger.info("Warm-started RGB-STC from %s", component)

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
    model.requires_grad_(True)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        brushnet.enable_gradient_checkpointing()

    dataset = make_dataset(args, tokenizer)
    sampler = EpochRandomSampler(dataset, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        collate_fn=collate_teacher_flow_clips,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )

    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ImportError("Install bitsandbytes for --use_8bit_adam") from error
        optimizer_class = bnb.optim.AdamW8bit
    optimizer = optimizer_class(
        model.parameters(),
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
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    updates_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    estimated_epochs = math.ceil(args.max_train_steps / updates_per_epoch)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    for module in (vae, text_encoder, image_encoder):
        module.to(accelerator.device, dtype=weight_dtype)
    for module in (unet, brushnet, image_proj_model, fusion_module):
        module.to(accelerator.device, dtype=torch.float32)
    vae.eval()
    text_encoder.eval()
    image_encoder.eval()
    image_proj_model.eval()
    fusion_module.eval()
    # Required for gradient-checkpointed frozen paths to pass gradients to STC.
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
            "Frozen baseline became trainable: " + ", ".join(accidentally_trainable[:5])
        )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    model_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if optimizer_ids != model_ids:
        raise RuntimeError("Optimizer must contain exactly RGB-STC + flow-head parameters")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process and args.report_to != "none":
        tracker_config = {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    global_step = int(resume_metadata.get("global_step", 0))
    first_epoch = int(resume_metadata.get("epoch", 0))
    resume_batch_index = int(resume_metadata.get("next_batch_index", 0))
    if resume_path:
        accelerator.load_state(str(resume_path / "accelerator_state"))
        logger.info("Resumed v3 state from %s at step %d", resume_path, global_step)

    trainable_count = sum(
        parameter.numel()
        for parameter in accelerator.unwrap_model(model).parameters()
        if parameter.requires_grad
    )
    logger.info("***** %s *****", TRAINING_LOG_TITLE)
    logger.info("Frozen baseline: %s", args.baseline_checkpoint)
    logger.info("IP mapping: %s", ip_report)
    logger.info(
        "Dataset: clips=%d teacher_pairs=%d covered=%d/%d",
        len(dataset),
        dataset.required_teacher_pair_count,
        dataset.covered_frame_count,
        dataset.frame_count,
    )
    logger.info(
        "Dropped clips crossing manifest source segments: %d",
        dataset.dropped_cross_segment_clips,
    )
    logger.info(
        "Mask raw BG=0/ROI=1 -> internal M_BG=1/M_ROI=0; flow_region=%s",
        args.flow_region,
    )
    logger.info(
        "Loss: L_total=L_diff + %.6g*L_flow; flow required at inference=%s",
        args.flow_loss_weight,
        FLOW_INFERENCE_DEPENDENCY,
    )
    logger.info("Trainable parameters: %s", f"{trainable_count:,}")
    logger.info("Estimated epochs: %d", estimated_epochs)

    progress = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="RGB-STC-flow steps",
    )
    optimizer.zero_grad(set_to_none=True)
    fresh_zero_identity_checked = bool(resume_path or args.init_stc_adapter)
    stop_training = global_step >= args.max_train_steps
    epoch = first_epoch

    while not stop_training:
        sampler.set_epoch(epoch)
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)
        model.train()
        for batch_index, batch in enumerate(dataloader):
            if epoch == first_epoch and batch_index < resume_batch_index:
                continue
            with accelerator.accumulate(model):
                num_clips = int(batch["clip_batch_size"])
                num_frames = int(batch["num_frames"])
                if num_frames != args.clip_length:
                    raise ValueError(
                        f"Batch T={num_frames}, configured T={args.clip_length}"
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
                            accelerator.device,
                            dtype=weight_dtype,
                            non_blocking=True,
                        )
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    base_condition_latents = vae.encode(
                        batch["conditioning_pixel_values"].to(
                            accelerator.device,
                            dtype=weight_dtype,
                            non_blocking=True,
                        )
                    ).latent_dist.sample()
                    base_condition_latents = (
                        base_condition_latents * vae.config.scaling_factor
                    )
                    latent_bg_mask = F.interpolate(
                        batch["masks"].to(
                            accelerator.device,
                            dtype=latents.dtype,
                            non_blocking=True,
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
                        epoch if args.sequence_shared_noise_refresh == "epoch" else 0
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
                            model=model,
                            base_condition_latents=base_condition_latents,
                            rgb_sequence=rgb_sequence,
                            bg_mask_sequence=bg_mask_sequence,
                            injection_scale=args.stc_injection_scale,
                            predict_flow=True,
                        )
                    )
                    if not fresh_zero_identity_checked:
                        if torch.count_nonzero(stc_output.delta_bg).item() != 0:
                            raise RuntimeError("Fresh STC ZeroConv must output exact zero")
                        if not torch.equal(
                            augmented_condition.flatten(0, 1), base_condition_latents
                        ):
                            raise RuntimeError("Fresh v3 must reproduce V8 condition")
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
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type: {noise_scheduler.config.prediction_type}"
                        )
                    loss_diff = F.mse_loss(
                        model_prediction.float(), target.float(), reduction="mean"
                    )

                flow_output = compute_teacher_flow_loss(
                    predicted_forward=stc_output.predicted_flow_forward,
                    predicted_backward=stc_output.predicted_flow_backward,
                    teacher_forward=batch["teacher_flow_forward"],
                    teacher_backward=batch["teacher_flow_backward"],
                    bg_mask_sequence=bg_mask_sequence,
                    valid_forward=batch["teacher_valid_forward"],
                    valid_backward=batch["teacher_valid_backward"],
                    region=args.flow_region,
                    charbonnier_eps=args.flow_charbonnier_eps,
                )
                loss_flow = flow_output.loss
                loss = loss_diff + args.flow_loss_weight * loss_flow

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scale_before = None
                if accelerator.sync_gradients and accelerator.scaler is not None:
                    scale_before = float(accelerator.scaler.get_scale())
                optimizer.step()
                if scale_before is not None:
                    optimizer_step_was_skipped = (
                        float(accelerator.scaler.get_scale()) < scale_before
                    )
                else:
                    optimizer_step_was_skipped = accelerator.optimizer_step_was_skipped
                if not optimizer_step_was_skipped:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients and optimizer_step_was_skipped:
                logger.warning("AMP skipped update; global_step unchanged")
            if accelerator.sync_gradients and not optimizer_step_was_skipped:
                global_step += 1
                progress.update(1)
                if global_step % args.logging_steps == 0:
                    metrics = torch.stack(
                        (
                            loss.detach().float(),
                            loss_diff.detach().float(),
                            loss_flow.detach().float(),
                            (args.flow_loss_weight * loss_flow).detach().float(),
                            flow_output.epe.detach().float(),
                            flow_output.valid_forward_ratio.detach().float(),
                            flow_output.valid_backward_ratio.detach().float(),
                            flow_output.predicted_magnitude.detach().float(),
                            stc_output.delta_bg.detach().float().abs().mean(),
                            stc_output.latent_bg_mask.detach().float().mean(),
                        )
                    )
                    metrics = accelerator.gather(metrics.unsqueeze(0)).mean(0)
                    logs = {
                        "train/loss_total": float(metrics[0]),
                        "train/loss_diff": float(metrics[1]),
                        "train/loss_flow": float(metrics[2]),
                        "train/loss_flow_weighted": float(metrics[3]),
                        "train/flow_epe": float(metrics[4]),
                        "train/flow_valid_forward": float(metrics[5]),
                        "train/flow_valid_backward": float(metrics[6]),
                        "train/flow_pred_magnitude": float(metrics[7]),
                        "train/delta_abs_mean": float(metrics[8]),
                        "train/degraded_bg_ratio": float(metrics[9]),
                        "train/lr": float(lr_scheduler.get_last_lr()[0]),
                    }
                    if args.report_to != "none":
                        accelerator.log(logs, step=global_step)
                    progress.set_postfix(
                        total=f"{logs['train/loss_total']:.4f}",
                        diff=f"{logs['train/loss_diff']:.4f}",
                        flow=f"{logs['train/loss_flow']:.3f}",
                    )
                if global_step % args.checkpointing_steps == 0:
                    save_training_checkpoint(
                        accelerator,
                        model,
                        args,
                        global_step,
                        epoch,
                        batch_index + 1,
                    )
                if global_step >= args.max_train_steps:
                    stop_training = True
                    break
        resume_batch_index = 0
        epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        bare = accelerator.unwrap_model(model)
        bare.save_pretrained(
            Path(args.output_dir) / "stc_flow_model", safe_serialization=True
        )
        bare.stc_adapter.save_pretrained(
            Path(args.output_dir) / "stc_adapter", safe_serialization=True
        )
        final_metadata = checkpoint_metadata(
            args, accelerator, global_step, epoch, 0
        )
        final_metadata["status"] = "complete"
        _json_dump(Path(args.output_dir) / "final_metadata.json", final_metadata)
    accelerator.wait_for_everyone()
    if args.report_to != "none":
        accelerator.end_training()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.preflight_only:
        run_preflight(parsed)
    else:
        main(parsed)
