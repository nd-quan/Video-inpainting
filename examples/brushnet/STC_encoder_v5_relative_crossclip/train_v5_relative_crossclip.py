#!/usr/bin/env python
"""Train V5: V4++ + relative temporal bias + paired cross-clip memory."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from transformers import CLIPImageProcessor


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v3_rgb_flow import train_rgb_stc_flow_shared_noise as trainer
from STC_encoder_v4pp_bg_feature.feature_alignment import (
    compute_feature_alignment_loss,
)
from STC_encoder_v5_relative_crossclip.cross_clip_data import (
    CrossClipTeacherFlowV8Dataset,
    collate_cross_clip_teacher_flow,
)
from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
    RelativeCrossClipBGSTCAdapter,
    augment_brushnet_condition_v5,
)


EXPERIMENT_NAME = "rgb_stc_v5_relative_bias_cross_clip"
DEFAULT_OUTPUT = (
    trainer.REPO_ROOT
    / "experiments"
    / "train_stc_v5_relative_crossclip_T16_S12_sharedNoise_0.95"
)

_base_checkpoint_metadata = trainer.checkpoint_metadata
_base_resume_contract = trainer._resume_contract


def resolve_full_component(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.name in {"best.json", "latest.json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = Path(payload["checkpoint"])
        path = candidate if candidate.is_absolute() else path.parent / candidate
        path = path.resolve()
    for name in ("stc_v5_model", "stc_flow_model"):
        nested = path / name
        if (nested / "config.json").is_file():
            path = nested
            break
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"No complete STC model below {path}")
    if not any(
        (path / name).is_file()
        for name in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")
    ):
        raise FileNotFoundError(f"No complete STC weights below {path}")
    return path


def _add_variant_arguments(parser) -> None:
    parser.add_argument(
        "--init_v4pp_model",
        default=None,
        help=(
            "Full repaired V4++ checkpoint/component (or best.json). "
            "The legacy stc_adapter is intentionally rejected."
        ),
    )
    parser.add_argument("--relative_position_max_distance", type=int, default=32)
    parser.add_argument("--cross_clip_memory_frames", type=int, default=4)
    parser.add_argument(
        "--detach_cross_clip_memory",
        dest="detach_cross_clip_memory",
        action="store_true",
        default=True,
        help=(
            "Detach the predecessor branch (required by the stateless, "
            "DDP-safe V5 training contract)."
        ),
    )


def _dataset_signature(dataset) -> str:
    records = []
    for index, (branch, _, frame_ids) in enumerate(dataset.clips):
        previous = dataset.predecessor_indices[index]
        records.append(
            {
                "branch": str(branch),
                "frames": [int(value) for value in frame_ids],
                "previous": None if previous is None else int(previous),
                "overlap": int(dataset.predecessor_overlaps[index]),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_dataset(args, tokenizer):
    dataset = CrossClipTeacherFlowV8Dataset(
        dataset_root=args.dataset_root,
        split=args.train_split,
        teacher_flow_root=args.teacher_flow_root,
        tokenizer=tokenizer,
        clip_image_processor=CLIPImageProcessor(),
        clip_length=args.clip_length,
        stride=args.clip_stride,
        resolution=args.resolution,
    )
    args.cross_clip_pair_index_sha256 = _dataset_signature(dataset)
    args.cross_clip_transition_count = dataset.cross_clip_transition_count
    args.cross_clip_run_count = dataset.cross_clip_run_count
    return dataset


def _model_factory(args):
    if not args.init_v4pp_model:
        raise ValueError("Fresh V5 training requires --init_v4pp_model")
    component = resolve_full_component(args.init_v4pp_model)
    return RelativeCrossClipBGSTCAdapter.from_v4pp_pretrained(
        component,
        relative_position_max_distance=args.relative_position_max_distance,
        cross_clip_memory_frames=args.cross_clip_memory_frames,
        detach_cross_clip_memory=args.detach_cross_clip_memory,
        require_memory_overlap=True,
    )


def _augment_extra_kwargs(*, batch, device, num_clips, num_frames, resolution):
    previous_rgb = batch["previous_conditioning_pixel_values"].to(
        device=device, non_blocking=True
    )
    previous_mask = batch["previous_masks"].to(device=device, non_blocking=True)
    expected_rgb = (num_clips, num_frames, 3, resolution, resolution)
    expected_mask = (num_clips, num_frames, 1, resolution, resolution)
    if tuple(previous_rgb.shape) != expected_rgb:
        raise ValueError(
            f"Previous RGB shape {tuple(previous_rgb.shape)} != {expected_rgb}"
        )
    if tuple(previous_mask.shape) != expected_mask:
        raise ValueError(
            f"Previous mask shape {tuple(previous_mask.shape)} != {expected_mask}"
        )
    return {
        "frame_ids": batch["frame_ids"].to(device=device, non_blocking=True),
        "previous_rgb_sequence": previous_rgb,
        "previous_bg_mask_sequence": previous_mask,
        "previous_frame_ids": batch["previous_frame_ids"].to(
            device=device, non_blocking=True
        ),
        "previous_valid_mask": batch["previous_valid_mask"].to(
            device=device, non_blocking=True
        ),
    }


def _extra_train_metrics(*, stc_output, batch):
    has_predecessor = batch["previous_valid_mask"].any(dim=1).float().mean()
    configured_memory = max(1, int(stc_output.temporal_memory.frame_ids.shape[1]))
    return {
        "train/cross_clip_gate_abs_mean": stc_output.cross_clip_gate_abs_mean,
        "train/relative_bias_abs_mean": stc_output.relative_bias_abs_mean,
        "train/memory_overlap_mean": stc_output.memory_overlap_count.float().mean(),
        "train/memory_overlap_ratio": (
            stc_output.memory_overlap_count.float().mean() / configured_memory
        ),
        "train/predecessor_sample_ratio": has_predecessor,
    }


def _post_dataset_validation(*, dataset, args, resume_metadata) -> None:
    if not resume_metadata:
        return
    actual = getattr(args, "cross_clip_pair_index_sha256", None)
    expected = resume_metadata.get("cross_clip_pair_index_sha256")
    if not expected or actual != expected:
        raise ValueError(
            "V5 resume dataset/pair index changed: "
            f"saved={expected!r}, current={actual!r}"
        )


def _checkpoint_metadata(args, accelerator, global_step, epoch, next_batch_index):
    metadata = _base_checkpoint_metadata(
        args, accelerator, global_step, epoch, next_batch_index
    )
    metadata.update(
        {
            "experiment": EXPERIMENT_NAME,
            "model_variant": "v5_relative_temporal_bias_cross_clip_overlap",
            "loss": (
                "L_diff + flow_loss_weight*L_flow + "
                "effective_feature_weight*L_feature_alignment"
            ),
            "inference_component": "stc_v5_model",
            "init_v4pp_model": str(resolve_full_component(args.init_v4pp_model)),
            "relative_position_max_distance": args.relative_position_max_distance,
            "relative_position_bias_init": "zero",
            "cross_clip_mode": "self_contained_predecessor_current_pair",
            "cross_clip_memory_source": "detached_per_layer_post_temporal_overlap",
            "cross_clip_memory_frames": args.cross_clip_memory_frames,
            "detach_cross_clip_memory": args.detach_cross_clip_memory,
            "cross_clip_gate_init": "zero",
            "cross_clip_require_absolute_id_overlap": True,
            "cross_clip_transition_count": getattr(
                args, "cross_clip_transition_count", None
            ),
            "cross_clip_run_count": getattr(args, "cross_clip_run_count", None),
            "cross_clip_pair_index_sha256": getattr(
                args, "cross_clip_pair_index_sha256", None
            ),
            "cross_batch_mutable_state": False,
            "train_dataloader_drop_last": True,
            "ddp_find_unused_parameters": True,
            "loss_ownership": "all_losses_current_clip_only",
            "legacy_stc_adapter_export_complete": False,
        }
    )
    return metadata


def _resume_contract(args):
    contract = _base_resume_contract(args)
    contract.update(
        {
            "experiment": EXPERIMENT_NAME,
            "init_v4pp_model": str(resolve_full_component(args.init_v4pp_model)),
            "relative_position_max_distance": args.relative_position_max_distance,
            "cross_clip_memory_frames": args.cross_clip_memory_frames,
            "detach_cross_clip_memory": args.detach_cross_clip_memory,
            "train_dataloader_drop_last": True,
            "ddp_find_unused_parameters": True,
        }
    )
    return contract


def _install_variant() -> None:
    trainer.RGBSTCFlowAdapter = RelativeCrossClipBGSTCAdapter
    trainer.augment_brushnet_condition = augment_brushnet_condition_v5
    trainer.FEATURE_ALIGNMENT_LOSS_FN = compute_feature_alignment_loss
    trainer.ADD_VARIANT_ARGUMENTS_FN = _add_variant_arguments
    trainer.MODEL_FACTORY_FN = _model_factory
    trainer.AUGMENT_EXTRA_KWARGS_FN = _augment_extra_kwargs
    trainer.EXTRA_TRAIN_METRICS_FN = _extra_train_metrics
    trainer.POST_DATASET_VALIDATION_FN = _post_dataset_validation
    trainer.TRAIN_DATALOADER_DROP_LAST = True
    trainer.FULL_MODEL_COMPONENT_NAME = "stc_v5_model"
    trainer.SAVE_LEGACY_STC_ADAPTER = False
    # Run-first clips intentionally have no predecessor memory, so the
    # cross-clip gate is conditionally unused. DDP must discover that dynamic
    # graph instead of assuming every parameter participates in every sample.
    trainer.DDP_FIND_UNUSED_PARAMETERS = True
    trainer.make_dataset = _make_dataset
    trainer.collate_teacher_flow_clips = collate_cross_clip_teacher_flow
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.EXPERIMENT_NAME = EXPERIMENT_NAME
    trainer.FLOW_INFERENCE_DEPENDENCY = True
    trainer.INFERENCE_COMPONENT = "stc_v5_model"
    trainer.TRAINING_LOG_TITLE = (
        "RGB-STC V5: relative temporal bias + paired cross-clip overlap memory"
    )
    trainer.checkpoint_metadata = _checkpoint_metadata
    trainer._resume_contract = _resume_contract


def parse_args(input_args=None):
    _install_variant()
    args = trainer.parse_args(input_args)
    if args.init_stc_adapter:
        raise ValueError("V5 rejects --init_stc_adapter; use --init_v4pp_model")
    if args.init_v4pp_model:
        args.init_v4pp_model = str(
            Path(args.init_v4pp_model).expanduser().resolve()
        )
    if not args.init_v4pp_model:
        raise ValueError(
            "V5 requires --init_v4pp_model for both fresh and exact-resume "
            "contract validation"
        )
    if not args.detach_cross_clip_memory:
        raise ValueError(
            "V5 training requires detached predecessor memory so shuffled/DDP "
            "samples remain self-contained"
        )
    if args.relative_position_max_distance < args.clip_length - 1:
        raise ValueError(
            "relative_position_max_distance must cover at least T-1 local offsets"
        )
    overlap = args.clip_length - args.clip_stride
    if overlap <= 0:
        raise ValueError("V5 requires clip_stride < clip_length")
    if not 1 <= args.cross_clip_memory_frames <= overlap:
        raise ValueError(
            "cross_clip_memory_frames must be in [1, clip_length-clip_stride]"
        )
    return args


def run_preflight(args) -> None:
    _install_variant()
    trainer.run_preflight(args)
    component = resolve_full_component(args.init_v4pp_model)
    model = _model_factory(args)
    gate_max = max(
        float(block.cross_clip_gate.detach().abs())
        for block in model.stc_adapter.temporal_blocks
    )
    bias_max = max(
        float(block.attention.relative_position_bias.detach().abs().max())
        for block in model.stc_adapter.temporal_blocks
    )
    report = {
        "v5_preflight": "ok",
        "init_full_component": str(component),
        "relative_position_max_distance": args.relative_position_max_distance,
        "cross_clip_memory_frames": args.cross_clip_memory_frames,
        "cross_clip_gate_max_at_init": gate_max,
        "relative_bias_max_at_init": bias_max,
        "zero_init_identity": gate_max == 0.0 and bias_max == 0.0,
        "cross_clip_transition_count": getattr(
            args, "cross_clip_transition_count", None
        ),
        "cross_clip_run_count": getattr(args, "cross_clip_run_count", None),
        "cross_clip_pair_index_sha256": getattr(
            args, "cross_clip_pair_index_sha256", None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main(args) -> None:
    _install_variant()
    trainer.main(args)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.preflight_only:
        run_preflight(parsed)
    else:
        main(parsed)
