#!/usr/bin/env python
"""Evaluate V6 deformable STC with clip-local CGE/VCM-RS guidance.

The V6 condition encoder and the V8 IP-Adapter/BrushNet inference contract are
unchanged.  This entry point only replaces DDIM with ``CustomDDIMScheduler``
and configures its per-frame VCM-RS degradation operator:

    D_M(x) = M_ROI * C_QP20(x) + (1 - M_ROI) * C_QP52(x)

The V6 dataset uses ``M_BG=1`` internally, so it is explicitly inverted before
being passed to CGE.  VCM-RS is one-frame only; a V6 clip is therefore guided
frame-by-frame while its original shared-noise and V6 cross-clip condition are
preserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch.utils.data import Subset


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from diffusers.schedulers.scheduling_ddim_CGE import (  # noqa: E402
    CustomDDIMScheduler,
    cond_fn,
)
from STC_encoder_v2_rgb import evaluate_rgb_stc_shared_noise as evaluator  # noqa: E402
from STC_encoder_v4_flow_aligned import evaluate_flow_aligned_stc as v4_evaluator  # noqa: E402
from STC_encoder_v5_relative_crossclip import (  # noqa: E402
    evaluate_v5_relative_crossclip as v5_evaluator,
)
from STC_encoder_v6_flow_deformable import (  # noqa: E402
    evaluate_v6_flow_deformable as v6_evaluator,
)
from STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (  # noqa: E402
    FlowGuidedDeformableBGSTCAdapter,
)
from vcmrs_codec_adapter import VCMRSDualRegionCodec  # noqa: E402


_BASE_V6_PREFLIGHT = v6_evaluator.preflight
_BASE_LOAD_MODELS = evaluator.load_models
DEFAULT_OUTPUT = (
    evaluator.PROJECT_ROOT / "experiments" / "eval_stc_v6_deformable_cge"
)


def add_cge_arguments(parser) -> None:
    """Add V6 arguments plus the CGE settings to the shared evaluator parser."""

    v6_evaluator._add_evaluation_arguments(parser)
    # Do not let a CGE run resume into, or overwrite, the plain V6 output tree.
    parser.set_defaults(output_dir=DEFAULT_OUTPUT)
    parser.add_argument("--cge_guidance_scale", type=float, default=1.0e-4)
    parser.add_argument("--cge_start_step", type=int, default=35)
    parser.add_argument(
        "--cge_end_step",
        type=int,
        default=None,
        help="Exclusive CGE end step; default is --num_inference_steps.",
    )
    parser.add_argument("--cge_every_n_steps", type=int, default=1)
    parser.add_argument(
        "--cge_max_evals",
        type=int,
        default=2,
        help=(
            "Maximum costly VCM-RS evaluations per clip; -1 means unlimited. "
            "Each evaluation runs every frame in the clip."
        ),
    )
    parser.add_argument("--cge_decode_chunk_size", type=int, default=1)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of independent clip shards/processes. Default: 1.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based shard rank in [0, --num_shards). Default: 0.",
    )
    parser.add_argument(
        "--include_sequences",
        nargs="+",
        metavar="SEQUENCE",
        default=None,
        help=(
            "Restrict inference to named video branches, for example "
            "--include_sequences BasketballPass. Works for flat and hierarchical datasets."
        ),
    )


def _effective_end_step(args) -> int:
    return (
        int(args.num_inference_steps)
        if args.cge_end_step is None
        else int(args.cge_end_step)
    )


def validate_cge_arguments(args) -> None:
    end_step = _effective_end_step(args)
    if args.cge_guidance_scale < 0.0:
        raise ValueError("cge_guidance_scale must be non-negative")
    if args.cge_start_step < 0:
        raise ValueError("cge_start_step must be non-negative")
    if not args.cge_start_step < end_step:
        raise ValueError("CGE window must satisfy cge_start_step < cge_end_step")
    if end_step > args.num_inference_steps:
        raise ValueError("cge_end_step cannot exceed num_inference_steps")
    if args.cge_every_n_steps < 1:
        raise ValueError("cge_every_n_steps must be positive")
    if args.cge_max_evals < -1:
        raise ValueError("cge_max_evals must be -1 or non-negative")
    if args.cge_decode_chunk_size < 1:
        raise ValueError("cge_decode_chunk_size must be positive")
    if args.num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")


def shard_output_dir(output_dir: Path, shard_index: int, num_shards: int) -> Path:
    """Use isolated outputs so concurrent shard processes never race on JSON."""

    if num_shards == 1:
        return output_dir
    return output_dir / f"shard-{shard_index:02d}-of-{num_shards:02d}"


def select_sequence_clip_indices(
    dataset, include_sequences: Optional[Sequence[str]]
) -> Tuple[int, ...]:
    """Return original clip indices for the requested V6 video branches."""

    all_indices = tuple(range(len(dataset)))
    if not include_sequences:
        return all_indices
    requested = {str(name) for name in include_sequences}
    by_branch = {}
    for index, clip in enumerate(dataset.clips):
        branch = str(clip[0])
        by_branch.setdefault(branch, []).append(index)
    unknown = sorted(requested - set(by_branch))
    if unknown:
        raise ValueError(
            f"Unknown --include_sequences value(s): {unknown}; "
            f"available={sorted(by_branch)}"
        )
    return tuple(
        index for index in all_indices if str(dataset.clips[index][0]) in requested
    )


def select_clip_shard(
    dataset,
    source_indices: Sequence[int],
    shard_index: int,
    num_shards: int,
):
    """Partition clip indices without changing V6 predecessor references.

    Rebuilding ``dataset.clips`` after splitting would drop predecessors that
    happen to be assigned to another shard. ``Subset`` instead forwards each
    selected index to the original cross-clip dataset, preserving its complete
    predecessor/current context and only distributing independent inference
    calls.
    """

    total_clips = len(source_indices)
    indices = tuple(source_indices[int(shard_index) :: int(num_shards)])
    if not indices:
        raise ValueError(
            f"Shard {shard_index}/{num_shards} has no clips; selected set has {total_clips} clips"
        )
    return Subset(dataset, indices), indices


def cge_preflight(args):
    validate_cge_arguments(args)
    args.output_dir = shard_output_dir(
        args.output_dir, int(args.shard_index), int(args.num_shards)
    )
    dataset, paths = _BASE_V6_PREFLIGHT(args)
    total_clips = len(dataset)
    selected_indices = select_sequence_clip_indices(dataset, args.include_sequences)
    dataset, clip_indices = select_clip_shard(
        dataset,
        selected_indices,
        int(args.shard_index),
        int(args.num_shards),
    )
    # Constructing the adapter is read-only and validates the VCM-RS root,
    # Python executable, profile, and any explicitly configured descriptors.
    codec = VCMRSDualRegionCodec.from_env()
    cge_operator = str(getattr(codec, "cge_operator", "dual_region_qp20_qp52"))
    if cge_operator == "direct_roi_plus_bg_qp52":
        vcmrs_qp = {"bg": codec.bg_quality}
        roi_fidelity = "direct_image_loss_against_input_qp20"
        codec_calls_per_frame_per_eval = 3
    else:
        vcmrs_qp = {"roi": codec.roi_quality, "bg": codec.bg_quality}
        roi_fidelity = "VCM-RS_QP20_roundtrip"
        codec_calls_per_frame_per_eval = 6
    print(
        json.dumps(
            {
                "v6_cge_preflight": "ok",
                "scheduler": "CustomDDIMScheduler",
                "per_frame_cge": True,
                "cge_window": [int(args.cge_start_step), _effective_end_step(args)],
                "cge_every_n_steps": int(args.cge_every_n_steps),
                "cge_max_evals_per_clip": int(args.cge_max_evals),
                "cge_guidance_scale": float(args.cge_guidance_scale),
                "cge_operator": cge_operator,
                "codec_calls_per_frame_per_eval": codec_calls_per_frame_per_eval,
                "roi_fidelity": roi_fidelity,
                "shard_index": int(args.shard_index),
                "num_shards": int(args.num_shards),
                "source_clip_count": int(total_clips),
                "selected_clip_count": int(len(selected_indices)),
                "include_sequences": list(args.include_sequences or []),
                "assigned_clip_count": int(len(clip_indices)),
                "first_assigned_clip": int(clip_indices[0]),
                "last_assigned_clip": int(clip_indices[-1]),
                "shard_output_dir": str(args.output_dir),
                "vcmrs_profile": codec.profile,
                "vcmrs_qp_used_by_cge": vcmrs_qp,
                "mask_for_cge": "M_ROI = 1 - V6_M_BG",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return dataset, paths


def load_models_with_cge_scheduler(args, paths, device: torch.device):
    loaded = _BASE_LOAD_MODELS(args, paths, device)
    pipe = loaded[0]
    scheduler = CustomDDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.per_frame_cge = True
    scheduler.decode_chunk_size = int(args.cge_decode_chunk_size)
    scheduler.vae_scaling_factor = float(pipe.vae.config.scaling_factor)
    scheduler.cge_codec = VCMRSDualRegionCodec.from_env()
    scheduler.guidance_scale_cge = float(args.cge_guidance_scale)
    scheduler.cge_start_step = int(args.cge_start_step)
    scheduler.cge_end_step = _effective_end_step(args)
    scheduler.cge_every_n_steps = int(args.cge_every_n_steps)
    scheduler.cge_max_evals = int(args.cge_max_evals)
    scheduler.direct_cge_guidance = True
    pipe.scheduler = scheduler
    return loaded


def before_pipeline_call(
    *, pipe, sample: Mapping[str, torch.Tensor], args, device: torch.device
) -> Dict[str, object]:
    scheduler = pipe.scheduler
    if not isinstance(scheduler, CustomDDIMScheduler):
        raise TypeError("V6-CGE evaluator requires CustomDDIMScheduler")

    # ``conditioning_pixel_values`` is the exact degraded input in [-1, 1].
    # The V6/STC convention is M_BG=1, while VCM-RS expects M_ROI=1.
    x_lr = sample["conditioning_pixel_values"].to(device=device, dtype=torch.float32)
    roi_mask = (1.0 - sample["masks"].to(device=device, dtype=torch.float32))
    roi_mask = (roi_mask > 0.5).to(dtype=torch.float32)
    if x_lr.ndim != 4 or roi_mask.ndim != 4:
        raise ValueError("V6-CGE requires clip tensors with shape [T,C,H,W]")
    if x_lr.shape[0] != roi_mask.shape[0] or x_lr.shape[-2:] != roi_mask.shape[-2:]:
        raise ValueError("V6-CGE input and ROI mask clip shapes do not match")
    if x_lr.shape[0] > 1 and (
        scheduler.cge_codec.roi_descriptor is not None
        or scheduler.cge_codec.bg_descriptor is not None
    ):
        raise ValueError(
            "A V6 clip has a different ROI mask per frame. Do not set "
            "CGE_VCMRS_ROI_DESCRIPTOR/CGE_VCMRS_BG_DESCRIPTOR here; use the "
            "default CGE_VCMRS_AUTO_DESCRIPTORS=1 instead."
        )

    # VCM-RS takes batch size one. Pre-create/cache a descriptor pair for each
    # clip frame; cond_fn then performs the same per-frame calls during DDIM.
    for frame_index in range(x_lr.shape[0]):
        scheduler.cge_codec.prepare_region_mask(
            roi_mask[frame_index], x_lr[frame_index]
        )

    scheduler.x_lr = x_lr
    scheduler.mask = roi_mask
    scheduler.decoder = pipe.vae.decode
    scheduler.cond_fn = cond_fn
    scheduler.cge_codec_eval_count = 0
    scheduler.cge_denoise_step_count = 0
    return {
        "cge_enabled": 1,
        "cge_operator": str(
            getattr(scheduler.cge_codec, "cge_operator", "dual_region_qp20_qp52")
        ),
        "cge_clip_frame_count": int(x_lr.shape[0]),
        "cge_roi_ratio": float(roi_mask.mean().detach().cpu()),
        "cge_profile": scheduler.cge_codec.profile,
        "cge_bg_qp": int(scheduler.cge_codec.bg_quality),
        "cge_roi_qp": (
            None
            if getattr(scheduler.cge_codec, "cge_operator", "")
            == "direct_roi_plus_bg_qp52"
            else int(scheduler.cge_codec.roi_quality)
        ),
    }


def after_pipeline_call(
    *, pipe, sample: Mapping[str, torch.Tensor], args, device: torch.device
) -> Dict[str, object]:
    del sample, args, device
    scheduler = pipe.scheduler
    return {
        "cge_codec_evaluations": int(scheduler.cge_codec_eval_count),
        "cge_denoise_steps_seen": int(scheduler.cge_denoise_step_count),
    }


def clear_pipeline_call(*, pipe) -> None:
    """Drop clip tensors so an overlapping V6 clip cannot reuse them."""

    scheduler = pipe.scheduler
    if isinstance(scheduler, CustomDDIMScheduler):
        scheduler.cond_fn = None
        scheduler.decoder = None
        scheduler.x_lr = None
        scheduler.mask = None
        scheduler.cge_codec_eval_count = 0
        scheduler.cge_denoise_step_count = 0


def main() -> None:
    # Replicate V6's runtime class/condition wiring, then install CGE hooks on
    # the shared V2 evaluator before delegating into the V5/V4/V2 entry chain.
    evaluator.ADD_EVALUATION_ARGUMENTS_FN = add_cge_arguments
    evaluator.CONDITION_EXTRA_KWARGS_FN = v6_evaluator._condition_extra_kwargs
    evaluator.BEFORE_PIPELINE_CALL_FN = before_pipeline_call
    evaluator.AFTER_PIPELINE_CALL_FN = after_pipeline_call
    evaluator.CLEAR_PIPELINE_CALL_FN = clear_pipeline_call
    evaluator.load_models = load_models_with_cge_scheduler

    v5_evaluator.RelativeCrossClipBGSTCAdapter = FlowGuidedDeformableBGSTCAdapter
    v5_evaluator.preflight = cge_preflight
    v5_evaluator.build_v5_condition = v6_evaluator.build_v6_condition
    v4_evaluator.FlowAlignedRGBSTCAdapter = FlowGuidedDeformableBGSTCAdapter
    v4_evaluator.preflight = cge_preflight
    v4_evaluator.build_flow_aligned_condition = v6_evaluator.build_v6_condition
    v5_evaluator.main()


if __name__ == "__main__":
    main()
