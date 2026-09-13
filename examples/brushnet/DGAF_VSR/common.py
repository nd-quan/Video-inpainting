"""Loading/data contracts shared by the new trainer and evaluator."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import DDIMScheduler
from ip_adapter.ip_adapter import ImageProjModel
from shared_bg_noise_training import HierarchicalV8ClipDataset, FlatV8TestClipDataset
from STC_encoder_v2_rgb.frozen_v8 import install_and_load_ip_adapter, load_fusion_module
from STC_encoder_v8_raft_deformable.raft_flow_provider import resolve_raft_student_component
from DGAF_VSR.brushnet_dgaf import DGAFBrushNetModel
from DGAF_VSR.pipeline_dgaf import DGAFBrushNetPipeline
from DGAF_VSR.warping import GuidanceConfig

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE = ROOT / "experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250"
DEFAULT_BASE_MODEL = ROOT / "examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_RAFT = ROOT / "experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"


class BooleanOptionalAction(argparse.Action):
    """Python 3.8-compatible --flag / --no-flag pair."""
    def __init__(self, option_strings, dest, default=None, **kwargs):
        options = [option for flag in option_strings for option in (flag, "--no-" + flag[2:])]
        super().__init__(options, dest, nargs=0, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, not option_string.startswith("--no-"))


def add_common_arguments(parser):
    parser.add_argument("--pretrained_model_name_or_path", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--baseline_checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--raft_student_path", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--image_encoder_name_or_path", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--dataset_root", type=Path, default=ROOT.parents[1] / "SFU_STC_flow")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_layout", choices=("auto", "hierarchical", "flat_test"), default="auto")
    parser.add_argument("--include_branches", nargs="+")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=12)
    parser.add_argument("--shared_bg_noise_strength", type=float, default=0.95)
    parser.add_argument("--fusion_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--raft_pair_batch_size", type=int, default=1)
    parser.add_argument("--raft_mixed_precision", action=BooleanOptionalAction, default=True)
    parser.add_argument("--preflight_only", action="store_true")


def add_guidance_arguments(parser):
    parser.add_argument("--warp_mode", choices=("direct", "dgaf"), required=True)
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument("--use_confidence_mask", action=BooleanOptionalAction, default=True,
                        help="Forward/backward consistency occlusion proxy; disable with --no-use_confidence_mask")
    parser.add_argument("--fb_alpha", type=float, default=0.01)
    parser.add_argument("--fb_beta", type=float, default=0.5, help="Squared RGB-pixel consistency threshold")
    parser.add_argument("--direction", choices=("alternating", "bidirectional", "previous", "next"), default="alternating")


def guidance_from_args(args):
    return GuidanceConfig(**{key: getattr(args, key) for key in GuidanceConfig.__dataclass_fields__})


def json_dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def resolve_paths(args):
    for key in ("pretrained_model_name_or_path", "baseline_checkpoint", "dataset_root", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.raft_student_path = resolve_raft_student_component(args.raft_student_path)
    if args.clip_length < 2 or not 1 <= args.clip_stride <= args.clip_length:
        raise ValueError("Require T >= 2 and 1 <= stride <= T")
    if args.resolution < 64 or args.resolution % 64:
        raise ValueError("resolution must be a positive multiple of 64")
    if not 0 <= args.shared_bg_noise_strength <= 1 or args.raft_pair_batch_size < 1:
        raise ValueError("Invalid noise strength or RAFT batch size")
    if not math.isfinite(args.fusion_scale) or args.fusion_scale < 0:
        raise ValueError("fusion_scale must be finite and nonnegative")
    for suffix in ("brushnet/config.json", "brushnet/diffusion_pytorch_model.safetensors",
                   "ipadapter/model.safetensors", "ipadapter/fusion_module.safetensors"):
        if not (args.baseline_checkpoint / suffix).is_file():
            raise FileNotFoundError(args.baseline_checkpoint / suffix)
    config = json.loads((args.baseline_checkpoint / "brushnet/config.json").read_text())
    if config["conditioning_channels"] != 5:
        raise ValueError("This experiment requires the 5-channel shared-noise baseline")


def make_dataset(args, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(args.pretrained_model_name_or_path), subfolder="tokenizer", use_fast=False)
    hierarchical = all((args.dataset_root / args.split / kind).is_dir() for kind in ("GT", "input", "mask"))
    layout = args.dataset_layout
    if layout == "auto":
        layout = "hierarchical" if hierarchical else "flat_test"
    if args.include_branches and layout != "hierarchical":
        raise ValueError("include_branches is supported only for hierarchical datasets")
    cls = HierarchicalV8ClipDataset if layout == "hierarchical" else FlatV8TestClipDataset
    kwargs = {"include_branches": args.include_branches} if layout == "hierarchical" else {}
    dataset = cls(dataset_root=args.dataset_root, split=args.split, tokenizer=tokenizer,
                  clip_image_processor=CLIPImageProcessor(), clip_length=args.clip_length,
                  stride=args.clip_stride, resolution=args.resolution, **kwargs)
    if not len(dataset):
        raise ValueError("Dataset contains no clips")
    return dataset


def dataset_fingerprint(dataset):
    return hashlib.sha256(repr(dataset.clips).encode()).hexdigest()


def load_stack(args, device, *, model_path=None, inference=False):
    model = (DGAFBrushNetModel.from_pretrained(str(model_path)) if model_path else
             DGAFBrushNetModel.from_baseline(args.baseline_checkpoint / "brushnet"))
    dtype = torch.float16 if inference else torch.float32
    pipe = DGAFBrushNetPipeline.from_pretrained(str(args.pretrained_model_name_or_path),
        brushnet=model, torch_dtype=dtype, low_cpu_mem_usage=False,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False, thresholding=False)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_name_or_path)
    projection = ImageProjModel(cross_attention_dim=pipe.unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim, clip_extra_context_tokens=4)
    report = install_and_load_ip_adapter(pipe.unet, projection, args.baseline_checkpoint / "ipadapter/model.safetensors")
    fusion = load_fusion_module(args.baseline_checkpoint / "ipadapter/fusion_module.safetensors",
                               embed_dim=image_encoder.config.projection_dim)
    pipe.to(device=device, dtype=dtype)
    for module in (pipe.unet, pipe.vae, pipe.text_encoder, model, image_encoder, projection, fusion):
        module.to(device=device, dtype=dtype).requires_grad_(False).eval()
    return pipe, image_encoder, projection, fusion, report
