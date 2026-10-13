#!/usr/bin/env python
"""Native-size shared-noise finetuning; temporal guidance remains inference-only.

Reuses the existing BrushNet/IP-Adapter/fusion architecture and checkpoint format.
No spatial resizing or cropping of diffusion inputs. CLIP conditioning retains
its pretrained preprocessing. Run --preflight_only before allocating GPU models.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from shared_bg_noise_training import (
    HierarchicalV8ClipDataset,
    collate_shared_noise_clips,
    make_sequence_shared_background_noise,
    sample_clip_timesteps,
    sample_shared_background_noise,
)

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1] / "experiments"
DEFAULT_CHECKPOINT = EXPERIMENTS / "train_sharedNoise_sameBG_0.95_T8/checkpoint-2250"


class NativeClipDataset(HierarchicalV8ClipDataset):
    """One native geometry per clip, including sources with mixed-size segments."""

    def __init__(self, *args, max_height=1080, max_width=1920, **kwargs):
        self.max_height, self.max_width = max_height, max_width  

        super().__init__(*args, resolution=None, **kwargs)
        sizes = {}
        counts = Counter()
        for path in sorted(self.roots["GT"].rglob("*.png")):
            relative = path.relative_to(self.roots["GT"])
            if self.include_branches and relative.parent.as_posix() not in self.include_branches:
                continue
            triplet_sizes = []
            for kind in ("GT", "input", "mask"):
                with Image.open(self.roots[kind] / relative) as image:
                    triplet_sizes.append(image.size)
            if len(set(triplet_sizes)) != 1:
                raise ValueError(f"GT/input/mask sizes differ: {relative}: {triplet_sizes}")
            sizes[relative] = triplet_sizes[0]
            counts[f"{triplet_sizes[0][0]}x{triplet_sizes[0][1]}"] += 1
        before = len(self.clips)
        self.clips = [clip for clip in self.clips if len({sizes[p] for p in clip[1]}) == 1]
        if not self.clips:
            raise ValueError("No contiguous clips with consistent native dimensions")
        self.covered_frame_count = len({p for _, paths, _ in self.clips for p in paths})
        self.sizes = sizes
        self.report = {
            "dataset_root": str(self.dataset_root), "split": self.split,
            "resize": False, "crop": False, "pad_multiple": 64,
            "rgb_padding": "replicate right/bottom", "roi_padding": 0,
            "train_eval_overlap_policy": "keep all requested training frames",
            "frames": self.frame_count, "covered_frames": self.covered_frame_count,
            "clips": len(self.clips), "clip_length": self.clip_length,
            "clip_stride": self.stride, "frames_by_stored_size_wh": dict(sorted(counts.items())),
            "clips_dropped_at_size_changes": before - len(self.clips),
            "note": "512x512 source files remain 512x512; native detail cannot be recovered by padding.",
        }

    def _resize_if_needed(self, image, is_mask=False):
        w, h = image.size

        scale = min(
            1.0,
            self.max_width / w,
            self.max_height / h,
        )

        if scale >= 1.0:
            return image

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resample = (
            Image.Resampling.NEAREST
            if is_mask
            else Image.Resampling.BILINEAR
        )

        return image.resize((new_w, new_h), resample=resample,)


    @staticmethod
    def _rgb_tensor(image):
        array = np.asarray(image, dtype=np.float32)
        h, w = array.shape[:2]
        array = np.pad(array, ((0, -h % 64), (0, -w % 64), (0, 0)), mode="edge")
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).div_(127.5).sub_(1)

    @staticmethod
    def _bg_mask_tensor(image):
        roi = np.asarray(image, dtype=np.uint8) >= 128
        h, w = roi.shape
        roi = np.pad(roi, ((0, -h % 64), (0, -w % 64)), constant_values=False)
        return torch.from_numpy(np.ascontiguousarray((~roi).astype(np.float32)[None]))

    def __getitem__(self, item):
        branch, paths, ids = self.clips[item]

        gt = self._load_images(paths, "GT", "RGB")
        decoded = self._load_images(paths, "input", "RGB")
        masks = self._load_images(paths, "mask", "L")

        src_w, src_h = self.sizes[paths[0]]

        gt = [self._resize_if_needed(x, is_mask=False) for x in gt]
        decoded = [self._resize_if_needed(x, is_mask=False) for x in decoded]
        masks = [self._resize_if_needed(x, is_mask=True) for x in masks]

        # w, h = self.sizes[paths[0]]

        w, h = gt[0].size

        historical_fg, historical_bg = self._masked_clip_images(decoded, masks)
        return {
                "pixel_values": torch.stack([self._rgb_tensor(x) for x in gt]),
                "conditioning_pixel_values": torch.stack(
                    [self._rgb_tensor(x) for x in decoded]
                ),
                "masks": torch.stack(
                    [self._bg_mask_tensor(x) for x in masks]
                ),
                "input_ids": self.empty_input_ids.unsqueeze(0).repeat(
                    self.clip_length, 1
                ),
                "clip_images": self._clip_pixels(decoded),
                "fg_clip_images": self._clip_pixels(historical_fg),
                "bg_clip_images": self._clip_pixels(historical_bg),
                "drop_image_embeds": torch.zeros(
                    self.clip_length, dtype=torch.long
                ),

                "video": f"{branch}@src{src_w}x{src_h}_train{w}x{h}",

                "frame_ids": torch.tensor(ids, dtype=torch.long),

                # IMPORTANT: resized size before padding
                "original_hw": torch.tensor([h, w], dtype=torch.long),
        }


def collate_native(examples):
    if len(examples) != 1:
        raise ValueError("Native-size training uses exactly one clip per device")
    batch = collate_shared_noise_clips(examples)
    batch["original_hw"] = examples[0]["original_hw"]
    return batch


def latent_valid_weights(original_hw, latent_hw, scale_factor, device):
    """Fraction of each latent cell covered by real pixels (not padded pixels)."""
    h, w = original_hw.to(device=device, dtype=torch.float32)
    lh, lw = latent_hw
    rows = ((h - torch.arange(lh, device=device) * scale_factor) / scale_factor).clamp(0, 1)
    cols = ((w - torch.arange(lw, device=device) * scale_factor) / scale_factor).clamp(0, 1)
    return (rows[:, None] * cols[None, :])[None, None]


def masked_noise_loss(prediction, target, weights):
    error = (prediction.float() - target.float()).square()
    return (error * weights).sum() / (weights.sum() * error.shape[0] * error.shape[1]).clamp_min(1)


class TrainModel(torch.nn.Module):
    """One DDP wrapper for all trainable components and gradient accumulation."""

    def __init__(self, brushnet, ip_adapter, fusion, fusion_scale):
        super().__init__()
        self.brushnet, self.ip_adapter, self.fusion = brushnet, ip_adapter, fusion
        self.fusion_scale = fusion_scale

    def forward(self, noisy, timestep, text, cond, base, fg, bg):
        image = base + self.fusion_scale * self.fusion(fg, bg)
        down, mid, up = self.brushnet(
            noisy, timestep, encoder_hidden_states=text, brushnet_cond=cond, return_dict=False
        )
        return self.ip_adapter(
            noisy, timestep, encoder_hidden_states=text, image_embeds=image,
            down_block_add_samples=[x.to(noisy.dtype) for x in down],
            mid_block_add_sample=mid.to(noisy.dtype),
            up_block_add_samples=[x.to(noisy.dtype) for x in up],
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow"))
    parser.add_argument("--train_split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--include_branches", nargs="+", default=None)
    parser.add_argument("--initial_checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained_model", type=Path, default=HERE / "base_model/stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--image_encoder", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--output_dir", type=Path, default=EXPERIMENTS / "finetune_shared_noise_original_size")
    parser.add_argument("--clip_length", type=int, default=4)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--max_height", type=int, default=1080)
    parser.add_argument("--max_width", type=int, default=1920)
    parser.add_argument("--frame_microbatch", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_warmup_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--shared_bg_noise_strength", type=float, default=0.95)
    parser.add_argument("--variance_preserving_shared_noise", action="store_true")
    parser.add_argument("--sequence_shared_noise_refresh", choices=["epoch", "never"], default="epoch")
    parser.add_argument("--fusion_scale", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--vae_tiling", action="store_true", help="Optional tiled VAE encoding; changes encoding near tile seams")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None, help="Resume this native trainer's state; otherwise only initial model weights are loaded")
    parser.add_argument("--preflight_only", action="store_true", help="Inspect all source image headers without loading GPU models")
    parser.add_argument("--preflight_report", type=Path, default=None)
    args = parser.parse_args()
    for name in ("frame_microbatch", "gradient_accumulation_steps", "max_train_steps", "checkpointing_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not 0 <= args.shared_bg_noise_strength <= 1:
        parser.error("--shared_bg_noise_strength must be in [0,1]")
    if args.learning_rate <= 0 or args.lr_warmup_steps < 0 or args.max_grad_norm <= 0:
        parser.error("Invalid learning rate, warmup or gradient clipping value")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader_num_workers must be nonnegative")
    return args


def build_models(args):
    # Import existing architecture/loaders without executing its training main().
    import train_brushnet_VCM_ipadapter_v8_coco_nulltext_sharedNoise_sameBG_v0_0 as legacy
    from safetensors.torch import load_file

    base = str(args.pretrained_model)
    tokenizer = legacy.AutoTokenizer.from_pretrained(base, subfolder="tokenizer", use_fast=False)
    text = legacy.CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    vae = legacy.AutoencoderKL.from_pretrained(base, subfolder="vae")
    unet = legacy.UNet2DConditionModel.from_pretrained(base, subfolder="unet")
    image = legacy.CLIPVisionModelWithProjection.from_pretrained(args.image_encoder)
    brushnet = legacy.BrushNetModel.from_pretrained(str(args.initial_checkpoint / "brushnet"))
    scheduler = legacy.DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    for frozen in (text, vae, unet, image):
        frozen.requires_grad_(False)
    projection = legacy.ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image.config.projection_dim, clip_extra_context_tokens=4,
    )
    processors = {}
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            processors[name] = legacy.AttnProcessor()
            continue
        if name.startswith("mid_block"):
            hidden = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hidden = list(reversed(unet.config.block_out_channels))[int(name.split(".")[1])]
        else:
            hidden = unet.config.block_out_channels[int(name.split(".")[1])]
        processors[name] = legacy.IPAttnProcessor(hidden_size=hidden, cross_attention_dim=unet.config.cross_attention_dim)
    unet.set_attn_processor(processors)
    adapter = legacy.IPAdapter(unet, projection, torch.nn.ModuleList(unet.attn_processors.values()),
                               str(args.initial_checkpoint / "ipadapter/model.safetensors"))
    fusion = legacy.FGBGFeatureFusion(image.config.projection_dim, num_heads=8)
    fusion.load_state_dict(load_file(str(args.initial_checkpoint / "ipadapter/fusion_module.safetensors")), strict=True)
    brushnet.requires_grad_(True)
    unet.enable_gradient_checkpointing()
    brushnet.enable_gradient_checkpointing()
    model = TrainModel(brushnet, adapter, fusion, args.fusion_scale)
    model.train()  # Checkpointing in these blocks is gated by module.training.
    for frozen in (text, vae, image):
        frozen.eval()
    if args.vae_tiling:
        vae.enable_tiling()
    return model, vae, text, image, tokenizer, scheduler


def save_checkpoint(accelerator, model, args, step, epoch, next_batch):
    from safetensors.torch import save_file

    root = args.output_dir / f"checkpoint-{step}"
    accelerator.wait_for_everyone()
    accelerator.save_state(str(root))  # Every rank saves its own RNG state.
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(model)
        raw.brushnet.save_pretrained(str(root / "brushnet"), safe_serialization=True)
        (root / "ipadapter").mkdir(exist_ok=True)
        state = {f"image_proj_model.{k}": v.detach().cpu().contiguous() for k, v in raw.ip_adapter.image_proj_model.state_dict().items()}
        for name, processor in raw.ip_adapter.unet.attn_processors.items():
            if name.endswith("attn2.processor"):
                state.update({f"unet.{name}.{k}": v.detach().cpu().contiguous() for k, v in processor.state_dict().items()})
        save_file(state, str(root / "ipadapter/model.safetensors"))
        save_file({k: v.detach().cpu().contiguous() for k, v in raw.fusion.state_dict().items()}, str(root / "ipadapter/fusion_module.safetensors"))
        state = {"global_step": step, "epoch": epoch, "next_batch": next_batch,
                 "world_size": accelerator.num_processes,
                 "gradient_accumulation_steps": args.gradient_accumulation_steps}
        (root / "native_training_state.json").write_text(json.dumps(state, indent=2) + "\n")
        print(f"Saved {root}", flush=True)
    accelerator.wait_for_everyone()


def remap_resume_batch(state, new_world_size, new_accumulation, loader_length):
    """Keep the global shuffled-data cursor when changing DDP process count."""
    old_world_size = state["world_size"]
    old_accumulation = state["gradient_accumulation_steps"]
    if old_world_size * old_accumulation != new_world_size * new_accumulation:
        raise ValueError("Resume would change the effective clips per optimizer update")
    global_cursor = state["next_batch"] * old_world_size  # One clip per rank/batch.
    if global_cursor % new_world_size:
        raise ValueError("Checkpoint data cursor cannot be represented with the new GPU count")
    next_batch = global_cursor // new_world_size
    if next_batch > loader_length or (next_batch < loader_length and next_batch % new_accumulation):
        raise ValueError("Checkpoint data cursor is not at an optimizer boundary in the new loader")
    return next_batch, global_cursor


def main(args):
    from transformers import CLIPImageProcessor

    # Dataset indexing/preflight does not need model weights or a GPU.
    placeholder_tokenizer = lambda *a, **k: SimpleNamespace(input_ids=torch.zeros(1, 77, dtype=torch.long))
    placeholder_tokenizer.model_max_length = 77
    dataset = NativeClipDataset(
        dataset_root=args.dataset_root, 
        split=args.train_split,
        tokenizer=placeholder_tokenizer, 
        clip_image_processor=CLIPImageProcessor(),
        clip_length=args.clip_length, 
        stride=args.clip_stride, 
        include_branches=args.include_branches,
        max_height=args.max_height,
        max_width=args.max_width
    )
    if args.preflight_only:
        print(json.dumps(dataset.report, indent=2))
        if args.preflight_report:
            args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_report.write_text(json.dumps(dataset.report, indent=2) + "\n")
        return

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from diffusers.optimization import get_scheduler

    for path in (args.initial_checkpoint / "brushnet/config.json",
                 args.initial_checkpoint / "ipadapter/model.safetensors",
                 args.initial_checkpoint / "ipadapter/fusion_module.safetensors"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.glob("checkpoint-*")) and not args.resume_from_checkpoint:
        raise ValueError("Output already contains checkpoints; choose a new output_dir or resume explicitly")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True)],
    )
    # Accelerate 0.21 creates the trackers attribute here, even with no log backend.
    accelerator.init_trackers("shared_noise_original_size")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        print(json.dumps(dataset.report, indent=2), flush=True)
        (args.output_dir / "dataset_report.json").write_text(json.dumps(dataset.report, indent=2) + "\n")
        config_name = (f"training_config_resume_{args.resume_from_checkpoint.name}.json"
                       if args.resume_from_checkpoint else "training_config.json")
        (args.output_dir / config_name).write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    model, vae, text, image, tokenizer, scheduler = build_models(args)
    dataset.empty_input_ids = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    generator = torch.Generator()
    loader_kwargs = dict(batch_size=1, shuffle=True, generator=generator, collate_fn=collate_native,
                         num_workers=args.dataloader_num_workers, pin_memory=True)
    if args.dataloader_num_workers:
        loader_kwargs["prefetch_factor"] = 1
    loader = DataLoader(dataset, **loader_kwargs)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-2)
    # Schedule is stepped once per optimizer update (including DDP).
    lr_scheduler = get_scheduler("constant_with_warmup", optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps)
    accelerator.register_for_checkpointing(lr_scheduler)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    for frozen in (vae, text, image):
        frozen.to(accelerator.device, dtype=dtype)
    vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
    step, first_epoch, skip_batches = 0, 0, 0
    if args.resume_from_checkpoint:
        state = json.loads((args.resume_from_checkpoint / "native_training_state.json").read_text())
        skip_batches, global_cursor = remap_resume_batch(
            state, accelerator.num_processes, args.gradient_accumulation_steps, len(loader)
        )
        accelerator.load_state(str(args.resume_from_checkpoint))
        step, first_epoch = state["global_step"], state["epoch"]
        if state["world_size"] != accelerator.num_processes:
            # The checkpoint has RNG files only for the original ranks. Give all
            # ranks distinct stochastic streams; the sampler is seeded by epoch below.
            set_seed(args.seed + step * 1009, device_specific=True)
        if accelerator.is_main_process:
            migration = {"checkpoint": str(args.resume_from_checkpoint), "global_step": step,
                         "epoch": first_epoch, "global_clip_cursor": global_cursor,
                         "old_world_size": state["world_size"], "new_world_size": accelerator.num_processes,
                         "old_accumulation": state["gradient_accumulation_steps"],
                         "new_accumulation": args.gradient_accumulation_steps,
                         "old_next_batch": state["next_batch"], "new_next_batch": skip_batches}
            (args.output_dir / f"resume_migration_{step}.json").write_text(json.dumps(migration, indent=2) + "\n")
            print(f"Resumed checkpoint {step}: {migration}", flush=True)
    if step >= args.max_train_steps:
        accelerator.print("Checkpoint has already reached max_train_steps")
        return
    optimizer.zero_grad(set_to_none=True)
    epoch = first_epoch
    while step < args.max_train_steps:
        generator.manual_seed(args.seed + epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        loss_sum, accumulated_clips = 0.0, 0
        update_started = time.monotonic()
        for batch_index, batch in enumerate(loader):
            if epoch == first_epoch and batch_index < skip_batches:
                continue
            with accelerator.accumulate(model):
                with torch.no_grad():
                    # Frozen VAE encodes one frame at a time; no full-clip activations.
                    latents = torch.cat([vae.encode(x[None].to(dtype)).latent_dist.sample() for x in batch["pixel_values"]]) * vae.config.scaling_factor
                    cond = torch.cat([vae.encode(x[None].to(dtype)).latent_dist.sample() for x in batch["conditioning_pixel_values"]]) * vae.config.scaling_factor
                    masks = F.interpolate(batch["masks"].to(dtype), size=latents.shape[-2:], mode="nearest")
                    cond = torch.cat([cond, masks], dim=1)
                    layout = latents.unsqueeze(0)
                    shared = make_sequence_shared_background_noise(layout, batch["videos"], args.seed,
                              epoch if args.sequence_shared_noise_refresh == "epoch" else 0)
                    noise = sample_shared_background_noise(layout, (masks.unsqueeze(0) >= 0.5).to(dtype),
                              strength=args.shared_bg_noise_strength, variance_preserving=args.variance_preserving_shared_noise,
                              shared_bg_noise=shared).squeeze(0)
                    timesteps = sample_clip_timesteps(1, latents.shape[0], scheduler.config.num_train_timesteps, latents.device, share_across_clip=True)
                    noisy = scheduler.add_noise(latents, noise, timesteps)
                    text_embeds = text(batch["input_ids"], return_dict=False)[0]
                    base_embeds = image(batch["clip_images"].to(dtype)).image_embeds
                    fg_embeds = image(batch["fg_clip_images"].to(dtype)).image_embeds
                    bg_embeds = image(batch["bg_clip_images"].to(dtype)).image_embeds
                    if scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif scheduler.config.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unsupported prediction type: {scheduler.config.prediction_type}")
                    weights = latent_valid_weights(batch["original_hw"], latents.shape[-2:], vae_scale, latents.device)
                frame_count = latents.shape[0]
                # Accelerator divides backward losses by the configured K.
                # Correct the last group when the epoch contains fewer than K clips.
                group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
                group_size = min(args.gradient_accumulation_steps, len(loader) - group_start)
                accumulation_correction = args.gradient_accumulation_steps / group_size
                clip_loss = 0.0
                for start in range(0, frame_count, args.frame_microbatch):
                    end = min(frame_count, start + args.frame_microbatch)
                    section = slice(start, end)
                    # At a synchronization step, reduce DDP gradients only on
                    # the final frame chunk. Other accumulation steps stay no_sync.
                    sync_context = accelerator.no_sync(model) if end < frame_count else contextlib.nullcontext()
                    with sync_context:
                        prediction = model(noisy[section], timesteps[section], text_embeds[section], cond[section],
                                           base_embeds[section], fg_embeds[section], bg_embeds[section])
                        loss = masked_noise_loss(prediction, target[section], weights) * ((end - start) / frame_count)
                        accelerator.backward(loss * accumulation_correction)
                    clip_loss += float(loss.detach())
                    del prediction, loss
                loss_sum += clip_loss
                accumulated_clips += 1
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                    optimizer.step()
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                averaged_loss = accelerator.reduce(torch.tensor(loss_sum / accumulated_clips, device=accelerator.device), reduction="mean").item()
                if accelerator.is_main_process:
                    record = {"step": step, "epoch": epoch, "loss": averaged_loss, "lr": lr_scheduler.get_last_lr()[0],
                              "original_hw": batch["original_hw"].tolist(), "latent_hw": list(latents.shape[-2:]),
                              "update_seconds": time.monotonic() - update_started,
                              "optimizer_step_skipped": accelerator.optimizer_step_was_skipped}
                    print(json.dumps(record), flush=True)
                    with (args.output_dir / "train_log.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                loss_sum, accumulated_clips = 0.0, 0
                if step % args.checkpointing_steps == 0 or step == args.max_train_steps:
                    save_checkpoint(accelerator, model, args, step, epoch, batch_index + 1)
                update_started = time.monotonic()
                if step >= args.max_train_steps:
                    break
        epoch += 1
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
