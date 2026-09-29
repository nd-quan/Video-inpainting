#!/usr/bin/env python
"""Train from a random adjacent source and target at the same diffusion timestep."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from shared_bg_noise_training import collate_shared_noise_clips, make_sequence_shared_background_noise, sample_shared_background_noise
from STC_encoder_v2_rgb.frozen_v8 import build_frozen_v8_context, frozen_v8_predict
from STC_encoder_v8_raft_deformable.raft_flow_provider import FrozenV7RAFTFlowProvider
from DGAF_VSR_original.common import (add_common_arguments, add_guidance_arguments, guidance_from_args,
    resolve_paths, make_dataset, dataset_fingerprint, load_stack, json_dump, BooleanOptionalAction)
from DGAF_VSR_original.warping import SequenceWarper, prediction_to_clean


class EpochSampler(Sampler):
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(torch.randperm(len(self), generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    add_guidance_arguments(parser)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Clips per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--gradient_checkpointing", action=BooleanOptionalAction, default=True)
    parser.add_argument("--report_to", choices=("none", "tensorboard"), default="tensorboard")
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--stop_after_steps", type=int, default=None,
                        help="Save and exit at this global step without changing the exact-resume training contract")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run exactly one update, verify temporal gradient; no checkpoint writes")
    args = parser.parse_args(argv)
    guidance_from_args(args)
    for name in ("train_batch_size", "gradient_accumulation_steps", "max_train_steps", "checkpointing_steps", "num_diffusion_steps"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.clip_length != 3:
        parser.error("Original-style training uses triplets: --clip_length 3")
    if args.dataloader_num_workers < 0 or any(not math.isfinite(x) or x <= 0 for x in (args.learning_rate, args.max_grad_norm)):
        parser.error("Invalid workers, learning rate, or max_grad_norm")
    if args.smoke_test:
        args.max_train_steps = 1
        if args.resume_from_checkpoint:
            parser.error("Smoke test does not support resume")
    if args.stop_after_steps is not None and not 1 <= args.stop_after_steps <= args.max_train_steps:
        parser.error("stop_after_steps must be between 1 and max_train_steps")
    resolve_paths(args)
    return args


def training_contract(args, dataset, num_processes):
    # Run-control paths do not change gradients. Everything else is enforced.
    excluded = {"output_dir", "resume_from_checkpoint", "preflight_only", "checkpointing_steps", "report_to", "stop_after_steps"}
    values = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in excluded}
    values.update(num_processes=num_processes, dataset_fingerprint=dataset_fingerprint(dataset),
        schema_version=2, stc_encoder=False, trainable="brushnet_all",
        guidance_source="frozen_baseline_neighbor_clean_same_timestep",
        guidance_update="sequential_same_timestep_sequence_sweep", training_method="random_left_right_neighbor_same_timestep_MSE",
        mask_scope="destination_BG_and_sampling_validity", raft_frozen=True,
        scheduler="DDIM_eta0_no_clipping", lr_scheduler="constant")
    return values


def resolve_resume(args):
    if args.resume_from_checkpoint is None:
        return None
    if args.resume_from_checkpoint == "latest":
        candidates = [p for p in args.output_dir.glob("checkpoint-*") if (p / "metadata.json").is_file()]
        if not candidates:
            raise FileNotFoundError("No complete checkpoint to resume")
        return max(candidates, key=lambda p: int(p.name.split("-")[-1]))
    path = Path(args.resume_from_checkpoint).expanduser().resolve()
    if not (path / "metadata.json").is_file():
        raise FileNotFoundError(f"Incomplete checkpoint: {path}")
    return path


def save_checkpoint(accelerator, model, args, contract, step, epoch, next_batch):
    path = args.output_dir / f"checkpoint-{step}"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
    # Everyone checks before rank zero creates the directory.
    accelerator.wait_for_everyone()
    path.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(path / "training_state"))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(path / "brushnet_dgaf", safe_serialization=True)
        json_dump(path / "guidance_config.json", guidance_from_args(args).to_dict())
        metadata = dict(contract=contract, global_step=step, epoch=epoch, next_batch_index=next_batch)
        # Completion marker is last: latest resolution ignores partial saves.
        json_dump(path / "metadata.json", metadata)
        json_dump(args.output_dir / "latest.json", {"checkpoint": path.name, "global_step": step})
    accelerator.wait_for_everyone()


def main(args):
    dataset = make_dataset(args)
    if args.preflight_only:
        print(json.dumps({"status": "ok", "clips": len(dataset), "guidance": guidance_from_args(args).to_dict(),
            "baseline": str(args.baseline_checkpoint), "raft": str(args.raft_student_path),
            "stc_encoder": False, "trainable": "BrushNet including new zero input channels"}, indent=2))
        return
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision, log_with=None if args.report_to == "none" else args.report_to,
        project_config=ProjectConfiguration(project_dir=str(args.output_dir), logging_dir=str(args.output_dir / "tensorboard")),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    if accelerator.device.type != "cuda":
        raise RuntimeError("Full training requires CUDA for RAFT")
    contract = training_contract(args, dataset, accelerator.num_processes)
    resume = resolve_resume(args)
    saved = json.loads((resume / "metadata.json").read_text()) if resume else None
    if saved and saved["contract"] != contract:
        differences = [k for k in set(contract) | set(saved["contract"]) if contract.get(k) != saved["contract"].get(k)]
        raise ValueError(f"Exact resume contract mismatch: {differences}")
    if not resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new output_dir or explicit resume; existing runs are protected")
    if resume:
        newer = [p for p in args.output_dir.glob("checkpoint-*") if (p / "metadata.json").exists() and int(p.name.split("-")[-1]) > saved["global_step"]]
        if newer:
            raise ValueError("Refusing to rewind over newer checkpoints")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, device_specific=True)
    pipe, image_encoder, projection, fusion, ip_report = load_stack(args, accelerator.device,
        model_path=(resume / "brushnet_dgaf") if resume else None)
    scheduler = pipe.scheduler
    scheduler.set_timesteps(args.num_diffusion_steps, device=accelerator.device)
    if scheduler.config.prediction_type not in ("epsilon", "v_prediction", "sample"):
        raise ValueError("Unsupported scheduler prediction type")
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    for module in (pipe.vae, pipe.text_encoder, image_encoder):
        module.to(dtype=dtype)
    model = pipe.brushnet.requires_grad_(True).train()
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        pipe.unet.enable_gradient_checkpointing()
        # Diffusers gates checkpointing on .training. U-Net weights remain frozen.
        pipe.unet.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    sampler = EpochSampler(dataset, args.seed)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=sampler,
        collate_fn=collate_shared_noise_clips, num_workers=args.dataloader_num_workers,
        pin_memory=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if len(loader) < 1:
        raise ValueError("Not enough clips for the configured training batch")
    provider = FrozenV7RAFTFlowProvider(args.raft_student_path, device=accelerator.device,
        pair_batch_size=args.raft_pair_batch_size, mixed_precision=args.raft_mixed_precision)
    if resume:
        accelerator.load_state(str(resume / "training_state"))
    step = saved["global_step"] if saved else 0
    epoch = saved["epoch"] if saved else 0
    skip_batches = saved["next_batch_index"] if saved else 0
    if accelerator.is_main_process:
        json_dump(args.output_dir / "run_config.json", contract)
        json_dump(args.output_dir / "model_contract.json", {**contract, "ip_loading": ip_report,
            "condition": "[input_latent(4), M_BG(1), aligned_clean(4), support(1)]"})
    accelerator.init_trackers("DGAF_VSR", config={"warp_mode": args.warp_mode, "learning_rate": args.learning_rate})
    accelerator.print(f"Training {args.warp_mode}; {len(dataset)} clips; {accelerator.num_processes} GPU(s); no STC")
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    last_saved = step if saved else -1
    stop_step = args.stop_after_steps or args.max_train_steps
    while step < stop_step:
        sampler.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        active_loader = accelerator.skip_first_batches(loader, skip_batches) if skip_batches else loader
        for batch_index, batch in enumerate(active_loader, start=skip_batches):
            with accelerator.accumulate(model):
                b, frames = int(batch["clip_batch_size"]), int(batch["num_frames"])
                rgb = batch["conditioning_pixel_values"].to(accelerator.device)
                with torch.no_grad():
                    clean_gt = pipe.vae.encode(batch["pixel_values"].to(accelerator.device, dtype=dtype)).latent_dist.sample().float() * pipe.vae.config.scaling_factor
                    base = pipe.vae.encode(rgb.to(dtype)).latent_dist.sample().float() * pipe.vae.config.scaling_factor
                    h, w = base.shape[-2:]
                    bg = (F.interpolate(batch["masks"].to(accelerator.device).float(), size=(h, w), mode="nearest") >= 0.5).float()
                    base_condition = torch.cat((base, bg), dim=1)
                    rgb_clips = rgb.reshape(b, frames, 3, *rgb.shape[-2:])
                    flows = provider.predict_sequence(rgb_clips)
                    warper = SequenceWarper(flows.forward, flows.backward, bg.reshape(b, frames, 1, h, w), guidance_from_args(args))
                    template = clean_gt.reshape(b, frames, 4, h, w)
                    shared = make_sequence_shared_background_noise(template, batch["videos"], args.seed, refresh_index=epoch)
                    noise = sample_shared_background_noise(template, warper.bg, strength=args.shared_bg_noise_strength,
                        variance_preserving=True, shared_bg_noise=shared).flatten(0, 1)
                    text, context = build_frozen_v8_context(batch, image_encoder, fusion, projection,
                        pipe.text_encoder, args.fusion_scale, accelerator.device, dtype, accelerator.autocast)
                    # Paper training: independent source/target forward noise at one t.
                    # rho defaults to zero; nonzero rho is an explicit BrushNet ablation.
                    timestep = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=accelerator.device)
                    source_frame = torch.randint(0, 2, (b,), device=accelerator.device) * 2
                    target_ids = torch.arange(b, device=accelerator.device) * frames + 1
                    source_ids = torch.arange(b, device=accelerator.device) * frames + source_frame
                    source_state = scheduler.add_noise(clean_gt[source_ids], noise[source_ids], timestep)
                    state = scheduler.add_noise(clean_gt[target_ids], noise[target_ids], timestep)
                    with accelerator.autocast():
                        source_prediction = frozen_v8_predict(pipe.baseline_brushnet, pipe.unet,
                            source_state.to(dtype), timestep, base_condition[source_ids].to(dtype),
                            text[source_ids], context[source_ids])
                    alpha = scheduler.alphas_cumprod.to(accelerator.device)[timestep].reshape(b, 1, 1, 1)
                    source_clean = prediction_to_clean(source_state, source_prediction, alpha, scheduler.config.prediction_type)
                    # Geometry may differ per sequence, so select its chosen neighbor explicitly.
                    pieces = []
                    for row in range(b):
                        row_warper = SequenceWarper(flows.forward[row:row+1], flows.backward[row:row+1],
                            warper.bg[row:row+1], guidance_from_args(args))
                        pieces.append(row_warper.frame_guidance(source_clean[row:row+1], 1, int(source_frame[row])))
                    guidance = torch.cat(pieces)
                    condition = torch.cat((base_condition[target_ids], guidance), dim=1)
                    if scheduler.config.prediction_type == "epsilon":
                        target = noise[target_ids]
                    elif scheduler.config.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(clean_gt[target_ids], noise[target_ids], timestep)
                    else:
                        target = clean_gt[target_ids]
                    text, context = text[target_ids], context[target_ids]
                with accelerator.autocast():
                    prediction = frozen_v8_predict(model, pipe.unet, state.to(dtype), timestep, condition.to(dtype), text, context)
                    loss = F.mse_loss(prediction.float(), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                accelerator.backward(loss)
                gradient_norm = 0.0
                temporal_gradient = 0.0
                if accelerator.sync_gradients:
                    gradient_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    gradient = accelerator.unwrap_model(model).conv_in_condition.weight.grad
                    temporal_gradient = float(gradient[:, 9:].float().norm())
                    if args.smoke_test and (not math.isfinite(temporal_gradient) or temporal_gradient <= 0):
                        raise RuntimeError("Smoke test needs finite nonzero gradients into new guidance channels")
                scale_before = float(accelerator.scaler.get_scale()) if accelerator.scaler is not None else None
                optimizer.step()
                overflow = (float(accelerator.scaler.get_scale()) < scale_before) if scale_before is not None else accelerator.optimizer_step_was_skipped
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients and not overflow:
                step += 1
                metrics = {"loss": float(accelerator.gather(loss.detach()).mean()), "temporal_input_grad_norm": temporal_gradient,
                    "grad_norm": float(gradient_norm), "support_mean": float(guidance[:, 4:].mean()),
                    "global_step": step, "timestep": float(timestep.float().mean()), "elapsed_seconds": time.monotonic() - start}
                accelerator.log(metrics, step=step)
                if accelerator.is_main_process:
                    print(json.dumps(metrics), flush=True)
                    with (args.output_dir / "train_metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(metrics) + "\n")
                if not args.smoke_test and (step % args.checkpointing_steps == 0 or step == stop_step):
                    save_checkpoint(accelerator, model, args, contract, step, epoch, batch_index + 1)
                    last_saved = step
                if step >= stop_step:
                    break
        skip_batches = 0
        epoch += 1
    if accelerator.is_main_process:
        json_dump(args.output_dir / "final_metadata.json", {"global_step": step, "smoke_test": args.smoke_test,
            "last_saved_step": last_saved, "contract": contract, "elapsed_seconds": time.monotonic() - start})
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
