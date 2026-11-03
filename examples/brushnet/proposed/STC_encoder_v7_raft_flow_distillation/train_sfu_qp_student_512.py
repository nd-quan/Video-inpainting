#!/usr/bin/env python3
"""Square-512 QP training with unique validation samples and checkpoint pause."""
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from bg56_512_setup import read_json, write_json
import train_raft_student_flow as trainer

original_save = trainer.save_checkpoint
original_validate = trainer.validate


def validate_unique(model, loader, args, accelerator):
    # Accelerator pads validation batches by default. For 112 pairs on three
    # ranks this repeats samples; use disjoint subsets with unwrapped RAFT.
    dataset = loader.dataset
    indices = list(range(accelerator.process_index, len(dataset), accelerator.num_processes))
    unique_loader = DataLoader(Subset(dataset, indices), batch_size=args.train_batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    return original_validate(model, unique_loader, args, accelerator)


def save_checkpoint(accelerator, model, output_dir, step, best_score, args, validation, is_best):
    best_path = output_dir / "best.json"
    if best_path.is_file():
        historical = trainer._metric_score(read_json(best_path), args.best_metric)
        best_score = min(best_score, historical)
        is_best = trainer._metric_score(validation, args.best_metric) < historical
    initial = args.init_from_checkpoint
    args.init_from_checkpoint = read_json(output_dir.parent / "experiment_config.json")["initial_checkpoint"]
    try:
        original_save(accelerator, model, output_dir, step, best_score, args, validation, is_best)
    finally:
        args.init_from_checkpoint = initial
    if (output_dir.parent / "pause.request.json").is_file():
        if accelerator.is_main_process:
            write_json(output_dir.parent / "paused_checkpoint.json", {"step": step, "checkpoint": str(output_dir / "checkpoint-{:07d}".format(step))})
            print("PAUSED_AFTER_CHECKPOINT {}".format(step), flush=True)
        accelerator.wait_for_everyone()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(0)


if __name__ == "__main__":
    args = trainer.parse_args()
    plan = read_json(args.output_dir.resolve().parent / "experiment_config.json")
    if (args.resolution, args.spatial_mode, args.max_input_side) != (512, "square", 512):
        raise ValueError("Only square 512x512 is allowed")
    if args.student_backend != "propainter_raft" or args.raft_iterations != 20 or plan["bg_qp"] not in (46, 48):
        raise ValueError("Unexpected student model or QP")
    teacher = read_json(args.teacher_flow_root / "metadata.json")
    if (teacher["height"], teacher["width"], teacher["source_bg_qp"]) != (512, 512, plan["bg_qp"]):
        raise ValueError("Teacher cache contract mismatch")
    if args.init_from_checkpoint and str(Path(args.init_from_checkpoint).resolve()) != plan["initial_checkpoint"]:
        raise ValueError("Student initialization differs from the stage plan")
    trainer.validate = validate_unique
    trainer.save_checkpoint = save_checkpoint
    trainer.main(args)
