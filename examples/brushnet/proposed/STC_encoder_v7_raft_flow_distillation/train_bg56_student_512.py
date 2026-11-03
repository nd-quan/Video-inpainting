#!/usr/bin/env python3
"""Square-512 entry point with checkpoint-boundary pause and stable best selection."""
from bg56_512_setup import INITIAL, read_json, write_json
import torch.distributed as dist
import train_raft_student_flow as trainer

original_save = trainer.save_checkpoint


def save_checkpoint(accelerator, model, output_dir, step, best_score, args, validation, is_best):
    best_path = output_dir / "best.json"
    if best_path.is_file():
        historical = trainer._metric_score(read_json(best_path), args.best_metric)
        best_score = min(best_score, historical)
        is_best = trainer._metric_score(validation, args.best_metric) < historical
    # Preserve the original weight initialization in metadata after a resume.
    initial = args.init_from_checkpoint
    args.init_from_checkpoint = str(INITIAL)
    try:
        original_save(accelerator, model, output_dir, step, best_score, args, validation, is_best)
    finally:
        args.init_from_checkpoint = initial
    request = output_dir.parent / "pause.request.json"
    if request.is_file():
        if accelerator.is_main_process:
            write_json(output_dir.parent / "paused_checkpoint.json", {"step": step, "checkpoint": str(output_dir / "checkpoint-{:07d}".format(step)), "reason": "requested pause after all checkpoint files were saved"})
            print("PAUSED_AFTER_CHECKPOINT {}".format(step), flush=True)
        accelerator.wait_for_everyone()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(0)


if __name__ == "__main__":
    args = trainer.parse_args()
    if args.resolution != 512 or args.spatial_mode != "square" or args.max_input_side != 512:
        raise ValueError("This experiment requires square 512x512 processing")
    if args.student_backend != "propainter_raft" or args.raft_iterations != 20:
        raise ValueError("This experiment requires ProPainter RAFT, 20 iterations")
    metadata = read_json(args.teacher_flow_root / "metadata.json")
    if (metadata["height"], metadata["width"]) != (512, 512):
        raise ValueError("Teacher cache must be exactly 512x512")
    trainer.save_checkpoint = save_checkpoint
    trainer.main(args)
