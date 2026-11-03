#!/usr/bin/env python3
"""Check Accelerator full-state restoration from four GPUs onto two GPUs."""
import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, TensorDataset

from bg56_512_setup import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import os
    ranks = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation = args.global_batch // (args.batch * ranks)
    assert accumulation > 0 and args.global_batch == args.batch * ranks * accumulation
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=accumulation)
    set_seed(1234, device_specific=True)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    loader = DataLoader(TensorDataset(torch.ones(128, 2)), batch_size=args.batch)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    if args.resume:
        accelerator.load_state(str(args.output / "state_4gpu"))
        saved = read_json(args.output / "saved_4gpu.json")
        flat = torch.cat([p.detach().flatten() for p in accelerator.unwrap_model(model).parameters()]).cpu()
        assert torch.equal(flat, torch.tensor(saved["parameters"], dtype=flat.dtype))
        assert all(int(state["step"]) == 1 for state in optimizer.state.values())
        assert scheduler.get_last_lr()[0] == 1e-5
    for (inputs,) in loader:
        with accelerator.accumulate(model):
            # Keep the tiny fp16 probe below the initial GradScaler overflow
            # threshold so it verifies a real AdamW update, not a skipped one.
            loss = model(inputs).float().square().mean() * 0.001
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if accelerator.sync_gradients:
            break
    assert optimizer.state, "The probe must perform a real optimizer update"
    assert all(int(state["step"]) == (2 if args.resume else 1) for state in optimizer.state.values())
    if not args.resume:
        accelerator.save_state(str(args.output / "state_4gpu"))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        parameters = torch.cat([p.detach().flatten() for p in accelerator.unwrap_model(model).parameters()]).cpu().tolist()
        record = {"ranks": ranks, "batch_per_gpu": args.batch, "accumulation": accumulation, "global_batch": args.global_batch, "optimizer_step": 2 if args.resume else 1, "parameters": parameters, "success": True}
        write_json(args.output / ("resumed_2gpu.json" if args.resume else "saved_4gpu.json"), record)
        print(record, flush=True)
    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
