#!/usr/bin/env python3
"""Backfill and stream native fine-tuning JSONL metrics to TensorBoard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from torch.utils.tensorboard import SummaryWriter


def add_new_steps(source: Path, writer: SummaryWriter, last_step: int) -> int:
    if not source.exists():
        return last_step
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # The trainer may be appending this line right now.
            step = int(row["step"])
            if step <= last_step:
                continue
            for key in ("loss", "lr", "update_seconds", "epoch"):
                if key in row:
                    writer.add_scalar(f"train/{key}", float(row[key]), step)
            if "original_hw" in row:
                writer.add_scalar("train/rank0_height", int(row["original_hw"][0]), step)
                writer.add_scalar("train/rank0_width", int(row["original_hw"][1]), step)
            if "optimizer_step_skipped" in row:
                writer.add_scalar(
                    "train/optimizer_step_skipped",
                    int(bool(row["optimizer_step_skipped"])),
                    step,
                )
            last_step = step
    writer.flush()
    return last_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--tmux-session", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    config = json.loads((args.experiment / "training_config.json").read_text())
    max_step = int(config["max_train_steps"])
    source = args.experiment / "train_log.jsonl"
    log_dir = args.experiment / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    last_step = 0
    with SummaryWriter(log_dir=str(log_dir), flush_secs=5) as writer:
        while True:
            new_step = add_new_steps(source, writer, last_step)
            if new_step != last_step:
                last_step = new_step
                print(f"TensorBoard synced through step {last_step}/{max_step}", flush=True)
            if last_step >= max_step:
                break
            running = subprocess.run(
                ["tmux", "has-session", "-t", args.tmux_session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
            if not running:
                # One final pass catches a log append just before tmux exited.
                add_new_steps(source, writer, last_step)
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
