#!/usr/bin/env python3
"""Copy and verify selected experiment checkpoints, then replace them with links."""

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SOURCE_ROOT = Path("/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments")
TARGET_ROOT = Path("/user/Cilab/Quan/experiment")
EXPERIMENTS = (
    "finetune_coco_bg56_from_bg52_step2500_native_area_bucket_6000steps",
    "finetune_coco_bg52_native_area_bucket_ckpt200000_3000steps",
    "finetune_coco_bg48_from_bg46_step2000_native_area_bucket_3000steps",
    "finetune_coco_bg46_native_area_bucket_ckpt200000_3000steps",
)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(path, records):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"updated_at": timestamp(), "checkpoints": records}, indent=2) + "\n")
    temporary.replace(path)


def verify_copy(source, target):
    result = subprocess.run(
        ["rsync", "-r", "--size-only", "--dry-run", "--itemize-changes",
         str(source) + "/", str(target) + "/"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("Files differ after copy: {}".format(result.stdout[:1000]))
    source_files = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
    target_files = {p.relative_to(target) for p in target.rglob("*") if p.is_file()}
    if source_files != target_files:
        raise RuntimeError("Source and destination file lists differ: {}".format(source))


def move_one(source, target):
    if source.is_symlink():
        if source.resolve(strict=True) != target.resolve(strict=True):
            raise RuntimeError("Checkpoint already links elsewhere: {}".format(source))
        print("ALREADY_MOVED {}".format(source), flush=True)
        return "already_moved"
    if not source.is_dir():
        raise RuntimeError("Missing checkpoint directory: {}".format(source))
    if any(path.is_symlink() for path in source.rglob("*")):
        raise RuntimeError("Unexpected symlink inside checkpoint: {}".format(source))
    backup = source.with_name(source.name + ".root_copy_before_move")
    if backup.exists():
        raise RuntimeError("A previous move left a backup that needs inspection: {}".format(backup))

    target.mkdir(parents=True, exist_ok=True)
    print("COPY {} -> {} at {}".format(source, target, timestamp()), flush=True)
    subprocess.run(["rsync", "-rt", "--info=stats2", str(source) + "/", str(target) + "/"], check=True)
    verify_copy(source, target)

    source.rename(backup)
    try:
        source.symlink_to(target, target_is_directory=True)
        if source.resolve(strict=True) != target.resolve(strict=True):
            raise RuntimeError("Symlink verification failed: {}".format(source))
    except BaseException:
        if source.is_symlink():
            source.unlink()
        backup.rename(source)
        raise
    shutil.rmtree(backup)
    print("MOVED {} at {}".format(source, timestamp()), flush=True)
    return "moved"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS)
    args = parser.parse_args()
    args.target_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.target_root / "checkpoint_move_manifest.json"
    records = {}
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text()).get("checkpoints", {})

    # Snapshot the currently existing checkpoints so newly written ones are
    # never copied while a trainer is still saving them.
    checkpoints = []
    for experiment in args.experiments:
        source_dir = args.source_root / experiment
        if not source_dir.is_dir():
            raise RuntimeError("Missing experiment directory: {}".format(source_dir))
        checkpoints.extend(sorted(
            (path for path in source_dir.glob("checkpoint-*")
             if path.is_dir() and path.name.split("-")[-1].isdigit()),
            key=lambda path: int(path.name.split("-")[-1]),
        ))

    for source in checkpoints:
        target = args.target_root / source.parent.name / source.name
        status = move_one(source, target)
        records[str(source)] = {"target": str(target), "status": status, "at": timestamp()}
        write_manifest(manifest_path, records)
    print("DONE moved or verified {} checkpoints".format(len(checkpoints)), flush=True)


if __name__ == "__main__":
    main()
