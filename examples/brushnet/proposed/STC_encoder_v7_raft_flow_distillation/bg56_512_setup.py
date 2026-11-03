"""Contracts and preparation for the independent BG56 square-512 experiment."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
BRUSHNET = HERE.parents[3]
SOURCE = BRUSHNET / "examples/brushnet/dataset/sfu_train_test_80_20/bg56"
INITIAL = BRUSHNET / "experiments/train_v7_raft_student_flow_constant_from4750/checkpoint-0007500"
PROPAINTER = Path("/home/cilab/ndquan/videoInpainting/pretrained/ProPainter")
TEACHER_WEIGHTS = PROPAINTER / "weights/raft-things.pth"
DEFAULT_RUN = Path("/user/Cilab/Quan/experiment/train_v7_raft_student_sfu_bg56_512_from7500")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".{}.tmp".format(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fixed_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Existing experiment contract differs: {}".format(path))
    else:
        write_json(path, value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb512(path):
    """Exactly matches RAFTTeacherFlowPairDataset._rgb in square mode."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).div_(127.5).sub_(1.0)


def pairs(manifest):
    for sequence in manifest["sequences"]:
        branch = Path(sequence["class"]) / sequence["name"]
        for split in ("train", "valid"):
            for segment in sequence["splits"][split]["segments"]:
                for first in range(segment["start"], segment["end"]):
                    yield split, branch, first, first + 1


def cache_path(run, pair):
    split, branch, first, second = pair
    return run / "teacher_flows_512x512" / split / branch / "{:06d}_{:06d}.npz".format(first, second)


def check_cache_file(path):
    if path.is_symlink():
        raise ValueError("Teacher labels must be generated inside this experiment: {}".format(path))
    with np.load(path, allow_pickle=False) as data:
        for key, channels in (("teacher_f", 2), ("teacher_b", 2), ("valid_f", 1), ("valid_b", 1)):
            array = data[key]
            if array.shape != (channels, 512, 512) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError("Invalid 512 teacher tensor {}:{}".format(path, key))
            if channels == 1 and (array.min() < 0 or array.max() > 1):
                raise ValueError("Invalid teacher validity mask: {}".format(path))


def prepare(run):
    run = Path(run).resolve()
    if read_json(SOURCE / "summary.json")["bg_qp"] != 56:
        raise ValueError("Source is not BG56")
    if not (INITIAL / "raft_student/pytorch_model.bin").is_file():
        raise FileNotFoundError(INITIAL)
    if read_json(INITIAL / "raft_student/config.json")["architecture"] != "propainter_raft_large":
        raise ValueError("Wrong initialization architecture")
    sequences = {}
    inventory = {}
    selected = {}
    for source_split, target_split in (("train", "train"), ("test", "valid")):
        with (SOURCE / (source_split + "_pairs.csv")).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = 4824 if source_split == "train" else 1206
        if len(rows) != expected:
            raise ValueError("Unexpected BG56 frame count for {}".format(source_split))
        grouped = defaultdict(list)
        keys = set()
        for row in rows:
            if int(row["bg_qp"]) != 56:
                raise ValueError("Mixed source QPs")
            branch = Path(row["class"]) / row["sequence"]
            first = int(row["source_frame"])
            key = (branch.as_posix(), first)
            if key in keys:
                raise ValueError("Duplicate source frame: {}".format(key))
            keys.add(key)
            grouped[branch.as_posix()].append(first)
            sizes = []
            for column, directory in (("input", "input"), ("gt", "GT"), ("mask", "mask")):
                relative = Path(source_split) / directory / branch / "{:06d}.png".format(first)
                if Path(row[column]) != relative:
                    raise ValueError("CSV path/frame mismatch: {}".format(row))
                path = SOURCE / relative
                stat = path.stat()
                if stat.st_size == 0:
                    raise ValueError("Empty source frame: {}".format(path))
                with Image.open(path) as image:
                    sizes.append(image.size)
                inventory[relative.as_posix()] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if len(set(sizes)) != 1:
                raise ValueError("Source RGB/mask dimensions differ: {}".format(key))
        selected[source_split] = keys
        for branch_name, frame_ids in sorted(grouped.items()):
            frame_ids.sort()
            if frame_ids != list(range(frame_ids[0], frame_ids[-1] + 1)):
                raise ValueError("Noncontiguous source frames: {}".format(branch_name))
            branch = Path(branch_name)
            record = sequences.setdefault(branch_name, {"class": branch.parts[0], "name": branch.parts[1], "splits": {}})
            record["splits"][target_split] = {"segments": [{"start": frame_ids[0], "end": frame_ids[-1]}], "frame_count": len(frame_ids)}
    if selected["train"] & selected["test"] or len(sequences) != 14:
        raise ValueError("Invalid BG56 split or sequence coverage")
    run.mkdir(parents=True, exist_ok=True)
    dataset = run / "dataset"
    dataset.mkdir(exist_ok=True)
    for source_split, target_split in (("train", "train"), ("test", "valid")):
        link = dataset / target_split
        target = SOURCE / source_split
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError("Conflicting dataset link: {}".format(link))
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(target, target_is_directory=True)
    manifest = {"source_dataset": str(SOURCE), "validation_policy": "entire source test split is used for validation", "sequences": [sequences[key] for key in sorted(sequences)]}
    fixed_json(dataset / "manifest.json", manifest)
    fixed_json(run / "source_inventory.json", inventory)
    all_pairs = list(pairs(manifest))
    counts = {split: sum(pair[0] == split for pair in all_pairs) for split in ("train", "valid")}
    if counts != {"train": 4810, "valid": 1192}:
        raise ValueError("Unexpected adjacent pair coverage: {}".format(counts))
    metadata = {"version": 1, "dataset_root": str(dataset), "manifest": str(dataset / "manifest.json"), "manifest_sha256": sha256(dataset / "manifest.json"), "raft_checkpoint": str(TEACHER_WEIGHTS), "raft_sha256": sha256(TEACHER_WEIGHTS), "teacher_backend": "propainter_raft", "height": 512, "width": 512, "spatial_mode": "fixed", "iters": 20, "rgb_resize": "Pillow BILINEAR 512x512, same as student square loader", "normalization": "RGB uint8 -> float32 /127.5 -1", "mask_semantics": "source 0=BG,255=ROI; internal M_BG=1", "format": "npz teacher_f/teacher_b [2,512,512], valid_f/valid_b [1,512,512], float32", "source_bg_qp": 56, "pairs": counts}
    fixed_json(run / "teacher_flows_512x512/metadata.json", metadata)
    plan = {"source": str(SOURCE), "initial_checkpoint": str(INITIAL), "initial_weights_sha256": sha256(INITIAL / "raft_student/pytorch_model.bin"), "resolution": [512, 512], "spatial_mode": "square", "max_train_steps": 4000, "learning_rate": 1e-5, "lr_scheduler": "constant", "teacher_weight": 1.0, "fb_weight": 0.1, "smoothness_weight": 0.01, "flow_loss_region": "all", "raft_iterations": 20, "raft_sequence_gamma": 0.8, "mixed_precision": "fp16", "freeze_batchnorm": True, "seed": 20260903, "checkpointing_steps": 250, "validation_steps": 250, "valid_pairs_per_sequence": 8, "full_pair_counts": counts, "resume_policy": "restore full state; preserve effective global batch by changing accumulation when GPU count changes"}
    fixed_json(run / "experiment_config.json", plan)
    return manifest


def verify_source(run):
    for relative, original in read_json(run / "source_inventory.json").items():
        stat = (SOURCE / relative).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != original:
            raise ValueError("Source changed since preparation: {}".format(relative))


def batch_layout(global_batch, gpu_count, preferred_batch):
    for batch in range(min(preferred_batch, global_batch // gpu_count), 0, -1):
        if global_batch % (batch * gpu_count) == 0:
            return batch, global_batch // (batch * gpu_count)
    raise ValueError("Cannot preserve global batch {} on {} GPUs".format(global_batch, gpu_count))
