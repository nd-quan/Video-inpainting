"""Prepare square-512 SFU cases without changing the completed BG56 run."""
import csv
import filecmp
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image

from bg56_512_setup import BRUSHNET, DEFAULT_RUN, HERE, INITIAL, TEACHER_WEIGHTS, cache_path, check_cache_file, fixed_json, load_rgb512, pairs, read_json, sha256, write_json

BASE = BRUSHNET / "examples/brushnet/dataset/sfu_train_test_80_20"
OUTPUT = Path("/user/Cilab/Quan/experiment")
RUN48 = OUTPUT / "train_v7_raft_student_sfu_bg48_512_from7500"
RUN46 = OUTPUT / "train_v7_raft_student_sfu_bg46_512_from_bg48_best"
QUEUE = OUTPUT / "train_v7_raft_student_sfu_bg48_then_bg46_512_queue"


def prepare_case(run, qp, initial):
    source = BASE / "bg{}".format(qp)
    if read_json(source / "summary.json")["bg_qp"] != qp:
        raise ValueError("Source QP mismatch")
    initial = Path(initial).resolve()
    if read_json(initial / "raft_student/config.json")["architecture"] != "propainter_raft_large":
        raise ValueError("Initialization must be a ProPainter RAFT student")
    weights_hash = sha256(initial / "raft_student/pytorch_model.bin")
    sequences, inventory, selected = {}, {}, {}
    for src_split, dst_split in (("train", "train"), ("test", "valid")):
        with (source / (src_split + "_pairs.csv")).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != (4824 if src_split == "train" else 1206):
            raise ValueError("Unexpected source frame count")
        grouped = defaultdict(list)
        keys = set()
        for row in rows:
            if int(row["bg_qp"]) != qp:
                raise ValueError("Mixed source QPs")
            branch = Path(row["class"]) / row["sequence"]
            frame = int(row["source_frame"])
            key = (branch.as_posix(), frame)
            if key in keys:
                raise ValueError("Duplicate frame")
            keys.add(key)
            grouped[branch.as_posix()].append(frame)
            sizes = []
            for column, directory in (("input", "input"), ("gt", "GT"), ("mask", "mask")):
                relative = Path(src_split) / directory / branch / "{:06d}.png".format(frame)
                if Path(row[column]) != relative:
                    raise ValueError("CSV frame/path mismatch")
                path = source / relative
                stat = path.stat()
                if not stat.st_size:
                    raise ValueError("Empty source frame: {}".format(path))
                inventory[relative.as_posix()] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                with Image.open(path) as image:
                    sizes.append(image.size)
            if len(set(sizes)) != 1:
                raise ValueError("Source input/GT/mask size mismatch: {}".format(key))
        selected[src_split] = keys
        for name, ids in sorted(grouped.items()):
            ids.sort()
            if ids != list(range(ids[0], ids[-1] + 1)):
                raise ValueError("Missing source frames: {}".format(name))
            cls, seq = Path(name).parts
            record = sequences.setdefault(name, {"class": cls, "name": seq, "splits": {}})
            record["splits"][dst_split] = {"segments": [{"start": ids[0], "end": ids[-1]}], "frame_count": len(ids)}
    if len(sequences) != 14 or selected["train"] & selected["test"]:
        raise ValueError("Sequence coverage or split overlap is invalid")
    run.mkdir(parents=True, exist_ok=True)
    (run / "dataset").mkdir(exist_ok=True)
    for src_split, dst_split in (("train", "train"), ("test", "valid")):
        link = run / "dataset" / dst_split
        if link.is_symlink():
            if link.resolve() != (source / src_split).resolve():
                raise ValueError("Conflicting dataset link")
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(source / src_split, target_is_directory=True)
    manifest = {"source_dataset": str(source), "validation_policy": "entire source test split is validation; eight pairs per sequence sampled periodically", "sequences": [sequences[key] for key in sorted(sequences)]}
    fixed_json(run / "dataset/manifest.json", manifest)
    fixed_json(run / "source_inventory.json", inventory)
    counts = {split: sum(pair[0] == split for pair in pairs(manifest)) for split in ("train", "valid")}
    if counts != {"train": 4810, "valid": 1192}:
        raise ValueError("Adjacent pair count mismatch")
    reference = read_json(DEFAULT_RUN / "teacher_flows_512x512/metadata.json")
    metadata = {key: reference[key] for key in ("version", "format", "height", "width", "iters", "mask_semantics", "normalization", "raft_checkpoint", "raft_sha256", "rgb_resize", "spatial_mode", "teacher_backend")}
    metadata.update(dataset_root=str(run / "dataset"), manifest=str(run / "dataset/manifest.json"), manifest_sha256=sha256(run / "dataset/manifest.json"), source_bg_qp=qp, pairs=counts)
    if (metadata["height"], metadata["width"], metadata["iters"]) != (512, 512, 20) or metadata["raft_sha256"] != sha256(TEACHER_WEIGHTS):
        raise ValueError("Unexpected teacher model or geometry")
    fixed_json(run / "teacher_flows_512x512/metadata.json", metadata)
    plan = {"source": str(source), "bg_qp": qp, "initial_checkpoint": str(initial), "initial_weights_sha256": weights_hash, "resolution": [512, 512], "spatial_mode": "square", "max_train_steps": 4000, "learning_rate": 1e-5, "lr_scheduler": "constant", "teacher_weight": 1.0, "fb_weight": 0.1, "smoothness_weight": 0.01, "flow_loss_region": "all", "raft_iterations": 20, "raft_sequence_gamma": 0.8, "mixed_precision": "fp16", "freeze_batchnorm": True, "seed": 20260903, "checkpointing_steps": 250, "validation_steps": 250, "valid_pairs_per_sequence": 8, "full_pair_counts": counts, "initial_gpu_count": 3, "batch_per_gpu": 4, "gradient_accumulation_steps": 1, "global_batch": 12, "validation_sharding": "unique rank-strided indices; no distributed padding duplicates", "optimizer_initialization": "fresh optimizer and scheduler for each QP stage"}
    fixed_json(run / "experiment_config.json", plan)
    return manifest


def verify_case_source(run):
    source = Path(read_json(run / "dataset/manifest.json")["source_dataset"])
    for relative, expected in read_json(run / "source_inventory.json").items():
        stat = (source / relative).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError("Source changed after preparation: {}".format(source / relative))


def reuse_teacher(run, manifest):
    proof_path = run / "teacher_cache_provenance.json"
    reference = read_json(DEFAULT_RUN / "dataset/manifest.json")
    report = read_json(DEFAULT_RUN / "preflight_audit.json")
    if manifest["sequences"] != reference["sequences"] or not report.get("success") or report["teacher_pairs_checked"] != 6002:
        raise ValueError("Reference teacher coverage is incompatible")
    source = Path(manifest["source_dataset"])
    original = Path(reference["source_dataset"])
    reference_inventory = read_json(DEFAULT_RUN / "source_inventory.json")
    checked, same_inode = 0, 0
    for relative, expected in read_json(run / "source_inventory.json").items():
        if Path(relative).parts[1] not in ("GT", "mask"):
            continue
        a, b = source / relative, original / relative
        stat = b.stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != reference_inventory[relative]:
            raise ValueError("Reference clean GT/mask changed since its audited run")
        if os.path.samefile(a, b):
            same_inode += 1
        elif not filecmp.cmp(a, b, shallow=False):
            write_json(proof_path, {"reused": False, "reason": "GT/mask bytes differ", "first_difference": str(a), "checked_files_before_difference": checked})
            return False
        checked += 1
    cutoff = datetime.fromisoformat(report["checked_at"]).timestamp()
    entries = list(pairs(manifest))
    for pair in entries:
        old = cache_path(DEFAULT_RUN, pair)
        if old.is_symlink() or old.stat().st_mtime > cutoff or old.stat().st_size == 0:
            raise ValueError("Reference teacher label changed after its audit: {}".format(old))
    for pair in entries:
        old, new = cache_path(DEFAULT_RUN, pair), cache_path(run, pair)
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            if not os.path.samefile(old, new):
                raise ValueError("Conflicting teacher label: {}".format(new))
        else:
            os.link(old, new)
    write_json(proof_path, {"reused": True, "reference_run": str(DEFAULT_RUN), "reference_audit_sha256": sha256(DEFAULT_RUN / "preflight_audit.json"), "gt_and_mask_files_checked": checked, "same_inode_files": same_inode, "remaining_files_byte_identical": checked - same_inode, "linked_teacher_pairs": len(entries), "identical_segments": True, "rgb_preprocessing": "same Pillow BILINEAR square 512x512"})
    return True


def audit_case(run, manifest, reused):
    import torch
    from raft_teacher_pair_data import RAFTTeacherFlowPairDataset
    entries = list(pairs(manifest))
    if reused:
        for pair in entries:
            if not os.path.samefile(cache_path(run, pair), cache_path(DEFAULT_RUN, pair)):
                raise ValueError("Reused teacher label identity changed")
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check_cache_file, [cache_path(run, pair) for pair in entries]))
    samples = 0
    for split in ("train", "valid"):
        dataset = RAFTTeacherFlowPairDataset(run / "dataset", run / "teacher_flows_512x512", split, 512, spatial_mode="square", max_input_side=512)
        seen = set()
        for index, (branch, first, second, path) in enumerate(dataset.entries):
            if branch in seen:
                continue
            sample = dataset[index]
            gt = run / "dataset" / split / "GT" / branch / "{:06d}.png".format(first)
            if sample["degraded0"].shape != (3, 512, 512) or not torch.equal(dataset._rgb(gt), load_rgb512(gt)):
                raise ValueError("Square shape or teacher/student preprocessing mismatch")
            check_cache_file(path)
            seen.add(branch)
            samples += 1
    verify_case_source(run)
    write_json(run / "preflight_audit.json", {"success": True, "resolution": [512, 512], "teacher_pairs": len(entries), "teacher_reused_after_exact_identity_verification": reused, "student_split_branch_samples": samples, "source_files_unchanged": True, "teacher_student_resize_exactly_equal": True})
