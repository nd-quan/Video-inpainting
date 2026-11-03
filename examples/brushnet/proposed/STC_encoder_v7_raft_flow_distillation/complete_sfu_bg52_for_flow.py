#!/usr/bin/env python3
"""Complete three full BG52 SFU sequences before resuming the flow queue."""

import ast
import csv
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = Path("/home/cilab/ndquan")
VCM = WORKSPACE / "vcm-rs"
BRUSHNET = WORKSPACE / "videoInpainting/code/BrushNet"
DATASET = BRUSHNET / "examples/brushnet/dataset/sfu_train_test_80_20"
BG52 = DATASET / "bg52"
BG56 = DATASET / "bg56"
RUN52 = BRUSHNET / "experiments/train_v7_raft_student_sfu_bg52_max960_from_previous_best_4000"
RUN46 = BRUSHNET / "experiments/train_v7_raft_student_sfu_bg46_max960_4000"
RUN48 = BRUSHNET / "experiments/train_v7_raft_student_sfu_bg48_max960_from_bg46_3500_4000"
QUEUE = BRUSHNET / "experiments/train_v7_raft_student_sfu_qp48_52_56_queue"
STATUS = QUEUE / "bg52_completion_status.json"
SUPERVISOR_PID = 711880
GUIDED_PYTHON = WORKSPACE / "envs/guided_diff/bin/python"
VCM_PYTHON = WORKSPACE / "envs/vcm/bin/python"
NAMES = ("BasketballDrill", "BQSquare", "BlowingBubbles")
ASSIGNMENTS = {4: ("BasketballDrill", "BlowingBubbles"), 5: ("BQSquare",)}
SOURCE_FIELDS = ("relative_path", "class", "sequence", "bg_qp", "source_frame",
                 "source_png", "source_type")
PAIR_FIELDS = ("input", "gt", "mask", "class", "sequence", "bg_qp", "source_frame")

sys.path.insert(0, str(VCM / "Scripts"))
sys.path.insert(0, str(VCM / "Plot"))
import run_sfu_full_bg_qp52_priority as priority  # noqa: E402
import postprocess_sfu_full_qp0_bg_multiqp as post  # noqa: E402


def status(phase, **details):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    data = {"phase": phase, "updated_at": datetime.now().astimezone().isoformat(), **details}
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(STATUS)
    print(json.dumps(data), flush=True)


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def wait_for_qp48():
    final = RUN48 / "student/checkpoint-0004000/raft_student/pytorch_model.bin"
    while True:
        latest = RUN48 / "student/latest.json"
        step = json.loads(latest.read_text()).get("global_step", 0) if latest.is_file() else 0
        if step == 4000 and final.is_file():
            break
        if not Path(f"/proc/{SUPERVISOR_PID}").exists():
            raise RuntimeError("Paused queue supervisor disappeared")
        status("waiting_for_qp48", step=step)
        time.sleep(30)
    while True:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
        used = [int(line.strip().split()[0]) for line in result.splitlines()]
        if used[4] < 2048 and used[5] < 2048:
            break
        status("waiting_for_gpus", memory_mib={"4": used[4], "5": used[5]})
        time.sleep(30)


def encode():
    priority.ASSIGNMENTS = ASSIGNMENTS
    infos = priority.preflight()
    status("encoding", sequences=list(NAMES))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(encode_gpu, gpu, names, infos): gpu
                   for gpu, names in ASSIGNMENTS.items()}
        for future in as_completed(futures):
            future.result()


def encode_gpu(gpu, names, infos):
    for name in names:
        if not priority.encode(infos[name], gpu):
            raise RuntimeError(f"BG52 full encode failed: {name}")


def postprocess():
    manifest = json.loads((VCM / "Descriptors/SFU_full/manifest.json").read_text())
    entries = {entry["name"]: entry for entry in manifest["sequences"]}
    status("postprocessing")

    def process(name):
        info = post.source_info(name, entries[name])
        roi_video, _, _, _, gt_video = post.paths(info, 52)
        if not post.valid_mp4(roi_video, info["width"], info["height"],
                              info["fps"], info["frames"]):
            raise RuntimeError(f"Full ROI MP4 missing: {roi_video}")
        if not post.valid_mp4(gt_video, info["width"], info["height"],
                              info["fps"], info["frames"]):
            raise RuntimeError(f"Full GT MP4 missing: {gt_video}")
        info["roi_yuv"] = roi_video
        info["roi_source_type"] = "mp4"
        info["roi_masks"] = ast.literal_eval(info["roi_descriptor"].read_text())
        if set(info["roi_masks"]) != set(range(info["frames"])):
            raise RuntimeError(f"Incomplete ROI descriptor: {name}")
        source = post.bg_source(info, 52, entries)
        if source is None or source["type"] != "decode_new_qp52":
            raise RuntimeError(f"Full new BG52 bitstream missing: {name}")
        return post.process_sequence(name, [(info, 52, source)])[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(process, name): name for name in NAMES}
        rows = [future.result() for future in as_completed(futures)]
    current = json.loads(post.manifest_path().read_text())
    existing = {(row["sequence"], row["bg_qp"]): row for row in current["cases"]}
    existing.update({(row["sequence"], row["bg_qp"]): row for row in rows})
    skipped = [row for row in current["skipped_cases"]
               if (row["sequence"], row["bg_qp"]) not in existing]
    post.write_manifest(list(existing.values()), skipped, current.get("cleanup"))
    return {row["sequence"]: row for row in rows}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def link(source, target):
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        a, b = source.stat(), target.stat()
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            raise FileExistsError(target)
    else:
        os.link(source, target)


def add_dataset(cases):
    status("adding_dataset")
    for name in NAMES:
        case = cases[name]
        if not post.valid_png(Path(case["png_dir"]), case["frames"]):
            raise RuntimeError(f"Incomplete input PNG set: {name}")
    for split in ("train", "test"):
        existing = {(row["sequence"], row["source_frame"]): row
                    for row in read_csv(BG52 / f"{split}.csv")}
        sources, pairs = [], []
        for ref, pair in zip(read_csv(BG56 / f"{split}.csv"),
                             read_csv(BG56 / f"{split}_pairs.csv")):
            name = ref["sequence"]
            key = (name, ref["source_frame"])
            frame = int(ref["source_frame"])
            if pair["input"] != ref["relative_path"]:
                raise RuntimeError(f"Mismatched reference row: {key}")
            if name in NAMES:
                source = Path(cases[name]["png_dir"]) / f"{frame:06d}.png"
                link(source, BG52 / ref["relative_path"])
                link(BG56 / pair["gt"], BG52 / pair["gt"])
                link(BG56 / pair["mask"], BG52 / pair["mask"])
                sources.append({**ref, "bg_qp": "52", "source_png": str(source),
                                "source_type": "decode_new_qp52"})
            else:
                source_row = existing.get(key)
                if source_row is None or source_row["relative_path"] != ref["relative_path"]:
                    raise RuntimeError(f"Existing BG52 row missing: {key}")
                for kind in ("input", "gt", "mask"):
                    if not (BG52 / pair[kind]).is_file():
                        raise FileNotFoundError(BG52 / pair[kind])
                sources.append(source_row)
            pairs.append({**pair, "bg_qp": "52"})
        sources.sort(key=lambda row: (row["class"], row["sequence"], int(row["source_frame"])))
        pairs.sort(key=lambda row: (row["class"], row["sequence"], int(row["source_frame"])))
        if len(sources) != len(pairs) or any(
                src["relative_path"] != pair["input"] for src, pair in zip(sources, pairs)):
            raise RuntimeError(f"BG52 {split} metadata misaligned")
        expected = 4824 if split == "train" else 1206
        if len(sources) != expected:
            raise RuntimeError(f"BG52 {split} has {len(sources)} rows, expected {expected}")
        write_csv(BG52 / f"{split}.csv", SOURCE_FIELDS, sources)
        write_csv(BG52 / f"{split}_pairs.csv", PAIR_FIELDS, pairs)
    summary = json.loads((BG56 / "summary.json").read_text())
    summary["bg_qp"] = 52
    (BG52 / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (BG52 / "triplet_summary.json").write_text(
        json.dumps({"train": 4824, "test": 1206}, indent=2) + "\n")
    status("dataset_complete", sequences=14, train_images=4824, test_images=1206)


def prepare_teacher_and_resume():
    stage = RUN52 / "dataset"
    teacher = RUN52 / "teacher_flows"
    run([GUIDED_PYTHON, HERE / "prepare_sfu_flow_split.py",
         "--source", BG52, "--output", stage, "--expected-qp", "52"])
    run([GUIDED_PYTHON, HERE / "reuse_sfu_teacher_cache.py",
         "--source-dataset", RUN46 / "dataset",
         "--source-cache", RUN46 / "teacher_flows",
         "--dataset", stage, "--output", teacher])
    run([GUIDED_PYTHON, HERE / "train_raft_student_flow.py",
         "--dataset_root", stage, "--teacher_flow_root", teacher,
         "--output_dir", RUN52 / "student", "--spatial_mode", "max_side",
         "--max_input_side", "960", "--raft_checkpoint",
         WORKSPACE / "videoInpainting/pretrained/ProPainter/weights/raft-things.pth",
         "--preflight_only"])
    manifest = json.loads((stage / "manifest.json").read_text())
    if len(manifest["sequences"]) != 14:
        raise RuntimeError("BG52 staged dataset is not complete")
    process = Path(f"/proc/{SUPERVISOR_PID}/status")
    if not process.is_file() or "State:\tT" not in process.read_text():
        raise RuntimeError("Queue supervisor is no longer stopped; refusing to signal")
    status("ready_to_resume", sequences=14)
    os.kill(SUPERVISOR_PID, signal.SIGCONT)
    status("queue_resumed", supervisor_pid=SUPERVISOR_PID)


def main():
    wait_for_qp48()
    encode()
    cases = postprocess()
    add_dataset(cases)
    prepare_teacher_and_resume()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=str(error))
        raise
