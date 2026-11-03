#!/usr/bin/env python3
"""GPU smoke validation of deployment inference with the three additional BG models."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "examples/brushnet/inference/restoration_engine.py"
DATA = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test")
OUTPUT = ROOT / "Quan_test/inference_restore_512_validation"
MODES = (("tg", "tg_reference"), ("cge", "cge_step25"), ("cge_then_tg", "combined_step25"))
LOCK = threading.Lock()
STATUS = {}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def update_status(key, value):
    with LOCK:
        STATUS[key] = value
        save_json(OUTPUT / "additional_bg_validation_status.json", STATUS)
        print(key, json.dumps(value), flush=True)


def verify(bg, mode, folder):
    report = json.loads((folder / "metadata/Traffic/clip_000000.json").read_text())
    manifest = json.loads((folder / "run_config.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["frame_count"] == 4
    assert [s["step_index"] for s in report["steps"]] == list(range(50))
    cge = [s["step_index"] for s in report["steps"] if s["cge_applied"]]
    tg = [s["step_index"] for s in report["steps"] if s["temporal_applied"]]
    assert cge == ([] if mode == "tg" else [25]), cge
    assert tg == (list(range(50)) if mode == "tg" else [25] if mode == "cge_then_tg" else []), tg
    assert manifest["settings"]["cge_bg_quality"] == bg
    assert len(report["frames"]) == 4
    for step in report["steps"]:
        if step["temporal_applied"]:
            assert math.isfinite(step["temporal_loss"]) and step["temporal_loss"] >= 0
        if step["cge_applied"]:
            assert math.isfinite(step["cge_delta_norm"]) and step["cge_delta_norm"] > 0
        if mode == "cge_then_tg" and step["temporal_applied"]:
            assert step["temporal_input"] == "equivalent_clean_after_cge"
    rows = []
    for name in report["frames"]:
        dataset = DATA / ("bg" + str(bg)) / "test"
        native = np.array(Image.open(dataset / "input/Class_A/Traffic" / name).convert("RGB"))
        mask = np.array(Image.open(dataset / "mask/Class_A/Traffic" / name).convert("RGB"))
        roi = mask.astype(np.uint16).sum(axis=2) > 255
        png = Path(name).stem + ".png"
        with Image.open(folder / "512/Traffic" / png) as im:
            generated = im.convert("RGB")
            assert generated.size == (512, 512)
            expected_upscale = np.array(generated.resize((native.shape[1], native.shape[0]), Image.Resampling.LANCZOS))
            std_512 = float(np.array(generated).std())
            assert std_512 > 0
        upscaled = np.array(Image.open(folder / "resized/Traffic" / png).convert("RGB"))
        restored = np.array(Image.open(folder / "final/Traffic" / png).convert("RGB"))
        expected_native = expected_upscale.copy()
        expected_native[roi] = native[roi]
        assert restored.shape == native.shape
        assert np.array_equal(upscaled, expected_upscale)
        assert np.array_equal(restored[roi], native[roi])
        assert np.array_equal(restored, expected_native)
        rows.append(dict(frame=name, native_size_wh=[native.shape[1], native.shape[0]],
                         roi_pixels=int(roi.sum()), roi_exact=True, lanczos_exact=True,
                         native_matches_legacy_postprocess=True, image_512_std=std_512))
    result = dict(bg=bg, mode=mode, passed=True, frame_count=4, ddim_steps=50,
                  cge_steps=cge, tg_steps=tg, generation_seconds=report["generation_seconds"],
                  generation_seconds_per_frame=report["generation_seconds_per_frame"],
                  checkpoint=manifest["settings"]["checkpoint"],
                  flow_checkpoint=manifest["settings"]["flow_checkpoint"], frames=rows)
    save_json(folder / "validation.json", result)
    return result


def run_bg(bg, gpu, entries):
    results = []
    for mode, label in MODES:
        key = "bg{}_{}".format(bg, mode)
        folder = OUTPUT / ("bg" + str(bg)) / label
        log_path = OUTPUT / "logs" / (key + ".log")
        command = [sys.executable, "-u", str(ENTRY),
                   "--checkpoint", entries["ckpt_bg" + str(bg)],
                   "--flow-checkpoint", entries["student_flow_bg" + str(bg)],
                   "--input-dir", str(DATA / ("bg" + str(bg)) / "test/input/Class_A/Traffic"),
                   "--mask-dir", str(DATA / ("bg" + str(bg)) / "test/mask/Class_A/Traffic"),
                   "--output-dir", str(folder), "--guidance", mode, "--gpu", str(gpu),
                   "--bg-qp", str(bg), "--max-frames", "4", "--save-intermediates",
                   "--steps", "50", "--clip-size", "4", "--clip-stride", "4",
                   "--cge-scale", "0.001", "--tg-scale", "0.1",
                   "--cge-start", "25", "--cge-end", "26",
                   "--tg-start", "0" if mode == "tg" else "25",
                   "--tg-end", "50" if mode == "tg" else "26",
                   "--codec-workers", "4", "--codec-max-parallel", "4"]
        save_json(OUTPUT / "commands" / (key + ".json"), command)
        state = dict(status="running", gpu=gpu, started_at=datetime.now().astimezone().isoformat(), log=str(log_path))
        update_status(key, state)
        started = time.monotonic()
        try:
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = verify(bg, mode, folder)
            result.update(gpu=gpu, log=str(log_path), wall_seconds=time.monotonic() - started)
            state.update(status="passed", generation_seconds=result["generation_seconds"], wall_seconds=result["wall_seconds"])
        except Exception as error:
            result = dict(bg=bg, mode=mode, passed=False, error=repr(error), log=str(log_path))
            state.update(status="failed", error=repr(error))
        update_status(key, state)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=3, default=[0, 3, 5], help="One currently idle GPU per BG")
    args = parser.parse_args()
    if len(set(args.gpus)) != 3:
        parser.error("Choose three distinct GPUs")
    # This catalog contains only simple key: /absolute/path entries.
    catalog = ROOT / "experiments/EVAL_FOR_FINAL/ckpt.yaml"
    entries = dict(line.split(": ", 1) for line in catalog.read_text().splitlines() if line.strip())
    for bg in (46, 48, 56):
        assert not (OUTPUT / ("bg" + str(bg))).exists(), "Do not overwrite previous validation"
        for prefix in ("ckpt_bg", "student_flow_bg"):
            assert Path(entries[prefix + str(bg)]).exists()
    started_at = datetime.now().astimezone().isoformat()
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_bg, bg, gpu, entries) for bg, gpu in zip((46, 48, 56), args.gpus)]
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda r: (r["bg"], r["mode"]))
    summary = dict(started_at=started_at, finished_at=datetime.now().astimezone().isoformat(),
                   source=str(ENTRY), source_sha256=hashlib.sha256(ENTRY.read_bytes()).hexdigest(),
                   catalog=str(catalog), catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
                   scope="First four Traffic frames per BG/mode; 50 DDIM steps. TG-only [0,50); CGE-only and combined [25,26).",
                   limitations="Smoke compatibility and exact native postprocessing validation; no whole-dataset quality evaluation or identical-reference-image comparison. Timings share CPU/codec resources across GPUs.",
                   all_passed=all(r["passed"] for r in results), runs=results)
    save_json(OUTPUT / "additional_bg_validation_summary.json", summary)
    print("ALL PASSED:", summary["all_passed"], flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
