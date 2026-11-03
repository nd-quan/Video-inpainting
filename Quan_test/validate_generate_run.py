#!/usr/bin/env python3
"""Audit real one-clip-per-case generate.py outputs, including exact native ROI."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/brushnet/inference"))
from media_io import frame_bytes
from restoration_engine import read_native


def validate(root, require_complete=True):
    request = json.loads((root / "request.json").read_text())
    compact = request.get("output_layout") == "png_yuv"
    public = Path(request["output_dir"]) if compact else root
    if compact:
        allowed = {"png", "yuv"} | ({request["public_log"]} if request.get("public_log") else set())
        assert {p.name for p in public.iterdir()} == allowed
        assert not list(public.rglob("*.json")) and not list(public.rglob("*.jsonl"))
    rows, unfinished = [], []
    for job in request["jobs"]:
        relative = Path(job["relative"])
        case = root / "frames" / ("bg" + job["qp"])
        final = public / "png" / ("bg" + job["qp"]) / relative if compact else case / "final" / relative
        metadata = case / "metadata" / relative
        complete = metadata / "complete.json"
        if not complete.exists():
            unfinished.append(job["qp"] + "/" + job["sequence"])
            continue
        record = json.loads(complete.read_text())
        clip = json.loads((metadata / "clip_000000.json").read_text())
        steps = clip["steps"]
        assert record["frames"] == 8 and record["clips"] == 1
        assert len(steps) == 50
        assert [s["step_index"] for s in steps if s["cge_applied"]] == [20, 30]
        assert [s["step_index"] for s in steps if s["temporal_requested"]] == list(range(50))
        assert [s["step_index"] for s in steps if s["temporal_applied"]] == list(range(50))
        assert all(s["temporal_input"] == "equivalent_clean_after_cge" for s in steps)
        assert all(math.isfinite(s["temporal_loss"]) for s in steps)
        assert all(math.isfinite(s["cge_delta_norm"]) and s["cge_delta_norm"] > 0 for s in steps if s["cge_applied"])
        assert record["codec_tasks"] == 48 and 1 <= record["peak_codec_encoders"] <= 24
        assert len(record["codec_reports"]) == 2
        assert all(r["frames"] == 8 and r["tasks"] == 24 and r["pool_workers"] == 24 and r["all_results_joined"]
                   for r in record["codec_reports"])
        assert record["codec_equivalence"]["passed"]
        assert record["codec_equivalence"]["gradient_max_abs_difference"] == 0
        names = record["media"]["frame_names"]
        source = Path(job["input"])
        if not source.is_dir():
            source = root / "decoded" / ("bg" + job["qp"]) / relative
        roi_pixels = 0
        for name in names:
            original, roi = read_native(source / name, Path(job["masks"]) / name)
            with Image.open(final / name) as image:
                restored = np.asarray(image.convert("RGB"))
            assert restored.shape == original.shape
            assert np.array_equal(restored[roi], original[roi]), str(relative / name)
            roi_pixels += int(roi.sum())
            if request["save_intermediates"]:
                with Image.open(case / "512" / relative / name) as image:
                    assert image.size == (512, 512)
                with Image.open(case / "resized" / relative / name) as image:
                    resized = np.asarray(image.convert("RGB"))
                assert np.array_equal(restored[~roi], resized[~roi])
        assert len(list(final.glob("*.png"))) == 8
        raw = Path(record["yuv"])
        sidecar = record["yuv_metadata"] if compact else json.loads(raw.with_suffix(".yuv.json").read_text())
        if compact:
            assert not raw.with_suffix(".yuv.json").exists()
        assert sidecar["frame_count"] == 8 and sidecar["frames"] == names
        assert raw.stat().st_size == frame_bytes(sidecar["width"], sidecar["height"], sidecar["pixel_format"]) * 8
        assert (sidecar["width"], sidecar["height"]) == (original.shape[1], original.shape[0])
        assert str(record["settings"]["cge_bg_quality"]) == job["qp"]
        assert record["settings"]["checkpoint"] == request["config"]["models"][job["qp"]]["checkpoint"]
        assert record["settings"]["flow_checkpoint"] == request["config"]["models"][job["qp"]]["flow_checkpoint"]
        assert (record["settings"]["temporal_clip_size"], record["settings"]["temporal_clip_stride"]) == (8, 6)
        rows.append(dict(qp=job["qp"], sequence=job["sequence"], input_type=record["media"]["conversion"],
                         frames=8, cge_tg_step_seconds=record["mean_cge_then_tg_step_seconds"],
                         generate_seconds_per_frame=record["generation_seconds_per_computed_frame"],
                         generate_seconds_per_clip=record["generation_seconds"],
                         total_sequence_seconds=record["total_sequence_seconds"], native_roi_pixels_exact=roi_pixels,
                         codec_tasks=48, peak_codec_encoders=record["peak_codec_encoders"]))
    if require_complete and unfinished:
        raise AssertionError("Unfinished: " + str(unfinished))
    if require_complete:
        assert json.loads((root / "status.json").read_text())["phase"] == "complete"
        assert json.loads((root / "summary.json").read_text())["cases"] == len(rows)
    events, completed_encoders = [], 0
    for path in (root / "workers").glob("*.codec.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            starting = event["event"] == "encoder_start"
            events.append((event["monotonic"], 1 if starting else -1))
            if not starting:
                assert event["success"]
                completed_encoders += 1
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
        assert active >= 0
    if require_complete:
        state = json.loads((root / "status.json").read_text())
        limit = len(state["gpus"]) * state["sequences_per_gpu"] * request["config"]["settings"]["cge_max_parallel"]
        assert active == 0 and peak <= limit
        assert completed_encoders == len(rows) * 48
    destination = root.parent / "validation"
    destination.mkdir(exist_ok=True)
    if rows:
        with (destination / "timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(passed_cases=len(rows), total_cases=len(request["jobs"]), unfinished=unfinished,
                  frames_verified=8 * len(rows), all_complete=not unfinished, cases=rows,
                  global_peak_encoders=peak, completed_encoders=completed_encoders,
                  output_layout=request.get("output_layout", "legacy"), public_output=str(public))
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Official generate.py smoke validation", "",
             "Verified {}/{} cases ({} native frames).".format(len(rows), len(request["jobs"]), 8 * len(rows)), "",
             "Completed encoder calls: {}; global peak concurrent encoders: {}.".format(completed_encoders, peak), "",
             "Each verified case: CGE at loop indices 20/30, TG at all 50 indices, 48 codec calls, "
             "24-task pools, exact native RGB ROI, 512 inference images, and an 8-frame native YUV.", "",
             "Generation includes denoising and final VAE decode; preparation/export/model loading are excluded.", "",
             "CGE→TG step time measures the scheduler call, including codec/VAE/TG and excluding the preceding UNet/BrushNet forward.", "",
             "| QP | Sequence | Input | CGE→TG s/step | Generate s/frame | Generate s/clip | Sequence total s |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {qp} | {sequence} | {input_type} | {cge_tg_step_seconds:.3f} | "
                     "{generate_seconds_per_frame:.3f} | {generate_seconds_per_clip:.3f} | "
                     "{total_sequence_seconds:.3f} |".format(**row))
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    validate(a.run.resolve(), not a.partial)
