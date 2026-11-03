#!/usr/bin/env python3
"""Compare complete x3/x10 BG52 runs, adding x1 when its metrics are available."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512"
FULL = ROOT / "experiments/EVAL_FOR_FINAL/ablation_bg52_cge_x3_x10_tg0p1_C25-40_T25-40_FULL"


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = FULL / "comparison"
    output.mkdir(exist_ok=True)
    configs = {factor: json.loads((FULL / "cge_x{}".format(factor) / "run_config.json").read_text()) for factor in (3, 10)}
    for key in ("dataset", "checkpoint", "flow_checkpoint", "temporal_window", "cge_window",
                "temporal_guidance_scale", "main_noise_seed", "shared_noise_seed", "shared_bg_noise_strength"):
        if configs[3][key] != configs[10][key]:
            raise ValueError("Unmatched experiment setting: " + key)
    dataset = Path(configs[3]["dataset"])
    paths = {"cge_x3": FULL / "cge_x3", "cge_x10": FULL / "cge_x10"}
    if (BASE / "test/metrics_full_reference/metrics.csv").is_file():
        paths = {"cge_x1": BASE, **paths}
    all_rows, aggregate_rows, by_run = [], [], {}
    for label, folder in paths.items():
        with (folder / "test/metrics_full_reference/metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        aggregate = next(row for row in rows if row["sequence"] == "ALL")
        if int(aggregate["frames"]) != 1198 or len(rows) != 15:
            raise ValueError("Expected 14 complete sequences / 1198 frames for " + label)
        by_run[label] = {row["sequence"]: row for row in rows}
        timing_path = folder / "timing_summary.json"
        timings = json.loads(timing_path.read_text()) if timing_path.is_file() else None
        timing_by_seq = {row["sequence"]: row for row in timings["sequences"]} if timings else {}
        for row in rows:
            joined = {"run": label, **row}
            timing = timings if row["sequence"] == "ALL" else timing_by_seq.get(row["sequence"])
            joined.update({key: timing.get(key) if timing else None for key in (
                "generation_seconds", "generation_seconds_per_saved_image", "total_clip_seconds")})
            all_rows.append(joined)
            if row["sequence"] == "ALL":
                aggregate_rows.append(joined)
    write_csv(output / "metrics_by_sequence.csv", all_rows)
    write_csv(output / "metrics_summary.csv", aggregate_rows)
    deltas = []
    for sequence in by_run["cge_x3"]:
        a, b = by_run["cge_x3"][sequence], by_run["cge_x10"][sequence]
        deltas.append({"sequence": sequence, "frames": int(a["frames"]),
                       **{key + "_x10_minus_x3": float(b[key]) - float(a[key])
                          for key in ("PSNR", "SSIM", "LPIPS", "WE", "WE_L2", "FS", "VFID_Clips")}})
    write_csv(output / "x10_minus_x3.csv", deltas)

    pixel_rows = []
    for class_dir in sorted((dataset / "input").glob("Class_*")):
        for source in sorted(class_dir.iterdir()):
            if not source.is_dir():
                continue
            relative = source.relative_to(dataset / "input")
            abs_sum = bg_abs_sum = bg_count = pixels = changed = frames = 0
            for name in sorted(p.name for p in source.glob("*.png")):
                with Image.open(FULL / "cge_x3/test" / relative / name) as im:
                    a = np.asarray(im.convert("RGB")).astype(np.float32)
                with Image.open(FULL / "cge_x10/test" / relative / name) as im:
                    b = np.asarray(im.convert("RGB")).astype(np.float32)
                with Image.open(dataset / "mask" / relative / name) as im:
                    background = np.asarray(im.convert("L").resize((512, 512), Image.Resampling.NEAREST)) <= 127
                difference = np.abs(a-b)
                abs_sum += float(difference.sum())
                bg_abs_sum += float(difference[background].sum())
                bg_count += int(background.sum())
                changed += int(np.any(difference > 0, axis=2).sum())
                pixels += a.shape[0] * a.shape[1]
                frames += 1
            pixel_rows.append({"sequence": source.name, "frames": frames,
                               "mae_x3_vs_x10": abs_sum / (pixels*3),
                               "background_mae_x3_vs_x10": bg_abs_sum / max(bg_count*3, 1),
                               "changed_pixels_pct": 100*changed/pixels,
                               "absolute_difference_sum": abs_sum, "background_difference_sum": bg_abs_sum,
                               "background_pixels": bg_count, "pixels": pixels, "changed_pixels": changed})
    total = {key: sum(row[key] for row in pixel_rows) for key in (
        "frames", "absolute_difference_sum", "background_difference_sum", "background_pixels", "pixels", "changed_pixels")}
    total.update(sequence="ALL", mae_x3_vs_x10=total["absolute_difference_sum"]/(total["pixels"]*3),
                 background_mae_x3_vs_x10=total["background_difference_sum"]/max(total["background_pixels"]*3, 1),
                 changed_pixels_pct=100*total["changed_pixels"]/total["pixels"])
    pixel_rows.append(total)
    write_csv(output / "pixel_difference_by_sequence.csv", pixel_rows)
    (output / "summary.json").write_text(json.dumps({
        "scope": "Complete BG52/test: 14 sequences, 1198 frames per variant",
        "metrics": aggregate_rows, "x10_minus_x3": deltas,
        "pixel_differences": pixel_rows,
        "timing_semantics": "Sum of synchronized clip wall times divided by unique saved frame count, including shifted-tail overhead and reused pilot clip timing; model loading excluded. Not parallel end-to-end wall time.",
        "metric_definitions": str(FULL / "cge_x3/test/metrics_full_reference/metrics.json"),
    }, indent=2) + "\n")
    print(json.dumps(aggregate_rows, indent=2))
    print("Comparison saved: " + str(output), flush=True)


if __name__ == "__main__":
    main()
