#!/usr/bin/env python3
"""Add only the missing WE-L2 scores to the completed SFU BG56 PNG metrics."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


BRUSHNET = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRUSHNET / "Quan_test"))
from evaluate_temporal_metrics import compute_flow, read_rgb, warp_with_flow


EVAL = BRUSHNET / "experiments/eval_sfu_bg56_table_window_bvi_ckpt2000_rgb_tg"
OUTPUT = EVAL / "metrics_full_reference_no_we"
CASE_CACHE = OUTPUT / "we_l2_cases"
DEFINITION = (
    "Farneback flow from next prediction to previous prediction; warp previous "
    "RGB frame; mean squared RGB error over valid pixels in [0,1]"
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def case_score(record: dict) -> dict:
    import cv2

    cv2.setNumThreads(1)
    name, cls = record["sequence"], record["class"]
    folder = EVAL / "test" / cls / name
    ids = list(range(record["frame_range"][0], record["frame_range"][1] + 1))
    paths = [folder / f"{frame_id:06d}.png" for frame_id in ids]
    if len(paths) != record["frame_count"] or any(not path.is_file() for path in paths):
        raise ValueError(f"Missing prediction frames for {name}")
    previous = read_rgb(paths[0])
    values = []
    for path in paths[1:]:
        current = read_rgb(path)
        if current.shape != previous.shape:
            raise ValueError(f"Frame geometry changes within {name}")
        # Identical flow, warp, valid-pixel mask and L2 formula to
        # evaluate_temporal_metrics.warping_error; the WE (L1) term is omitted.
        backward_flow = compute_flow(current, previous)
        warped_previous, valid = warp_with_flow(previous, backward_flow)
        if not np.any(valid):
            raise ValueError(f"No valid warped pixels for {name}, {path.name}")
        difference = warped_previous[valid] - current[valid]
        values.append(float(np.mean(difference * difference)))
        previous = current
    result = {"sequence": name, "class": cls, "frames": len(paths),
              "pairs": len(values), "WE_L2": float(np.mean(values))}
    if not math.isfinite(result["WE_L2"]):
        raise ValueError(f"Non-finite WE-L2 for {name}")
    write_json(CASE_CACHE / f"{name}.json", result)
    return result


def cached_score(record: dict) -> dict | None:
    path = CASE_CACHE / f"{record['sequence']}.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value["sequence"] == record["sequence"]
                and value["frames"] == record["frame_count"]
                and value["pairs"] == record["frame_count"] - 1
                and math.isfinite(float(value["WE_L2"]))):
            return value
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return None


def update_case_files(record: dict, score: dict) -> None:
    name = record["sequence"]
    folder = OUTPUT / "cases" / name
    csv_path = folder / "metrics.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if "WE" in fields:
        raise ValueError(f"Unexpected WE column for {name}")
    if "WE_L2" not in fields:
        fields.insert(fields.index("FS"), "WE_L2")
    for row in rows:
        row["WE_L2"] = score["WE_L2"]
    write_csv(csv_path, fields, rows)
    json_path = folder / "metrics.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["definitions"].pop("WE", None)
    payload["definitions"]["WE_L2"] = DEFINITION
    for row in payload["results"]:
        row["WE_L2"] = score["WE_L2"]
    write_json(json_path, payload)
    row_path = folder / f"{name}.json"
    if row_path.is_file():
        row = json.loads(row_path.read_text(encoding="utf-8"))
        row["WE_L2"] = score["WE_L2"]
        write_json(row_path, row)


def update_aggregates(scores: dict[str, dict]) -> None:
    csv_path = OUTPUT / "per_sequence.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if len(rows) != 14 or "WE" in fields:
        raise ValueError("Unexpected per-sequence metrics schema")
    for key in ("WE_L2", "WE-L2 (×10⁻²)"):
        if key in fields:
            fields.remove(key)
    position = fields.index("FS")
    fields[position:position] = ["WE_L2", "WE-L2 (×10⁻²)"]
    for row in rows:
        score = scores[row["sequence"]]["WE_L2"]
        row["WE_L2"] = score
        row["WE-L2 (×10⁻²)"] = score * 100
    write_csv(csv_path, fields, rows)

    macro = float(np.mean([scores[row["sequence"]]["WE_L2"] for row in rows]))
    weighted = float(np.average(
        [scores[row["sequence"]]["WE_L2"] for row in rows],
        weights=[int(row["pairs"]) for row in rows],
    ))
    summary_path = OUTPUT / "summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        summaries = list(reader)
        summary_fields = list(reader.fieldnames or ())
    for key in ("WE_L2", "WE-L2 (×10⁻²)"):
        if key in summary_fields:
            summary_fields.remove(key)
    position = summary_fields.index("FS")
    summary_fields[position:position] = ["WE_L2", "WE-L2 (×10⁻²)"]
    for row in summaries:
        score = macro if row["aggregation"] == "sequence_macro_mean" else weighted
        row["WE_L2"] = score
        row["WE-L2 (×10⁻²)"] = score * 100
    write_csv(summary_path, summary_fields, summaries)

    summary_json = OUTPUT / "summary.json"
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    payload["definitions"].pop("WE_and_WE_L2", None)
    payload["definitions"]["WE"] = "not computed"
    payload["definitions"]["WE_L2"] = DEFINITION
    payload["definitions"]["WE_L2_display_multiplier"] = 100
    for row in payload["results"]:
        score = macro if row["aggregation"] == "sequence_macro_mean" else weighted
        row["WE_L2"] = score
        row["WE-L2 (×10⁻²)"] = score * 100
    write_json(summary_json, payload)

    table = OUTPUT / "comparison_row.csv"
    with table.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        table_rows = list(reader)
        table_fields = list(reader.fieldnames or ())
    if len(table_rows) != 1:
        raise ValueError("Expected one comparison table row")
    if "WE-L2 (×10⁻²)" not in table_fields:
        table_fields.insert(table_fields.index("FS"), "WE-L2 (×10⁻²)")
    table_rows[0]["WE-L2 (×10⁻²)"] = macro * 100
    write_csv(table, table_fields, table_rows)
    label = table_rows[0]["Method"]
    markdown = (
        "| Method | Sequences | PSNR ↑ | SSIM ↑ | LPIPS ↓ | WE-L2 (×10⁻²) ↓ | FS ↑ | VFID ↓ |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| {label} | 14 | {float(table_rows[0]['PSNR']):.4f} | "
        f"{float(table_rows[0]['SSIM']):.5f} | {float(table_rows[0]['LPIPS']):.5f} | "
        f"{macro * 100:.5f} | {float(table_rows[0]['FS']):.5f} | "
        f"{float(table_rows[0]['VFID']):.4f} |\n"
    )
    path = OUTPUT / "comparison_row.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(markdown, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    status = json.loads((EVAL / "logs/status.json").read_text(encoding="utf-8"))
    if len(status) != 14 or any(row["state"] != "complete" for row in status):
        raise ValueError("Expected 14 complete BG56 sequences")
    CASE_CACHE.mkdir(parents=True, exist_ok=True)
    scores = {row["sequence"]: cached_score(row) for row in status}
    pending = [row for row in status if scores[row["sequence"]] is None]
    print(f"WE-L2: {len(status) - len(pending)} cached, {len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=min(8, len(pending)) or 1) as pool:
        futures = {pool.submit(case_score, row): row["sequence"] for row in pending}
        for future in as_completed(futures):
            score = future.result()
            scores[score["sequence"]] = score
            print(f"[DONE] {score['sequence']}: {score['WE_L2']:.8f}", flush=True)
    if any(value is None for value in scores.values()):
        raise ValueError("Some WE-L2 cases are still missing")
    for record in status:
        update_case_files(record, scores[record["sequence"]])
    update_aggregates(scores)
    print(f"Updated {OUTPUT / 'comparison_row.md'}", flush=True)


if __name__ == "__main__":
    main()
