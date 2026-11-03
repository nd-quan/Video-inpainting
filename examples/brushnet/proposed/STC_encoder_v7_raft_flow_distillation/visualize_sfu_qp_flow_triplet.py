#!/usr/bin/env python3
"""Compare pretrained coarse RAFT, QP student, and cached clean teacher flow."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

BRUSHNET = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BRUSHNET / "examples/brushnet"))
import visualize_raft_student_flow as vis  # noqa: E402
EXPERIMENTS = BRUSHNET / "experiments"
RUNS = {
    46: "train_v7_raft_student_sfu_bg46_max960_4000",
    48: "train_v7_raft_student_sfu_bg48_max960_from_bg46_3500_4000",
    52: "train_v7_raft_student_sfu_bg52_max960_from_previous_best_4000",
    56: "train_v7_raft_student_sfu_bg56_max960_from_previous_best_4000",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qp", type=int, choices=sorted(RUNS), required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--pairs_per_sequence", type=int, default=2)
    parser.add_argument("--evaluate_all_valid", action="store_true")
    parser.add_argument("--include_branches", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def save_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def main():
    args = parse_args()
    torch.set_num_threads(4)
    run = EXPERIMENTS / RUNS[args.qp]
    checkpoint, component = vis.resolve_checkpoint(run / "student/best.json")
    cfg = json.loads((component / "config.json").read_text())
    if cfg.get("architecture") != "propainter_raft_large" or int(cfg["iterations"]) != 20:
        raise ValueError("Expected the 20-iteration ProPainter RAFT student")
    dataset = vis.RAFTTeacherFlowPairDataset(
        run / "dataset", run / "teacher_flows", "valid", 512,
        args.include_branches, spatial_mode="max_side", max_input_side=960,
    )
    montage_indices = set(vis.select_pair_indices(dataset, args.pairs_per_sequence))
    selected = list(range(len(dataset))) if args.evaluate_all_valid else sorted(montage_indices)
    output = args.output_root.expanduser().resolve() / f"qp{args.qp}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists: {output}")
    montages = output / "montages"
    montages.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    coarse = vis.RAFTStudentFlowPredictor(
        cfg["propainter_root"], cfg["source_checkpoint"],
        iterations=20, mixed_precision=False,
        freeze_batchnorm=cfg.get("freeze_batchnorm", True),
    ).to(device).eval().requires_grad_(False)
    student = vis.RAFTStudentFlowPredictor.from_pretrained(
        component, mixed_precision=False,
    ).to(device).eval().requires_grad_(False)

    rows = []
    for count, index in enumerate(selected, 1):
        sample = dataset[index]
        a = sample["degraded0"].unsqueeze(0).to(device)
        b = sample["degraded1"].unsqueeze(0).to(device)
        with torch.inference_mode():
            coarse_flows = [flow[0].float() for flow in
                            coarse.predict_bidirectional(a, b, pair_batch_size=1)]
            student_flows = [flow[0].float() for flow in
                             student.predict_bidirectional(a, b, pair_batch_size=1)]
        teachers, valids, bgs = [], [], []
        for direction, bg_key in (("forward", "bg0"), ("backward", "bg1")):
            teacher, valid = vis.prepare_teacher_flow(
                sample["teacher_" + direction][None, None].to(device),
                a.shape[-2:], sample["valid_" + direction][None, None].to(device),
            )
            teachers.append(teacher[0, 0].float())
            valids.append(valid[0, 0].float())
            bgs.append(sample[bg_key].to(device).float())

        # Teacher is identical across QPs for a semantic frame pair; this
        # produces the same flow color scale across the three QP runs.
        flow_scale = max(vis.percentile_scale(
            [flow.norm(dim=0) for flow in teachers], 99), 1.0)
        errors = [[(predictions[d] - teachers[d]).norm(dim=0, keepdim=True)
                   for d in range(2)] for predictions in (coarse_flows, student_flows)]
        error_scale = vis.percentile_scale(errors[0], 99)
        row = {
            "qp": args.qp, "sequence": str(sample["video"]),
            "frame_t": int(sample["frame0"]), "frame_t1": int(sample["frame1"]),
            "height": int(a.shape[-2]), "width": int(a.shape[-1]),
            "flow_scale_px": flow_scale, "epe_scale_px": error_scale,
        }
        for d, direction in enumerate(("forward", "backward")):
            for predictions, name in ((coarse_flows, "coarse"),
                                      (student_flows, "student")):
                stats = vis.direction_metrics(
                    predictions[d], teachers[d], valids[d], bgs[d])
                row.update({f"{direction}_{name}_{key}": value
                            for key, value in stats.items()})
            row[f"{direction}_bg_epe_reduction_pct"] = 100 * (
                row[f"{direction}_coarse_bg_epe"]
                - row[f"{direction}_student_bg_epe"]
            ) / max(row[f"{direction}_coarse_bg_epe"], 1e-8)

        if index not in montage_indices:
            rows.append(row)
            print(f"QP{args.qp} [{count}/{len(selected)}] {row['sequence']} "
                  f"{row['frame_t']}->{row['frame_t1']}", flush=True)
            continue
        tiles = [
            ("Degraded frame t", vis.rgb_to_bgr(a[0])),
            ("Degraded frame t+1", vis.rgb_to_bgr(b[0])),
            ("BG mask t", vis.mask_to_bgr(bgs[0])),
            ("BG mask t+1", vis.mask_to_bgr(bgs[1])),
        ]
        for d, direction in enumerate(("forward", "backward")):
            short = "F" if d == 0 else "B"
            tiles.extend([
                (f"Coarse {short} | BG EPE {row[f'{direction}_coarse_bg_epe']:.3f}",
                 vis.flow_to_bgr(coarse_flows[d], flow_scale)),
                (f"Student {short} | BG EPE {row[f'{direction}_student_bg_epe']:.3f}",
                 vis.flow_to_bgr(student_flows[d], flow_scale)),
                (f"Teacher {short} | flow scale {flow_scale:.1f}px",
                 vis.flow_to_bgr(teachers[d], flow_scale)),
                (f"Valid BG {short}", vis.mask_to_bgr(valids[d] * bgs[d])),
            ])
        tiles.extend([
            ("Coarse forward EPE", vis.heatmap(errors[0][0], error_scale)),
            ("Student forward EPE", vis.heatmap(errors[1][0], error_scale)),
            ("Coarse backward EPE", vis.heatmap(errors[0][1], error_scale)),
            ("Student backward EPE", vis.heatmap(errors[1][1], error_scale)),
        ])
        name = f"{vis.sanitize(row['sequence'])}_f{row['frame_t']:06d}_{row['frame_t1']:06d}.png"
        if not cv2.imwrite(str(montages / name), vis.make_montage(tiles)):
            raise OSError(montages / name)
        rows.append(row)
        print(f"QP{args.qp} [{count}/{len(selected)}] {name}", flush=True)

    save_csv(output / "per_pair_metrics.csv", rows)
    sequence_names = sorted({row["sequence"] for row in rows})
    keys = [key for key in rows[0] if key not in
            {"qp", "sequence", "frame_t", "frame_t1", "height", "width",
             "flow_scale_px", "epe_scale_px"}]
    per_sequence = [
        {"qp": args.qp, "sequence": name, "pair_count": sum(row["sequence"] == name for row in rows),
         **{key: mean([row for row in rows if row["sequence"] == name], key) for key in keys}}
        for name in sequence_names
    ]
    save_csv(output / "per_sequence_metrics.csv", per_sequence)
    summary = {
        "qp": args.qp, "checkpoint": str(checkpoint),
        "coarse_checkpoint": cfg["source_checkpoint"],
        "teacher_cache": str((run / "teacher_flows").resolve()),
        "dataset": str((run / "dataset").resolve()),
        "split": "valid", "spatial_mode": "max_side", "max_input_side": 960,
        "selected_pairs_per_sequence": args.pairs_per_sequence,
        "evaluation_scope": "all valid pairs" if args.evaluate_all_valid else "visualized pairs only",
        "pair_count": len(rows), "montage_count": len(montage_indices),
        "sequence_count": len(sequence_names),
        "flow_color_scale": "Teacher forward/backward p99, minimum 1 px; same across QPs for matching pairs",
        "epe_color_scale": "Coarse forward/backward p99 for each pair; shared with student EPE tiles",
        "coarse_definition": "Frozen pretrained ProPainter RAFT on degraded RGB",
        "student_definition": "Best QP-specific fine-tuned RAFT on the same degraded RGB",
        "teacher_definition": "Cached frozen RAFT flow from clean GT RGB; pseudo ground truth",
        "metric_support": "Teacher-valid pixels intersected with the directional BG mask",
        "metric_caveat": "Teacher is pseudo ground truth, not physical ground truth",
        "mean_per_pair": {key: mean(rows, key) for key in keys},
        "sequence_macro_average": {key: mean(per_sequence, key) for key in keys},
        "selected_pairs": [{k: row[k] for k in ("sequence", "frame_t", "frame_t1")}
                           for row in rows],
    }
    vis.json_dump(output / "summary.json", summary)
    print(json.dumps({"qp": args.qp, "pair_count": len(rows),
                      "coarse_forward_bg_epe": summary["sequence_macro_average"]["forward_coarse_bg_epe"],
                      "student_forward_bg_epe": summary["sequence_macro_average"]["forward_student_bg_epe"],
                      "coarse_backward_bg_epe": summary["sequence_macro_average"]["backward_coarse_bg_epe"],
                      "student_backward_bg_epe": summary["sequence_macro_average"]["backward_student_bg_epe"]}),
          flush=True)


if __name__ == "__main__":
    main()
