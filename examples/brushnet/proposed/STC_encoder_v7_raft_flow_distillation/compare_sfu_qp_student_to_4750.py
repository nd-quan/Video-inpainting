#!/usr/bin/env python3
"""Compare the old 512-trained V7 student with one QP-specific student."""

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
OLD = EXPERIMENTS / "train_v7_raft_student_flow/checkpoint-0004750"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qp", type=int, choices=sorted(RUNS), required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--old_checkpoint", type=Path, default=OLD)
    parser.add_argument("--pairs_per_sequence", type=int, default=2)
    parser.add_argument("--include_branches", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def save_csv(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def main():
    args = parse_args()
    torch.set_num_threads(4)
    run = EXPERIMENTS / RUNS[args.qp]
    old_root, old_component = vis.resolve_checkpoint(args.old_checkpoint)
    new_root, new_component = vis.resolve_checkpoint(run / "student/best.json")
    old_cfg = json.loads((old_component / "config.json").read_text())
    new_cfg = json.loads((new_component / "config.json").read_text())
    for key in ("architecture", "iterations", "freeze_batchnorm", "source_checkpoint"):
        if old_cfg[key] != new_cfg[key]:
            raise ValueError(f"The student architectures differ at {key}")
    if old_cfg["architecture"] != "propainter_raft_large" or int(old_cfg["iterations"]) != 20:
        raise ValueError("Expected ProPainter RAFT with 20 iterations")
    dataset = vis.RAFTTeacherFlowPairDataset(
        run / "dataset", run / "teacher_flows", "valid", 512,
        include_branches=args.include_branches,
        spatial_mode="max_side", max_input_side=960,
    )
    montage_indices = set(vis.select_pair_indices(dataset, args.pairs_per_sequence))
    output = args.output_root.expanduser().resolve() / f"qp{args.qp}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    montages = output / "montages"
    montages.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    old = vis.RAFTStudentFlowPredictor.from_pretrained(
        old_component, mixed_precision=False,
    ).to(device).eval().requires_grad_(False)
    new = vis.RAFTStudentFlowPredictor.from_pretrained(
        new_component, mixed_precision=False,
    ).to(device).eval().requires_grad_(False)

    rows = []
    for count, index in enumerate(range(len(dataset)), 1):
        sample = dataset[index]
        a = sample["degraded0"].unsqueeze(0).to(device)
        b = sample["degraded1"].unsqueeze(0).to(device)
        with torch.inference_mode():
            old_flows = [flow[0].float() for flow in
                         old.predict_bidirectional(a, b, pair_batch_size=1)]
            new_flows = [flow[0].float() for flow in
                         new.predict_bidirectional(a, b, pair_batch_size=1)]
        teachers, valids, bgs = [], [], []
        for direction, bg_key in (("forward", "bg0"), ("backward", "bg1")):
            teacher, valid = vis.prepare_teacher_flow(
                sample["teacher_" + direction][None, None].to(device),
                a.shape[-2:], sample["valid_" + direction][None, None].to(device),
            )
            teachers.append(teacher[0, 0].float())
            valids.append(valid[0, 0].float())
            bgs.append(sample[bg_key].to(device).float())
        row = {
            "qp": args.qp, "sequence": str(sample["video"]),
            "frame_t": int(sample["frame0"]), "frame_t1": int(sample["frame1"]),
            "height": int(a.shape[-2]), "width": int(a.shape[-1]),
        }
        for direction_index, direction in enumerate(("forward", "backward")):
            for predictions, label in ((old_flows, "old_4750"), (new_flows, "qp_student")):
                stats = vis.direction_metrics(
                    predictions[direction_index], teachers[direction_index],
                    valids[direction_index], bgs[direction_index])
                row.update({f"{direction}_{label}_{key}": value
                            for key, value in stats.items()})
            old_epe = row[f"{direction}_old_4750_bg_epe"]
            new_epe = row[f"{direction}_qp_student_bg_epe"]
            row[f"{direction}_bg_epe_reduction_pct"] = 100 * (old_epe - new_epe) / max(old_epe, 1e-8)

        if index in montage_indices:
            flow_scale = max(vis.percentile_scale(
                [flow.norm(dim=0) for flow in teachers], 99), 1.0)
            errors = [[(predictions[d] - teachers[d]).norm(dim=0, keepdim=True)
                       for d in range(2)] for predictions in (old_flows, new_flows)]
            error_scale = vis.percentile_scale(errors[0], 99)
            tiles = [
                ("Degraded frame t", vis.rgb_to_bgr(a[0])),
                ("Degraded frame t+1", vis.rgb_to_bgr(b[0])),
                ("BG mask t", vis.mask_to_bgr(bgs[0])),
                ("BG mask t+1", vis.mask_to_bgr(bgs[1])),
            ]
            for d, direction in enumerate(("forward", "backward")):
                short = "F" if d == 0 else "B"
                tiles.extend([
                    (f"Old 4750 {short} | BG EPE {row[f'{direction}_old_4750_bg_epe']:.3f}",
                     vis.flow_to_bgr(old_flows[d], flow_scale)),
                    (f"QP student {short} | BG EPE {row[f'{direction}_qp_student_bg_epe']:.3f}",
                     vis.flow_to_bgr(new_flows[d], flow_scale)),
                    (f"Clean teacher {short} | {flow_scale:.1f}px",
                     vis.flow_to_bgr(teachers[d], flow_scale)),
                    (f"Valid BG {short}", vis.mask_to_bgr(valids[d] * bgs[d])),
                ])
            tiles.extend([
                ("Old 4750 forward EPE", vis.heatmap(errors[0][0], error_scale)),
                ("QP student forward EPE", vis.heatmap(errors[1][0], error_scale)),
                ("Old 4750 backward EPE", vis.heatmap(errors[0][1], error_scale)),
                ("QP student backward EPE", vis.heatmap(errors[1][1], error_scale)),
            ])
            name = f"{vis.sanitize(row['sequence'])}_f{row['frame_t']:06d}_{row['frame_t1']:06d}.png"
            if not cv2.imwrite(str(montages / name), vis.make_montage(tiles)):
                raise OSError(montages / name)
        rows.append(row)
        print(f"QP{args.qp} [{count}/{len(dataset)}] {row['sequence']} "
              f"{row['frame_t']}->{row['frame_t1']}", flush=True)

    save_csv(output / "per_pair_metrics.csv", rows)
    sequences = sorted({row["sequence"] for row in rows})
    keys = [key for key in rows[0] if key not in
            {"qp", "sequence", "frame_t", "frame_t1", "height", "width"}]
    sequence_rows = [
        {"qp": args.qp, "sequence": name,
         "pair_count": sum(row["sequence"] == name for row in rows),
         **{key: mean([row for row in rows if row["sequence"] == name], key)
            for key in keys}}
        for name in sequences
    ]
    save_csv(output / "per_sequence_metrics.csv", sequence_rows)
    summary = {
        "qp": args.qp, "old_checkpoint": str(old_root),
        "qp_checkpoint": str(new_root),
        "dataset": str((run / "dataset").resolve()),
        "teacher_cache": str((run / "teacher_flows").resolve()),
        "split": "valid", "spatial_mode": "max_side", "max_input_side": 960,
        "pair_count": len(rows), "sequence_count": len(sequences),
        "include_branches": args.include_branches,
        "montage_count": len(montage_indices),
        "precision": "float32; no external autocast",
        "comparison": "Both checkpoints on identical degraded RGB pairs, geometry and frozen clean teacher",
        "old_training_geometry": "512x512 square",
        "current_training_geometry": "native aspect ratio capped at side 960",
        "teacher_interpretation": "Pseudo ground truth, not physical optical flow",
        "metric_support": "Teacher-valid pixels intersected with directional BG mask",
        "sequence_macro_average": {key: mean(sequence_rows, key) for key in keys},
        "mean_per_pair": {key: mean(rows, key) for key in keys},
        "selected_montages": [
            {"sequence": rows[index]["sequence"],
             "frame_t": rows[index]["frame_t"],
             "frame_t1": rows[index]["frame_t1"]}
            for index in sorted(montage_indices)
        ],
    }
    vis.json_dump(output / "summary.json", summary)
    print(json.dumps({"qp": args.qp, "pair_count": len(rows),
                      "old_forward_bg_epe": summary["sequence_macro_average"]["forward_old_4750_bg_epe"],
                      "new_forward_bg_epe": summary["sequence_macro_average"]["forward_qp_student_bg_epe"],
                      "old_backward_bg_epe": summary["sequence_macro_average"]["backward_old_4750_bg_epe"],
                      "new_backward_bg_epe": summary["sequence_macro_average"]["backward_qp_student_bg_epe"]}),
          flush=True)


if __name__ == "__main__":
    main()
