#!/usr/bin/env python3
"""Compare matched four-frame BG52 CGE scales; no whole-dataset claims."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512"
ABLATION = ROOT / "experiments/EVAL_FOR_FINAL/ablation_bg52_cge_x3_x10_tg0p1_C25-40_T25-40_Traffic4"


def main():
    output = ABLATION / "comparison"
    output.mkdir(exist_ok=True)
    dataset = Path(json.loads((BASELINE / "run_config.json").read_text())["dataset"])
    names = ["{:06d}.png".format(i) for i in range(117, 121)]
    runs = {"temporal_only": BASELINE / "diagnostics/temporal_control",
            "cge_x1": BASELINE, "cge_x3": ABLATION / "cge_x3", "cge_x10": ABLATION / "cge_x10"}
    images = {}
    for label, folder in runs.items():
        images[label] = []
        for name in names:
            with Image.open(folder / "test/Class_A/Traffic" / name) as image:
                if image.size != (512, 512):
                    raise ValueError("Unexpected image dimensions")
                images[label].append(np.array(image.convert("RGB")).astype(np.float32))
    references, masks = [], []
    for name in names:
        with Image.open(dataset / "GT/Class_A/Traffic" / name) as image:
            references.append(np.array(image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)).astype(np.float32))
        with Image.open(dataset / "mask/Class_A/Traffic" / name) as image:
            masks.append(np.array(image.convert("L").resize((512, 512), Image.Resampling.NEAREST)) > 127)
    per_frame, summary = [], []
    def psnr(a, b):
        return float(10 * np.log10(255**2 / max(float(np.mean((a-b)**2)), 1e-12)))
    for label, predictions in images.items():
        rows = []
        for i, (pred, gt, mask) in enumerate(zip(predictions, references, masks)):
            difference = np.abs(pred - images["cge_x1"][i])
            row = {"run": label, "frame": names[i], "psnr_gt": psnr(pred, gt),
                   "background_psnr_gt": psnr(pred[~mask], gt[~mask]),
                   "mae_vs_x1": float(difference.mean()),
                   "background_mae_vs_x1": float(difference[~mask].mean()),
                   "changed_pixels_pct_vs_x1": float(100 * np.any(difference > 0, axis=2).mean())}
            rows.append(row)
            per_frame.append(row)
        aggregate = {"run": label, "frames": 4,
                     **{k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("run", "frame")}}
        timing_path = runs[label] / "test/Class_A/Traffic/generation_timing.jsonl"
        timing = json.loads(timing_path.read_text().splitlines()[-1]) if timing_path.is_file() else {}
        for key in ("generation_seconds", "generation_seconds_per_image", "total_clip_seconds", "total_seconds_per_image"):
            aggregate[key] = timing.get(key)
        summary.append(aggregate)
    (output / "per_frame.json").write_text(json.dumps(per_frame, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps({
        "scope": "Traffic frames 000117..000120 only; all runs TGE 0.1, windows [25,40), identical seeds/checkpoints",
        "gt_preprocessing": "PIL RGB bilinear resize to 512x512, matching the established baseline metric domain",
        "timing_semantics": "Seconds per image are synchronized batch generation wall time / 4, not independent per-frame latency; model loading excluded. Historical x1/control timings unavailable.",
        "results": summary}, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i, name in enumerate(names):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for col, label in enumerate(("cge_x1", "cge_x3", "cge_x10")):
            axes[0, col].imshow(images[label][i].astype(np.uint8))
            axes[0, col].set_title(label + "; TGE=0.1")
            difference = np.abs(images[label][i] - images["cge_x1"][i])
            axes[1, col].imshow(np.clip(difference * 20, 0, 255).astype(np.uint8))
            axes[1, col].set_title("Difference vs x1, amplified x20")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle("Traffic {} — identical guidance windows [25,40)".format(name))
        fig.tight_layout()
        fig.savefig(output / (Path(name).stem + "_comparison.png"), dpi=150)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
