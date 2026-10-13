#!/usr/bin/env python3
"""Recover the exact V8-R student evaluation inputs from saved clip references."""
import argparse
import filecmp
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "experiments/metrics_three_methods_comparison_20260918/sources.json"
RELOCATED = ROOT / "experiments/eval_tg_stc"
OLDER_SAVED_REFERENCES = ROOT / "experiments/latent_full_test1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    frozen = output / "frozen_reference"
    frozen.mkdir(parents=True, exist_ok=True)
    records = []
    for source in json.loads(SOURCES.read_text())["R×4"]:
        name = Path(source).parts[-3]
        student_dir = RELOCATED / name
        config_path = student_dir / "run_config.json"
        metric_path = student_dir / "metrics_full_reference/metrics.csv"
        if not config_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(f"Student source unavailable: {student_dir}")
        config = json.loads(config_path.read_text())
        (sequence,) = config["include_branches"]
        split = Path(config["dataset_root"]).name
        if split not in {"test_1", "test_2"}:
            raise ValueError(f"Unexpected dataset split: {split}")
        clips = sorted(x for x in student_dir.glob(f"{sequence}__*") if x.is_dir())
        if not clips:
            raise ValueError(f"No saved student clips: {student_dir}")
        ids = set()
        for clip in clips:
            metadata = json.loads((clip / "clip_metrics.json").read_text())
            for frame in metadata["frame_ids"]:
                frame = int(frame)
                ids.add(frame)
                filename = f"{frame:06d}.png"
                for saved_name, flat_name in (("gt", "gt"), ("input", "inputs"),
                                              ("mask_roi", "masks")):
                    src = clip / saved_name / filename
                    if not src.is_file():
                        if sequence in {"ParkScene", "Traffic"}:
                            src = OLDER_SAVED_REFERENCES / clip.name / saved_name / filename
                        elif sequence == "BasketballPass":
                            src = Path(config["dataset_root"]) / sequence / flat_name / filename
                        if not src.is_file() or src.stat().st_mtime > config_path.stat().st_mtime:
                            raise FileNotFoundError(f"No pre-student-run reference: {src}")
                    dst = frozen / split / sequence / flat_name / filename
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.is_symlink():
                        if not filecmp.cmp(src, dst, shallow=False):
                            raise ValueError(f"Overlapping student clips disagree: {src} vs {dst}")
                    elif dst.exists():
                        raise FileExistsError(dst)
                    else:
                        dst.symlink_to(src.resolve())
        first, last = min(ids), max(ids)
        if sorted(ids) != list(range(first, last + 1)):
            raise ValueError(f"Saved student references have missing frames: {sequence}")
        span = last - first + 1
        starts = list(range(0, span - config["clip_length"] + 1, config["clip_stride"]))
        if starts[-1] != span - config["clip_length"]:
            starts.append(span - config["clip_length"])
        expected_clips = len(starts)
        if len(clips) != expected_clips:
            raise ValueError(f"Clip schedule differs for {sequence}: {len(clips)} != {expected_clips}")
        records.append({"sequence": sequence, "split": split, "source_dir": str(student_dir),
                        "student_config": str(config_path), "student_metrics": str(metric_path),
                        "frozen_dataset_root": str(frozen / split), "frame_range": [first, last],
                        "unique_frames": len(ids), "clips": len(clips)})
        print(f"{split}/{sequence}: {len(ids)} saved frames, {len(clips)} clips", flush=True)
    manifest = {"source_index": str(SOURCES), "snapshot_source": "saved input/gt/mask_roi PNGs from each student clip",
                "records": records}
    (output / "reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {output / 'reference_manifest.json'}")


if __name__ == "__main__":
    main()
