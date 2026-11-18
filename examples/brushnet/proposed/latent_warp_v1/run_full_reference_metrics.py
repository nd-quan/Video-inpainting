#!/usr/bin/env python
"""Adapt sharded latent-warp outputs to the supplied full-reference evaluator.

A CPU controller waits for all requested evaluations, then submits independent
one-GPU metric jobs. The original evaluator computes all metrics and ALL rows.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from infer_latent_warp_v1 import REPO, discover, save_json
from export_inference_videos import wait_for_inference

EVALUATOR = REPO / "Quan_test/metric/full_reference_metrics/evaluate.py"
WEIGHTS = EVALUATOR.parent / "weights"


def stage_inputs(output):
    output = Path(output).resolve()
    config = json.loads((output / "resolved_inference_config.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    rows = json.loads((output / "frame_metrics.json").read_text())
    if not summary.get("complete_evaluation") or len(rows) != summary["frames"]:
        raise ValueError("Incomplete source inference: " + str(output))
    names = sorted({r["sequence"] for r in rows})
    sequences = discover(config, names)
    if len({s["name"].split("/")[-1] for s in sequences}) != len(sequences):
        raise ValueError("Metric evaluator requires unique sequence names")
    sources = [Path(p).resolve() for p in summary["prediction_dirs"]]
    pred = output / "full_reference_inputs/pred"
    gt = output / "full_reference_inputs/gt"
    for sequence in sequences:
        name = sequence["name"]
        short = name.split("/")[-1]
        folders = [p / name for p in sources if (p / name).is_dir()]
        if len(folders) != 1:
            raise ValueError("Ambiguous predictions for " + name)
        expected = set(sequence["stems"])
        if {p.stem for p in folders[0].glob("*.png")} != expected:
            raise ValueError("Missing/extra prediction frames for " + name)
        if {r["frame"] for r in rows if r["sequence"] == name} != expected:
            raise ValueError("Missing metric records for " + name)
        for link, target in (
            (pred / short, folders[0]),
            (gt / short / "gt", next(iter(sequence["files"]["gt"].values())).parent),
        ):
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                if link.resolve() != target.resolve():
                    raise ValueError("Wrong existing metric link: " + str(link))
            elif link.exists():
                raise FileExistsError(link)
            else:
                link.symlink_to(target.resolve(), target_is_directory=True)
    save_json(
        output / "full_reference_inputs/manifest.json",
        dict(
            sequences=names,
            frames=summary["frames"],
            pred_root=str(pred),
            gt_root=str(gt),
            source="original generated PNGs; no MP4 re-encoding",
        ),
    )
    return pred, gt


def run_metrics(output):
    output = Path(output).resolve()
    pred, gt = stage_inputs(output)
    state = output / "full_reference_metrics_status.json"
    save_json(state, dict(state="running"))
    command = [
        sys.executable,
        str(EVALUATOR),
        "--pred-root",
        str(pred),
        "--gt-root",
        str(gt),
        "--layout",
        "flat",
        "--output-dir",
        str(output / "metrics_full_reference"),
        "--weights-dir",
        str(WEIGHTS),
        "--require-native-size",
        "--device",
        "cuda",
        "--resume",
    ]
    try:
        subprocess.run(command, check=True)
        result = json.loads(
            (output / "metrics_full_reference/metrics.json").read_text()
        )["results"]
        expected = json.loads((output / "summary.json").read_text())
        sequence_rows = [r for r in result if r["sequence"] != "ALL"]
        overall = [r for r in result if r["sequence"] == "ALL"]
        if (
            len(sequence_rows) != expected["sequences"]
            or len(overall) != 1
            or overall[0]["frames"] != expected["frames"]
        ):
            raise ValueError("Full-reference metric coverage is incomplete")
        save_json(
            state,
            dict(
                state="complete",
                sequences=len(sequence_rows),
                frames=overall[0]["frames"],
                overall=overall[0],
            ),
        )
    except BaseException as error:
        save_json(state, dict(state="failed", error=str(error)))
        raise


def submit_metrics(output):
    output = Path(output).resolve()
    record = output / "full_reference_submission.json"
    with output.joinpath(".full_reference_submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if record.exists():
            existing = json.loads(record.read_text())
            if existing.get("state") == "submitted":
                return existing["task_id"]
            raise RuntimeError("Previous submission needs inspection: " + str(record))
        stage_inputs(output)
        save_json(record, dict(state="submitting"))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run",
            "--eval-roots",
            str(output),
        ]
        result = subprocess.run(
            [
                "job",
                "run",
                "-g",
                "1",
                "-e",
                str(Path(sys.executable).parents[1]),
                "-n",
                "Full-reference metrics: " + output.name,
                shlex.join(command),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(r"Task #(\d+)", result.stdout)
        if not match:
            raise RuntimeError(result.stdout)
        task_id = int(match.group(1))
        save_json(record, dict(state="submitted", task_id=task_id))
        print(result.stdout, flush=True)
        return task_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--after-task", type=int)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    for path in (
        WEIGHTS / "rgb_imagenet.pt",
        WEIGHTS / "torch_home/hub/checkpoints/alexnet-owt-7be5be79.pth",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.preflight:
        for root in args.eval_roots:
            print(stage_inputs(root))
        return
    if args.run:
        if len(args.eval_roots) != 1:
            parser.error("--run requires exactly one root")
        run_metrics(args.eval_roots[0])
        return
    # Hold no GPUs while either experiment is still running.
    for root in args.eval_roots:
        wait_for_inference(root, args.after_task, poll_seconds=30)
    jobs = {str(root.resolve()): submit_metrics(root) for root in args.eval_roots}
    save_json(args.eval_roots[0].parent / "full_reference_metric_jobs.json", jobs)
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == "__main__":
    main()
