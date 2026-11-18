"""Shared partitioning, result merging and scheduler status for inference."""
import json
import subprocess

from infer_latent_warp_v1 import clip_windows, save_json


def partition_sequences(sequences, workers, config):
    """Greedy clip-balanced sharding; each sequence stays on one GPU."""
    shards = [[] for _ in range(workers)]
    loads = [0] * workers
    weighted = [
        (
            len(
                list(
                    clip_windows(
                        s["stems"], config["clip_length"], config["clip_stride"]
                    )
                )
            ),
            s,
        )
        for s in sequences
    ]
    for cost, sequence in sorted(
        weighted, key=lambda item: (-item[0], item[1]["name"])
    ):
        index = min(range(workers), key=lambda i: (loads[i], i))
        shards[index].append(sequence)
        loads[index] += cost
    if any(not shard for shard in shards):
        raise ValueError("Need at least one sequence per worker")
    return shards


def merge_results(output, shards, config):
    rows = []
    for rank, sequences in enumerate(shards):
        path = output / "shards" / ("shard-%02d" % rank)
        metrics = json.loads((path / "frame_metrics.json").read_text())
        expected = {(s["name"], stem) for s in sequences for stem in s["stems"]}
        actual = {(r["sequence"], r["frame"]) for r in metrics}
        if expected != actual or len(metrics) != len(actual):
            raise ValueError("Missing/duplicate output frames in shard " + str(rank))
        rows.extend(metrics)
    rows.sort(key=lambda row: (row["sequence"], int(row["frame"])))
    save_json(output / "frame_metrics.json", rows)
    summary = {}
    for key in ("psnr_full", "psnr_bg", "psnr_roi", "ssim_full"):
        values = [r[key] for r in rows if r[key] is not None]
        if values:
            summary[key] = sum(values) / len(values)
    summary.update(
        frames=len(rows),
        sequences=sum(map(len, shards)),
        workers=len(shards),
        checkpoint=config["checkpoint_dir"],
        oracle_gt_flow=config["flow_source"] == "gt",
        raft_student_checkpoint=config.get("raft_student_checkpoint"),
        cge=False,
        raft_weights=config["raft_weights"],
        complete_evaluation=True,
        prediction_dirs=[
            str(output / "shards" / ("shard-%02d" % i)) for i in range(len(shards))
        ],
        latent_warp_enabled=config["latent_warp_enabled"],
        temporal_guidance_enabled=config["temporal_guidance_enabled"],
        effective_temporal_guidance_scale=(
            config["temporal_guidance_scale"]
            if config["temporal_guidance_enabled"]
            else 0.0
        ),
    )
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def query_task(task_id):
    result = subprocess.run(
        ["job", "status", str(task_id), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if value.get("id") != task_id or "status" not in value:
        raise RuntimeError("Invalid scheduler status for task " + str(task_id))
    return value
