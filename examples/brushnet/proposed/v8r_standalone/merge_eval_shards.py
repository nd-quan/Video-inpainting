"""Combine equal-protocol branch shards without copying generated frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_root", type=Path)
    parser.add_argument("--num_shards", type=int, default=4)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("num_shards must be positive")
    summaries = []
    for index in range(args.num_shards):
        path = args.split_root / f"shard-{index}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed shard summary: {path}")
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    counts = [int(item["clips"]) for item in summaries]
    if any(count < 1 for count in counts):
        raise ValueError(f"All shards must contain clips: {counts}")
    keys = set.intersection(*(set(item["mean"]) for item in summaries))
    total = sum(counts)
    merged = {
        "clips": total,
        "num_shards": args.num_shards,
        "shard_clips": counts,
        "mean": {
            key: sum(count * float(item["mean"][key]) for count, item in zip(counts, summaries)) / total
            for key in sorted(keys)
        },
    }
    output = args.split_root / "summary.json"
    output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(output), "clips": total, "shard_clips": counts}))


if __name__ == "__main__":
    main()
