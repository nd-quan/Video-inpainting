#!/usr/bin/env python
"""Aggregate clip metrics produced by whole-video V8 evaluation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_root", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("num_shards must be positive")

    records: List[Dict[str, object]] = []
    shard_summaries: Dict[str, object] = {}
    seen_clips = set()
    for index in range(args.num_shards):
        shard_root = args.split_root / f"shard-{index}"
        shard_summary_path = shard_root / "summary.json"
        if not shard_summary_path.is_file():
            raise FileNotFoundError(f"Missing completed shard summary: {shard_summary_path}")
        shard_summaries[str(index)] = json.loads(
            shard_summary_path.read_text(encoding="utf-8")
        )
        paths = sorted(shard_root.glob("*/clip_metrics.json"))
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            clip_key = (str(record["video"]), tuple(record["frame_ids"]))
            if clip_key in seen_clips:
                raise RuntimeError(f"Duplicate clip across shards: {clip_key}")
            seen_clips.add(clip_key)
            records.append(record)

    expected_clips = sum(int(summary["clips"]) for summary in shard_summaries.values())
    if len(records) != expected_clips:
        raise RuntimeError(
            f"Found {len(records)} clip metrics, but shard summaries report {expected_clips}"
        )

    numeric_keys = sorted(
        {
            key
            for record in records
            for key, value in record.get("metrics", {}).items()
            if isinstance(value, (int, float))
        }
    )
    summary = {
        "clips": len(records),
        "num_shards": args.num_shards,
        "shards": shard_summaries,
        "mean": {
            key: float(np.mean([record["metrics"][key] for record in records]))
            for key in numeric_keys
            if all(key in record.get("metrics", {}) for record in records)
        },
    }
    output_path = args.split_root / "summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
