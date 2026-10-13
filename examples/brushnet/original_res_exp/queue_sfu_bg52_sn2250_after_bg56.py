#!/usr/bin/env python3
"""Start the SFU BG52 shared-noise eval after the BG56 eval completes."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


BRUSHNET = Path(__file__).resolve().parents[2]
PREDECESSOR = BRUSHNET / "experiments/eval_sfu_bg56_table_window_bvi_ckpt2000_rgb_tg/logs/status.json"
OUTPUT = BRUSHNET / "experiments/eval_sfu_bg52_table_window_sn2250_rgb_tg"
STATUS = OUTPUT / "logs/queue_status.json"
RUNNER = Path(__file__).with_name("run_sfu_bg52_table_window_sn2250_rgb_tg.py")
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
CHECKPOINT = BRUSHNET / "experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250"
DATASET = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test")


def write_status(state: str, **details: object) -> None:
    payload = {
        "state": state,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "predecessor_status": str(PREDECESSOR),
        "dataset": str(DATASET),
        "checkpoint": str(CHECKPOINT),
        "output": str(OUTPUT),
        "runner": str(RUNNER),
        "gpus": list(range(6)),
        "gpu_memory_ceiling_mib": 47000,
        **details,
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def main() -> int:
    for path in (DATASET, CHECKPOINT / "brushnet",
                 CHECKPOINT / "ipadapter/model.safetensors",
                 CHECKPOINT / "ipadapter/fusion_module.safetensors", RUNNER, PYTHON):
        if not path.exists():
            raise FileNotFoundError(path)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if PREDECESSOR.exists():
            records = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
            counts = dict(Counter(record.get("state", "unknown") for record in records))
            if len(records) == 14 and counts.get("complete") == 14:
                break
            state = "waiting_bg56_recovery" if any(
                counts.get(value, 0) for value in ("failed", "stopped_vram")
            ) else "waiting_for_bg56"
        else:
            counts = {}
            state = "waiting_for_bg56"
        write_status(state, predecessor_counts=counts)
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {state}: {counts}",
              flush=True)
        time.sleep(60)

    write_status("running", predecessor_counts={"complete": 14},
                 started_at=datetime.now().isoformat(timespec="seconds"))
    print("BG56 complete (14/14). Starting BG52 on the six-GPU guarded runner.",
          flush=True)
    result = subprocess.run([str(PYTHON), "-u", str(RUNNER)], cwd=BRUSHNET,
                            check=False)
    write_status("complete" if result.returncode == 0 else "failed",
                 predecessor_counts={"complete": 14},
                 exit_code=result.returncode,
                 finished_at=datetime.now().isoformat(timespec="seconds"))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
