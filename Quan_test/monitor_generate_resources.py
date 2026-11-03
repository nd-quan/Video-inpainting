#!/usr/bin/env python3
"""Sample one generate launcher and its children, plus host and selected GPUs."""
import argparse
import json
from pathlib import Path
import statistics
import time

import psutil
import pynvml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pynvml.nvmlInit()
    handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in args.gpus}
    root = psutil.Process(args.pid)
    tracked, samples = {}, []
    psutil.cpu_percent(None)
    try:
        with (args.output / "resource_samples.jsonl").open("w") as stream:
            while root.is_running() and root.status() != psutil.STATUS_ZOMBIE:
                started = time.monotonic()
                try:
                    processes = [root] + root.children(recursive=True)
                except psutil.NoSuchProcess:
                    break
                cpu = rss = encoders = threads = 0
                for process in processes:
                    try:
                        key = (process.pid, process.create_time())
                        if key not in tracked:
                            tracked[key] = process
                            process.cpu_percent(None)
                        else:
                            process = tracked[key]
                            cpu += process.cpu_percent(None)
                        rss += process.memory_info().rss
                        threads += process.num_threads()
                        encoders += int("vcmrs.encoder" in " ".join(process.cmdline()))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                memory = psutil.virtual_memory()
                gpu = {}
                for g, handle in handles.items():
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu[str(g)] = dict(utilization=utilization.gpu,
                                       memory_mib=pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20)
                sample = dict(time=time.time(), host_cpu_percent=psutil.cpu_percent(None),
                    tree_cpu_percent=cpu, tree_logical_cpu_equivalents=cpu / 100,
                    tree_processes=len(processes), tree_threads=threads, encoder_processes=encoders,
                    tree_rss_sum_gib=rss / 2**30, host_ram_used_gib=memory.used / 2**30,
                    host_ram_available_gib=memory.available / 2**30, gpus=gpu)
                samples.append(sample)
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                time.sleep(max(0.1, 1 - (time.monotonic() - started)))
    except psutil.NoSuchProcess:
        pass
    finally:
        pynvml.nvmlShutdown()
    summary = dict(samples=len(samples), physical_cpu_cores=psutil.cpu_count(logical=False),
                   logical_cpus=psutil.cpu_count(), ram_total_gib=psutil.virtual_memory().total / 2**30)
    for key in ("host_cpu_percent", "tree_cpu_percent", "tree_logical_cpu_equivalents", "tree_processes",
                "tree_threads", "encoder_processes", "tree_rss_sum_gib", "host_ram_used_gib"):
        values = [s[key] for s in samples]
        summary[key] = dict(mean=statistics.mean(values), peak=max(values)) if values else None
    summary["gpus"] = {str(g): dict(peak_memory_mib=max(s["gpus"][str(g)]["memory_mib"] for s in samples),
        mean_utilization=statistics.mean(s["gpus"][str(g)]["utilization"] for s in samples),
        peak_utilization=max(s["gpus"][str(g)]["utilization"] for s in samples)) for g in args.gpus}
    summary["note"] = ("1-second process-tree samples. CPU 100% equals one logical CPU; process counts are not core counts. "
                       "Short-lived children may be missed. RSS sums include shared pages; host RAM includes other work.")
    (args.output / "resource_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
