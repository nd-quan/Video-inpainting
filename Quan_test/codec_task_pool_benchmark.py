"""Isolated codec-call pool; the production scheduler is not modified on disk."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time


class EventWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, value):
        with self.lock, self.path.open("a") as stream:
            stream.write(json.dumps(value) + "\n")


class CodecTaskPool:
    """Submit one task for each (frame, base/plus/minus) instead of each frame.

    Tensor preparation and finite-difference arithmetic follow grad_codec exactly.
    Only codec round trips execute in the thread pool. Results retain frame order.
    """

    def __init__(self, core, workers=24, report_path=None, verify_first=True):
        self.core = core
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="codec-call")
        self.original_batch = core.grad_codec_batch
        self.original_roundtrip = core._codec_roundtrip01
        self.report = EventWriter(report_path) if report_path else None
        self.verify_first = verify_first
        self.replay_validation = None
        self.call_count = 0
        self.reports = []

    def __call__(self, x_in, x_lr, roi_mask, curStep, codec, call_prefix):
        import torch
        started = time.perf_counter()
        n = x_in.shape[0]
        if x_lr.shape[0] != n:
            if x_lr.shape[0] != 1:
                raise ValueError("Observation batch differs from prediction batch")
            x_lr = x_lr.repeat(n, 1, 1, 1)
        if roi_mask.shape[0] != n:
            if roi_mask.shape[0] != 1:
                raise ValueError("Mask batch differs from prediction batch")
            roi_mask = roi_mask.repeat(n, 1, 1, 1)
        h = 16 / 255
        inputs = {}
        masks = {}
        for i in range(n):
            image = x_in[i]
            perturb = torch.ones_like(image)
            plus = torch.clamp(image + h * perturb, -1, 1)
            minus = torch.clamp(image - h * perturb, -1, 1)
            for variant, value in (("base", image), ("plus", plus), ("minus", minus)):
                key = "{}_b{:02d}_{}".format(call_prefix, i, variant)
                inputs[key] = ((value + 1) / 2).clamp(0, 1)
                masks[key] = roi_mask[i]
        if x_in.is_cuda:
            torch.cuda.synchronize(x_in.device)

        def execute(key):
            context = torch.cuda.device(x_in.device) if x_in.is_cuda else nullcontext()
            with context:
                return self.original_roundtrip(codec, inputs[key], masks[key], key)

        futures = {key: self.pool.submit(execute, key) for key in inputs}
        # This barrier is essential: no CGE gradient or TGE sees partial results.
        results = {key: future.result() for key, future in futures.items()}
        gradients, reconstructions = [], []
        for i in range(n):
            prefix = "{}_b{:02d}".format(call_prefix, i)
            base, plus, minus = (results[prefix + "_" + v] for v in ("base", "plus", "minus"))
            residual_mask = (torch.ones_like(roi_mask[i]) if self.core._codec_uses_region_composite(codec)
                             else 1 - roi_mask[i])
            grad_Dx = 2 * residual_mask * (base - ((x_lr[i] + 1) / 2))
            grad_x = (plus - minus) * torch.ones_like(x_in[i]) / (2 * h)
            gradients.append(grad_Dx * grad_x)
            reconstructions.append(base * 2 - 1)
        gradient = torch.stack(gradients, dim=0)
        degraded = torch.stack(reconstructions, dim=0)
        self.call_count += 1
        validation_seconds = 0.0
        if self.verify_first and self.replay_validation is None:
            validation_started = time.perf_counter()
            original = self.core._codec_roundtrip01
            old_workers = os.environ.get("CGE_CODEC_WORKERS")
            replayed = []

            def replay(unused_codec, image, mask, key):
                if not torch.equal(image, inputs[key]) or not torch.equal(mask, masks[key]):
                    raise AssertionError("Flat task changed the original codec input or mask: " + key)
                replayed.append(key)
                return results[key]

            self.core._codec_roundtrip01 = replay
            os.environ["CGE_CODEC_WORKERS"] = "1"
            try:
                ref_gradient, ref_degraded = self.original_batch(
                    x_in, x_lr, roi_mask, curStep, codec, call_prefix)
            finally:
                self.core._codec_roundtrip01 = original
                if old_workers is None:
                    os.environ.pop("CGE_CODEC_WORKERS", None)
                else:
                    os.environ["CGE_CODEC_WORKERS"] = old_workers
            passed = (len(replayed) == 3 * n and torch.equal(gradient, ref_gradient)
                      and torch.equal(degraded, ref_degraded))
            self.replay_validation = {
                "passed": passed, "codec_inputs_and_masks_exact": True, "codec_calls_replayed": len(replayed),
                "gradient_max_abs_difference": float((gradient.float() - ref_gradient.float()).abs().max()),
                "reconstruction_max_abs_difference": float((degraded.float() - ref_degraded.float()).abs().max()),
                "scope": "Real codec outputs replayed through the unchanged serial grad_codec_batch. "
                         "Validates task indexing, input tensors and finite-difference math; not serial codec determinism.",
            }
            if not passed:
                raise AssertionError("Codec task equivalence failed: " + str(self.replay_validation))
            validation_seconds = time.perf_counter() - validation_started
        if x_in.is_cuda:
            torch.cuda.synchronize(x_in.device)
        report = {"call_index": self.call_count - 1, "diffusion_timestep": int(curStep),
                  "frames": n, "tasks": 3 * n, "pool_workers": self.workers,
                  "seconds": time.perf_counter() - started, "validation_seconds": validation_seconds,
                  "all_results_joined": True, "prefix": call_prefix}
        self.reports.append(report)
        if self.report:
            self.report.write(report)
        return gradient, degraded

    def install(self):
        self.core.grad_codec_batch = self

    def close(self):
        self.pool.shutdown(wait=True)
        self.core.grad_codec_batch = self.original_batch


def track_encoder(codec, events, step_index):
    """Instrument inside the codec semaphore: measure real encoder intervals."""
    target = codec.regional_codec._bg_codec
    original = target._run_encoder
    lock = threading.Lock()
    counts = {"active": 0, "peak": 0, "started": 0, "completed": 0}

    @wraps(original)
    def tracked(ini_path, job_dir):
        with lock:
            counts["active"] += 1
            counts["started"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            events.write({"event": "encoder_start", "time": time.time(), "monotonic": time.perf_counter(),
                          "step_index": step_index(), "job": str(job_dir), "active": counts["active"]})
        success = False
        try:
            result = original(ini_path, job_dir)
            success = True
            return result
        finally:
            with lock:
                counts["active"] -= 1
                counts["completed"] += int(success)
                events.write({"event": "encoder_end", "time": time.time(), "monotonic": time.perf_counter(),
                              "step_index": step_index(), "job": str(job_dir), "active": counts["active"],
                              "success": success})

    target._run_encoder = tracked
    return counts
