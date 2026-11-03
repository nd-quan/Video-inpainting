"""Check the new task boundaries against the unchanged serial CGE code."""
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/brushnet"))
import torch
import diffusers.schedulers.scheduling_ddim_CGE as core
from codec_task_pool_benchmark import CodecTaskPool


class FakeCodec:
    def roundtrip_background01(self, image, *, roi_mask, call_key):
        # Different completion orders exercise the frame/variant mapping.
        time.sleep(0.001 * (int(call_key.split("_b")[1][:2]) % 3))
        return (image.square() * 31).round() / 31


class FlatCodecContracts(unittest.TestCase):
    def test_eight_frames_24_tasks_equal_original_serial_gradient_and_cge(self):
        torch.manual_seed(17)
        x = torch.rand(8, 3, 7, 11) * 2 - 1
        target = torch.rand_like(x) * 2 - 1
        mask = (torch.rand(8, 1, 7, 11) > 0.7).float()
        codec = FakeCodec()
        os.environ["CGE_CODEC_WORKERS"] = "1"
        expected = core.grad_codec_batch(x, target, mask, 480, codec, "check")
        pool = CodecTaskPool(core, workers=24)
        actual = pool(x, target, mask, 480, codec, "check")
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        self.assertTrue(pool.replay_validation["passed"])
        self.assertEqual(pool.reports[0]["tasks"], 24)
        options = SimpleNamespace(cge_codec=codec, guidance_scale_cge=.001, cge_scale_schedule="fixed",
                                  per_frame_cge=False, decode_chunk_size=1, vae_scaling_factor=1,
                                  cge_codec_eval_count=0)
        decoder = lambda z: SimpleNamespace(sample=z * .75)
        reference_cge = core.cond_fn(x, torch.tensor(480), target, mask, decoder, options)
        pool.install()
        try:
            actual_cge = core.cond_fn(x, torch.tensor(480), target, mask, decoder, options)
        finally:
            pool.close()
        self.assertTrue(torch.equal(reference_cge, actual_cge))

    def test_broadcast_observation_and_mask_keep_serial_behavior(self):
        x = torch.linspace(-1, 1, 8 * 3 * 3 * 5).reshape(8, 3, 3, 5)
        target, mask = torch.zeros_like(x[:1]), torch.zeros(1, 1, 3, 5)
        os.environ["CGE_CODEC_WORKERS"] = "1"
        reference = core.grad_codec_batch(x, target, mask, 480, FakeCodec(), "broadcast")
        pool = CodecTaskPool(core, workers=24)
        try:
            actual = pool(x, target, mask, 480, FakeCodec(), "broadcast")
        finally:
            pool.close()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, reference)))


if __name__ == "__main__":
    unittest.main()
