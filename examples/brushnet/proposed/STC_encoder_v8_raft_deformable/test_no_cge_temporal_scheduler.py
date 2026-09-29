"""Regression: disabling CGE must not disable temporal guidance or crash DDIM."""
import unittest

import torch

from evaluate_v8_raft_deformable_cge_temporal_bg_only import (
    CGETemporalDDIMScheduler, _configure_cge,
)


class NoCGESchedulerTests(unittest.TestCase):
    def test_first_clip_and_stale_condition(self):
        for stale in (False, True):
            scheduler = CGETemporalDDIMScheduler()
            scheduler.set_timesteps(50)
            scheduler.cge_max_evals = 0
            if stale:
                def forbidden(*args, **kwargs):
                    raise AssertionError("CGE must not run")
                scheduler.cond_fn = forbidden
            report = _configure_cge(scheduler, None, {}, torch.device("cpu"))
            self.assertEqual(report["cge_enabled"], 0)
            self.assertIsNone(scheduler.cond_fn)
            scheduler.set_temporal_guidance(
                flow_backward=torch.zeros(1, 2, 8, 8),
                stable_bg=torch.ones(1, 1, 8, 8),
                guidance_scale=0.001, start_step=25, end_step=35,
            )
            sample = torch.randn(2, 4, 8, 8)
            for timestep in scheduler.timesteps:
                sample = scheduler.step(torch.zeros_like(sample), timestep,
                                        sample, return_dict=False)[0]
            self.assertTrue(torch.isfinite(sample).all())
            self.assertEqual(scheduler.cge_codec_eval_count, 0)
            self.assertEqual(scheduler.temporal_guidance_applied_steps, 10)


if __name__ == "__main__":
    unittest.main()
