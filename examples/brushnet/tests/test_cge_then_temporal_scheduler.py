"""Check sequential guidance algebra, gradient input, and window boundaries."""
import unittest

import torch

from diffusers.schedulers.scheduling_ddim_CGE import CustomDDIMScheduler
from diffusers.schedulers.scheduling_ddim_cge_temporal import CGETemporalDDIMScheduler
from diffusers.schedulers.scheduling_ddim_cge_then_temporal import CGEThenTemporalDDIMScheduler


def configure(cls, delta=None, temporal_scale=0.1, start=0, end=50):
    scheduler = cls(clip_sample=False)
    scheduler.set_timesteps(50)
    scheduler.direct_cge_guidance = True
    scheduler.x_lr = None
    scheduler.mask = None
    scheduler.decoder = None
    scheduler.cge_start_step = start
    scheduler.cge_end_step = end
    scheduler.cge_every_n_steps = 1
    scheduler.cge_max_evals = -1
    scheduler.cond_fn = None if delta is None else lambda x0, **kw: delta.to(x0)
    if hasattr(scheduler, "set_temporal_guidance"):
        scheduler.set_temporal_guidance(
            flow_backward=torch.zeros(1, 2, 4, 4), stable_bg=torch.ones(1, 1, 4, 4),
            guidance_scale=temporal_scale, start_step=start, end_step=end,
        )
    return scheduler


class SequentialGuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.sample = torch.randn(2, 4, 4, 4)
        self.noise = torch.randn_like(self.sample) * 0.1
        self.delta = torch.randn_like(self.sample) * 0.03

    def test_temporal_sees_corrected_clean_and_cge_is_not_added_twice(self):
        scheduler = configure(CGEThenTemporalDDIMScheduler, self.delta)
        baseline = configure(CustomDDIMScheduler)
        timestep = scheduler.timesteps[25]
        base = baseline.step(self.noise, timestep, self.sample)
        previous = int(timestep) - 20
        a = scheduler.alphas_cumprod[previous].sqrt()
        expected_input = base.pred_original_sample + self.delta / a
        observed = []

        def temporal_probe(x0):
            observed.append(x0.clone())
            return x0 - 0.02 * x0  # Input-dependent correction distinguishes old/new algorithms.

        scheduler._apply_latent_temporal_guidance = temporal_probe
        result = scheduler.step(self.noise, timestep, self.sample)
        torch.testing.assert_close(observed[0], expected_input)
        torch.testing.assert_close(result.prev_sample,
                                   base.prev_sample + self.delta - a * 0.02 * expected_input)
        torch.testing.assert_close(result.pred_original_sample, expected_input * 0.98)

    def test_zero_temporal_preserves_original_cge_transition(self):
        sequential = configure(CGEThenTemporalDDIMScheduler, self.delta, temporal_scale=0)
        original = configure(CustomDDIMScheduler, self.delta)
        for step in (0, 25, 49):
            timestep = sequential.timesteps[step]
            actual = sequential.step(self.noise, timestep, self.sample)
            expected = original.step(self.noise, timestep, self.sample)
            torch.testing.assert_close(actual.prev_sample, expected.prev_sample, rtol=0, atol=0)

    def test_zero_cge_matches_parallel_temporal(self):
        sequential = configure(CGEThenTemporalDDIMScheduler)
        parallel = configure(CGETemporalDDIMScheduler)
        for step in (0, 25, 49):
            timestep = sequential.timesteps[step]
            actual = sequential.step(self.noise, timestep, self.sample)
            expected = parallel.step(self.noise, timestep, self.sample)
            torch.testing.assert_close(actual.prev_sample, expected.prev_sample)

    def test_windows_and_no_stale_cge_correction(self):
        scheduler = configure(CGEThenTemporalDDIMScheduler, self.delta, start=25, end=40)
        sample = self.sample.clone()
        for timestep in scheduler.timesteps:
            sample = scheduler.step(torch.zeros_like(sample), timestep, sample).prev_sample
        reports = scheduler.sequential_step_reports
        self.assertEqual([r["step_index"] for r in reports if r["cge_applied"]], list(range(25, 40)))
        self.assertEqual([r["step_index"] for r in reports if r["temporal_requested"]], list(range(25, 40)))
        self.assertEqual(reports[40]["equivalent_clean_delta_norm"], 0)
        self.assertTrue(torch.isfinite(sample).all())
        scheduler.set_timesteps(50)
        self.assertEqual(scheduler.sequential_step_reports, [])

    def test_cge_callback_restored_on_failure_and_eta_rejected(self):
        scheduler = configure(CGEThenTemporalDDIMScheduler)

        def broken(**kwargs):
            raise RuntimeError("codec failure")

        scheduler.cond_fn = broken
        with self.assertRaisesRegex(RuntimeError, "codec failure"):
            scheduler.step(self.noise, scheduler.timesteps[0], self.sample)
        self.assertIs(scheduler.cond_fn, broken)
        with self.assertRaisesRegex(ValueError, "eta=0"):
            scheduler.step(self.noise, scheduler.timesteps[0], self.sample, eta=1)


if __name__ == "__main__":
    unittest.main()
