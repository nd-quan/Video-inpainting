"""CPU tests for geometry, guidance causality, checkpoint and gradient contracts."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import DDIMScheduler
from diffusers.models.brushnet import BrushNetModel
from DGAF_VSR_original.brushnet_dgaf import DGAFBrushNetModel
from DGAF_VSR_original.warping import GuidanceConfig, ClipWarper, warp_clean, resize_flow, prediction_to_clean, rollout_target
from DGAF_VSR_original.pipeline_dgaf import DGAFBrushNetPipeline


class GeometryTests(unittest.TestCase):
    def test_zero_flow_identity_both_versions(self):
        source = torch.randn(2, 4, 8, 12)
        flow = torch.zeros(2, 2, 32, 48)
        for mode in ("direct", "dgaf"):
            actual, valid = warp_clean(source, flow, GuidanceConfig(warp_mode=mode))
            torch.testing.assert_close(actual, source)
            self.assertTrue(bool(valid.all()))

    def test_flow_resize_scales_xy_separately(self):
        flow = torch.ones(1, 2, 12, 24)
        resized = resize_flow(flow, (6, 6))
        torch.testing.assert_close(resized[:, 0], torch.full((1, 6, 6), 0.25))
        torch.testing.assert_close(resized[:, 1], torch.full((1, 6, 6), 0.5))

    def test_previous_uses_backward_and_next_uses_forward(self):
        clean = torch.arange(8).float()[None, None, None, None, :].expand(1, 2, 4, 8, 8).clone()
        clean[:, 1] += 100
        forward = torch.zeros(1, 1, 2, 8, 8)
        forward[:, :, 0] = 1
        backward = -forward
        bg = torch.ones(1, 2, 1, 8, 8)
        prev = ClipWarper(forward, backward, bg, GuidanceConfig(direction="previous", use_confidence_mask=False)).guidance(clean, 1)
        torch.testing.assert_close(prev[:, 1, :4, :, 1:], clean[:, 0, :, :, :-1])
        self.assertEqual(float(prev[:, 0].abs().sum()), 0)
        nxt = ClipWarper(forward, backward, bg, GuidanceConfig(direction="next", use_confidence_mask=False)).guidance(clean, 1)
        torch.testing.assert_close(nxt[:, 0, :4, :, :-1], clean[:, 1, :, :, 1:])

    def test_confidence_toggle_and_bounds(self):
        clean = torch.ones(1, 2, 4, 8, 8)
        f = torch.zeros(1, 1, 2, 8, 8)
        b = f.clone()
        b[:, :, 0] = 2  # Inconsistent with the zero opposite flow.
        bg = torch.ones(1, 2, 1, 8, 8)
        enabled = ClipWarper(f, b, bg, GuidanceConfig(direction="previous")).guidance(clean, 1)
        disabled = ClipWarper(f, b, bg, GuidanceConfig(direction="previous", use_confidence_mask=False)).guidance(clean, 1)
        self.assertEqual(float(enabled.sum()), 0)
        self.assertGreater(float(disabled.sum()), 0)
        self.assertEqual(float(disabled[:, 1, :, :, -2:].sum()), 0)

    def test_destination_mask_source_roi_and_no_cross_clip(self):
        clean = torch.ones(2, 2, 4, 8, 8, requires_grad=True)
        clean = clean * torch.tensor([2., 7.])[:, None, None, None, None]
        flow = torch.zeros(2, 1, 2, 8, 8)
        bg = torch.ones(2, 2, 1, 8, 8)
        bg[:, 0] = 0  # Source is ROI and is allowed to guide the destination.
        result = ClipWarper(flow, flow, bg, GuidanceConfig(direction="previous")).guidance(clean, 1)
        self.assertFalse(result.requires_grad)
        self.assertEqual(float(result[:, 0].sum()), 0)
        self.assertEqual(float(result[0, 1, 0, 0, 0]), 2)
        self.assertEqual(float(result[1, 1, 0, 0, 0]), 7)

    def test_bootstrap_and_bidirectional_normalization(self):
        clean = torch.ones(1, 3, 4, 8, 8) * 3
        flow = torch.zeros(1, 2, 2, 8, 8)
        warper = ClipWarper(flow, flow, torch.ones(1, 3, 1, 8, 8), GuidanceConfig(direction="bidirectional"))
        self.assertEqual(float(warper.guidance(None, 0).sum()), 0)
        result = warper.guidance(clean, 1)
        torch.testing.assert_close(result[:, :, :4], clean)
        self.assertTrue(bool((result[:, :, 4] == 1).all()))

    def test_rescaled_fractional_warp_differs_from_direct(self):
        clean = torch.arange(8).remainder(2).float()[None, None, None, :].expand(1, 4, 8, 8)
        flow = torch.zeros(1, 2, 8, 8)
        flow[:, 0] = .25
        direct, _ = warp_clean(clean, flow, GuidanceConfig(warp_mode="direct"))
        dgaf, _ = warp_clean(clean, flow, GuidanceConfig(warp_mode="dgaf"))
        self.assertGreater(float((direct - dgaf).abs().sum()), 0)

    def test_nonfinite_flow_rejected(self):
        flow = torch.full((1, 1, 2, 8, 8), float("nan"))
        with self.assertRaises(ValueError):
            ClipWarper(flow, flow, torch.ones(1, 2, 1, 8, 8), GuidanceConfig())

    def test_effective_target_recovers_gt_after_model_rollout(self):
        scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=False)
        scheduler.set_timesteps(10)
        clean = torch.randn(2, 4, 8, 8)
        initial_noise = torch.randn_like(clean)
        state = scheduler.add_noise(clean, initial_noise, scheduler.timesteps[:1])
        state = scheduler.step(torch.randn_like(state), scheduler.timesteps[0], state).prev_sample
        alpha = scheduler.alphas_cumprod[scheduler.timesteps[1]]
        for mode in ("epsilon", "v_prediction", "sample"):
            target = rollout_target(state, clean, alpha, mode)
            recovered = prediction_to_clean(state, target, alpha, mode)
            torch.testing.assert_close(recovered, clean, atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(rollout_target(state, clean, alpha, "epsilon"), initial_noise))


class ModelTests(unittest.TestCase):
    def test_sequential_warper_flow_confidence_and_batch_isolation(self):
        from DGAF_VSR_original.warping import SequenceWarper
        source = torch.arange(8).float()[None,None,None,:].expand(2,4,8,8).clone()
        source[1] += 100
        forward = torch.zeros(2,2,2,8,8)
        forward[:,:,0] = 1
        backward = -forward
        bg = torch.ones(2,3,1,8,8)
        for mode in ("direct", "dgaf"):
            warper = SequenceWarper(forward, backward, bg, GuidanceConfig(warp_mode=mode))
            actual = warper.frame_guidance(source, 1, 0)
            torch.testing.assert_close(actual[:,:4,:,1:], source[:,:,:,:-1])
            self.assertEqual(float(actual[:,:,:,:1].abs().sum()), 0)
            actual = warper.frame_guidance(source, 1, 2)
            torch.testing.assert_close(actual[:,:4,:,:-1], source[:,:,:,1:])
        enabled = SequenceWarper(forward*0, backward, bg, GuidanceConfig())
        disabled = SequenceWarper(forward*0, backward, bg, GuidanceConfig(use_confidence_mask=False))
        self.assertEqual(float(enabled.frame_guidance(source,1,0).abs().sum()), 0)
        self.assertGreater(float(disabled.frame_guidance(source,1,0).abs().sum()), 0)

    def test_native_save_load_identity_and_guidance_gradient(self):
        torch.manual_seed(9)
        base = BrushNetModel(conditioning_channels=5, in_channels=4,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            block_out_channels=(16, 32), layers_per_block=1,
            norm_num_groups=8, cross_attention_dim=16, attention_head_dim=4,
            mid_block_type="MidBlock2D")
        # A trained baseline has nonzero output projections. Simulate that to
        # verify new conditioning channels actually receive gradients.
        with torch.no_grad():
            for name, parameter in base.named_parameters():
                if name.startswith(("brushnet_down_blocks", "brushnet_mid_block", "brushnet_up_blocks")):
                    parameter.normal_(std=0.01)
        model = DGAFBrushNetModel.from_baseline_model(base)
        sample, condition = torch.randn(2, 4, 8, 8), torch.randn(2, 5, 8, 8)
        guidance = torch.randn(2, 5, 8, 8)
        context = torch.randn(2, 3, 16)
        base.eval()
        model.eval()
        expected = base(sample, 20, context, condition, return_dict=False)
        actual = model(sample, 20, context, torch.cat((condition, guidance), dim=1), return_dict=False)
        for e, a in zip((*expected[0], expected[1], *expected[2]), (*actual[0], actual[1], *actual[2])):
            torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)
        loss = sum(x.square().mean() for x in (*actual[0], actual[1], *actual[2]))
        loss.backward()
        self.assertGreater(float(model.conv_in_condition.weight.grad[:, 9:].abs().sum()), 0)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored = DGAFBrushNetModel.from_pretrained(directory)
            self.assertEqual(restored.config.conditioning_channels, 10)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)

    def test_same_step_sweeps_cfg_and_reset(self):
        from DGAF_VSR_original.warping import SequenceWarper
        scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=False)
        base, temporal = object(), object()
        fake = SimpleNamespace(scheduler=scheduler, unet=SimpleNamespace(dtype=torch.float32),
            brushnet=temporal, baseline_brushnet=base, progress_bar=lambda steps: steps)
        flow = torch.zeros(2, 3, 2, 8, 8)
        warper = SequenceWarper(flow, flow, torch.ones(2, 4, 1, 8, 8), GuidanceConfig())
        calls, estimates = [], []
        original_step = scheduler.step
        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            estimates.append(result.pred_original_sample.clone())
            return result
        def predict(model, unet, state, timestep, condition, text, context):
            frame = int(context[2, 0, 0])
            calls.append((frame, model is base, int(timestep)))
            torch.testing.assert_close(condition[:2], condition[2:])
            if model is temporal:
                # Must consume the estimate just produced within this sweep.
                torch.testing.assert_close(condition[:2, 5:9], estimates[-1])
            else:
                self.assertEqual(condition.shape[1], 5)
            return torch.zeros_like(state)
        context = torch.arange(4).repeat(2)[:, None, None].expand(8, 3, 8).float()
        kwargs = dict(latents=torch.randn(8, 4, 8, 8), base_condition=torch.ones(8, 5, 8, 8),
            warper=warper, prompt_embeds=context, negative_prompt_embeds=-context,
            brushnet_prompt_embeds=context, negative_brushnet_prompt_embeds=-context,
            num_inference_steps=3, output_type="latent")
        with patch("DGAF_VSR_original.pipeline_dgaf.frozen_v8_predict", side_effect=predict), patch.object(scheduler, "step", side_effect=step):
            for _ in range(2):
                result = DGAFBrushNetPipeline.__call__(fake, **kwargs)
                self.assertTrue(bool(torch.isfinite(result.images).all()))
        self.assertEqual([v[0] for v in calls], [0,1,2,3,3,2,1,0,0,1,2,3] * 2)
        self.assertEqual([i for i,v in enumerate(calls) if v[1]], [0,4,8,12,16,20])

    def test_sequence_dataset_merges_overlap_and_preserves_gaps(self):
        from DGAF_VSR_original.common import make_sequence_dataset
        dataset = SimpleNamespace(clips=[
            ("a", ("p0","p1"), (0,1)), ("a", ("p1","p2"), (1,2)),
            ("a", ("p4","p5"), (4,5)), ("b", ("q0","q1"), (0,1))])
        with patch("DGAF_VSR_original.common.make_dataset", return_value=dataset):
            result = make_sequence_dataset(SimpleNamespace())
        self.assertEqual([c[2] for c in result.clips], [(0,1,2),(4,5),(0,1)])

    def test_neighbor_target_at_same_timestep(self):
        scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=False)
        clean, noise = torch.randn(2,4,8,8), torch.randn(2,4,8,8)
        timesteps = torch.tensor([10,80])
        state = scheduler.add_noise(clean, noise, timesteps)
        alpha = scheduler.alphas_cumprod[timesteps].reshape(2,1,1,1)
        torch.testing.assert_close(rollout_target(state, clean, alpha, "epsilon"), noise)
        torch.testing.assert_close(rollout_target(state, clean, alpha, "v_prediction"),
                                   scheduler.get_velocity(clean, noise, timesteps))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
