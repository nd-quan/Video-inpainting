import json
import tempfile
import unittest
from pathlib import Path

import torch

from diffusers.models.stc_flow_training import (
    charbonnier_flow_loss,
    compute_flow_training_losses,
    edge_aware_smoothness_loss,
    endpoint_error,
    forward_backward_consistency_loss,
    prepare_teacher_flow,
    save_stage1_checkpoint,
)
from train_stc_flow_vcm import encode_deterministic_latents


class STCFlowTrainingTest(unittest.TestCase):
    def test_vae_encoding_uses_posterior_mode_not_sample(self):
        class Distribution:
            def __init__(self, value):
                self.value = value

            def mode(self):
                return self.value

            def sample(self):
                raise AssertionError("Stage-1 VAE encoding must not sample")

        class VAE:
            config = type("Config", (), {"scaling_factor": 0.5})()

            def encode(self, images):
                value = torch.cat((images, images[:, :1]), dim=1)
                return type("Output", (), {"latent_dist": Distribution(value)})()

        images = torch.ones(2, 3, 3, 4, 5)
        latents = encode_deterministic_latents(
            VAE(), images, torch.device("cpu"), torch.float32
        )
        self.assertEqual(tuple(latents.shape), (2, 3, 4, 4, 5))
        self.assertTrue(torch.equal(latents, torch.full_like(latents, 0.5)))

    def test_endpoint_error_known_vector(self):
        predicted = torch.zeros(1, 1, 2, 4, 5)
        target = torch.zeros_like(predicted)
        target[:, :, 0] = 3.0
        target[:, :, 1] = 4.0
        self.assertAlmostEqual(float(endpoint_error(predicted, target)), 5.0, places=5)

    def test_charbonnier_respects_valid_mask(self):
        predicted = torch.zeros(1, 1, 2, 2, 2)
        target = torch.zeros_like(predicted)
        target[..., 0, 0] = 10.0
        valid = torch.ones(1, 1, 1, 2, 2)
        valid[..., 0, 0] = 0.0
        loss = charbonnier_flow_loss(predicted, target, valid, eps=1e-3)
        self.assertAlmostEqual(float(loss), 1e-3, places=6)

    def test_teacher_resize_scales_displacements_and_masks_bounds(self):
        teacher = torch.zeros(1, 1, 2, 2, 4)
        teacher[:, :, 0] = 1.0
        resized, valid = prepare_teacher_flow(teacher, (4, 8))
        self.assertEqual(tuple(resized.shape), (1, 1, 2, 4, 8))
        self.assertTrue(torch.allclose(resized[:, :, 0], torch.full((1, 1, 4, 8), 2.0)))
        # The last two columns point outside after the x displacement of two.
        self.assertEqual(float(valid[..., -1].sum()), 0.0)
        self.assertGreater(float(valid.mean()), 0.0)

    def test_nonfinite_teacher_values_are_zeroed_and_invalid(self):
        teacher = torch.zeros(1, 1, 2, 3, 3)
        teacher[:, :, 0, 1, 1] = float("inf")
        prepared, valid = prepare_teacher_flow(teacher, (3, 3))
        self.assertTrue(torch.isfinite(prepared).all())
        self.assertEqual(float(prepared[:, :, :, 1, 1].abs().sum()), 0.0)
        self.assertEqual(float(valid[:, :, :, 1, 1]), 0.0)

    def test_inverse_constant_flows_have_near_zero_fb_loss(self):
        forward = torch.zeros(1, 2, 2, 8, 8)
        backward = torch.zeros_like(forward)
        forward[:, :, 0] = 1.0
        backward[:, :, 0] = -1.0
        loss = forward_backward_consistency_loss(forward, backward, eps=1e-3)
        self.assertLess(float(loss), 1.1e-3)

    def test_constant_flow_has_zero_edge_smoothness(self):
        flow = torch.ones(2, 3, 2, 8, 8)
        frames = torch.randn(2, 3, 3, 32, 32)
        loss = edge_aware_smoothness_loss(flow, frames)
        self.assertEqual(float(loss), 0.0)

    def test_combined_loss_backpropagates(self):
        predicted_forward = torch.zeros(1, 2, 2, 8, 8, requires_grad=True)
        predicted_backward = torch.zeros(1, 2, 2, 8, 8, requires_grad=True)
        teacher_forward = torch.randn_like(predicted_forward)
        teacher_backward = torch.randn_like(predicted_backward)
        frames = torch.randn(1, 3, 3, 32, 32)
        losses = compute_flow_training_losses(
            predicted_forward,
            predicted_backward,
            teacher_forward,
            teacher_backward,
            frames,
        )
        losses["total"].backward()
        self.assertIsNotNone(predicted_forward.grad)
        self.assertIsNotNone(predicted_backward.grad)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.isfinite(predicted_forward.grad).all())
        self.assertTrue(torch.isfinite(predicted_backward.grad).all())

    def test_combined_loss_rejects_all_invalid_teacher(self):
        predicted = torch.zeros(1, 1, 2, 4, 4)
        # Every vector samples outside the opposite frame.
        teacher = torch.full_like(predicted, 1000.0)
        frames = torch.randn(1, 2, 3, 16, 16)
        with self.assertRaisesRegex(ValueError, "no finite in-bounds"):
            compute_flow_training_losses(
                predicted,
                predicted,
                teacher,
                -teacher,
                frames,
            )

    def test_checkpoint_contains_resume_state_and_pointers(self):
        model = torch.nn.Conv2d(3, 2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_stage1_checkpoint(
                root,
                model,
                optimizer,
                scheduler,
                None,
                step=12,
                best_valid_epe=1.25,
                config={"seed": 2026},
                is_best=True,
                extra_state={"epoch": 3, "batches_in_epoch": 7},
            )
            self.assertTrue((checkpoint / "flow_predictor" / "pytorch_model.bin").is_file())
            state = torch.load(checkpoint / "trainer_state.pt", map_location="cpu")
            self.assertEqual(state["step"], 12)
            self.assertEqual(state["best_valid_epe"], 1.25)
            self.assertEqual(state["epoch"], 3)
            self.assertEqual(state["batches_in_epoch"], 7)
            latest = json.loads((root / "latest.json").read_text())
            best = json.loads((root / "best.json").read_text())
            self.assertEqual(Path(latest["checkpoint"]), checkpoint)
            self.assertEqual(best["valid_epe"], 1.25)


if __name__ == "__main__":
    unittest.main()
