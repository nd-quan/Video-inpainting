"""Focused geometry, identity, matching, and gradient tests for the refiner."""

import unittest

import torch

from flow_refinement.losses import refinement_losses
from flow_refinement.model import (
    FlowRefinementBlock, RefinerConfig, local_matching_cost, resize_flow,
)


class FlowRefinementTests(unittest.TestCase):
    def test_resize_scales_dx_dy_separately(self):
        flow = torch.zeros(1, 2, 24, 32)
        flow[:, 0] = 8
        flow[:, 1] = -6
        reduced = resize_flow(flow, (12, 8))
        self.assertTrue(torch.allclose(reduced[:, 0], torch.full_like(reduced[:, 0], 2)))
        self.assertTrue(torch.allclose(reduced[:, 1], torch.full_like(reduced[:, 1], -3)))
        self.assertTrue(torch.allclose(resize_flow(reduced, (24, 32)), flow))

    def test_matching_looks_around_current_correspondence(self):
        torch.manual_seed(5)
        target = torch.randn(1, 8, 8, 8)
        source = torch.roll(target, shifts=1, dims=-1)
        cost, valid = local_matching_cost(target, source, torch.zeros(1, 2, 8, 8), 1)
        # Offset order: dy=-1..1, dx=-1..1. Center is 4; right is 5.
        self.assertEqual(tuple(cost.shape), (1, 9, 8, 8))
        self.assertEqual(tuple(valid.shape), tuple(cost.shape))
        self.assertLess(float(cost[0, 5, 3, 3]), 1e-5)
        self.assertGreater(float(cost[0, 4, 3, 3]), float(cost[0, 5, 3, 3]))
        self.assertEqual(float(valid[0, 5, 3, -1]), 0.0)

    def test_zero_residual_preserves_coarse_and_both_heads_receive_gradients(self):
        torch.manual_seed(7)
        model = FlowRefinementBlock(RefinerConfig(search_radius=1))
        image0, image1 = torch.randn(2, 3, 32, 40), torch.randn(2, 3, 32, 40)
        flow_f, flow_b = torch.randn(2, 2, 32, 40), torch.randn(2, 2, 32, 40)
        refined_f, refined_b = model(image0, image1, flow_f, flow_b)
        self.assertTrue(torch.equal(refined_f, flow_f))
        self.assertTrue(torch.equal(refined_b, flow_b))
        teacher_f, teacher_b = flow_f + 1, flow_b - 1
        valid = torch.ones(2, 1, 32, 40)
        losses = refinement_losses(refined_f, refined_b, teacher_f, teacher_b,
                                   valid, valid, valid, valid, edge_weight=.05)
        losses["loss_total"].backward()
        for head in (model.quarter_head, model.half_head):
            grad = head[-1].weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)

    def test_edge_requires_both_teacher_pixels_valid(self):
        predicted = torch.zeros(1, 2, 8, 8, requires_grad=True)
        teacher = torch.zeros_like(predicted)
        teacher[:, 0, :, 4:] = 3
        valid = torch.zeros(1, 1, 8, 8)
        valid[..., 4:] = 1
        bg = torch.ones_like(valid)
        losses = refinement_losses(predicted, predicted, teacher, teacher,
                                   valid, valid, bg, bg, edge_threshold=1)
        self.assertEqual(float(losses["edge_support"]), 0)
        self.assertEqual(float(losses["loss_edge"]), 0)
        valid.fill_(1)
        losses = refinement_losses(predicted, predicted, teacher, teacher,
                                   valid, valid, bg, bg, edge_threshold=1)
        self.assertGreater(float(losses["edge_support"]), 0)
        self.assertGreater(float(losses["loss_edge"]), 0)

    def test_edge_loss_reaches_both_residual_heads(self):
        torch.manual_seed(11)
        model = FlowRefinementBlock(RefinerConfig(search_radius=1))
        image0, image1 = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
        coarse = torch.zeros(1, 2, 32, 32)
        teacher = torch.zeros_like(coarse)
        teacher[:, 0, :, 16:] = 3
        valid = torch.ones(1, 1, 32, 32)
        forward, backward = model(image0, image1, coarse, coarse)
        loss = refinement_losses(forward, backward, teacher, teacher,
                                 valid, valid, valid, valid)["loss_edge"]
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        for head in (model.quarter_head, model.half_head):
            grad = head[-1].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
