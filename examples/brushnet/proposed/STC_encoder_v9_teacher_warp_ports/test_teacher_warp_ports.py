"""Small contract tests for direct x4 warping of BrushNet residual ports."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROPOSED = Path(__file__).resolve().parent.parent
if str(PROPOSED) not in sys.path:
    sys.path.insert(0, str(PROPOSED))

from STC_encoder_v9_teacher_warp_ports.condition import (
    precompute_brushnet_residuals,
    predict_with_feature_ports,
)
from STC_encoder_v9_teacher_warp_ports.model import PortProjection, TeacherWarpPortModel


def test_backward_teacher_flow_moves_previous_feature_right():
    source = torch.zeros(1, 1, 8, 8)
    source[0, 0, 4, 2] = 1
    backward = torch.zeros(1, 2, 32, 32)
    backward[:, 0] = -4  # one low-grid pixel, defined at the target frame
    result, _, valid = PortProjection._warp_pair(
        source, torch.ones_like(source), backward, torch.ones(1, 1, 32, 32)
    )
    assert result[0, 0, 4, 3].item() == 1
    assert result[0, 0, 4, 2].item() == 0
    assert valid[0, 0, 4, 3].item()


def test_no_warp_control_keeps_teacher_reliability():
    torch.manual_seed(7)
    head = PortProjection(residual_channels=1, guide_channels=2, bottleneck=4)
    nn.init.normal_(head.output.weight, std=0.1)
    residual = torch.zeros(2, 1, 8, 8)
    residual[0, 0, 4, 2] = 1
    residual[1, 0, 4, 3] = 0.5
    guide = torch.zeros(1, 2, 2, 8, 8)
    bg = torch.ones(1, 2, 1, 32, 32)
    forward = torch.zeros(1, 1, 2, 32, 32)
    backward = forward.clone()
    forward[:, :, 0] = 4
    backward[:, :, 0] = -4
    valid = torch.ones(1, 1, 1, 32, 32)
    common = (residual, guide, bg, forward, backward, valid, valid,
              torch.tensor(1.0))
    corrected, prev_on, next_on = head(*common, 1.0)
    control, prev_off, next_off = head(*common, 0.0)
    assert torch.equal(prev_on, prev_off)
    assert torch.equal(next_on, next_off)
    assert not torch.allclose(corrected, control)


class _BrushNet(nn.Module):
    def forward(self, noisy_latents, timesteps, **kwargs):
        del timesteps, kwargs
        n = noisy_latents.shape[0]
        device = noisy_latents.device
        down = [torch.zeros(n, 4, 8, 8, device=device) for _ in range(3)]
        down += [torch.zeros(n, 4, 4, 4, device=device) for _ in range(2)]
        down += [torch.zeros(n, 6, 4, 4, device=device)]
        down += [torch.zeros(n, 6, 2, 2, device=device) for _ in range(2)]
        down += [torch.zeros(n, 8, 2, 2, device=device)]
        down[2][1] = 1
        return down, torch.zeros(n, 8, 1, 1, device=device), []


class _UNet(nn.Module):
    def forward(self, noisy_latents, timesteps, **kwargs):
        del timesteps
        values = kwargs["down_block_add_samples"]
        values = values + [kwargs["mid_block_add_sample"]]
        total = sum(value.mean(dim=(1, 2, 3), keepdim=True) for value in values)
        return (noisy_latents + total,)


def test_zero_identity_shapes_and_diffusion_gradient():
    torch.set_num_threads(1)
    model = TeacherWarpPortModel(
        hidden_channels=8, num_heads=2,
        down_port_channels=(4, 6, 8), mid_port_channels=8,
        port_bottleneck_channels=4,
    )
    rgb = torch.randn(1, 2, 3, 64, 64)
    bg = torch.ones(1, 2, 1, 64, 64)
    flow = torch.zeros(1, 1, 2, 64, 64)
    valid = torch.ones(1, 1, 1, 64, 64)
    noisy = torch.zeros(2, 4, 8, 8)
    residuals = precompute_brushnet_residuals(
        brushnet=_BrushNet(), noisy_latents=noisy, timesteps=torch.zeros(2),
        base_condition_latents=torch.zeros(2, 4, 8, 8),
        bg_mask_sequence=bg,
        brushnet_text_hidden_states=torch.zeros(2, 1, 8),
    )["brushnet_residuals"]
    output = model(
        rgb, bg,
        teacher_flow_forward_rgb=flow,
        teacher_flow_backward_rgb=flow,
        teacher_valid_forward_rgb=valid,
        teacher_valid_backward_rgb=valid,
        brushnet_residuals=residuals,
    )
    class Args:
        stc_injection_scale = 1.0
    prediction = predict_with_feature_ports(
        model=model, stc_output=output, args=Args(), brushnet=_BrushNet(),
        unet=_UNet(), noisy_latents=noisy, timesteps=torch.zeros(2),
        brushnet_condition=torch.zeros(2, 5, 8, 8),
        brushnet_text_hidden_states=torch.zeros(2, 1, 8),
        unet_hidden_states=torch.zeros(2, 1, 8),
    )
    assert torch.equal(prediction[0], torch.zeros_like(prediction[0]))
    assert torch.equal(prediction[1], torch.ones_like(prediction[1]))
    assert [tuple(value.shape) for value in output.port_down_corrections] == [
        (2, 4, 8, 8), (2, 6, 4, 4), (2, 8, 2, 2),
    ]
    assert tuple(output.port_mid_correction.shape) == (2, 8, 1, 1)
    assert all(torch.count_nonzero(value) == 0 for value in output.port_down_corrections)
    assert torch.count_nonzero(output.port_mid_correction) == 0
    prediction.square().mean().backward()
    assert model.port_adapter.down[0].output.weight.grad is not None
    assert model.port_adapter.down[0].output.weight.grad.abs().sum() > 0
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert not missing, missing
