"""Teacher-valid dense flow and motion-boundary supervision."""

from __future__ import annotations

from typing import Dict

import torch


def _validate(predicted: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor,
              bg: torch.Tensor):
    if predicted.ndim != 4 or predicted.shape[1] != 2 or predicted.shape != teacher.shape:
        raise ValueError("predicted and teacher flow must be matching [N,2,H,W]")
    expected = (predicted.shape[0], 1, *predicted.shape[-2:])
    if valid.shape != expected or bg.shape != expected:
        raise ValueError(f"valid and BG masks must be {expected}")


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _one_direction(predicted: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor,
                   bg: torch.Tensor, *, bg_weight: float, edge_threshold: float,
                   charbonnier_eps: float) -> Dict[str, torch.Tensor]:
    _validate(predicted, teacher, valid, bg)
    prediction = predicted.float()
    target = teacher.float().detach()
    q = valid.float().detach().clamp(0, 1)
    b = bg.float().detach().clamp(0, 1)
    weighted = q * (1 + bg_weight*b)
    residual = prediction - target
    squared_error = residual.square().sum(1, keepdim=True)
    # Do not differentiate through sqrt(0): its derivative is infinite and
    # can make a perfect zero-residual initialization backpropagate NaNs.
    epe = squared_error.detach().sqrt()
    robust = (squared_error + charbonnier_eps**2).sqrt() - charbonnier_eps
    flow_loss = _masked_mean(robust, weighted)
    all_epe = _masked_mean(epe, q)
    bg_epe = _masked_mean(epe, q*b)

    # Edge weights are fixed pseudo-targets. Require BOTH pixels of each finite
    # difference to be teacher-valid. Empty support produces a zero loss.
    edge_numerator = prediction.new_zeros(())
    edge_denominator = prediction.new_zeros(())
    boundary_epe_num = prediction.new_zeros(())
    boundary_epe_den = prediction.new_zeros(())
    for dim in (-1, -2):
        dp = torch.diff(prediction, dim=dim)
        dt = torch.diff(target, dim=dim)
        if dim == -1:
            pair_valid = q[..., :, 1:] * q[..., :, :-1]
            edge_epe = .5*(epe[..., :, 1:] + epe[..., :, :-1])
        else:
            pair_valid = q[..., 1:, :] * q[..., :-1, :]
            edge_epe = .5*(epe[..., 1:, :] + epe[..., :-1, :])
        teacher_edge = dt.square().sum(1, keepdim=True).sqrt()
        edge_mask = pair_valid * (teacher_edge >= edge_threshold).to(pair_valid.dtype)
        edge_numerator = edge_numerator + ((dp-dt).abs().mean(1, keepdim=True)*edge_mask).sum()
        edge_denominator = edge_denominator + edge_mask.sum()
        boundary_epe_num = boundary_epe_num + (edge_epe*edge_mask).sum()
        boundary_epe_den = boundary_epe_den + edge_mask.sum()
    return {
        "loss_flow": flow_loss,
        "loss_edge": edge_numerator / edge_denominator.clamp_min(1.0),
        "epe": all_epe,
        "bg_epe": bg_epe,
        "boundary_epe": boundary_epe_num / boundary_epe_den.clamp_min(1.0),
        "edge_support": edge_denominator,
        "valid_support": q.sum(),
    }


def refinement_losses(predicted_forward: torch.Tensor, predicted_backward: torch.Tensor,
                      teacher_forward: torch.Tensor, teacher_backward: torch.Tensor,
                      valid_forward: torch.Tensor, valid_backward: torch.Tensor,
                      bg0: torch.Tensor, bg1: torch.Tensor,
                      *, edge_weight: float = .05, bg_weight: float = 1.0,
                      edge_threshold: float = 1.0,
                      charbonnier_eps: float = 1e-3) -> Dict[str, torch.Tensor]:
    """Flow loss plus L_edge on final RGB-grid flow in both directions."""
    if edge_weight < 0 or bg_weight < 0 or edge_threshold < 0 or charbonnier_eps <= 0:
        raise ValueError("loss weights/thresholds must be nonnegative and epsilon positive")
    directions = (
        _one_direction(predicted_forward, teacher_forward, valid_forward, bg0,
                       bg_weight=bg_weight, edge_threshold=edge_threshold,
                       charbonnier_eps=charbonnier_eps),
        _one_direction(predicted_backward, teacher_backward, valid_backward, bg1,
                       bg_weight=bg_weight, edge_threshold=edge_threshold,
                       charbonnier_eps=charbonnier_eps),
    )
    result = {key: .5*(directions[0][key] + directions[1][key]) for key in directions[0]}
    result["loss_total"] = result["loss_flow"] + edge_weight*result["loss_edge"]
    return result
