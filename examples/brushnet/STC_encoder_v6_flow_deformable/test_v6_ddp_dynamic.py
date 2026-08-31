#!/usr/bin/env python
"""Two-rank CPU regression for V6's dynamic predecessor-memory graph."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
    RelativeCrossClipBGSTCAdapter,
)
from STC_encoder_v6_flow_deformable.deformable_alignment_loss import (
    compute_deformable_alignment_loss,
)
from STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (
    FlowGuidedDeformableBGSTCAdapter,
)


def build_model() -> FlowGuidedDeformableBGSTCAdapter:
    torch.manual_seed(2026)
    source = RelativeCrossClipBGSTCAdapter(
        hidden_channels=8,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        flow_max_displacement=(2.0, 2.0),
        relative_position_max_distance=8,
        cross_clip_memory_frames=2,
    )
    model = FlowGuidedDeformableBGSTCAdapter(
        hidden_channels=8,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        flow_max_displacement=(2.0, 2.0),
        relative_position_max_distance=8,
        cross_clip_memory_frames=2,
        deform_hidden_channels=16,
        deform_groups=2,
    )
    model.load_state_dict(source.state_dict(), strict=False)
    return model


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.set_num_threads(1)
    model = build_model().train()
    wrapped = DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)
    checkpoint = Path(tempfile.gettempdir()) / (
        f"v6_ddp_dynamic_{os.environ.get('MASTER_PORT', 'default')}.pt"
    )
    previous_ids = torch.tensor([[0, 1, 2, 3]])
    current_ids = torch.tensor([[2, 3, 4, 5]])

    for step in range(5):
        generator = torch.Generator().manual_seed(1000 + 10 * step + rank)
        current_rgb = torch.randn(1, 4, 3, 32, 32, generator=generator)
        bg = torch.ones(1, 4, 1, 32, 32)
        memory = None
        # Deliberately give the two ranks different graph branches.
        if (step + rank) % 2:
            previous_rgb = torch.randn(1, 4, 3, 32, 32, generator=generator)
            with torch.no_grad():
                previous = wrapped.module(
                    previous_rgb,
                    bg,
                    output_size=(4, 4),
                    frame_ids=previous_ids,
                )
            memory = previous.temporal_memory.detach()
        output = wrapped(
            current_rgb,
            bg,
            output_size=(4, 4),
            frame_ids=current_ids,
            temporal_memory=memory,
        )
        batch, pairs, _, height, width = output.predicted_flow_forward.shape
        teacher = torch.zeros(batch, pairs, 2, height, width)
        valid = torch.ones(batch, pairs, 1, height, width)
        deform = compute_deformable_alignment_loss(
            spatial_features=output.spatial_features,
            deformed_previous=output.deformed_previous_features,
            deformed_next=output.deformed_next_features,
            residual_offset_backward=output.residual_offset_backward,
            residual_offset_forward=output.residual_offset_forward,
            reliability_backward=output.deform_reliability_backward,
            reliability_forward=output.deform_reliability_forward,
            teacher_forward=teacher,
            teacher_backward=teacher,
            valid_forward=valid,
            valid_backward=valid,
        )
        # Real training consumes delta_bg through the frozen BrushNet/U-Net;
        # include it here so output_norm/zero_conv follow the same DDP graph.
        loss = output.delta_bg.float().square().mean() + deform.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"rank {rank} step {step}: non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1:
            if rank == 0:
                torch.save(
                    {
                        "model": wrapped.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    checkpoint,
                )
            dist.barrier()
            state = torch.load(checkpoint, map_location="cpu")
            wrapped.module.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            dist.barrier()

    final_head = wrapped.module.deformable_alignment.offset_mask_head[-1]
    if final_head.weight.grad is None or not torch.isfinite(final_head.weight.grad).all():
        raise RuntimeError(f"rank {rank}: missing/non-finite deform-head gradient")
    dist.barrier()
    if rank == 0:
        checkpoint.unlink(missing_ok=True)
        print("V6 DDP dynamic-memory 5-step + resume regression: ok")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
