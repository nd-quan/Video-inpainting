"""CPU/Gloo regression: DDP ranks may have different predecessor availability."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from model import StandaloneV8RAdapter


def _worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(1234)
        model = StandaloneV8RAdapter(
            hidden_channels=8, deform_hidden_channels=16, deform_groups=2,
            rescaled_warp_scale=2,
        )
        model.requires_grad_(False)
        model.deformable_alignment.requires_grad_(True)
        model.deformable_fusion.requires_grad_(True)
        model.stc_adapter.temporal_blocks.requires_grad_(True)
        model.stc_adapter.output_norm.requires_grad_(True)
        model.stc_adapter.zero_conv.requires_grad_(True)
        rgb = torch.randn(1, 2, 3, 64, 64)
        bg = torch.ones(1, 2, 1, 64, 64)
        flow = torch.zeros(1, 1, 2, 64, 64)
        ids = torch.tensor([[0, 1]])
        with torch.no_grad():
            memory = model(rgb, bg, frame_ids=ids,
                           raft_flow_forward_rgb=flow,
                           raft_flow_backward_rgb=flow).temporal_memory
        ddp = DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD((p for p in ddp.parameters() if p.requires_grad), lr=1e-3)
        for step in range(2):
            # One rank has memory while the other does not, then swap.
            has_memory = (rank + step) % 2 == 1
            output = ddp(rgb, bg, frame_ids=ids,
                         temporal_memory=memory if has_memory else None,
                         raft_flow_forward_rgb=flow,
                         raft_flow_backward_rgb=flow)
            loss = output.delta_bg.square().mean()
            loss = loss + 0.1 * output.deformed_previous_features.square().mean()
            loss = loss + 0.1 * output.deformed_next_features.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gate = ddp.module.stc_adapter.temporal_blocks[0].cross_clip_gate
            assert gate.grad is not None, f"rank={rank}, step={step}: gate has no gradient"
            optimizer.step()
        print(f"rank={rank}: two mixed-memory DDP steps passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        url = f"file://{Path(directory) / 'rendezvous'}"
        mp.spawn(_worker, args=(url,), nprocs=2, join=True)
