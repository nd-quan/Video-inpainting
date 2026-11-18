"""Load the original ProPainter RAFT-Large teacher, not torchvision RAFT."""

import argparse
import importlib
from pathlib import Path
import sys

import torch


class RAFTArguments(argparse.Namespace):
    # Original RAFT uses both args.foo and 'foo' in args.
    def __contains__(self, key):
        return hasattr(self, key)


def load_original_raft(config, device):
    root = Path(config["propainter_root"]).resolve()
    if not (root / "RAFT/raft.py").is_file():
        raise FileNotFoundError(root / "RAFT/raft.py")
    sys.path.insert(0, str(root))
    module = importlib.import_module("RAFT.raft")
    if root not in Path(module.__file__).resolve().parents:
        raise RuntimeError(
            "A different RAFT package is already imported: " + module.__file__
        )
    checkpoint = Path(config["raft_weights"])
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    state = {
        (
            key.removeprefix("module.")
            if hasattr(key, "removeprefix")
            else (key[7:] if key.startswith("module.") else key)
        ): value
        for key, value in state.items()
    }
    model = module.RAFT(
        RAFTArguments(
            small=False, mixed_precision=False, alternate_corr=False, dropout=0.0
        )
    )
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).eval().to(device)
