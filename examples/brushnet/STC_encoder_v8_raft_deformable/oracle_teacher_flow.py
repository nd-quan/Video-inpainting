"""Strict V7-format clean-teacher cache reader; never estimates missing flow."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class OracleTeacherFlowCache:
    def __init__(self, root, split, resolution):
        self.root = Path(root).resolve()
        self.split = str(split)
        self.resolution = int(resolution)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if (self.metadata["height"], self.metadata["width"]) != (resolution, resolution):
            raise ValueError("Teacher cache resolution differs from evaluation; no implicit resize")
        self.directories = {}

    def directory(self, video):
        if video not in self.directories:
            branch = Path(video)
            if branch.is_absolute() or ".." in branch.parts:
                raise ValueError(f"Invalid video branch: {video}")
            exact = self.root / self.split / branch
            candidates = [exact] if exact.is_dir() else list(
                (self.root / self.split).glob(f"*/{branch.name}")
            )
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected one teacher branch for {video} in {self.root / self.split}; "
                    f"found {candidates}. Do not substitute a different split."
                )
            self.directories[video] = candidates[0]
        return self.directories[video]

    def pair_path(self, video, a, b):
        if b != a + 1:
            raise ValueError(f"Non-adjacent teacher request: {video} {a}->{b}")
        path = self.directory(video) / f"{a:06d}_{b:06d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing clean teacher pair: {path}")
        return path

    def validate_clips(self, clips):
        pairs = set()
        for video, _, ids in clips:
            for a, b in zip(ids, ids[1:]):
                a, b = int(a), int(b)
                if a != b:  # Repeated tail padding represents the same frame.
                    pairs.add((str(video), a, b))
        for video, a, b in sorted(pairs):
            self.pair_path(video, a, b)
        return len(pairs)

    def sequence(self, video, frame_ids, device):
        ids = [int(x) for x in frame_ids.detach().cpu().tolist()]
        forward, backward = [], []
        shape = (2, self.resolution, self.resolution)
        for a, b in zip(ids, ids[1:]):
            if a == b:
                f, r = torch.zeros(shape), torch.zeros(shape)
            else:
                path = self.pair_path(str(video), a, b)
                with np.load(path, allow_pickle=False) as data:
                    arrays = [np.asarray(data[key], dtype=np.float32) for key in ("teacher_f", "teacher_b")]
                    for value in arrays:
                        if value.shape != shape or not np.isfinite(value).all():
                            raise ValueError(f"Invalid RGB-pixel teacher flow in {path}")
                    f, r = [torch.from_numpy(value.copy()) for value in arrays]
            forward.append(f)
            backward.append(r)
        if not forward:
            empty = torch.empty((1, 0, *shape), device=device)
            return SimpleNamespace(forward=empty, backward=empty.clone())
        return SimpleNamespace(
            forward=torch.stack(forward).unsqueeze(0).to(device),
            backward=torch.stack(backward).unsqueeze(0).to(device),
        )
