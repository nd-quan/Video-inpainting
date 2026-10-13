"""Current and predecessor clip teacher flow for explicit cross-clip memory."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence

import torch

BRUSHNET_EXAMPLES = Path(__file__).resolve().parents[2]
if str(BRUSHNET_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_EXAMPLES))

try:
    from STC_encoder_v3_rgb_flow.teacher_flow_data import TEACHER_KEYS
    from STC_encoder_v5_relative_crossclip.cross_clip_data import (
        CrossClipTeacherFlowV8Dataset,
        collate_cross_clip_teacher_flow,
    )
except ModuleNotFoundError:  # Imported through examples.brushnet.proposed.
    from ..STC_encoder_v3_rgb_flow.teacher_flow_data import TEACHER_KEYS
    from ..STC_encoder_v5_relative_crossclip.cross_clip_data import (
        CrossClipTeacherFlowV8Dataset,
        collate_cross_clip_teacher_flow,
    )


PREVIOUS_TEACHER_KEYS = tuple("previous_" + key for key in TEACHER_KEYS)


class CrossClipOracleFlowDataset(CrossClipTeacherFlowV8Dataset):
    """Add predecessor teacher tensors; the V5 loader only adds predecessor RGB."""

    def __getitem__(self, item: int) -> Dict:
        sample = super().__getitem__(item)
        if not bool(sample["previous_valid_mask"].any()):
            for key, previous_key in zip(TEACHER_KEYS, PREVIOUS_TEACHER_KEYS):
                sample[previous_key] = torch.zeros_like(sample[key])
            return sample

        branch = str(sample["previous_video"])
        ids = [int(value) for value in sample["previous_frame_ids"]]
        pairs = []
        for a, b in zip(ids[:-1], ids[1:]):
            if b != a + 1:
                raise ValueError(f"Predecessor teacher pair must be adjacent: {branch} {a}->{b}")
            pairs.append(self._load_teacher_pair(self._teacher_path(branch, a, b)))
        for index, previous_key in enumerate(PREVIOUS_TEACHER_KEYS):
            sample[previous_key] = torch.stack([pair[index] for pair in pairs])
        return sample


def collate_oracle_flow_clips(examples: Sequence[Dict]) -> Dict:
    batch = collate_cross_clip_teacher_flow(examples)
    for key in PREVIOUS_TEACHER_KEYS:
        batch[key] = torch.stack([example[key] for example in examples], dim=0)
    return batch
