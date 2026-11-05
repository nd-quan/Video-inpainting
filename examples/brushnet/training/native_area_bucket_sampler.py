"""Deterministic, rank-aligned batches for bounded native-resolution images."""

import math
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Sampler


BUCKETS = {
    "small": {"max_area": 512 * 1024, "batch_per_gpu": 6, "accumulation": 2},
    "medium": {"max_area": 1024 * 1024, "batch_per_gpu": 3, "accumulation": 4},
    "large": {"max_area": None, "batch_per_gpu": 1, "accumulation": 12},
}


def padded_shape(source_width, source_height, max_width, max_height, multiple):
    scale = min(1.0, max_width / source_width, max_height / source_height)
    width = max(1, round(source_width * scale))
    height = max(1, round(source_height * scale))
    return (math.ceil(height / multiple) * multiple,
            math.ceil(width / multiple) * multiple)


def bucket_for_shape(height, width):
    area = height * width
    if area <= BUCKETS["small"]["max_area"]:
        return "small"
    if area <= BUCKETS["medium"]["max_area"]:
        return "medium"
    return "large"


class DistributedAreaBucketBatchSampler(Sampler):
    """One shape per physical batch; synchronized update boundaries on all ranks.

    Each padded-shape group is shuffled independently. An incomplete global
    physical batch repeats indices from that shape. An incomplete accumulation
    group repeats microbatches from its area bucket. These repeats are counted
    in ``stats`` so the sampling bias remains visible.
    """

    def __init__(self, dataset, rank, world_size, seed=0):
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if len(set(dataset.image_ids)) != len(dataset.image_ids):
            raise ValueError("Area bucket training expects unique image IDs")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shape_indices = defaultdict(list)
        for index, image_id in enumerate(dataset.image_ids):
            with Image.open(dataset.root / "gt" / f"{image_id}.png") as image:
                shape = padded_shape(*image.size, dataset.max_width,
                                     dataset.max_height, dataset.pad_multiple)
            self.shape_indices[shape].append(index)
        self.epoch = None
        self.schedule = []
        self.updates_per_epoch = 0
        self.stats = {}
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        rng = random.Random(self.seed + epoch)
        by_bucket = defaultdict(list)
        padded_physical_samples = 0
        for shape, original_indices in sorted(self.shape_indices.items()):
            indices = list(original_indices)
            rng.shuffle(indices)
            bucket = bucket_for_shape(*shape)
            physical_per_gpu = BUCKETS[bucket]["batch_per_gpu"]
            global_physical = physical_per_gpu * self.world_size
            for start in range(0, len(indices), global_physical):
                full = indices[start:start + global_physical]
                deficit = global_physical - len(full)
                if deficit:
                    full.extend(rng.choices(original_indices, k=deficit))
                    padded_physical_samples += deficit
                by_bucket[bucket].append({
                    "shape": shape,
                    "indices": full[
                        self.rank * physical_per_gpu:(self.rank + 1) * physical_per_gpu
                    ],
                })

        update_groups = []
        repeated_microbatches = 0
        for bucket in BUCKETS:
            microbatches = by_bucket[bucket]
            rng.shuffle(microbatches)
            accumulation = BUCKETS[bucket]["accumulation"]
            for start in range(0, len(microbatches), accumulation):
                group = list(microbatches[start:start + accumulation])
                deficit = accumulation - len(group)
                if deficit:
                    group.extend(rng.choices(microbatches, k=deficit))
                    repeated_microbatches += deficit
                update_groups.append((bucket, group))
        rng.shuffle(update_groups)

        self.schedule = []
        for bucket, group in update_groups:
            for position, microbatch in enumerate(group):
                self.schedule.append({
                    "indices": microbatch["indices"],
                    "shape": microbatch["shape"],
                    "bucket": bucket,
                    "accumulation": len(group),
                    "is_update_end": position == len(group) - 1,
                })
        self.updates_per_epoch = len(update_groups)
        self.stats = {
            "epoch": epoch,
            "original_images": len(self.dataset),
            "shape_count": len(self.shape_indices),
            "microbatches_by_bucket": {key: len(by_bucket[key]) for key in BUCKETS},
            "updates_by_bucket": {
                key: sum(group_bucket == key for group_bucket, _ in update_groups)
                for key in BUCKETS
            },
            "padded_physical_samples": padded_physical_samples,
            "repeated_microbatches": repeated_microbatches,
            "updates_per_epoch": self.updates_per_epoch,
            "local_microbatches_per_epoch": len(self.schedule),
        }

    def __len__(self):
        return len(self.schedule)

    def __iter__(self):
        for entry in self.schedule:
            yield entry["indices"]

    def microbatch_offset_after_updates(self, completed_updates):
        if not 0 <= completed_updates <= self.updates_per_epoch:
            raise ValueError("completed_updates out of range")
        if completed_updates == 0:
            return 0
        found = 0
        for index, entry in enumerate(self.schedule):
            if entry["is_update_end"]:
                found += 1
                if found == completed_updates:
                    return index + 1
        raise AssertionError("Schedule has fewer updates than expected")
