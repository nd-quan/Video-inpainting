#!/usr/bin/env python3
"""Rebuild native-resolution test frames to match legacy test_1/test_2 IDs.

The legacy test_1 files reset some source frame IDs to zero. The offsets below
were checked against the corresponding 512x512 GT images, rather than inferred
from filenames alone. The existing original_size tree is kept as a backup.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
from PIL import Image


BRUSHNET = Path(__file__).resolve().parents[1]
DATASET = BRUSHNET / "examples/brushnet/dataset"
SOURCE = Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow")
CLEAN_YUV = Path("/home/cilab/ndquan/vcm-rs/Data/SFU")
TARGET = DATASET / "original_size"
KINDS = {"gt": "GT", "inputs": "input", "masks": "mask"}
CASES = (
    ("test_1", "BasketballPass", "BasketballPass", 0),
    ("test_1", "ParkScene", "ParkScene", 0),
    ("test_1", "PartyScene", "PartyScene", 200),
    ("test_1", "RaceHorses", "RaceHorsesC", 100),
    ("test_1", "Traffic", "Traffic", 0),
    ("test_2", "BQMall", "BQMall", 0),
    ("test_2", "BQSquare", "BQSquare", 0),
    ("test_2", "BQTerrace", "BQTerrace", 0),
    ("test_2", "Cactus", "Cactus", 0),
    ("test_2", "FourPeople", "FourPeople", 0),
    ("test_2", "Kimono", "Kimono", 0),
    ("test_2", "KristenAndSara", "KristenAndSara", 0),
    ("test_2", "PeopleOnStreet", "PeopleOnStreet", 0),
)


def frame_ids(directory: Path) -> list[int]:
    paths = sorted(directory.glob("*.png"))
    if not paths or any(not p.stem.isdecimal() or len(p.stem) != 6 for p in paths):
        raise ValueError(f"Missing or invalid frame names: {directory}")
    ids = [int(p.stem) for p in paths]
    if ids != list(range(ids[0], ids[-1] + 1)):
        raise ValueError(f"Frame IDs are not contiguous: {directory}")
    return ids


def source_owners(source_entry: dict) -> dict[int, str]:
    owners = {}
    for split, split_data in source_entry["splits"].items():
        for segment in split_data["segments"]:
            for frame_id in range(segment["start"], segment["end"] + 1):
                if frame_id in owners:
                    raise ValueError(f"Overlapping source frame {source_entry['name']}/{frame_id}")
                owners[frame_id] = split
    return owners


def sample_mae(reference: Path, source: Path) -> float:
    with Image.open(reference) as image:
        a = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.int16)
    with Image.open(source) as image:
        b = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.int16)
    return float(np.abs(a - b).mean())


def plan() -> list[dict]:
    source_entries = {
        entry["name"]: entry
        for entry in json.loads((SOURCE / "manifest.json").read_text())["sequences"]
    }
    jobs = []
    for split, label, source_name, offset in CASES:
        reference = DATASET / split / label
        ids = frame_ids(reference / "gt")
        for kind in KINDS:
            if frame_ids(reference / kind) != ids:
                raise ValueError(f"Legacy {split}/{label}/{kind} has different frame IDs")
        source_entry = source_entries[source_name]
        owners = source_owners(source_entry)
        width, height = source_entry["native_width"], source_entry["native_height"]
        source_paths = []
        source_counts = Counter()
        for legacy_id in ids:
            source_id = legacy_id + offset
            owner = owners.get(source_id)
            if owner is None:
                raise FileNotFoundError(f"Missing source ID {source_name}/{source_id}")
            paths = {
                kind: SOURCE / owner / source_kind / source_entry["class"] / source_name / f"{source_id:06d}.png"
                for kind, source_kind in KINDS.items()
            }
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            source_paths.append((legacy_id, source_id, owner, paths))
            source_counts[owner] += 1

        for index in (0, len(ids) // 2, len(ids) - 1):
            legacy_id, _, _, paths = source_paths[index]
            score = sample_mae(reference / "gt" / f"{legacy_id:06d}.png", paths["gt"])
            if score > 10.0:
                raise ValueError(f"GT content mismatch {split}/{label}/{legacy_id}: MAE={score:.2f}")

        first_paths = source_paths[0][3]
        with Image.open(first_paths["gt"]) as image:
            first_size = image.size
        derived_prefix = label == "BasketballPass" and split == "test_1"
        if derived_prefix and first_size != (512, 512):
            raise ValueError(f"Expected 512x512 early BasketballPass source, got {first_size}")
        if not derived_prefix and first_size != (width, height):
            raise ValueError(f"Expected native source size for {label}, got {first_size}")
        if derived_prefix and not (CLEAN_YUV / source_entry["clean_yuv"]).is_file():
            raise FileNotFoundError(CLEAN_YUV / source_entry["clean_yuv"])
        jobs.append({
            "split": split, "label": label, "source_name": source_name,
            "source_entry": source_entry, "offset": offset, "ids": ids,
            "width": width, "height": height, "source_paths": source_paths,
            "source_split_counts": dict(source_counts),
        })
    return jobs


def save_native_gt_from_yuv(yuv_file, frame_id: int, width: int, height: int, output: Path) -> None:
    frame_bytes = width * height * 3 // 2
    yuv_file.seek(frame_id * frame_bytes)
    raw = np.frombuffer(yuv_file.read(frame_bytes), dtype=np.uint8)
    if raw.size != frame_bytes:
        raise ValueError(f"Clean YUV ended before frame {frame_id}")
    bgr = cv2.cvtColor(raw.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420)
    if not cv2.imwrite(str(output), bgr):
        raise OSError(f"Could not write {output}")


def build(stage: Path, jobs: list[dict], backup: Path) -> None:
    entries = []
    for job in jobs:
        split, label = job["split"], job["label"]
        out = stage / split / label
        for kind in KINDS:
            (out / kind).mkdir(parents=True)
        source_entry = job["source_entry"]
        width, height = job["width"], job["height"]
        yuv_path = CLEAN_YUV / source_entry["clean_yuv"]
        yuv_file = yuv_path.open("rb") if label == "BasketballPass" and split == "test_1" else None
        derived_count = 0
        try:
            for legacy_id, source_id, _, paths in job["source_paths"]:
                name = f"{legacy_id:06d}.png"
                if yuv_file is not None and source_id < 150:
                    # SFU_STC_flow has only square 512x512 input/mask here.
                    # Recover native GT from the clean YUV; reconstruct the
                    # two conditioning files at native canvas resolution.
                    save_native_gt_from_yuv(yuv_file, source_id, width, height, out / "gt" / name)
                    with Image.open(paths["inputs"]) as image:
                        image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR).save(
                            out / "inputs" / name
                        )
                    with Image.open(paths["masks"]) as image:
                        mask = np.asarray(
                            image.convert("L").resize((width, height), Image.Resampling.NEAREST)
                        )
                    Image.fromarray(np.where(mask > 127, 255, 0).astype(np.uint8), "L").save(
                        out / "masks" / name
                    )
                    derived_count += 1
                else:
                    for kind in KINDS:
                        shutil.copy2(paths[kind], out / kind / name)
        finally:
            if yuv_file is not None:
                yuv_file.close()

        ids = job["ids"]
        entry = {
            "split": split, "sequence": label,
            "source_sequence": f"{source_entry['class']}/{job['source_name']}",
            "source_frame_offset": job["offset"],
            "fps": source_entry["fps"], "frames": len(ids),
            "first_frame": ids[0], "last_frame": ids[-1],
            "source_first_frame": ids[0] + job["offset"],
            "source_last_frame": ids[-1] + job["offset"],
            "width": width, "height": height,
            "seconds": len(ids) / source_entry["fps"],
            "source_split_counts": job["source_split_counts"],
            "derived_native_input_mask_frames": derived_count,
        }
        entries.append(entry)
        print(f"Built {split}/{label}: {len(ids)} frames, source offset {job['offset']}, derived input/mask {derived_count}", flush=True)

    manifest = {
        "source_root": str(SOURCE),
        "reference_roots": [str(DATASET / split) for split in ("test_1", "test_2")],
        "dataset_root": str(TARGET),
        "previous_dataset_backup": str(backup),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mask_convention": "0=background, 255=inpaint ROI",
        "frame_mapping": "legacy frame ID + source_frame_offset = SFU source frame ID",
        "note": "BasketballPass IDs 0..149 use native clean-YUV GT; native input/mask are resized from the only available 512x512 SFU_STC_flow files.",
        "sequences": entries,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def validate(stage: Path, jobs: list[dict]) -> None:
    for job in jobs:
        out = stage / job["split"] / job["label"]
        expected = job["ids"]
        expected_size = (job["width"], job["height"])
        for kind in KINDS:
            if frame_ids(out / kind) != expected:
                raise ValueError(f"Wrong frame IDs: {out / kind}")
            for frame_id in expected:
                with Image.open(out / kind / f"{frame_id:06d}.png") as image:
                    if image.size != expected_size:
                        raise ValueError(f"Wrong size: {out / kind / f'{frame_id:06d}.png'}")
    print("Validated all GT/input/mask frame IDs and native dimensions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    jobs = plan()
    for job in jobs:
        ids = job["ids"]
        print(f"{job['split']}/{job['label']}: {ids[0]}..{ids[-1]} -> source {ids[0]+job['offset']}..{ids[-1]+job['offset']}")
    if args.preflight_only:
        return

    backup = TARGET.with_name(f"original_size_before_legacy_alignment_{datetime.now():%Y%m%d}")
    index = 2
    while backup.exists():
        backup = TARGET.with_name(f"original_size_before_legacy_alignment_{datetime.now():%Y%m%d}_{index}")
        index += 1
    stage = Path(tempfile.mkdtemp(prefix=".original_size_aligned_", dir=DATASET))
    try:
        build(stage, jobs, backup)
        validate(stage, jobs)
        TARGET.rename(backup)
        try:
            stage.rename(TARGET)
        except Exception:
            backup.rename(TARGET)
            raise
        print(f"Installed: {TARGET}", flush=True)
        print(f"Previous dataset: {backup}", flush=True)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
