"""Load legacy SFU test sequences from hierarchical or flat dataset roots.

The long-test dataset uses ``input/Class_X/<source>`` and
``mask/Class_X/<source>``.  The legacy ``dataset/test_1`` and ``test_2``
exports instead use ``<label>/inputs`` and ``<label>/masks``.  This module
accepts either layout without requiring symlinks or copied frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_LONG_TEST_ROOT = Path(
    "/home/cilab/ndquan/videoInpainting/SFU_STC_flow/long_test"
)


@dataclass(frozen=True)
class SequenceSpec:
    split: str
    label: str
    class_name: str
    source_name: str


@dataclass(frozen=True)
class SequenceData:
    spec: SequenceSpec
    image_dir: Path
    mask_dir: Path
    output_dir: Path
    image_paths: Tuple[Path, ...]
    mask_paths: Tuple[Path, ...]
    frame_ids: Tuple[int, ...]


# Keep the legacy test_1/test_2 names requested by the evaluation workflow.
# The old ``RaceHorses`` sequence is the Class_C source, named RaceHorsesC in
# the hierarchical SFU dataset. RaceHorsesD is a distinct Class_D sequence.
LONG_SEQUENCE_SPECS: Tuple[SequenceSpec, ...] = (
    SequenceSpec("test_1", "BasketballPass", "Class_D", "BasketballPass"),
    SequenceSpec("test_1", "ParkScene", "Class_B", "ParkScene"),
    SequenceSpec("test_1", "PartyScene", "Class_C", "PartyScene"),
    SequenceSpec("test_1", "RaceHorses", "Class_C", "RaceHorsesC"),
    SequenceSpec("test_1", "Traffic", "Class_A", "Traffic"),
    SequenceSpec("test_2", "BQMall", "Class_C", "BQMall"),
    SequenceSpec("test_2", "BQSquare", "Class_D", "BQSquare"),
    SequenceSpec("test_2", "BQTerrace", "Class_B", "BQTerrace"),
    SequenceSpec("test_2", "FourPeople", "Class_E", "FourPeople"),
    SequenceSpec("test_2", "PeopleOnStreet", "Class_A", "PeopleOnStreet"),
)


def parse_sequence_names(value: Optional[str]) -> Optional[List[str]]:
    if value is None or not value.strip():
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    return names or None


def select_specs(names: Optional[Sequence[str]]) -> List[SequenceSpec]:
    if names is None:
        return list(LONG_SEQUENCE_SPECS)
    requested = set(names)
    available = {spec.label for spec in LONG_SEQUENCE_SPECS}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(
            f"Unknown sequence(s): {unknown}; available={sorted(available)}"
        )
    return [spec for spec in LONG_SEQUENCE_SPECS if spec.label in requested]


def _numeric_pngs(directory: Path) -> Tuple[Tuple[Path, ...], Tuple[int, ...]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {directory}")
    candidates = list(directory.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No PNG frames found in: {directory}")
    try:
        indexed_paths = sorted((int(path.stem), path) for path in candidates)
    except ValueError as exc:
        raise ValueError(f"PNG filenames must be numeric in: {directory}") from exc
    frame_ids = tuple(frame_id for frame_id, _ in indexed_paths)
    paths = tuple(path for _, path in indexed_paths)
    broken = [path for path in paths if not path.is_file()]
    if broken:
        raise FileNotFoundError(f"Unreadable or broken frame link: {broken[0]}")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"Duplicate numeric frame IDs in: {directory}")
    expected = tuple(range(frame_ids[0], frame_ids[-1] + 1))
    if frame_ids != expected:
        missing = sorted(set(expected) - set(frame_ids))
        raise ValueError(
            f"Frames are not contiguous in {directory}; missing={missing[:20]}"
        )
    return paths, frame_ids


def _detect_layout(root: Path) -> str:
    """Return the supported layout at ``root`` with actionable errors."""
    hierarchical_markers = (root / "input", root / "mask")
    if any(path.exists() for path in hierarchical_markers):
        return "hierarchical"

    flat_markers = tuple(
        root / spec.label / subdirectory
        for spec in LONG_SEQUENCE_SPECS
        for subdirectory in ("inputs", "masks")
    )
    if any(path.exists() for path in flat_markers):
        return "flat"

    raise FileNotFoundError(
        f"Unsupported or missing dataset root: {root}. Expected either "
        "input/Class_X/<sequence> plus mask/Class_X/<sequence>, or "
        "<sequence>/inputs plus <sequence>/masks."
    )


def _flat_specs_present(root: Path, specs: Sequence[SequenceSpec]) -> List[SequenceSpec]:
    """Return only complete flat-layout sequences available at ``root``."""
    present = []
    for spec in specs:
        sequence_root = root / spec.label
        if (sequence_root / "inputs").is_dir() and (sequence_root / "masks").is_dir():
            present.append(spec)
    return present


def _sequence_directories(root: Path, spec: SequenceSpec, layout: str) -> Tuple[Path, Path]:
    if layout == "hierarchical":
        return (
            root / "input" / spec.class_name / spec.source_name,
            root / "mask" / spec.class_name / spec.source_name,
        )
    return root / spec.label / "inputs", root / spec.label / "masks"


def load_sequences(
    long_test_root: Path,
    output_root: Path,
    names: Optional[Sequence[str]] = None,
) -> List[SequenceData]:
    long_test_root = Path(long_test_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    layout = _detect_layout(long_test_root)
    specs = select_specs(names)
    if layout == "flat" and names is None:
        specs = _flat_specs_present(long_test_root, specs)
        if not specs:
            raise FileNotFoundError(
                f"No complete <sequence>/inputs and <sequence>/masks pairs found in: "
                f"{long_test_root}"
            )

    loaded = []
    for spec in specs:
        image_dir, mask_dir = _sequence_directories(long_test_root, spec, layout)
        image_paths, frame_ids = _numeric_pngs(image_dir)
        mask_paths, mask_frame_ids = _numeric_pngs(mask_dir)
        image_names = tuple(path.name for path in image_paths)
        mask_names = tuple(path.name for path in mask_paths)
        if image_names != mask_names or frame_ids != mask_frame_ids:
            raise ValueError(
                f"Input/mask filenames do not match for {spec.label}: "
                f"{len(image_paths)} inputs vs {len(mask_paths)} masks"
            )
        loaded.append(
            SequenceData(
                spec=spec,
                image_dir=image_dir,
                mask_dir=mask_dir,
                output_dir=output_root / spec.split / spec.label,
                image_paths=image_paths,
                mask_paths=mask_paths,
                frame_ids=frame_ids,
            )
        )
    return loaded


def print_preflight(sequences: Iterable[SequenceData]) -> None:
    for data in sequences:
        spec = data.spec
        print(
            f"[Dataset] {spec.split}/{spec.label}: "
            f"source={spec.class_name}/{spec.source_name}, "
            f"frames={len(data.frame_ids)} "
            f"({data.frame_ids[0]:06d}..{data.frame_ids[-1]:06d}), "
            f"output={data.output_dir}"
        )
