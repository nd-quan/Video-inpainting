#!/usr/bin/env python3
"""Build 512px oracle flow from the exact clean GT frames of the reference study."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/eval_temporal_guidance_rgb_vs_latent_student_s0p1_20260918"
SOURCE = Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow/teacher_flows_512x512")
DATASET = ROOT / "examples/brushnet/dataset"
PROPAINTER = Path("/home/cilab/ndquan/videoInpainting/pretrained/ProPainter")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--all-from-flat-gt", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "build_report.json").unlink(missing_ok=True)
    source_metadata = json.loads((SOURCE / "metadata.json").read_text())
    metadata = dict(source_metadata)
    metadata.update(
        dataset_root=str(DATASET),
        manifest=None,
        reference_experiment=str(REFERENCE),
        teacher_model_reference_cache=str(SOURCE),
        format="npz teacher_f/teacher_b [2,512,512] float32 (no valid_f/valid_b)",
        layout="matched/<class>/<sequence>/<frame0>_<frame1>.npz",
        selection=("all pairs from reference flat GT" if args.all_from_flat_gt else
                   "exact clips in RGB/latent reference; missing pairs from flat GT"),
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    missing = []
    reused = 0
    for seq_dir in sorted(REFERENCE.iterdir()):
        if not seq_dir.is_dir():
            continue
        paths = list(seq_dir.glob("latent/test_*/" + seq_dir.name + "/source_metadata.json"))
        if len(paths) != 1:
            raise ValueError(f"Missing source metadata: {seq_dir}")
        spec = json.loads(paths[0].read_text())
        first, last = spec["frame_range"]
        assert last - first + 1 == spec["frame_count"]
        seq = spec["source_sequence"]
        cls = spec["class"]
        branch = output / "matched" / cls / seq
        branch.mkdir(parents=True, exist_ok=True)
        # The inference script validates every adjacent pair before sampling,
        # including clip-boundary pairs that guidance itself does not use.
        for a in range(first, last):
            b = a + 1
            name = f"{a:06d}_{b:06d}.npz"
            dest = branch / name
            candidates = [SOURCE / split / cls / seq / name for split in ("train", "valid", "test")]
            present = [p for p in candidates if p.is_file()]
            if len(present) > 1:
                raise ValueError(f"Ambiguous cached pair: {candidates}")
            if args.all_from_flat_gt:
                if dest.is_symlink():
                    dest.unlink()
                if not dest.is_file():
                    gt = DATASET / spec["split"] / spec["label"] / "gt"
                    missing.append((gt / f"{a:06d}.png", gt / f"{b:06d}.png", dest))
            elif present:
                if dest.exists() or dest.is_symlink():
                    if dest.resolve() != present[0].resolve():
                        raise ValueError(f"Conflicting cache entry: {dest}")
                else:
                    dest.symlink_to(present[0])
                reused += 1
            elif not dest.is_file():
                gt = DATASET / spec["split"] / spec["label"] / "gt"
                missing.append((gt / f"{a:06d}.png", gt / f"{b:06d}.png", dest))
    print(f"Cached pairs linked: {reused}; missing pairs to estimate: {len(missing)}", flush=True)
    if missing:
        sys.path.insert(0, str(ROOT / "examples/brushnet"))
        sys.path.insert(0, str(PROPAINTER))
        from model.modules.flow_comp_raft import initialize_RAFT
        from proposed.STC_encoder_v1.precompute_sfu_stc_teacher_flows import load_frame, estimate_pair
        device = torch.device(args.device)
        model = initialize_RAFT(source_metadata["raft_checkpoint"], device=device).eval()
        for i, (a, b, dest) in enumerate(missing, 1):
            frame_a = load_frame(a, 512, 512, device)
            frame_b = load_frame(b, 512, 512, device)
            forward, backward = estimate_pair(model, frame_a, frame_b, 20)
            arrays = {"teacher_f": forward[0].cpu().numpy().astype(np.float32),
                      "teacher_b": backward[0].cpu().numpy().astype(np.float32)}
            if not all(np.isfinite(x).all() for x in arrays.values()):
                raise FloatingPointError(dest)
            temporary = dest.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(dest)
            if i == 1 or i % 25 == 0 or i == len(missing):
                print(f"Estimated {i}/{len(missing)} missing pairs", flush=True)
    (output / "build_report.json").write_text(json.dumps({"linked_this_invocation": reused,
        "estimated_this_invocation": len(missing),
        "total_pairs": len(list((output / "matched").glob("*/*/*.npz"))),
        "checkpoint": source_metadata["raft_checkpoint"],
        "raft_sha256": source_metadata["raft_sha256"],
        "source": "flat reference GT at 512x512",
        "flow_input": "clean GT", "teacher_is_oracle": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
