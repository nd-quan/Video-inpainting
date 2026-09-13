#!/usr/bin/env python
"""Branches B/C only: observational standard-DDIM z0 alignment, no guidance."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import torch
from STC_encoder_v8_raft_deformable import evaluate_v8_raft_deformable as v8
from STC_encoder_v8_raft_deformable.latent_warp_diagnostics import analyze, install_capture, mean_rows

BASE_ARGS = v8._add_evaluation_arguments
BASE_PREFLIGHT = v8.preflight
BASE_LOAD = v8.evaluator.load_models
STATE = {}
REQUIRED_GPU = "4"
BG_ONLY = False


def add_args(parser):
    BASE_ARGS(parser)
    parser.add_argument("--latent_capture_steps", default="0,10,25,40,49")
    parser.add_argument("--latent_warp_scale", type=int, default=4)
    parser.set_defaults(latent_bg_only=BG_ONLY)
    parser.set_defaults(dataset_root=HERE.parent / "dataset/test_1", dataset_layout="flat_test", split="test",
                        clip_length=16, clip_stride=12, shared_bg_noise_strength=0.95)


def preflight(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != REQUIRED_GPU:
        raise ValueError(f"Diagnostic requires CUDA_VISIBLE_DEVICES={REQUIRED_GPU} (single GPU)")
    if Path(args.dataset_root).resolve() != (HERE.parent / "dataset/test_1").resolve():
        raise ValueError("This diagnostic is restricted to dataset/test_1")
    if args.condition_flow_source != "student" or args.device not in ("cuda", "cuda:0"):
        raise ValueError("Use standard student V8 on the single visible GPU")
    steps = set(map(int, args.latent_capture_steps.split(',')))
    if not steps or min(steps)<0 or max(steps)>=args.num_inference_steps or args.latent_warp_scale<1:
        raise ValueError("Invalid inference-step indices or warp scale")
    STATE.update(args=args, steps=steps, pairs={}, clips=[])
    result = BASE_PREFLIGHT(args)
    dataset, _ = result
    # The inherited flat-test loader does not apply include_branches itself.
    if args.include_branches:
        selected = set(args.include_branches)
        available = {str(v) for v,_,_ in dataset.clips}
        if selected - available:
            raise ValueError(f"Unknown test_1 sequences: {selected-available}")
        dataset.clips = [c for c in dataset.clips if str(c[0]) in selected]
        dataset.rebuild_predecessors()
        dataset.branch_count = len(selected)
        dataset.covered_frame_count = len({p for _,paths,_ in dataset.clips for p in paths})
    print(json.dumps({"latent_diagnostic": "standard DDIM pred_original_sample; observational only",
                      "capture_inference_indices": sorted(steps), "CGE": False, "temporal_guidance": False,
                      "selected_sequences": sorted({str(v) for v,_,_ in dataset.clips}),
                      "selected_clip_count":len(dataset.clips),
                      "primary_FB_filter": False, "sampling_contract": "One original scheduler.step per step; unchanged prev_sample"}))
    return result


def load_models(args, paths, device):
    loaded = BASE_LOAD(args, paths, device)
    adapter = loaded[4]
    def capture_flow(module, inputs, output):
        # Called for predecessor then current. Last output is current clip.
        # Reuse EXACT V8 output, not another estimator/provider call.
        STATE["flow"] = output.raft_flow_backward_rgb.detach().float().cpu().clone().squeeze(0)
    adapter.register_forward_hook(capture_flow)
    return loaded


def before(*, pipe, sample, args, device):
    STATE["captured"] = {}
    STATE["undo"] = install_capture(pipe.scheduler, STATE["steps"], STATE["captured"])
    return {}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def after(*, pipe, sample, args, device):
    captured = STATE["captured"]
    assert set(captured) == STATE["steps"], "Missing captured steps"
    ids = list(map(int, sample["frame_ids"].tolist()))
    video = str(sample["video"])
    clip = {"video":video, "frame_ids":ids, "latent_analysis":{}}
    for index, (timestep,z) in sorted(captured.items()):
        rows = analyze(z, STATE["flow"], sample["masks"].cpu(), args.latent_warp_scale, bg_only=BG_ONLY)
        clip["latent_analysis"][f"step_{index}"] = {"training_timestep": timestep, "aggregate":mean_rows(rows), "pairs":[]}
        for i,row in enumerate(rows):
            record = {"video":video, "frame_ids":ids[i:i+2], "step":index, "metrics":row}
            clip["latent_analysis"][f"step_{index}"]["pairs"].append(record)
            STATE["pairs"].setdefault((video,ids[i],ids[i+1],index), record)
    root = Path(args.output_dir)/"latent_analysis"
    relative = f"clips/{video}/{ids[0]:06d}_{ids[-1]:06d}.json"
    dump(root/relative, clip)
    STATE["clips"].append(relative)
    records = list(STATE["pairs"].values())
    dump(root/"pairs.json", records)
    sequences = {}
    global_steps = {}
    for index in sorted(STATE["steps"]):
        step = f"step_{index}"
        selected = [r for r in records if r["step"]==index]
        per_video = {}
        for name in sorted({r["video"] for r in selected}):
            per_video[name] = mean_rows([r["metrics"] for r in selected if r["video"]==name])
            sequences.setdefault(name,{})[step] = per_video[name]
        macro = mean_rows(list(per_video.values()))
        macro.pop("pair_count",None)
        global_steps[step] = {"pair_weighted":mean_rows([r["metrics"] for r in selected]),
                              "sequence_macro":macro, "sequence_count":len(per_video)}
    dump(root/"sequences.json", sequences)
    dump(root/"summary.json", {"steps":global_steps, "clips":STATE["clips"],
        "pair_selection":"earliest clip occurrence per absolute adjacent pair; no cross-clip latent mixing",
        "primary_mask":"target BG * direct-warped source BG>=0.5 * direct validity * downsampled rescaled validity; no FB",
        "support_ratio_denominator":"all latent pixels; within-motion-bin pixels for bin support ratios",
        "warp_scale":args.latent_warp_scale, "bg_only_composite":BG_ONLY,
        "bg_only_formula":"A*warped+(1-A)*current; B uses primary, C uses common" if BG_ONLY else None,
        "optional_FB":"not implemented",
        "interpretation":"High errors can reflect disocclusion. Inspect support and motion bins, not error alone.",
        "status":"incremental; see standard evaluation summary for run completion"})
    return {"latent_capture_count":len(captured)}


def clear(*, pipe):
    undo = STATE.pop("undo",None)
    if undo: undo()
    STATE.pop("captured",None)
    STATE.pop("flow",None)


def main():
    v8._add_evaluation_arguments = add_args
    v8.preflight = preflight
    v8.evaluator.load_models = load_models
    v8.evaluator.BEFORE_PIPELINE_CALL_FN = before
    v8.evaluator.AFTER_PIPELINE_CALL_FN = after
    v8.evaluator.CLEAR_PIPELINE_CALL_FN = clear
    v8.main()


if __name__ == "__main__":
    main()
