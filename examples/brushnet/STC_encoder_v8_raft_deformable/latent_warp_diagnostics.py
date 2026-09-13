"""Read-only predicted-clean latent diagnostics. No FB filtering, no recursion.

FB filtering would hide difficult motion/occlusion, the subject of this test.
High primary error can mean disocclusion, not necessarily an interpolation bug.
"""
import functools
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim_temporal import resize_flow
from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import backward_warp_feature

BINS = (("motion_0_4", 0, 4), ("motion_4_16", 4, 16),
        ("motion_16_32", 16, 32), ("motion_32_plus", 32, float("inf")))


def install_capture(scheduler, steps, captured):
    """Return an undo callback. Original step executes exactly once, unchanged.

    Requesting its structured output exposes z0 without changing computations.
    Preserve the original tuple API when the pipeline requests return_dict=False.
    functools.wraps preserves the eta/generator signature inspected by pipelines.
    """
    original = scheduler.step
    counter = [0]
    @functools.wraps(original)
    def step(*args, **kwargs):
        # Pipeline passes only model_output, timestep and sample positionally.
        if len(args) > 3:
            raise ValueError("Diagnostic step expects optional arguments by keyword")
        return_dict = kwargs.get("return_dict", True)
        kwargs["return_dict"] = True
        result = original(*args, **kwargs)
        index = counter[0]
        if index in steps:
            captured[index] = (int(args[1]), result.pred_original_sample.detach().float().cpu().clone())
        counter[0] += 1
        return result if return_dict else (result.prev_sample,)
    scheduler.step = step
    return lambda: setattr(scheduler, "step", original)


def pair_metrics(target, previous, direct, rescaled, mask):
    n = int(mask.sum())
    if not n:
        return {"support_pixels": 0, **{k: None for k in (
            "no_warp_relative_mse", "direct_relative_mse", "direct_cosine", "direct_warp_gain",
            "rescaled_relative_mse", "rescaled_cosine", "rescaled_warp_gain", "rescaled_vs_direct_gain")}}
    m = mask.float()
    denominator = float((m * target.square().sum(0, keepdim=True)).sum()) + 1e-8
    def error(x):
        return float((m * (target-x).square().sum(0, keepdim=True)).sum()) / denominator
    def cosine(x):
        return float((m * (F.normalize(target, dim=0, eps=1e-6) * F.normalize(x, dim=0, eps=1e-6)).sum(0, keepdim=True).clamp(-1,1)).sum()) / n
    e0, eb, ec = map(error, (previous, direct, rescaled))
    return dict(support_pixels=n, no_warp_relative_mse=e0, direct_relative_mse=eb,
                direct_cosine=cosine(direct), direct_warp_gain=(e0-eb)/(e0+1e-8),
                rescaled_relative_mse=ec, rescaled_cosine=cosine(rescaled),
                rescaled_warp_gain=(e0-ec)/(e0+1e-8), rescaled_vs_direct_gain=(eb-ec)/(eb+1e-8))


def bg_only_composite(warped, current, support):
    """Apply the warp only on valid BG; keep current latent everywhere else."""
    result = torch.where(support, warped, current)
    assert torch.equal(result.masked_select(~support), current.masked_select(~support))
    return result


def analyze(z, flow_rgb, bg, scale, bg_only=False):
    assert z.ndim == 4 and z.shape[1] == 4, "Must capture predicted-clean [T,4,H,W]"
    assert flow_rgb.ndim == 4 and flow_rgb.shape[:2] == (len(z)-1, 2)
    assert bg.shape[:2] == (len(z), 1) and bg.shape[-2:] == flow_rgb.shape[-2:]
    assert scale >= 1 and int(scale) == scale
    assert all(torch.isfinite(x).all() for x in (z, flow_rgb, bg))
    z, flow_rgb, bg = z.float(), flow_rgb.float(), bg.float()
    h, w = z.shape[-2:]; up = (h*scale, w*scale)
    # Each source is the ORIGINAL captured latent, never a preceding warp.
    previous, target = z[:-1], z[1:]
    direct_flow = resize_flow(flow_rgb, (h,w))
    direct, valid_b = backward_warp_feature(previous, direct_flow)
    rescaled_up, valid_c_up = backward_warp_feature(F.interpolate(previous, size=up, mode="nearest"), resize_flow(flow_rgb, up))
    rescaled = F.interpolate(rescaled_up, size=(h,w), mode="nearest")
    valid_c = F.interpolate(valid_c_up.float(), size=(h,w), mode="nearest") > 0.5
    bg_native = F.interpolate(bg, size=(h,w), mode="nearest") >= 0.5
    source_bg, _ = backward_warp_feature(bg_native[:-1].float(), direct_flow)
    # One shared source-BG support, defined in native target coordinates.
    primary = bg_native[1:] & (source_bg >= 0.5) & valid_b
    common = primary & valid_c
    if bg_only:
        direct = bg_only_composite(direct, target, primary)
        rescaled = bg_only_composite(rescaled, target, common)
    # Magnitude measured BEFORE resize so bins remain original RGB pixel units.
    magnitude = F.interpolate(flow_rgb.square().sum(1,keepdim=True).sqrt(), size=(h,w), mode="nearest")
    results = []
    for i in range(len(previous)):
        row = pair_metrics(target[i], previous[i], direct[i], rescaled[i], common[i])
        row.update(primary_bg_valid_ratio=float(primary[i].float().mean()),
                   common_valid_ratio=float(common[i].float().mean()),
                   target_bg_ratio=float(bg_native[i+1].float().mean()))
        if bg_only:
            roi = ~bg_native[i+1]
            row["direct_roi_changed_elements"] = int(torch.count_nonzero((direct[i]-target[i])*roi))
            row["rescaled_roi_changed_elements"] = int(torch.count_nonzero((rescaled[i]-target[i])*roi))
        row["motion_bins"] = {}
        for name, low, high in BINS:
            region = (magnitude[i] >= low) & (magnitude[i] < high)
            item = pair_metrics(target[i], previous[i], direct[i], rescaled[i], common[i] & region)
            denominator = max(int(region.sum()), 1)
            item.update(motion_pixel_count=int(region.sum()),
                        primary_bg_valid_ratio=float((primary[i] & region).sum()) / denominator,
                        common_valid_ratio=float((common[i] & region).sum()) / denominator)
            row["motion_bins"][name] = item
        results.append(row)
    return results


def mean_rows(rows):
    """Equal pair weighting; unsupported errors are null, never zero."""
    result = {"pair_count": len(rows)}
    if not rows:
        return result
    for key in rows[0]:
        if key == "motion_bins":
            result[key] = {name: mean_rows([r[key][name] for r in rows]) for name,_,_ in BINS}
        elif isinstance(rows[0][key], (float,int)) or rows[0][key] is None:
            values = [r[key] for r in rows if r[key] is not None]
            result[key] = sum(values)/len(values) if values else None
    return result
