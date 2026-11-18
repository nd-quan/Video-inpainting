#!/usr/bin/env python3
"""Standalone full-reference video metrics: PSNR, SSIM, LPIPS, WE, WE_L2, FS, VFID_Clips.

Self-contained port of the BrushNet evaluation chain
    Quan_test/evaluate_flat_sequence_metrics.py
      -> examples/brushnet/proposed/DGAF_VSR_original/postprocess_eval.py
         (+ Quan_test/metric.py, Quan_test/evaluate_temporal_metrics.py,
            Quan_test/compute_vfid_clips_4datasets.py,
            video-inpainting-evaluation/src/fid/util.py)
with identical metric definitions. Only this folder is needed at runtime.

Usage:
    python evaluate.py --pred-root <generated>/test --gt-root <dataset>/test
"""
import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pytorch_i3d import InceptionI3d  # noqa: E402

VERSION = '1.0'
MIN_FRAMES = 10          # one full I3D window
I3D_WINDOW = 10
I3D_SIZE = (224, 224)
TEMPORAL_KEYS = ('WE', 'WE_L2', 'FS')

DEFINITIONS = {
    'scope': 'full-frame prediction vs GT at prediction resolution',
    'PSNR': 'RGB uint8; frame mean; higher is better',
    'SSIM': 'torchmetrics SSIM data_range=1; higher is better',
    'LPIPS': 'AlexNet; lower is better',
    'WE': 'backward Farneback warped adjacent prediction frames, valid-pixel RGB L1 in [0,1]; lower is better',
    'WE_L2': 'squared L2 warping error on adjacent prediction frames; lower is better',
    'FS': 'adjacent prediction-frame SSIM (11x11 Gaussian, sigma 1.5, per-channel mean); higher is better',
    'VFID_Clips': 'I3D rgb_imagenet, 10-frame sliding windows, stride 1, 224x224 bicubic, empirical Frechet; lower is better',
    'aggregation': 'frame-weighted image metrics; pair-weighted temporal metrics; pooled VFID features',
}


# ----------------------------------------------------------------------------
# Image metrics (Quan_test/metric.py)
# ----------------------------------------------------------------------------
def psnr_np(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def to_torch_01(img):
    """RGB uint8 HxWxC -> 1xCxHxW float tensor in [0, 1]."""
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0


# ----------------------------------------------------------------------------
# Temporal metrics (Quan_test/evaluate_temporal_metrics.py)
# ----------------------------------------------------------------------------
def _rgb_to_gray_uint8(image):
    return cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _resize_like(image, target_hw):
    h, w = target_hw
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def _compute_flow(prev_rgb, next_rgb):
    return cv2.calcOpticalFlowFarneback(
        _rgb_to_gray_uint8(prev_rgb), _rgb_to_gray_uint8(next_rgb), None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def _warp_with_flow(image, flow):
    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    return warped, valid


def warping_error(prev_rgb, next_rgb):
    """float32 RGB [0,1] frames -> (L1, squared L2) over valid warped pixels."""
    next_rgb = _resize_like(next_rgb, prev_rgb.shape[:2])
    warped_prev, valid = _warp_with_flow(prev_rgb, _compute_flow(next_rgb, prev_rgb))
    if not np.any(valid):
        return float('nan'), float('nan')
    diff = warped_prev[valid] - next_rgb[valid]
    return float(np.mean(np.abs(diff))), float(np.mean(diff * diff))


def frame_ssim(image_a, image_b):
    image_b = _resize_like(image_b, image_a.shape[:2])
    k1d = cv2.getGaussianKernel(11, 1.5)
    kernel = k1d @ k1d.T
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    scores = []
    for ch in range(image_a.shape[2]):
        a, b = image_a[..., ch], image_b[..., ch]
        mu_a, mu_b = cv2.filter2D(a, -1, kernel), cv2.filter2D(b, -1, kernel)
        mu_a_sq, mu_b_sq, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        sigma_a_sq = cv2.filter2D(a * a, -1, kernel) - mu_a_sq
        sigma_b_sq = cv2.filter2D(b * b, -1, kernel) - mu_b_sq
        sigma_ab = cv2.filter2D(a * b, -1, kernel) - mu_ab
        ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
            (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2))
        scores.append(float(np.mean(ssim_map)))
    return float(np.mean(scores))


# ----------------------------------------------------------------------------
# VFID (video-inpainting-evaluation/src/fid/util.py,
#       Quan_test/compute_vfid_clips_4datasets.py)
# ----------------------------------------------------------------------------
def _frame_to_i3d_tensor(pil_image):
    arr = np.array(pil_image, dtype=np.float32) / 255
    return torch.from_numpy(np.expand_dims(arr.transpose(2, 0, 1), axis=0))


def extract_clip_features(model, frames_fn, num_frames, device, window=I3D_WINDOW):
    """Nx1024 float32 I3D Logits-endpoint features, sliding window with stride 1."""
    features = []
    num_windows = max(1, num_frames - window + 1)
    cur = []
    for t in range(window):
        cur.append(_frame_to_i3d_tensor(frames_fn(t)) if t < num_frames else cur[-1])
    for t in range(num_windows):
        pred = model.extract_features(torch.stack(cur, dim=2).to(device), target_endpoints='Logits')
        features.append(pred[0].squeeze().cpu().numpy())
        if t + window < num_frames:
            del cur[0]
            cur.append(_frame_to_i3d_tensor(frames_fn(t + window)))
    return np.stack(features)


def frechet_distance_low_rank(features_a, features_b):
    """Exact empirical Frechet distance without a feature_dim x feature_dim sqrtm."""
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f'Incompatible feature shapes: {a.shape} and {b.shape}')
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError('At least two clips per distribution are required.')
    mean_diff = np.square(a.mean(axis=0) - b.mean(axis=0)).sum()
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    trace_a = np.square(a_c).sum() / (a.shape[0] - 1)
    trace_b = np.square(b_c).sum() / (b.shape[0] - 1)
    cross = a_c @ b_c.T
    cross /= np.sqrt((a.shape[0] - 1) * (b.shape[0] - 1))
    trace_covmean = np.linalg.svd(cross, compute_uv=False).sum()
    return float(max(mean_diff + trace_a + trace_b - 2.0 * trace_covmean, 0.0))


# ----------------------------------------------------------------------------
# Data discovery (Quan_test/evaluate_flat_sequence_metrics.py)
# ----------------------------------------------------------------------------
def numbered_pngs(folder):
    return {int(p.stem): p for p in folder.glob('*.png') if p.stem.isdigit()}


def discover(args):
    pred_root, gt_root = args.pred_root, args.gt_root
    layout = args.layout
    if layout == 'auto':
        layout = 'hierarchical' if any(d.is_dir() for d in pred_root.glob('Class_*')) else 'flat'
    if layout == 'hierarchical':
        sources = sorted(seq for cls in pred_root.glob('Class_*') if cls.is_dir()
                         for seq in cls.iterdir() if seq.is_dir())
    else:
        sources = sorted(d for d in pred_root.iterdir() if d.is_dir())

    records = []
    for source in sources:
        if args.sequences and source.name not in args.sequences:
            continue
        predictions = numbered_pngs(source)
        if not predictions:
            continue
        if layout == 'hierarchical':
            candidates = [gt_root / 'GT' / source.parent.name / source.name,
                          gt_root / source.parent.name / source.name]
        else:
            candidates = [gt_root / source.name / 'gt', gt_root / source.name]
        gt_dir = next((c for c in candidates if numbered_pngs(c)), None)
        if gt_dir is None:
            raise FileNotFoundError(f'No GT PNGs for {source.name}; looked in: ' +
                                    ', '.join(map(str, candidates)))
        references = numbered_pngs(gt_dir)
        ids = sorted(predictions)
        missing_gt = sorted(set(ids) - set(references))
        if missing_gt:
            raise ValueError(f'Missing GT for {source.name}: frames {missing_gt[:10]}...')
        if len(ids) < MIN_FRAMES or ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError(f'Need at least {MIN_FRAMES} continuous frames: {source.name}')
        complete = set(ids) == set(references)
        if not complete and not args.allow_partial:
            raise ValueError(f'Incomplete sequence (use --allow-partial): {source.name} '
                             f'{len(ids)}/{len(references)} frames')
        records.append({'name': source.name, 'pred_dir': source, 'gt_dir': gt_dir, 'ids': ids,
                        'predictions': predictions, 'references': references, 'complete': complete})
    if not records:
        raise ValueError(f'No generated sequences found under {pred_root} (layout={layout})')
    names = [r['name'] for r in records]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate sequence names across class directories')
    if args.sequences and set(names) != set(args.sequences):
        raise ValueError(f'Sequences not found: {sorted(set(args.sequences) - set(names))}')
    return layout, sorted(records, key=lambda r: r['name'])


def load_gt(path, size, require_native):
    """GT as RGB at prediction size: PIL bilinear resize when sizes differ."""
    with Image.open(path) as im:
        if im.size == size and im.mode == 'RGB':
            return im.copy()
        if require_native and im.size != size:
            raise ValueError(f'Prediction/GT size mismatch: {path}: {size} vs {im.size}')
        return im.convert('RGB').resize(size, Image.Resampling.BILINEAR)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def evaluate_sequence(rec, args, models, device, score_names):
    perceptual, i3d, ssim_fn = models
    pred_paths = [rec['predictions'][i] for i in rec['ids']]
    gt_paths = [rec['references'][i] for i in rec['ids']]
    scores = {k: [] for k in score_names}
    gt_small, pred_small = [], []
    prev = None
    for pred_path, gt_path in zip(pred_paths, gt_paths):
        with Image.open(pred_path) as im:
            pred_img = im.convert('RGB')
        gt_img = load_gt(gt_path, pred_img.size, args.require_native_size)
        pred, gt = np.array(pred_img), np.array(gt_img)
        if pred.shape != gt.shape:
            raise ValueError('Image shape mismatch')
        a, b = to_torch_01(pred).to(device), to_torch_01(gt).to(device)
        scores['PSNR'].append(psnr_np(gt, pred))
        scores['SSIM'].append(ssim_fn(a, b, data_range=1.0).item())
        scores['LPIPS'].append(perceptual(a * 2 - 1, b * 2 - 1).item())
        cur = pred.astype(np.float32) / 255
        if prev is not None:
            if not args.skip_warping:
                l1, l2 = warping_error(prev, cur)
                scores['WE'].append(l1)
                scores['WE_L2'].append(l2)
            scores['FS'].append(frame_ssim(prev, cur))
        prev = cur
        gt_small.append(gt_img.resize(I3D_SIZE, Image.Resampling.BICUBIC))
        pred_small.append(pred_img.resize(I3D_SIZE, Image.Resampling.BICUBIC))
    n = len(rec['ids'])
    gt_feat = extract_clip_features(i3d, gt_small.__getitem__, n, device)
    pred_feat = extract_clip_features(i3d, pred_small.__getitem__, n, device)
    row = {'sequence': rec['name'], 'frames': n, 'pairs': n - 1,
           **{k: float(np.mean(v)) for k, v in scores.items()},
           'VFID_Clips': frechet_distance_low_rank(gt_feat, pred_feat)}
    return row, gt_feat, pred_feat


def load_cached(out, rec, score_names):
    row_p = out / f'{rec["name"]}.json'
    gt_p, pred_p = out / f'{rec["name"]}_gt_i3d.npy', out / f'{rec["name"]}_final_i3d.npy'
    if not (row_p.is_file() and gt_p.is_file() and pred_p.is_file()):
        return None
    try:
        row = json.loads(row_p.read_text())
        n = len(rec['ids'])
        if (row['sequence'] != rec['name'] or row['frames'] != n or row['pairs'] != n - 1
                or not all(np.isfinite(row[k]) for k in (*score_names, 'VFID_Clips'))):
            return None
        gt_feat, pred_feat = np.load(gt_p), np.load(pred_p)
        if gt_feat.shape != pred_feat.shape or not gt_feat.size:
            return None
        return row, gt_feat, pred_feat
    except (OSError, ValueError, KeyError, TypeError):
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred-root', type=Path, required=True,
                   help='Generated PNGs: <root>/Class_*/<seq>/<frame>.png (hierarchical) or <root>/<seq>/<frame>.png (flat).')
    p.add_argument('--gt-root', type=Path, required=True,
                   help='GT PNGs: <root>/GT/Class_*/<seq> or <root>/Class_*/<seq> (hierarchical); '
                        '<root>/<seq>/gt or <root>/<seq> (flat).')
    p.add_argument('--layout', choices=('auto', 'hierarchical', 'flat'), default='auto')
    p.add_argument('--output-dir', type=Path, help='Default: <pred-root>/metrics_full_reference')
    p.add_argument('--sequences', nargs='+', help='Evaluate only these sequence names.')
    p.add_argument('--allow-partial', action='store_true',
                   help='Allow sequences whose predictions cover only part of the GT frames.')
    p.add_argument('--require-native-size', action='store_true',
                   help='Fail instead of resizing GT when sizes differ.')
    p.add_argument('--skip-warping', action='store_true', help='Omit WE and WE_L2.')
    p.add_argument('--resume', action='store_true',
                   help='Reuse per-sequence results already in --output-dir.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--weights-dir', type=Path, default=HERE / 'weights')
    return p.parse_args()


def main():
    args = parse_args()
    args.pred_root, args.gt_root = args.pred_root.resolve(), args.gt_root.resolve()
    out = (args.output_dir or args.pred_root / 'metrics_full_reference').resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f'{out} exists; pass --resume or choose another --output-dir')
    layout, records = discover(args)
    for r in records:
        print(f'{r["name"]}: {len(r["ids"])}/{len(r["references"])} frames; '
              f'complete={r["complete"]}; gt={r["gt_dir"]}', flush=True)

    # Offline AlexNet backbone for LPIPS if bundled (<TORCH_HOME>/hub/checkpoints layout).
    torch_home = args.weights_dir / 'torch_home'
    if any((torch_home / 'hub' / 'checkpoints').glob('alexnet-*.pth')):
        os.environ['TORCH_HOME'] = str(torch_home)
    else:
        print('Bundled AlexNet not found; LPIPS will use/download the default torch hub cache.', flush=True)
    import lpips
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

    cv2.setNumThreads(1)
    device = torch.device(args.device)
    perceptual = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(args.weights_dir / 'rgb_imagenet.pt', map_location='cpu'))
    i3d.to(device).eval()
    models = (perceptual, i3d, ssim_fn)

    out.mkdir(parents=True, exist_ok=True)
    score_names = (('PSNR', 'SSIM', 'LPIPS', 'FS') if args.skip_warping
                   else ('PSNR', 'SSIM', 'LPIPS', 'WE', 'WE_L2', 'FS'))
    rows, all_gt, all_pred = [], [], []
    with torch.inference_mode():
        for rec in records:
            cached = load_cached(out, rec, score_names) if args.resume else None
            if cached:
                print(f'Reused completed metrics: {rec["name"]}', flush=True)
                row, gt_feat, pred_feat = cached
            else:
                row, gt_feat, pred_feat = evaluate_sequence(rec, args, models, device, score_names)
                np.save(out / f'{rec["name"]}_gt_i3d.npy', gt_feat)
                np.save(out / f'{rec["name"]}_final_i3d.npy', pred_feat)
                (out / f'{rec["name"]}.json').write_text(json.dumps(row, indent=2))
                print(json.dumps(row), flush=True)
            rows.append(row)
            all_gt.append(gt_feat)
            all_pred.append(pred_feat)

    aggregate = {'sequence': 'ALL', 'frames': sum(r['frames'] for r in rows),
                 'pairs': sum(r['pairs'] for r in rows)}
    for k in score_names:
        weight = 'pairs' if k in TEMPORAL_KEYS else 'frames'
        aggregate[k] = float(np.average([r[k] for r in rows], weights=[r[weight] for r in rows]))
    aggregate['VFID_Clips'] = frechet_distance_low_rank(np.concatenate(all_gt), np.concatenate(all_pred))
    rows.append(aggregate)

    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    definitions = {k: v for k, v in DEFINITIONS.items()
                   if not (args.skip_warping and k in ('WE', 'WE_L2'))}
    (out / 'metrics.json').write_text(json.dumps({'definitions': definitions, 'results': rows}, indent=2))
    (out / 'source.json').write_text(json.dumps({
        'evaluator': f'full_reference_metrics/evaluate.py v{VERSION}',
        'created': datetime.now().isoformat(timespec='seconds'),
        'pred_root': str(args.pred_root), 'gt_root': str(args.gt_root), 'layout': layout,
        'device': str(device),
        'gt_preprocessing': ('native size required' if args.require_native_size else
                             'PIL RGB bilinear resize to prediction resolution when sizes differ'),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'opencv': cv2.__version__, 'numpy': np.__version__},
        'sequences': [{'sequence': r['name'], 'gt_dir': str(r['gt_dir']),
                       'evaluated_frames': len(r['ids']), 'expected_frames': len(r['references']),
                       'complete_sequence': r['complete'], 'frame_range': [r['ids'][0], r['ids'][-1]],
                       'missing_frame_ids': sorted(set(r['references']) - set(r['ids']))}
                      for r in records],
        'aggregate_scope': 'ALL aggregates available frames, including any incomplete sequences',
    }, indent=2) + '\n')
    print(f'Metrics saved: {out}', flush=True)


if __name__ == '__main__':
    main()
