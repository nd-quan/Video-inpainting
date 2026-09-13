#!/usr/bin/env python
"""Full-frame metrics on final PNGs, using Quan_test metric definitions."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Quan_test'))
from metric import psnr_np, to_torch_01, ssim_fn
from evaluate_temporal_metrics import warping_error, frame_ssim
from compute_vfid_clips_4datasets import frechet_distance_low_rank
import lpips
import cv2


def main():
    cv2.setNumThreads(1)
    p = argparse.ArgumentParser()
    p.add_argument('--eval_root', type=Path, required=True)
    p.add_argument('--evaluator_root', type=Path, default=ROOT.parent / 'video-inpainting-evaluation')
    args = p.parse_args()
    sys.path.insert(0, str(args.evaluator_root))
    from src.fid.util import extract_video_clip_features
    from src.models.i3d.pytorch_i3d import InceptionI3d
    out = args.eval_root / 'metrics'
    out.mkdir(exist_ok=True)
    manifests = sorted(args.eval_root.glob('*/clip_metrics.json'))
    expected = json.loads((args.eval_root / 'summary.json').read_text())
    if not manifests:
        raise ValueError('No completed sequences')
    perceptual = lpips.LPIPS(net='alex').cuda().eval()
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(args.evaluator_root / 'pretrained_models/rgb_imagenet.pt', map_location='cpu'))
    i3d.cuda().eval()
    rows, all_gt, all_pred = [], [], []
    with torch.inference_mode():
        for manifest in manifests:
            meta = json.loads(manifest.read_text())
            ids = meta['frame_ids']
            pred_paths = [manifest.parent / 'final' / f'{i:06d}.png' for i in ids]
            gt_paths = [manifest.parent / 'gt' / f'{i:06d}.png' for i in ids]
            scores = {k: [] for k in ('PSNR', 'SSIM', 'LPIPS', 'WE', 'WE_L2', 'FS')}
            prev = None
            for pred_path, gt_path in zip(pred_paths, gt_paths):
                with Image.open(pred_path) as im: pred = np.array(im.convert('RGB'))
                with Image.open(gt_path) as im: gt = np.array(im.convert('RGB'))
                if pred.shape != gt.shape: raise ValueError('Image shape mismatch')
                a, b = to_torch_01(pred).cuda(), to_torch_01(gt).cuda()
                scores['PSNR'].append(psnr_np(gt, pred))
                scores['SSIM'].append(ssim_fn(a, b, data_range=1.0).item())
                scores['LPIPS'].append(perceptual(a * 2 - 1, b * 2 - 1).item())
                cur = pred.astype(np.float32) / 255
                if prev is not None:
                    l1, l2 = warping_error(prev, cur)
                    scores['WE'].append(l1)
                    scores['WE_L2'].append(l2)
                    scores['FS'].append(frame_ssim(prev, cur))
                prev = cur
            features = []
            for kind, paths in [('gt', gt_paths), ('final', pred_paths)]:
                def get_frame(_name, index):
                    with Image.open(paths[index]) as im:
                        return im.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
                f = extract_video_clip_features(i3d, get_frame, [meta['video']], [len(ids)])
                np.save(out / f'{manifest.parent.name}_{kind}_i3d.npy', f)
                features.append(f)
            all_gt.append(features[0]); all_pred.append(features[1])
            row = {'sequence': meta['video'], 'frames': len(ids), 'pairs': len(ids)-1,
                   **{k: float(np.mean(v)) for k, v in scores.items()},
                   'VFID_Clips': frechet_distance_low_rank(*features)}
            rows.append(row)
            (out / f'{manifest.parent.name}.json').write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
    aggregate = {'sequence': 'ALL', 'frames': sum(r['frames'] for r in rows), 'pairs': sum(r['pairs'] for r in rows)}
    for k in scores:
        weight = 'pairs' if k in ('WE', 'WE_L2', 'FS') else 'frames'
        aggregate[k] = float(np.average([r[k] for r in rows], weights=[r[weight] for r in rows]))
    aggregate['VFID_Clips'] = frechet_distance_low_rank(np.concatenate(all_gt), np.concatenate(all_pred))
    rows.append(aggregate)
    with (out / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (out / 'metrics.json').write_text(json.dumps({'definitions': {
        'scope': 'full-frame final vs GT at evaluator resolution, before video encoding',
        'PSNR': 'RGB uint8; frame mean; higher is better', 'SSIM': 'torchmetrics SSIM data_range=1; higher is better',
        'LPIPS': 'AlexNet; lower is better', 'WE': 'backward Farneback warped adjacent final frames, valid-pixel RGB L1 in [0,1]; lower is better',
        'FS': 'adjacent final-frame SSIM from Quan_test/evaluate_temporal_metrics.py; higher is better',
        'VFID_Clips': 'I3D rgb_imagenet, 10-frame sliding windows, stride 1, 224x224 bicubic, empirical Frechet; lower is better',
        'aggregation': 'frame-weighted image metrics; pair-weighted temporal metrics; pooled VFID features'}, 'results': rows}, indent=2))
    print(f'Metrics complete: {out}', flush=True)

if __name__ == '__main__':
    main()
