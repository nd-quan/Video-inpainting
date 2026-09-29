#!/usr/bin/env python
"""Compare original RAFT and distilled V7 RAFT on identical degraded pairs."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import visualize_raft_student_flow as vis


def montage(tiles, gap=12):
    size = 320
    rows = (len(tiles) + 3) // 4
    canvas = np.full((rows * size + (rows + 1) * gap, 4 * size + 5 * gap, 3), 235, np.uint8)
    for index, (title, tile) in enumerate(tiles):
        row, col = divmod(index, 4)
        y, x = gap + row * (size + gap), gap + col * (size + gap)
        canvas[y:y+size, x:x+size] = vis.label(cv2.resize(tile, (size, size)), title)
    return canvas


def rgb_warp(source, target, source_bg, target_bg, flow):
    """Target-to-source sampling; BG plus geometric support, without FB filtering."""
    source = ((source.detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
    target = ((target.detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
    flow = flow.detach().cpu().numpy().transpose(1, 2, 0)
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x, y = x + flow[..., 0], y + flow[..., 1]
    warped = cv2.remap(source, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bg = cv2.remap(source_bg.detach().cpu().numpy().squeeze(), x, y,
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    support = ((target_bg.detach().cpu().numpy().squeeze() > .5) & (bg > .5)
               & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1))
    return target, warped, np.abs(target - warped).mean(axis=2), support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset_root', type=Path, required=True)
    parser.add_argument('--teacher_flow_root', type=Path, required=True)
    parser.add_argument('--pair_csv', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--split', default='valid', choices=('train', 'valid', 'test'))
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    _, component = vis.resolve_checkpoint(args.checkpoint)
    config = json.loads((component / 'config.json').read_text())
    if config.get('architecture') != 'propainter_raft_large':
        raise ValueError('This before/after comparison requires the original V7 ProPainter RAFT student')
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Use a new output directory: {root}')
    (root / 'montages').mkdir(parents=True)
    (root / 'warp_diagnostics').mkdir()
    device = torch.device(args.device)
    # Same architecture, preprocessing, iteration count, and inference precision.
    baseline = vis.RAFTStudentFlowPredictor(
        config['propainter_root'], config['source_checkpoint'],
        iterations=config['iterations'], mixed_precision=False,
        freeze_batchnorm=config.get('freeze_batchnorm', True),
    ).to(device).eval().requires_grad_(False)
    student = vis.RAFTStudentFlowPredictor.from_pretrained(
        component, mixed_precision=False,
    ).to(device).eval().requires_grad_(False)
    dataset = vis.RAFTTeacherFlowPairDataset(
        args.dataset_root, args.teacher_flow_root, args.split, args.resolution, None,
    )
    with args.pair_csv.open() as handle:
        selected = list(csv.DictReader(handle))
    # Match semantic IDs, not old dataset indices.
    requested = {(r['sequence'], int(r['frame_t']), int(r['frame_t1'])) for r in selected}
    indices = []
    for index in range(len(dataset)):
        entry = dataset.entries[index]
        if (str(entry[0]), int(entry[1]), int(entry[2])) in requested:
            indices.append(index)
    found, rows = set(), []
    for index in indices:
        sample = dataset[index]
        key = (str(sample['video']), int(sample['frame0']), int(sample['frame1']))
        if key not in requested:
            continue
        found.add(key)
        a = sample['degraded0'].unsqueeze(0).to(device)
        b = sample['degraded1'].unsqueeze(0).to(device)
        with torch.inference_mode():
            before = baseline.predict_bidirectional(a, b, pair_batch_size=1)
            after = student.predict_bidirectional(a, b, pair_batch_size=1)
        predictions = [[f[0].float() for f in result] for result in (before, after)]
        teachers, valids, masks = [], [], []
        for direction, bg_key in [('forward', 'bg0'), ('backward', 'bg1')]:
            flow, valid = vis.prepare_teacher_flow(
                sample['teacher_' + direction][None, None].to(device),
                a.shape[-2:], sample['valid_' + direction][None, None].to(device),
            )
            teachers.append(flow[0, 0])
            valids.append(valid[0, 0])
            masks.append(sample[bg_key].to(device))
        flow_scale = vis.percentile_scale(
            [f.norm(dim=0) for f in teachers + predictions[0] + predictions[1]], 99,
        )
        errors = [[(predictions[m][d] - teachers[d]).norm(dim=0, keepdim=True)
                   for d in range(2)] for m in range(2)]
        error_scale = vis.percentile_scale(errors[0] + errors[1], 99)
        row = dict(sequence=key[0], frame_t=key[1], frame_t1=key[2],
                   flow_scale_px=flow_scale, epe_scale_px=error_scale)
        tiles = [
            ('Degraded t (same input)', vis.rgb_to_bgr(a[0])),
            ('Degraded t+1 (same input)', vis.rgb_to_bgr(b[0])),
            ('BG t (white=BG)', vis.mask_to_bgr(masks[0])),
            ('BG t+1 (white=BG)', vis.mask_to_bgr(masks[1])),
        ]
        for d, direction in enumerate(('forward', 'backward')):
            for m, name in enumerate(('pretrained', 'student')):
                stats = vis.direction_metrics(predictions[m][d], teachers[d], valids[d], masks[d])
                row.update({f'{direction}_{name}_{k}': v for k, v in stats.items()})
            base_epe = row[f'{direction}_pretrained_bg_epe']
            student_epe = row[f'{direction}_student_bg_epe']
            row[f'{direction}_student_gain_vs_pretrained_bg'] = (
                (base_epe - student_epe) / max(base_epe, 1e-8)
            )
            tiles.extend([
                (f'Clean teacher {direction}', vis.flow_to_bgr(teachers[d], flow_scale)),
                (f'Before: pretrained RAFT {direction}', vis.flow_to_bgr(predictions[0][d], flow_scale)),
                (f'After: V7 student {direction}', vis.flow_to_bgr(predictions[1][d], flow_scale)),
                (f'{direction} valid BG support', vis.mask_to_bgr(valids[d] * masks[d])),
            ])
        tiles.extend([
            ('Pretrained forward EPE (px)', vis.heatmap(errors[0][0], error_scale)),
            ('Student forward EPE (px)', vis.heatmap(errors[1][0], error_scale)),
            ('Pretrained backward EPE (px)', vis.heatmap(errors[0][1], error_scale)),
            ('Student backward EPE (px)', vis.heatmap(errors[1][1], error_scale)),
        ])
        name = f'{vis.sanitize(key[0])}_f{key[1]:06d}_{key[2]:06d}.png'
        # Backward flow aligns t to t+1; forward aligns t+1 to t.
        # Compare both estimators on the intersection of their BG/in-bounds support.
        warp_tiles = []
        for d, direction in enumerate(('forward', 'backward')):
            source, target = (b[0], a[0]) if d == 0 else (a[0], b[0])
            source_bg, target_bg = (masks[1], masks[0]) if d == 0 else (masks[0], masks[1])
            results = [rgb_warp(source, target, source_bg, target_bg, predictions[m][d])
                       for m in range(2)]
            common = results[0][3] & results[1][3]
            scale = max(float(np.percentile(np.concatenate(
                [r[2][common] for r in results]), 99)), 1e-8) if common.any() else 1.0
            row[f'{direction}_warp_common_bg_ratio'] = float(common.mean())
            row[f'{direction}_warp_error_scale_rgb01'] = scale
            for m, model_name in enumerate(('pretrained', 'student')):
                target_np, warped, error, _ = results[m]
                mae = float(error[common].mean()) if common.any() else float('nan')
                row[f'{direction}_{model_name}_warp_rgb_mae'] = mae
                composite = np.where(common[..., None], warped, target_np)
                image = cv2.cvtColor((composite * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                heat = vis.heatmap(torch.from_numpy(error[None]), scale)
                heat[~common] = 235
                warp_tiles.extend([(f'{model_name} {direction} BG warp', image),
                                   (f'{model_name} warp MAE={mae:.5f} (RGB 0-1)', heat)])
        if not cv2.imwrite(str(root / 'warp_diagnostics' / name), montage(warp_tiles)):
            raise OSError(name)
        if not cv2.imwrite(str(root / 'montages' / name), montage(tiles)):
            raise OSError(name)
        rows.append(row)
        vis.write_csv(root / 'per_pair_metrics.csv', rows)
        print(f'[{len(rows)}/{len(requested)}] {name}', flush=True)
    if found != requested:
        raise ValueError(f'Missing requested pairs: {sorted(requested - found)}')
    numeric = [k for k in rows[0] if k not in ('sequence', 'frame_t', 'frame_t1')]
    seq_rows = [dict(sequence=seq, **{k: vis.finite_average(
        [r for r in rows if r['sequence'] == seq], k) for k in numeric})
        for seq in sorted({r['sequence'] for r in rows})]
    vis.write_csv(root / 'per_sequence_metrics.csv', seq_rows)
    vis.json_dump(root / 'run_config.json', vars(args))
    vis.json_dump(root / 'summary.json', dict(
        pair_count=len(rows), baseline_checkpoint=config['source_checkpoint'],
        student_component=str(component), iterations=config['iterations'],
        precision='float32; no external autocast',
        input_policy='Both models consume identical degraded RGB; teacher comes from cached clean RGB flow',
        support_policy='Same cached teacher-valid mask and directional BG mask for both models',
        color_policy='All six flows share per-pair p99 scale; all four EPE panels share per-pair p99 scale; gaps=12px',
        epe_interpretation='Agreement with pseudo-GT clean teacher, not true physical ground-truth flow',
        warp_error_interpretation='Degraded RGB photometric MAE in [0,1], not flow EPE or V8 feature error',
        warp_support_policy='Common pretrained/student target BG, sampled source BG >0.5 and in-bounds; no FB filtering; outside support retains target RGB',
        mean_per_pair={k: vis.finite_average(rows, k) for k in numeric},
        sequence_macro_average={k: vis.finite_average(seq_rows, k) for k in numeric},
    ))


if __name__ == '__main__':
    main()
