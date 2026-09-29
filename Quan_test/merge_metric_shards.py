#!/usr/bin/env python3
"""Merge metric shards and produce a report with WE-L2 in units of 10^-2."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
from compute_vfid_clips_4datasets import frechet_distance_low_rank


def merge(root):
    shards = sorted(root.glob('metrics_full_reference_shard*'))
    output = root / 'metrics_full_reference'
    if output.exists():
        raise FileExistsError(output)
    if not shards:
        raise ValueError(f'No metric shards: {root}')
    rows, sources = [], []
    definitions = None
    for shard in shards:
        payload = json.loads((shard / 'metrics.json').read_text())
        if definitions is not None and definitions != payload['definitions']:
            raise ValueError('Metric definition mismatch')
        definitions = payload['definitions']
        rows.extend(r for r in payload['results'] if r['sequence'] != 'ALL')
        sources.extend(json.loads((shard / 'source.json').read_text())['sequences'])
    rows.sort(key=lambda r: r['sequence'])
    if len({r['sequence'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sequences')
    scores = ('PSNR', 'SSIM', 'LPIPS', 'WE', 'WE_L2', 'FS')
    assert all(math.isfinite(r[k]) for r in rows for k in (*scores, 'VFID_Clips'))
    aggregate = {'sequence': 'ALL', 'frames': sum(r['frames'] for r in rows),
                 'pairs': sum(r['pairs'] for r in rows)}
    for key in scores:
        weight = 'pairs' if key in ('WE', 'WE_L2', 'FS') else 'frames'
        aggregate[key] = float(np.average([r[key] for r in rows], weights=[r[weight] for r in rows]))
    features = {}
    for kind in ('gt', 'final'):
        features[kind] = np.concatenate([np.load(path) for shard in shards
                                        for path in sorted(shard.glob(f'*_{kind}_i3d.npy'))])
    aggregate['VFID_Clips'] = frechet_distance_low_rank(features['gt'], features['final'])
    temporary = Path(tempfile.mkdtemp(prefix='.merged_metrics_', dir=str(root)))
    try:
        for shard in shards:
            for path in shard.iterdir():
                if path.name in {'metrics.json', 'metrics.csv', 'source.json'}:
                    continue
                if (temporary / path.name).exists():
                    raise ValueError(f'Duplicate artifact: {path.name}')
                shutil.copy2(path, temporary / path.name)
        with (temporary / 'metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows([*rows, aggregate])
        (temporary / 'metrics.json').write_text(json.dumps({'definitions': definitions,
                                                          'results': [*rows, aggregate]}, indent=2))
        (temporary / 'source.json').write_text(json.dumps({'eval_root': str(root),
            'input_shards': [str(p) for p in shards], 'sequences': sources,
            'gt_preprocessing': 'PIL RGB bilinear resize to generated resolution',
            'source_pngs_preserved': True}, indent=2))
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f'Merged {len(rows)} sequences: {output}')
    source_by_name = {r['sequence']: r for r in sources}
    return [{**r, 'split': root.name, 'source': source_by_name[r['sequence']]} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eval_roots', nargs='+', type=Path)
    parser.add_argument('--report-root', type=Path, required=True)
    args = parser.parse_args()
    rows = [row for root in args.eval_roots for row in merge(root.resolve())]
    report = args.report_root.resolve()
    report.mkdir(exist_ok=True, parents=True)
    headers = ['Split', 'Sequence', 'Frames', 'Expected frames', 'Complete',
               'PSNR', 'SSIM', 'LPIPS', 'WE', 'WE-L2 (×10⁻²)', 'FS', 'VFID']
    with (report / 'metrics_report.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for r in rows:
            source = r['source']
            writer.writerow([r['split'], r['sequence'], r['frames'], source['expected_frames'],
                source['complete_sequence'], r['PSNR'], r['SSIM'], r['LPIPS'], r['WE'],
                r['WE_L2'] * 100, r['FS'], r['VFID_Clips']])
    lines = ['| Sequence | Frames | PSNR ↑ | SSIM ↑ | LPIPS ↓ | WE ↓ | WE-L2 (×10⁻²) ↓ | FS ↑ | VFID ↓ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['sequence']} | {r['frames']} | {r['PSNR']:.4f} | {r['SSIM']:.5f} | {r['LPIPS']:.5f} | {r['WE']:.6f} | {r['WE_L2']*100:.5f} | {r['FS']:.5f} | {r['VFID_Clips']:.4f} |")
    lines += ['', 'WE-L2 actual value = displayed value × 10⁻².']
    (report / 'metrics_report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
