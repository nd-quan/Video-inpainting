#!/usr/bin/env python3
"""Render the sequence comparison with highlighted best metric values."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison_csv', type=Path)
    args = parser.parse_args()
    source = args.comparison_csv.resolve()
    with source.open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    methods = tuple(dict.fromkeys(r['Method'] for r in rows))
    labels = {'SN': 'SN', 'SN+TG': 'SN+TG', 'R×4': 'TCE', 'Frame-base': 'Frame-base',
              'V8R RGB-TG0.1': 'V8R RGB-TG0.1'}
    metrics = ('PSNR', 'SSIM', 'LPIPS', 'WE-L2 (×10⁻²)', 'FS', 'VFID')
    sequences = list(dict.fromkeys(r['Sequence'] for r in rows))
    grouped = {seq: {r['Method']: r for r in rows if r['Sequence'] == seq} for seq in sequences}
    assert len(rows) == len(methods) * len(sequences) and len(sequences) == 13
    assert all(set(group) == set(methods) for group in grouped.values())
    excluded = {seq for seq, group in grouped.items()
                if len({r['Frames'] for r in group.values()}) != 1}
    assert excluded == {'PartyScene'}
    best = {}
    for seq, group in grouped.items():
        for metric in metrics:
            values = {method: float(group[method][metric]) for method in methods}
            target = (max if metric in ('PSNR', 'SSIM', 'FS') else min)(values.values())
            best[seq, metric] = {m for m, v in values.items() if math.isclose(v, target, rel_tol=1e-12, abs_tol=1e-12)}

    row_count = len(sequences) * len(methods)
    fig = plt.figure(figsize=(14, 1.0 + 0.36 * row_count), facecolor='white')
    ax = fig.add_axes([0.025, 0.06, 0.95, 0.92])
    boundaries = (0, 2.85, 4.15, 5.80, 7.40, 9.00, 10.90, 12.45, 14.00)
    centers = [(a + b) / 2 for a, b in zip(boundaries, boundaries[1:])]
    ax.set_xlim(0, 14)
    ax.set_ylim(row_count + 0.7, -1.5)
    ax.axis('off')
    headers = ('Sequence', 'Method', 'PSNR ↑', 'SSIM ↑', 'LPIPS ↓',
               r'WE-L2 ($\times10^{-2}$) ↓', 'FS ↑', 'VFID ↓')
    for x, label in zip(centers, headers):
        ax.text(x, -0.7, label, ha='center', va='center', fontsize=13)
    ax.hlines(0.05, 0, 14, color='black', linewidth=1.25)
    lines = ['| Sequence | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | WE-L2 (×10⁻²) ↓ | FS ↑ | VFID ↓ |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    winner_records = []
    for group_index, seq in enumerate(sequences):
        base_y = group_index * len(methods) + 0.6
        ax.text(centers[0], base_y + (len(methods) - 1) / 2, seq,
                ha='center', va='center', fontsize=12.5)
        for method_index, method in enumerate(methods):
            y = base_y + method_index
            row = grouped[seq][method]
            method_label = labels[method]
            ax.text(centers[1], y, method_label, ha='center', va='center', fontsize=12.5)
            rendered = []
            for column, metric in enumerate(metrics, start=2):
                value = float(row[metric])
                text = f'{value:.4f}' if metric in ('PSNR', 'VFID') else f'{value:.5f}'
                winner = method in best.get((seq, metric), set())
                if winner:
                    ax.add_patch(Rectangle((boundaries[column] + 0.06, y - 0.40),
                        boundaries[column + 1] - boundaries[column] - 0.12, 0.80,
                        facecolor='#ddf2e3', edgecolor='none', zorder=0))
                    winner_records.append({'sequence': seq, 'metric': metric,
                                           'method': labels[method], 'value': value})
                ax.text(centers[column], y, text, ha='center', va='center', fontsize=12.5,
                        fontweight='bold' if winner else 'normal',
                        color='#146b35' if winner else 'black')
                rendered.append(f'**{text}**' if winner else text)
            lines.append('| ' + ' | '.join([seq if method_index == 0 else '', method_label, *rendered]) + ' |')
        ax.hlines(base_y + len(methods) - 0.5, 0, 14, color='black', linewidth=1.1)
    footnotes = ['Green bold values: best score within each sequence; ↑ higher is better, ↓ lower is better.',
                 'Frame-base: independent-noise frame-wise baseline; SN: shared noise only; '
                 'SN+TG: shared noise + temporal guidance; TCE: Rescaled ×4; '
                 'V8R RGB-TG0.1: standalone V8R with RGB temporal guidance 0.1.',
                 'PartyScene SN+TG has only 32/250 frames, so that row is not directly comparable.',
                 'Actual WE-L2 = displayed value × 10⁻².']
    fig.text(0.03, 0.042, '\n'.join(footnotes), fontsize=10.5, va='top', linespacing=1.4)
    output = source.parent / 'comparison_metrics_highlighted'
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output.with_suffix('.' + extension), dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    lines += ['', *footnotes]
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    output.with_suffix('.json').write_text(json.dumps({'winners': winner_records,
        'unequal_frame_sequences': sorted(excluded)}, indent=2), encoding='utf-8')
    print(f'Highlighted {len(winner_records)} best values across 13 sequences.')
    print(f'Saved PNG/PDF/SVG/Markdown: {output}')


if __name__ == '__main__':
    main()
