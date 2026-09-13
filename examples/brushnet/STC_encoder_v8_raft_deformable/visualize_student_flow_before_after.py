#!/usr/bin/env python
"""Visualize V8 features before/after student-flow alignment.

Uses the same arguments, checkpoints and pair selection as the three-model
visualizer; only the V8 and RAFT-student checkpoints are loaded.
"""
from pathlib import Path
import json
import cv2
import torch
import torch.nn.functional as F
import visualize_v5_v6_v8_feature_alignment as vis
from feature_alignment_detail import render_detail


def render(sample, pair, case, tile_size):
    keys = ('source', 'base_candidate', 'candidate', 'target',
            'aligned_target', 'temporal_target')
    pca = dict(zip(keys, vis.pca_feature_bgr_maps([case[k] for k in keys])))
    errors = {k: vis.cosine_distance(case[k], case['target'])
              for k in keys[:3]}
    tiles = [
        ('Input t', vis.rgb_to_bgr(sample['conditioning_pixel_values'][pair])),
        ('Input t+1', vis.rgb_to_bgr(sample['conditioning_pixel_values'][pair+1])),
        ('Student flow t+1 -> t', vis.flow_to_bgr(case['raft_flow_backward_rgb'],
             vis.percentile_scale([case['raft_flow_backward_rgb'].square().sum(0).sqrt()]))),
        ('Common BG support', vis.mask_to_bgr(case['metric_support'])),
        ('Before: source t PCA', pca['source']),
        ('After: student warp PCA', pca['base_candidate']),
        ('After: student + DCN PCA', pca['candidate']),
        ('Reference: target t+1 PCA', pca['target']),
    ]
    for k, title in zip(keys[:3], ('Before cosine error', 'Student warp cosine error', 'Student + DCN cosine error')):
        error = vis.scalar_to_bgr(errors[k], 2.0)
        error[case['metric_support'][0].cpu().numpy() < .5] = 0
        tiles.append((title, error))
    ramp = torch.linspace(0, 2, 256)[None, None, :].expand(1, 256, 256)
    tiles.extend([
        ('Error scale: 0 left -> 2 right', vis.scalar_to_bgr(ramp, 2.0)),
        ('Before fusion: target PCA', pca['target']),
        ('After alignment fusion PCA', pca['aligned_target']),
        ('After temporal encoder PCA', pca['temporal_target']),
        ('Target restore mask', vis.mask_to_bgr(sample['masks'][pair+1])),
    ])
    return vis.montage(tiles, tile_size, columns=4), errors


def main():
    args = vis.parse_args()
    if args.resolution < 8 or args.resolution % 8 or args.clip_length < 2 or args.clip_stride < 1:
        raise ValueError('Invalid resolution or clip geometry')
    if args.tile_size < 64 or args.raft_pair_batch_size < 1 or args.deformable_alignment_scale < 0:
        raise ValueError('Invalid tile size, RAFT batch size or alignment scale')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the RAFT student')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    root = args.output_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Use a new output directory: {root}')
    (root / 'montages').mkdir(parents=True, exist_ok=True)
    for subdir in ('channels', 'diagnostics', 'display_scales'):
        (root / subdir).mkdir(exist_ok=True)
    selected = vis.read_selected_pairs(args)
    dataset = vis.build_dataset(args, selected)
    vis.append_exact_pair_windows(dataset, selected)
    pairs = vis.locate_clip_pairs(dataset, selected, args.use_cross_clip)
    component = vis.resolve_component(args.v8_checkpoint, 'stc_v8_model')
    model = vis.RAFTGuidedDeformableBGSTCAdapter.from_pretrained(str(component)).to(device).eval()
    model.requires_grad_(False)
    provider = vis.FrozenV7RAFTFlowProvider(vis.resolve_raft_student_component(args.raft_student_path),
        device=device, pair_batch_size=args.raft_pair_batch_size, mixed_precision=not args.no_amp)
    vis.json_dump(root / 'run_config.json', vars(args))
    rows = []
    with torch.inference_mode():
        for ordinal, item in enumerate(pairs, 1):
            sample = dataset[item.dataset_index]
            p = item.local_pair_index
            assert tuple(map(int, sample['frame_ids'][p:p+2])) == (item.selected.frame_t, item.selected.frame_t1)
            def batch(key):
                return sample[key].unsqueeze(0).to(device)
            with vis.autocast_context(device, args.no_amp):
                output = vis.v8_forward(model, provider,
                    batch('conditioning_pixel_values'), batch('masks'), batch('frame_ids'),
                    previous_rgb=batch('previous_conditioning_pixel_values'), previous_bg=batch('previous_masks'),
                    previous_ids=batch('previous_frame_ids'), previous_valid=batch('previous_valid_mask'),
                    use_cross_clip=args.use_cross_clip, deformable_alignment_scale=args.deformable_alignment_scale)
            masks = F.interpolate(sample['masks'].float().to(device), size=output.spatial_features.shape[-2:], mode='nearest')
            case = vis.prepare_case('v8', output, p, masks,
                deform_groups=model.config.deform_groups, deform_kernel_size=model.config.deform_kernel_size)
            # Same pixels for all three comparisons, also requiring unwarped
            # source BG. Invalid warp fallback pixels never count as successes.
            case['metric_support'] = case['support'] * (masks[p] >= .5).float() * case['base_valid']
            canvas, errors = render(sample, p, case, args.tile_size)
            name = f'{vis.sanitize(item.selected.sequence)}_f{item.selected.frame_t:06d}_{item.selected.frame_t1:06d}.png'
            path = root / 'montages' / name
            if not cv2.imwrite(str(path), canvas):
                raise OSError(path)
            channel_image, diagnostic_image, display_scales = render_detail(case, args.tile_size)
            for subdir, image in [('channels', channel_image), ('diagnostics', diagnostic_image)]:
                if not cv2.imwrite(str(root / subdir / name), image):
                    raise OSError(root / subdir / name)
            vis.json_dump(root / 'display_scales' / Path(name).with_suffix('.json'), display_scales)
            support = case['metric_support']
            row = dict(sequence=item.selected.sequence, frame_t=item.selected.frame_t, frame_t1=item.selected.frame_t1,
                       support_pixels=int(support.sum()), output_montage=str(path))
            for key, prefix in [('source', 'before'), ('base_candidate', 'student_warp'), ('candidate', 'student_dcn')]:
                row[prefix + '_cosine'] = vis.weighted_mean(errors[key], support) if support.sum() > 0 else float('nan')
                row[prefix + '_l1'] = vis.weighted_mean(vis.mean_abs_channel(case[key], case['target']), support) if support.sum() > 0 else float('nan')
            rows.append(row)
            vis.write_csv(root / 'per_pair_metrics.csv', rows)
            print(f'[{ordinal}/{len(pairs)}] {path}', flush=True)
    summary = dict(pair_count=len(rows), v8_checkpoint=str(component), raft_student=provider.metadata(),
        mean_metrics={k: vis.finite_average(rows, k) for k in rows[0] if k.endswith(('_cosine', '_l1'))},
        interpretation='Lower cosine/L1 means closer to target spatial features; not a ground-truth restoration metric. PCA basis and normalization shared across all six feature maps within each pair. Original PCA montages use fixed [0,2] errors with black excluded support. Additional diagnostics use shared per-pair p99 scales with gray excluded support; signed gain is blue for improvement and red for worsening. Channel panels use shared per-row 1st-99th percentile ranges. Exact ranges are in display_scales/. Fusion and temporal panels show later stages, not pure flow warps.')
    vis.json_dump(root / 'summary.json', summary)
    (root / 'README.md').write_text('# Student-flow feature alignment\n\n'
        'Each montage: inputs/flow/support; unwarped source/student warp/student + DCN/target; '
        'cosine errors and scale; target before fusion/after alignment fusion/after temporal encoding/mask.\n\n'
        'PCA colors share one basis per pair. Errors compare all candidates to the same target on identical valid BG pixels. '
        'Lower error is better correspondence, not proof of improved restored RGB. White masks identify restore regions.\n\n'
        'Additional views: channels/ shows four channels selected by source/target spatial variance with shared 1st-99th percentile scaling per row. diagnostics/ shows shared p99-scaled cosine errors, signed improvement (blue=better, red=worse, gray=excluded), and absolute feature changes across intermediate stages. Changes are not quality scores. display_scales/ records all ranges. Scales differ between pairs.\n\n'
        'Checkpoint and arguments: run_config.json. Pair measurements: per_pair_metrics.csv. Means: summary.json.\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
