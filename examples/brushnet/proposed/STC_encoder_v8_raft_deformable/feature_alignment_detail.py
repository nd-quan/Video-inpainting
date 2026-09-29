"""Shared-scale channel, change and correspondence views for one feature pair."""
import cv2
import numpy as np
import torch
import visualize_v5_v6_v8_feature_alignment as vis


def render_detail(case, tile_size, tile_gap=0):
    keys = ('source', 'base_candidate', 'candidate', 'target')
    titles = ('Before', 'Student warp', 'Student + DCN', 'Target')
    support = case['metric_support'].bool()
    # Select informative channels using only the unaligned source and target,
    # never by how much the alignment appears to improve a channel.
    variance = (case['source'].flatten(1).var(1) + case['target'].flatten(1).var(1)) / 2
    channels = variance.argsort(descending=True)[:4].tolist()
    tiles, channel_scales = [], {}
    for channel in channels:
        values = torch.cat([case[k][channel].flatten() for k in keys])
        low, high = torch.quantile(values, values.new_tensor([.01, .99])).tolist()
        channel_scales[str(channel)] = [low, high]
        for key, title in zip(keys, titles):
            value = (case[key][channel:channel+1] - low) / max(high-low, 1e-8)
            tiles.append((f'Ch{channel} {title}', vis.scalar_to_bgr(value, 1)))
    channel_image = vis.montage(tiles, tile_size, columns=4, gap=tile_gap)
    errors = [vis.cosine_distance(case[k], case['target']) for k in keys[:3]]
    def scale(values, mask=None):
        selected = [v[mask] if mask is not None else v.flatten() for v in values]
        selected = [v for v in selected if v.numel()]
        return vis.percentile_scale(selected) if selected else 1.0
    error_scale = scale(errors, support)
    def masked(value, maximum):
        result = vis.scalar_to_bgr(value, maximum)
        result[~support[0].cpu().numpy()] = (70, 70, 70)
        return result
    def legend(maximum, signed=False):
        ramp = torch.linspace(-maximum if signed else 0, maximum, tile_size)[None, None, :].expand(1, tile_size, tile_size)
        return diverging(ramp, maximum) if signed else vis.scalar_to_bgr(ramp, maximum)
    tiles = [(f'{t} cosine error', masked(v, error_scale)) for t, v in zip(titles, errors)]
    tiles.append((f'Error 0 -> {error_scale:.3g}', legend(error_scale)))
    gains = [errors[0] - errors[1], errors[0] - errors[2]]
    gain_scale = scale([v.abs() for v in gains], support)
    for title, gain in zip(('Warp gain vs before', 'DCN gain vs before'), gains):
        image = diverging(gain, gain_scale)
        image[~support[0].cpu().numpy()] = (70, 70, 70)
        tiles.append((title, image))
    tiles.extend([(f'Red worse / blue better +/-{gain_scale:.2g}', legend(gain_scale, True)),
                  ('Evaluated support (white)', vis.mask_to_bgr(support.float()))])
    changes = [vis.mean_abs_channel(case[a], case[b]) for a,b in
               [('base_candidate','source'), ('candidate','base_candidate'),
                ('aligned_target','target'), ('temporal_target','aligned_target')]]
    change_scale = scale(changes)
    for title, value in zip(('Change: warp - source', 'Change: DCN - warp',
                             'Change: fusion - target', 'Change: temporal - fusion'), changes):
        tiles.append((title, vis.scalar_to_bgr(value, change_scale)))
    tiles.append((f'Change 0 -> {change_scale:.3g}', legend(change_scale)))
    metadata = dict(channels=channels, channel_percentile_1_99_ranges=channel_scales,
                    cosine_error_p99=error_scale, cosine_gain_abs_p99=gain_scale,
                    feature_change_p99=change_scale, support_pixels=int(support.sum()),
                    policy='Scales shared within each row/group and pair. Channel selection uses source/target spatial variance only. Percentile clipping enhances contrast; scales differ between pairs. Gray excludes unsupported pixels. Change maps are full-field and show magnitude, not quality. Positive cosine gain is blue; negative is red.')
    return channel_image, vis.montage(tiles, tile_size, columns=4, gap=tile_gap), metadata


def diverging(value, maximum):
    """Negative=red, zero=white, positive=blue (BGR for OpenCV)."""
    x = (value.detach().float().cpu().squeeze().numpy() / max(maximum, 1e-8)).clip(-1, 1)
    image = np.ones((*x.shape, 3), dtype=np.float32)
    image[..., 0] -= np.maximum(-x, 0)
    image[..., 1] -= np.abs(x)
    image[..., 2] -= np.maximum(x, 0)
    return (image * 255).astype(np.uint8)
