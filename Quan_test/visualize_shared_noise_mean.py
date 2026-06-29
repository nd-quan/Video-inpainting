import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-ndquan")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-cache-ndquan")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np


def summarize(name, tensor):
    flat = tensor.reshape(-1).astype(np.float64)
    return {
        "name": name,
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2": float(np.linalg.norm(flat, ord=2)),
    }


def save_histogram(noise, shared_mean, shared_mean_scaled, out_dir):
    plt.figure(figsize=(12, 7))

    for frame_idx in range(noise.shape[0]):
        values = noise[frame_idx].reshape(-1)
        plt.hist(values, bins=120, density=True, alpha=0.18, label=f"z frame {frame_idx}")

    plt.hist(
        shared_mean.reshape(-1),
        bins=120,
        density=True,
        alpha=0.65,
        label="z_shared = mean(z_i)",
    )
    plt.hist(
        shared_mean_scaled.reshape(-1),
        bins=120,
        density=True,
        alpha=0.45,
        label="z_shared_scaled = mean(z_i) * sqrt(N)",
    )

    plt.title("Latent Noise Distribution Before and After Clip Mean")
    plt.xlabel("latent value")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "noise_distribution_before_after_mean.png", dpi=180)
    plt.close()


def save_channel_maps(noise, shared_mean, shared_mean_scaled, out_dir, channel):
    num_frames = noise.shape[0]
    rows = 3
    cols = num_frames
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 8.0))

    vmin = float(noise[:, channel].min())
    vmax = float(noise[:, channel].max())

    for frame_idx in range(num_frames):
        ax = axes[0, frame_idx]
        ax.imshow(noise[frame_idx, channel], cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(f"original z_{frame_idx}")
        ax.axis("off")

        ax = axes[1, frame_idx]
        ax.imshow(shared_mean[0, channel], cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title("mean shared")
        ax.axis("off")

        ax = axes[2, frame_idx]
        ax.imshow(shared_mean_scaled[0, channel], cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title("mean * sqrt(N)")
        ax.axis("off")

    plt.suptitle(f"Spatial Map Comparison, Latent Channel {channel}")
    plt.tight_layout()
    plt.savefig(out_dir / f"noise_spatial_maps_channel_{channel}.png", dpi=180)
    plt.close()


def save_per_frame_stats(noise, shared_mean, shared_mean_scaled, out_dir):
    frame_stds = noise.reshape(noise.shape[0], -1).std(axis=1)
    frame_means = noise.reshape(noise.shape[0], -1).mean(axis=1)

    labels = [f"z{i}" for i in range(noise.shape[0])] + ["mean", "mean*sqrt(N)"]
    means = frame_means.tolist() + [float(shared_mean.mean()), float(shared_mean_scaled.mean())]
    stds = frame_stds.tolist() + [float(shared_mean.std()), float(shared_mean_scaled.std())]

    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(x, means)
    axes[0].set_title("Mean per noise tensor")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, stds)
    axes[1].set_title("Std per noise tensor")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25)
    axes[1].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_dir / "noise_stats_before_after_mean.png", dpi=180)
    plt.close()


def save_residual_histogram(noise, shared_mean, out_dir):
    residual = noise - shared_mean

    plt.figure(figsize=(10, 6))
    for frame_idx in range(noise.shape[0]):
        plt.hist(
            residual[frame_idx].reshape(-1),
            bins=120,
            density=True,
            alpha=0.22,
            label=f"z_{frame_idx} - z_shared",
        )

    plt.title("Residual Distribution: z_i - mean(z_i)")
    plt.xlabel("residual value")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "noise_residual_distribution.png", dpi=180)
    plt.close()


def write_stats(stats, out_dir):
    lines = []
    for item in stats:
        lines.append(
            f"{item['name']}: "
            f"mean={item['mean']:.6f}, std={item['std']:.6f}, "
            f"min={item['min']:.6f}, max={item['max']:.6f}, l2={item['l2']:.6f}"
        )
    (out_dir / "noise_stats.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize how clip-mean shared noise changes latent distribution."
    )
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/noise_mean_visualization"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(
        (args.num_frames, args.channels, args.height, args.width),
        dtype=np.float32,
    )

    shared_mean = noise.mean(axis=0, keepdims=True)
    shared_mean_scaled = shared_mean * math.sqrt(args.num_frames)
    residual = noise - shared_mean

    stats = []
    for frame_idx in range(args.num_frames):
        stats.append(summarize(f"z_frame_{frame_idx}", noise[frame_idx]))
    stats.append(summarize("z_shared_mean", shared_mean))
    stats.append(summarize("z_shared_mean_times_sqrt_N", shared_mean_scaled))
    stats.append(summarize("residual_z_i_minus_mean", residual))
    write_stats(stats, args.out_dir)

    save_histogram(noise, shared_mean, shared_mean_scaled, args.out_dir)
    save_channel_maps(noise, shared_mean, shared_mean_scaled, args.out_dir, args.channel)
    save_per_frame_stats(noise, shared_mean, shared_mean_scaled, args.out_dir)
    save_residual_histogram(noise, shared_mean, args.out_dir)

    print(f"Saved visualization files to: {args.out_dir}")
    print("Key expected observation:")
    print(f"- original z_i std is close to 1")
    print(f"- mean(z_i) std is close to 1/sqrt({args.num_frames}) = {1 / math.sqrt(args.num_frames):.4f}")
    print("- mean(z_i) * sqrt(N) restores the variance close to original z_i")


if __name__ == "__main__":
    main()
