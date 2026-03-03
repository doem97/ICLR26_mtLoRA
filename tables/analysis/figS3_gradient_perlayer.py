#!/usr/bin/env python3
"""
Compare per-layer gradient similarity between component and block models.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "sans-serif",
        "pdf.fonttype": 42,
    }
)


def load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component_json", type=str, required=True)
    parser.add_argument("--block_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    comp_data = load_results(args.component_json)
    block_data = load_results(args.block_json)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract per-layer data
    comp_layers = comp_data["per_layer"]
    block_layers = block_data["per_layer"]

    # Get common layers
    layers = sorted([int(l) for l in comp_layers.keys() if l in block_layers])

    comp_means = [comp_layers[str(l)]["mean"] for l in layers]
    comp_stds = [comp_layers[str(l)]["std"] for l in layers]
    block_means = [block_layers[str(l)]["mean"] for l in layers]
    block_stds = [block_layers[str(l)]["std"] for l in layers]

    # ============================================
    # Figure 1: Side-by-side comparison
    # ============================================
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.array(layers)
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        comp_means,
        width,
        yerr=comp_stds,
        label="Component (HydraLoRA)",
        color="#FF6B6B",
        alpha=0.8,
        capsize=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        block_means,
        width,
        yerr=block_stds,
        label="Block (mtLoRA)",
        color="#4ECDC4",
        alpha=0.8,
        capsize=3,
    )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Pairwise Gradient Similarity")
    ax.set_title("Per-Layer Gradient Similarity: Component vs Block", fontweight="bold")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "perlayer_comparison.png", bbox_inches="tight")
    plt.savefig(output_dir / "perlayer_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'perlayer_comparison.png'}")
    plt.close()

    # ============================================
    # Figure 2: Line plot with error bands
    # ============================================
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(
        layers,
        np.array(comp_means) - np.array(comp_stds),
        np.array(comp_means) + np.array(comp_stds),
        alpha=0.2,
        color="#FF6B6B",
    )
    ax.fill_between(
        layers,
        np.array(block_means) - np.array(block_stds),
        np.array(block_means) + np.array(block_stds),
        alpha=0.2,
        color="#4ECDC4",
    )

    ax.plot(layers, comp_means, "o-", color="#FF6B6B", linewidth=2, markersize=8, label="Component (HydraLoRA)")
    ax.plot(layers, block_means, "s-", color="#4ECDC4", linewidth=2, markersize=8, label="Block (mtLoRA)")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Pairwise Gradient Similarity")
    ax.set_title("Per-Layer Gradient Similarity Trend", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # Add annotations for interesting points
    # Find max difference layers
    diffs = [b - c for c, b in zip(comp_means, block_means)]
    max_idx = np.argmax(np.abs(diffs))
    ax.annotate(
        f"Δ={diffs[max_idx]:.2f}",
        xy=(layers[max_idx], (comp_means[max_idx] + block_means[max_idx]) / 2),
        xytext=(layers[max_idx] + 2, 0.3),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "perlayer_trend.png", bbox_inches="tight")
    plt.savefig(output_dir / "perlayer_trend.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'perlayer_trend.png'}")
    plt.close()

    # ============================================
    # Figure 3: Difference plot (block - component)
    # ============================================
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ["#4ECDC4" if d < 0 else "#FF6B6B" for d in diffs]
    bars = ax.bar(layers, diffs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Δ Similarity (Block − Component)")
    ax.set_title("Per-Layer Gradient Similarity Difference\n(Negative = Block has LOWER conflict)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add color legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#4ECDC4", alpha=0.8, label="Block lower (better)"),
        Patch(facecolor="#FF6B6B", alpha=0.8, label="Block higher"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Add overall stats
    avg_diff = np.mean(diffs)
    n_better = sum(1 for d in diffs if d < 0)
    stats_text = f"Avg Δ: {avg_diff:.3f}\nBlock lower in {n_better}/{len(diffs)} layers"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "perlayer_difference.png", bbox_inches="tight")
    plt.savefig(output_dir / "perlayer_difference.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'perlayer_difference.png'}")
    plt.close()

    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Component overall mean: {comp_data['overall_mean']:.4f}")
    print(f"Block overall mean:     {block_data['overall_mean']:.4f}")
    print(f"Average difference:     {avg_diff:.4f}")
    print(f"Block lower in:         {n_better}/{len(diffs)} layers")
    print()

    # Find most different layers
    sorted_diffs = sorted(zip(layers, diffs), key=lambda x: x[1])
    print("Layers where Block shows MOST improvement (lower similarity):")
    for l, d in sorted_diffs[:3]:
        print(
            f"  Layer {l}: Δ = {d:.3f} (comp={comp_layers[str(l)]['mean']:.3f}, block={block_layers[str(l)]['mean']:.3f})"
        )

    print("\nLayers where Block shows LEAST improvement (or worse):")
    for l, d in sorted_diffs[-3:]:
        print(
            f"  Layer {l}: Δ = {d:.3f} (comp={comp_layers[str(l)]['mean']:.3f}, block={block_layers[str(l)]['mean']:.3f})"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
