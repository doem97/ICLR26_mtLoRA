#!/usr/bin/env python3
"""
Compare gradient conflict results between component-level and block-level models.

Generates:
1. Side-by-side boxplot of off-diagonal similarity distributions
2. Difference heatmap (block - component)
3. Summary statistics

Usage:
    python compare.py \
        --component_json figures/gradient_conflict_component.json \
        --block_json figures/gradient_conflict_block.json \
        --output_dir figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Plot style
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
    """Load results from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def get_off_diagonal(matrix):
    """Extract off-diagonal elements from a similarity matrix."""
    n = len(matrix)
    mask = ~np.eye(n, dtype=bool)
    arr = np.array(matrix)
    return arr[mask]


def plot_comparison_boxplot(component_data, block_data, output_path):
    """Create side-by-side boxplot comparison."""
    comp_offdiag = get_off_diagonal(component_data["similarity_matrix"])
    block_offdiag = get_off_diagonal(block_data["similarity_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Boxplot
    ax1 = axes[0]
    bp = ax1.boxplot(
        [comp_offdiag, block_offdiag],
        labels=["Component\n(HydraLoRA)", "Block\n(mtLoRA)"],
        patch_artist=True,
        widths=0.6,
    )

    # Colors
    colors = ["#FF6B6B", "#4ECDC4"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel("Pairwise Gradient Cosine Similarity")
    ax1.set_title("Distribution of Task-Pair Gradient Similarity", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Add mean markers
    means = [np.mean(comp_offdiag), np.mean(block_offdiag)]
    ax1.scatter([1, 2], means, marker="D", color="black", s=50, zorder=5, label="Mean")
    ax1.legend(loc="upper right")

    # Add statistics text
    comp_mean, comp_std = np.mean(comp_offdiag), np.std(comp_offdiag)
    block_mean, block_std = np.mean(block_offdiag), np.std(block_offdiag)
    reduction = (comp_mean - block_mean) / comp_mean * 100

    stats_text = (
        f"Component: {comp_mean:.3f}±{comp_std:.3f}\n"
        f"Block: {block_mean:.3f}±{block_std:.3f}\n"
        f"Reduction: {reduction:.1f}%"
    )
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Right: Violin plot for better distribution view
    ax2 = axes[1]
    vp = ax2.violinplot([comp_offdiag, block_offdiag], positions=[1, 2], showmeans=True, showmedians=True, widths=0.7)

    # Color violins
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.7)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Component\n(HydraLoRA)", "Block\n(mtLoRA)"])
    ax2.set_ylabel("Pairwise Gradient Cosine Similarity")
    ax2.set_title("Similarity Distribution (Violin)", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved boxplot to: {output_path}")
    plt.close()

    return comp_mean, block_mean, reduction


def plot_difference_heatmap(component_data, block_data, output_path):
    """Create difference heatmap (block - component)."""
    comp_matrix = np.array(component_data["similarity_matrix"])
    block_matrix = np.array(block_data["similarity_matrix"])

    # Check if tasks match
    comp_tasks = component_data["tasks"]
    block_tasks = block_data["tasks"]

    if comp_tasks != block_tasks:
        print("Warning: Tasks don't match exactly. Using component tasks.")

    tasks = comp_tasks
    diff_matrix = block_matrix - comp_matrix

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Heatmap 1: Component
    ax1 = axes[0]
    im1 = ax1.imshow(comp_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax1.set_title("Component-Level\n(HydraLoRA)", fontweight="bold")
    ax1.set_xticks(range(len(tasks)))
    ax1.set_yticks(range(len(tasks)))
    ax1.set_xticklabels([t[:10] for t in tasks], rotation=45, ha="right", fontsize=7)
    ax1.set_yticklabels([t[:10] for t in tasks], fontsize=7)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Heatmap 2: Block
    ax2 = axes[1]
    im2 = ax2.imshow(block_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax2.set_title("Block-Level\n(mtLoRA)", fontweight="bold")
    ax2.set_xticks(range(len(tasks)))
    ax2.set_yticks(range(len(tasks)))
    ax2.set_xticklabels([t[:10] for t in tasks], rotation=45, ha="right", fontsize=7)
    ax2.set_yticklabels([t[:10] for t in tasks], fontsize=7)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Heatmap 3: Difference (block - component)
    ax3 = axes[2]
    # Use symmetric colormap centered at 0
    vmax = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    im3 = ax3.imshow(diff_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax3.set_title("Difference\n(Block − Component)", fontweight="bold")
    ax3.set_xticks(range(len(tasks)))
    ax3.set_yticks(range(len(tasks)))
    ax3.set_xticklabels([t[:10] for t in tasks], rotation=45, ha="right", fontsize=7)
    ax3.set_yticklabels([t[:10] for t in tasks], fontsize=7)
    cbar = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar.set_label("Δ Similarity")

    # Add values in difference cells
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            if i != j:  # Skip diagonal
                val = diff_matrix[i, j]
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax3.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved difference heatmap to: {output_path}")
    plt.close()


def plot_task_pair_comparison(component_data, block_data, output_path):
    """Bar chart comparing each task pair."""
    comp_matrix = np.array(component_data["similarity_matrix"])
    block_matrix = np.array(block_data["similarity_matrix"])
    tasks = component_data["tasks"]

    # Get all unique task pairs (upper triangle, excluding diagonal)
    n = len(tasks)
    pairs = []
    comp_vals = []
    block_vals = []

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(f"{tasks[i][:8]}—{tasks[j][:8]}")
            comp_vals.append(comp_matrix[i, j])
            block_vals.append(block_matrix[i, j])

    # Sort by difference (block - component)
    diffs = [b - c for b, c in zip(block_vals, comp_vals)]
    sorted_idx = np.argsort(diffs)

    pairs = [pairs[i] for i in sorted_idx]
    comp_vals = [comp_vals[i] for i in sorted_idx]
    block_vals = [block_vals[i] for i in sorted_idx]
    diffs = [diffs[i] for i in sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(pairs))
    width = 0.35

    bars1 = ax.barh(x - width / 2, comp_vals, width, label="Component (HydraLoRA)", color="#FF6B6B", alpha=0.8)
    bars2 = ax.barh(x + width / 2, block_vals, width, label="Block (mtLoRA)", color="#4ECDC4", alpha=0.8)

    ax.set_xlabel("Gradient Cosine Similarity")
    ax.set_ylabel("Task Pair")
    ax.set_title("Per-Task-Pair Gradient Similarity Comparison", fontweight="bold")
    ax.set_yticks(x)
    ax.set_yticklabels(pairs, fontsize=7)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Add vertical line at mean
    ax.axvline(np.mean(comp_vals), color="#FF6B6B", linestyle="--", alpha=0.5)
    ax.axvline(np.mean(block_vals), color="#4ECDC4", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved task pair comparison to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare gradient conflict results")
    parser.add_argument("--component_json", type=str, required=True, help="Path to component-level results JSON")
    parser.add_argument("--block_json", type=str, required=True, help="Path to block-level results JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for figures")
    args = parser.parse_args()

    print("=" * 60)
    print("Gradient Conflict Comparison")
    print("=" * 60)

    # Load results
    print(f"\nLoading component results from: {args.component_json}")
    component_data = load_results(args.component_json)

    print(f"Loading block results from: {args.block_json}")
    block_data = load_results(args.block_json)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparisons
    print("\n" + "-" * 40)
    print("Generating comparison plots...")
    print("-" * 40)

    # 1. Boxplot comparison
    comp_mean, block_mean, reduction = plot_comparison_boxplot(
        component_data, block_data, output_dir / "comparison_boxplot.png"
    )

    # 2. Difference heatmap
    plot_difference_heatmap(component_data, block_data, output_dir / "comparison_heatmap.png")

    # 3. Task pair comparison
    plot_task_pair_comparison(component_data, block_data, output_dir / "comparison_taskpairs.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Component-level mean similarity: {comp_mean:.4f}")
    print(f"Block-level mean similarity:     {block_mean:.4f}")
    print(f"Reduction:                       {reduction:.2f}%")
    print()

    if reduction > 0:
        print("✓ Block-level shows LOWER gradient similarity (less conflict)")
    else:
        print("✗ Block-level shows HIGHER gradient similarity (more conflict)")

    print("=" * 60)


if __name__ == "__main__":
    main()
