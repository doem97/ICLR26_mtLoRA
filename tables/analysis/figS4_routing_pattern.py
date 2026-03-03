#!/usr/bin/env python
"""
Visualize per-dimension routing pattern.
One clean figure for the paper.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def plot_routing_heatmap(task_data, save_path=None, figsize=(7, 5)):
    """
    Single clean heatmap showing dimension-wise routing weights.
    X-axis: Experts, Y-axis: Dimension Groups, Color: Routing Weight
    """
    first_task = list(task_data.keys())[0]
    layer_names = list(task_data[first_task].keys())

    # Use middle layer (most representative)
    mid_idx = len(layer_names) // 2
    layer_name = layer_names[mid_idx]

    # Average across all tasks
    all_weights = []
    for task, layers in task_data.items():
        if layer_name in layers:
            all_weights.append(layers[layer_name]["mean"])

    avg_weights = torch.stack(all_weights).mean(dim=0).numpy()
    num_groups, num_experts = avg_weights.shape

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(avg_weights, aspect="auto", cmap="YlOrRd")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Routing Weight", fontsize=11)

    # Axis labels
    ax.set_xlabel("Expert Index", fontsize=12)
    ax.set_ylabel("Dimension Group", fontsize=12)

    # X ticks: all experts
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([str(i) for i in range(num_experts)], fontsize=9)

    # Y ticks: sparse for readability
    yticks = [0, num_groups // 4, num_groups // 2, 3 * num_groups // 4, num_groups - 1]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=9)

    # Title
    layer_idx = [s for s in layer_name.split(".") if s.isdigit()]
    layer_num = layer_idx[0] if layer_idx else ""
    ax.set_title(f"Per-Dimension Routing Weights (Layer {layer_num})", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    plt.close()


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./scripts/analysis/routing_heatmap/routing_weights.pt")
    parser.add_argument("--output_dir", type=str, default="./scripts/analysis/routing_heatmap/figures")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    print(f"Loading: {args.input}")
    task_data = torch.load(args.input, weights_only=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # One figure only
    plot_routing_heatmap(task_data, save_path=output_dir / "fig_routing_pattern.png")

    print("Done.")


if __name__ == "__main__":
    main()
