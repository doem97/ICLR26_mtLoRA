#!/usr/bin/env python3
"""
Singular Value Spectrum Analysis for Spectral Regularization Visualization

This script compares the singular value distribution of B matrices:
1. Without spectral regularization (HydraLoRA baseline)
2. With spectral regularization (mtLoRA with spectral reg)

Key insight: The spectral-aware regularization uses w(σ) = exp(-σ/σ̄), which
weights low-SV components more heavily for orthogonalization. The effect is
SELECTIVE orthogonalization, not necessarily SV magnitude change.

Usage:
    python sv_spectrum_analysis.py

Output:
    - Singular value spectrum plots (PDF/PNG)
    - Per-band analysis (Top/Mid/Low SV components)
    - Cross-expert correlation analysis
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
# Configuration
# ==============================================================================

# Checkpoint paths (relative to PROJECT_ROOT)
CHECKPOINTS = {
    "baseline": {
        "path": "output/bbh/2_fgr_granularity/2a_component/g4096_scalar/sft_lora_model",
        "label": "HydraLoRA\n(w/o Spectral Reg)",
        "color": "#2196F3",  # Primary blue
        "marker": "o",
    },
    "spectral_0.5": {
        "path": "output/bbh/3_spectral/3a_lambda/lambda_0.5/sft_lora_model",
        "label": "Spectral Reg\n(λ=0.5)",
        "color": "#FF9800",  # Secondary orange
        "marker": "s",
    },
    "spectral_1.0": {
        "path": "output/bbh/3_spectral/3a_lambda/lambda_1.0/sft_lora_model",
        "label": "Spectral Reg\n(λ=1.0)",
        "color": "#FF9800",  # Secondary orange
        "marker": "^",
    },
}

# Use baseline and λ=1.0 for main comparison
MAIN_COMPARISON = ["baseline", "spectral_1.0"]

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "analysis" / "figures"

# Analysis config
TARGET_MODULE = "v_proj"  # Focus on one module for cleaner plot
SAMPLE_LAYERS = [5, 15, 25]  # Layers for per-layer plot

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
        "pdf.fonttype": 42,  # TrueType - keeps text editable in Illustrator
    }
)


def load_adapter_weights(ckpt_path):
    """Load adapter weights from checkpoint."""
    adapter_path = PROJECT_ROOT / ckpt_path / "adapter_model.bin"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    print(f"  Loading: {adapter_path.name}")
    state_dict = torch.load(adapter_path, map_location="cpu", weights_only=False)
    return state_dict


def extract_b_matrices(state_dict, module_name=None):
    """
    Extract B matrices from state dict.

    Returns: dict of {(layer, module, expert): weight_tensor}
    """
    b_matrices = {}

    for key, value in state_dict.items():
        if "lora_B" in key and ".weight" in key:
            parts = key.split(".")

            # Find layer index
            try:
                layer_pos = parts.index("layers")
                layer_id = int(parts[layer_pos + 1])
            except (ValueError, IndexError):
                continue

            # Find module name
            module = None
            for p in parts:
                if p in ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                    module = p
                    break
            if module is None:
                continue

            # Find expert index
            expert_idx = None
            for p in parts:
                if p.startswith("lora_B"):
                    expert_idx = int(p.replace("lora_B", ""))
                    break
            if expert_idx is None:
                continue

            # Filter if specified
            if module_name is not None and module != module_name:
                continue

            b_matrices[(layer_id, module, expert_idx)] = value

    return b_matrices


def compute_singular_values(weight_matrix):
    """Compute singular values of a weight matrix."""
    U, S, Vh = torch.linalg.svd(weight_matrix.float(), full_matrices=False)
    return S.numpy()


def analyze_checkpoint(ckpt_config):
    """Analyze a single checkpoint and return SV statistics."""
    print(f"\nAnalyzing: {ckpt_config['label'].replace(chr(10), ' ')}")

    state_dict = load_adapter_weights(ckpt_config["path"])
    b_matrices = extract_b_matrices(state_dict, module_name=TARGET_MODULE)

    print(f"  Found {len(b_matrices)} B matrices for {TARGET_MODULE}")

    # Collect singular values
    all_svs = []
    layer_svs = defaultdict(list)
    expert_svs = defaultdict(list)

    # Also store the actual B matrices for correlation analysis
    b_weights = {}

    for (layer, module, expert), weight in b_matrices.items():
        svs = compute_singular_values(weight)
        all_svs.append(svs)
        layer_svs[layer].append(svs)
        expert_svs[expert].append(svs)
        b_weights[(layer, expert)] = weight.float().numpy().flatten()

    if len(all_svs) == 0:
        print("  WARNING: No B matrices found")
        return None

    # Stack and compute statistics
    all_svs = np.stack(all_svs)  # (num_matrices, rank)
    mean_svs = np.mean(all_svs, axis=0)
    std_svs = np.std(all_svs, axis=0)

    # Compute per-band statistics (top 20%, mid 20-50%, bottom 50-100%)
    rank = all_svs.shape[1]
    top_idx = max(1, int(rank * 0.2))  # top 20%
    mid_idx = max(top_idx + 1, int(rank * 0.5))  # 20-50%

    band_stats = {
        "top": np.mean(all_svs[:, :top_idx]),  # top 20%
        "mid": np.mean(all_svs[:, top_idx:mid_idx]),  # 20-50%
        "bottom": np.mean(all_svs[:, mid_idx:]),  # 50-100%
    }

    print(
        f"  Mean SV by band: Top={band_stats['top']:.4f}, Mid={band_stats['mid']:.4f}, Bottom={band_stats['bottom']:.4f}"
    )

    return {
        "all_svs": all_svs,
        "mean_svs": mean_svs,
        "std_svs": std_svs,
        "layer_svs": dict(layer_svs),
        "expert_svs": dict(expert_svs),
        "band_stats": band_stats,
        "b_weights": b_weights,
        "config": ckpt_config,
    }


def compute_cross_expert_correlation(b_weights, layer):
    """Compute correlation matrix between experts' B matrices for a given layer."""
    experts = sorted([exp for (l, exp) in b_weights.keys() if l == layer])
    n_experts = len(experts)

    if n_experts == 0:
        return None

    corr_matrix = np.zeros((n_experts, n_experts))
    for i, exp_i in enumerate(experts):
        for j, exp_j in enumerate(experts):
            vec_i = b_weights[(layer, exp_i)]
            vec_j = b_weights[(layer, exp_j)]
            corr = np.corrcoef(vec_i, vec_j)[0, 1]
            corr_matrix[i, j] = corr

    return corr_matrix


def plot_main_comparison(results, output_path):
    """Create main comparison figure (publication quality)."""

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1, 1])

    baseline = results["baseline"]
    spectral = results["spectral_1.0"]

    rank = len(baseline["mean_svs"])
    x = np.arange(1, rank + 1)

    # === Panel A: SV Spectrum with Weighting Function ===
    ax1 = fig.add_subplot(gs[0])

    # Plot baseline SV spectrum
    ax1.plot(
        x,
        baseline["mean_svs"],
        color=baseline["config"]["color"],
        marker=baseline["config"]["marker"],
        markersize=6,
        linewidth=2,
        label="SV Magnitude σ",
    )
    ax1.fill_between(
        x,
        baseline["mean_svs"] - baseline["std_svs"],
        baseline["mean_svs"] + baseline["std_svs"],
        alpha=0.15,
        color=baseline["config"]["color"],
    )

    ax1.set_xlabel("Singular Value Index (sorted)")
    ax1.set_ylabel("Singular Value Magnitude", color=baseline["config"]["color"])
    ax1.tick_params(axis="y", labelcolor=baseline["config"]["color"])
    ax1.set_xticks([1, 4, 8, 12, 16])
    ax1.set_xlim(0.5, rank + 0.5)
    ax1.grid(True, alpha=0.3)

    # Secondary y-axis: weighting function w(σ) = exp(-σ/σ̄)
    ax1_twin = ax1.twinx()
    sigma_bar = np.mean(baseline["mean_svs"])  # mean SV as normalization
    weights = np.exp(-baseline["mean_svs"] / sigma_bar)
    ax1_twin.plot(
        x,
        weights,
        color="#FF9800",  # Orange for weight
        marker="D",
        markersize=5,
        linewidth=2,
        linestyle="--",
        label="Weight w(σ) = exp(-σ/σ̄)",
    )
    ax1_twin.set_ylabel("Regularization Weight w(σ)", color="#FF9800")
    ax1_twin.tick_params(axis="y", labelcolor="#FF9800")
    ax1_twin.set_ylim(0, 1.1)

    # Combined legend - upper right
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax1.set_title("(A) SV Spectrum & Reg Weight", fontweight="bold")

    # === Panel B: Relative Change by SV Index ===
    ax2 = fig.add_subplot(gs[1])

    ratio = (spectral["mean_svs"] - baseline["mean_svs"]) / baseline["mean_svs"] * 100

    # Use blue for negative (suppression), lighter orange for positive
    colors = ["#2196F3" if r < 0 else "#FFCC80" for r in ratio]
    bars = ax2.bar(x, ratio, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Add band annotations (top 20%, 20-50%, 50-100%)
    top_idx = max(1, int(rank * 0.2))
    mid_idx = max(top_idx + 1, int(rank * 0.5))

    # Subtle band shading
    ax2.axvspan(0.5, top_idx + 0.5, alpha=0.08, color="#2196F3", label="Top 20%")
    ax2.axvspan(top_idx + 0.5, mid_idx + 0.5, alpha=0.08, color="#FF9800", label="20-50%")
    ax2.axvspan(mid_idx + 0.5, rank + 0.5, alpha=0.12, color="#4CAF50", label="50-100%")

    ax2.set_xlabel("Singular Value Index")
    ax2.set_ylabel("Relative Change (%)")
    ax2.set_title("(B) Effect of Spectral Reg", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks([1, 4, 8, 12, 16])
    ax2.set_xlim(0.5, rank + 0.5)

    # Add mean change annotation
    mean_change = np.mean(ratio)
    ax2.text(
        0.95,
        0.95,
        f"Mean: {mean_change:.1f}%",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # === Panel C: Per-Band Statistics ===
    ax3 = fig.add_subplot(gs[2])

    bands = ["Top 20%\n(High SV)", "20-50%", "50-100%\n(Low SV)"]
    band_keys = ["top", "mid", "bottom"]

    baseline_vals = [baseline["band_stats"][k] for k in band_keys]
    spectral_vals = [spectral["band_stats"][k] for k in band_keys]

    bar_width = 0.35
    x_pos = np.arange(len(bands))

    bars1 = ax3.bar(
        x_pos - bar_width / 2,
        baseline_vals,
        bar_width,
        label="w/o Spectral Reg",
        color=baseline["config"]["color"],
        alpha=0.8,
    )
    bars2 = ax3.bar(
        x_pos + bar_width / 2,
        spectral_vals,
        bar_width,
        label="w/ Spectral Reg",
        color=spectral["config"]["color"],
        alpha=0.8,
    )

    # Add percentage change annotations
    for i, (b, s) in enumerate(zip(baseline_vals, spectral_vals)):
        change = (s - b) / b * 100
        ax3.annotate(
            f"{change:+.1f}%",
            xy=(x_pos[i] + bar_width / 2, s + 0.01),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#388E3C" if change < 0 else "#D32F2F",  # Green/red for change
        )

    ax3.set_ylabel("Mean Singular Value")
    ax3.set_title("(C) Per-Band Analysis", fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bands)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nSaved main comparison to: {output_path}")
    plt.close()


def plot_per_layer_analysis(results, output_path):
    """Plot per-layer singular value comparison."""

    baseline = results["baseline"]
    spectral = results["spectral_1.0"]

    fig, axes = plt.subplots(1, len(SAMPLE_LAYERS), figsize=(12, 4))

    for idx, layer in enumerate(SAMPLE_LAYERS):
        ax = axes[idx]

        # Get mean across experts for this layer
        if layer not in baseline["layer_svs"]:
            continue

        baseline_layer = np.mean(np.stack(baseline["layer_svs"][layer]), axis=0)
        spectral_layer = np.mean(np.stack(spectral["layer_svs"][layer]), axis=0)

        rank = len(baseline_layer)
        x = np.arange(1, rank + 1)

        ax.plot(
            x,
            baseline_layer,
            color=baseline["config"]["color"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label="w/o Spectral Reg",
        )
        ax.plot(
            x,
            spectral_layer,
            color=spectral["config"]["color"],
            marker="s",
            markersize=4,
            linewidth=1.5,
            label="w/ Spectral Reg",
        )

        ax.set_xlabel("SV Index")
        ax.set_ylabel("Magnitude" if idx == 0 else "")
        ax.set_title(f"Layer {layer}")
        if idx == len(SAMPLE_LAYERS) - 1:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 8, 16])

    plt.suptitle(f"Per-Layer Singular Value Spectrum ({TARGET_MODULE})", fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved per-layer plot to: {output_path}")
    plt.close()


def plot_expert_correlation(results, output_path):
    """Plot cross-expert correlation heatmaps."""

    baseline = results["baseline"]
    spectral = results["spectral_1.0"]

    layer = 15  # Middle layer

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for idx, (name, data) in enumerate([("baseline", baseline), ("spectral_1.0", spectral)]):
        ax = axes[idx]

        corr_matrix = compute_cross_expert_correlation(data["b_weights"], layer)
        if corr_matrix is None:
            continue

        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Correlation")

        ax.set_xlabel("Expert Index")
        ax.set_ylabel("Expert Index")
        label = data["config"]["label"].replace("\n", " ")
        ax.set_title(f"{label}")
        ax.set_xticks(range(0, 16, 4))
        ax.set_yticks(range(0, 16, 4))

        # Compute off-diagonal mean
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diag_mean = np.mean(np.abs(corr_matrix[mask]))
        ax.text(
            0.02,
            0.98,
            f"Mean |corr|: {off_diag_mean:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.suptitle(f"Cross-Expert B Matrix Correlation (Layer {layer}, {TARGET_MODULE})", fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved correlation plot to: {output_path}")
    plt.close()


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for name in MAIN_COMPARISON:
        data = results[name]
        label = data["config"]["label"].replace("\n", " ")
        print(f"\n{label}:")
        print(f"  Overall mean SV: {np.mean(data['all_svs']):.6f}")
        print(
            f"  Band breakdown: Top={data['band_stats']['top']:.4f}, "
            f"Mid={data['band_stats']['mid']:.4f}, "
            f"Bottom={data['band_stats']['bottom']:.4f}"
        )

    # Compute relative changes
    baseline = results["baseline"]
    spectral = results["spectral_1.0"]

    print("\nSpectral Reg Effect (λ=1.0 vs Baseline):")
    for band in ["top", "mid", "bottom"]:
        b_val = baseline["band_stats"][band]
        s_val = spectral["band_stats"][band]
        change = (s_val - b_val) / b_val * 100
        print(f"  {band.capitalize()} band: {change:+.2f}%")

    overall_change = (np.mean(spectral["all_svs"]) - np.mean(baseline["all_svs"])) / np.mean(baseline["all_svs"]) * 100
    print(f"  Overall: {overall_change:+.2f}%")


def main():
    print("=" * 60)
    print("Singular Value Spectrum Analysis")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Target module: {TARGET_MODULE}")

    # Check checkpoints exist
    for name, config in CHECKPOINTS.items():
        path = PROJECT_ROOT / config["path"]
        if not path.exists():
            print(f"WARNING: Checkpoint not found: {path}")

    # Analyze checkpoints
    results = {}
    for name, config in CHECKPOINTS.items():
        path = PROJECT_ROOT / config["path"]
        if path.exists():
            results[name] = analyze_checkpoint(config)

    if len(results) < 2:
        print("ERROR: Need at least 2 checkpoints for comparison")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_main_comparison(results, OUTPUT_DIR / "fig_sv_spectrum_main.png")
    plot_per_layer_analysis(results, OUTPUT_DIR / "fig_sv_spectrum_per_layer.png")
    plot_expert_correlation(results, OUTPUT_DIR / "fig_expert_correlation.png")

    # Print summary
    print_summary(results)

    print("\n" + "=" * 60)
    print("Done! Figures saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
