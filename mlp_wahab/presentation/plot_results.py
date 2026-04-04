"""
Generate comparison charts from SpecEE predictor benchmark results.

Produces two figures:
  1. Average exit layer across all datasets (lower = earlier exit = better)
  2. Accuracy on commonsenseqa and sst2 (higher = better)

Usage:
    cd EarlyExit
    source .venv/bin/activate
    python mlp_wahab/presentation/plot_results.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Benchmark data (2026-04-03, threshold=0.5, 10 samples/dataset, MPS device)
# ---------------------------------------------------------------------------

PREDICTORS = ["specee", "baseline", "augmented"]
PREDICTOR_LABELS = ["SpecEE\n(original)", "Baseline", "Data\nAugmentation"]

# --- Average Exit Layer (lower = exits earlier = faster) ---
EXIT_DATASETS = ["mt_bench", "alpaca", "gsm8k", "qa", "humaneval", "commonsenseqa", "sst2"]
EXIT_DATA = {
    "specee":    [24.6, 24.7, 24.9, 25.2, 24.5, 26.1, 28.2],
    "baseline":  [25.2, 24.0, 24.9, 25.5, 25.1, 26.2, 29.1],
    "augmented": [24.7, 24.6, 24.7, 26.0, 24.8, 26.1, 29.1],
}

# --- Accuracy (%) ---
ACC_DATASETS = ["commonsenseqa", "sst2"]
HF_BASELINE_ACC = [60.00, 90.00]
ACC_DATA = {
    "specee":    [59.74, 93.75],
    "baseline":  [60.00, 93.75],
    "augmented": [60.76, 93.75],
}

COLORS = {
    "specee":    "#888888",
    "baseline":  "#5CB85C",
    "augmented": "#2D8A2D",
}


def plot_avg_exit_layer():
    """Bar chart of average exit layer across all datasets for each predictor."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(EXIT_DATASETS))
    width = 0.15
    offsets = np.arange(len(PREDICTORS)) - (len(PREDICTORS) - 1) / 2

    for i, pred in enumerate(PREDICTORS):
        bars = ax.bar(
            x + offsets[i] * width,
            EXIT_DATA[pred],
            width,
            label=PREDICTOR_LABELS[i].replace("\n", " "),
            color=COLORS[pred],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, EXIT_DATA[pred]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{round(val)}",
                ha="center", va="bottom", fontsize=6, fontweight="bold",
            )

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Avg Exit Layer", fontsize=11)
    ax.set_title("Average Exit Layer by Predictor\n(lower = exits earlier = faster inference)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(EXIT_DATASETS, fontsize=9)
    ax.set_ylim(22, 31)
    ax.axhline(y=32, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="No exit (layer 32)")
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "avg_exit_layer.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_accuracy():
    """Grouped bar chart of accuracy on commonsenseqa and sst2."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(ACC_DATASETS))
    width = 0.13
    # Include HF baseline as first group
    all_labels = ["HF Baseline"] + [l.replace("\n", " ") for l in PREDICTOR_LABELS]
    all_colors = ["#D9534F"] + [COLORS[p] for p in PREDICTORS]
    all_data = [HF_BASELINE_ACC] + [ACC_DATA[p] for p in PREDICTORS]

    n = len(all_labels)
    offsets = np.arange(n) - (n - 1) / 2

    for i, (label, color, data) in enumerate(zip(all_labels, all_colors, all_data)):
        bars = ax.bar(
            x + offsets[i] * width,
            data,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, data):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy by Predictor\n(higher = better, HF Baseline = no early exit)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(ACC_DATASETS, fontsize=10)
    ax.set_ylim(50, 100)
    ax.legend(fontsize=8, loc="lower right", ncol=3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "accuracy.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_avg_exit_layer()
    plot_accuracy()
    print("\nAll charts generated.")
