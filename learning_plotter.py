#!/usr/bin/env python3
"""Recreate the learning benchmark figure from a saved metrics.npz file."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "graph":             "#9b2c2c",
    "hypergraph":        "#1f4f82",
    "random_hypergraph": "#7a6f5a",
}
LABELS = {
    "graph":             "Pairwise Fisher graph",
    "hypergraph":        "Triality-informed hypergraph",
    "random_hypergraph": "Random-triplet hypergraph control",
}
PLOT_MODES = ["graph", "hypergraph"]


def make_figure(npz_path: str, out_path: str) -> None:
    d = np.load(npz_path, allow_pickle=True)
    train_sizes      = d["train_sizes"]
    epochs           = d["epochs"]
    task_title       = str(d["task_title"])
    n_epochs         = int(d["n_epochs"])
    epoch_train_size = int(d["epoch_train_size"])

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    ylims = []
    for mode in PLOT_MODES:
        means = d[f"{mode}_mean_auc"]
        stds  = d[f"{mode}_std_auc"]
        ylims += [(means - stds).min() - 0.02, (means + stds).max() + 0.02]
        axes[0].errorbar(
            train_sizes, means, yerr=stds,
            color=COLORS[mode], linewidth=2.2, marker="o", capsize=3.5, label=LABELS[mode],
        )

    axes[0].set_xlabel("Labeled training events", fontsize=15)
    axes[0].set_ylabel("Test AUC", fontsize=15)
    axes[0].set_xticks(train_sizes)
    axes[0].set_ylim(min(ylims), max(ylims))
    axes[0].set_title(task_title, fontsize=16)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=12)

    band_extremes = []
    for mode in PLOT_MODES:
        mean_auc = d[f"{mode}_mean_val_auc"]
        std_auc  = d[f"{mode}_std_val_auc"]
        band_extremes += [(mean_auc - std_auc).min(), (mean_auc + std_auc).max()]
        axes[1].plot(epochs, mean_auc, color=COLORS[mode], linewidth=2.0, label=LABELS[mode])
        axes[1].fill_between(
            epochs, mean_auc - std_auc, mean_auc + std_auc,
            color=COLORS[mode], alpha=0.15,
        )

    axes[1].set_xlabel("Training epoch", fontsize=15)
    axes[1].set_ylabel("Validation AUC", fontsize=15)
    axes[1].set_xlim(1, n_epochs)
    axes[1].set_ylim(min(band_extremes) - 0.03, max(band_extremes) + 0.03)
    axes[1].set_title(f"Convergence at {epoch_train_size} labeled events", fontsize=15)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning benchmark from metrics.npz.")
    parser.add_argument("--npz", help="Path to metrics.npz")
    parser.add_argument("-o", "--output", default="learning_real_summary.pdf",
                        help="Output figure path (default: learning_real_summary.pdf)")
    args = parser.parse_args()
    make_figure(args.npz, args.output)


if __name__ == "__main__":
    main()
