"""Recreate the AUC-vs-features figure from a saved auc.npz file."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AUC scan results from auc.npz")
    parser.add_argument("--npz", type=Path, help="Path to auc.npz")
    parser.add_argument(
        "--type",
        dest="comparison_type",
        choices=["ttbar_vs_wqq", "ttbar_eft"],
        default=None,
        help="Comparison type (used for the plot title; inferred from directory name if omitted)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output path (PDF/PNG). Defaults to auc_scores.pdf next to the NPZ file.")
    args = parser.parse_args()

    data = np.load(args.npz)
    pair_means = data["pair_auc_mean"]
    pair_stds  = data["pair_auc_std"]
    hyper_means = data["hyper_auc_mean"]
    hyper_stds  = data["hyper_auc_std"]

    n_features = len(pair_means)
    ks = np.arange(2, n_features + 2)

    comparison_type = args.comparison_type
    if comparison_type is None:
        parent = args.npz.parent.name
        if "eft" in parent:
            comparison_type = "ttbar_eft"
        elif "wqq" in parent.lower() or "vs" in parent.lower():
            comparison_type = "ttbar_vs_wqq"

    title_map = {
        "ttbar_vs_wqq": r"AUC vs. number of features: $t\bar{t}$ vs. $W\!\to\!qq$",
        "ttbar_eft":    r"AUC vs. number of features: $t\bar{t}$ SM vs. EFT",
    }
    title = title_map.get(comparison_type, "AUC vs. retained observables")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for means, stds, color, marker, label in [
        (pair_means,  pair_stds,  "#9b2c2c", "o", "Pairwise graph selection"),
        (hyper_means, hyper_stds, "#1f4f82", "s", "Fisher hypergraph selection"),
    ]:
        ax.plot(ks, means, color=color, linewidth=2.2, marker=marker, markersize=5, label=label)
        if np.any(stds > 0):
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.18)

    ax.set_xlabel("Retained observables ($D$)", fontsize=13)
    ax.set_ylabel("Test AUC", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xticks(ks)
    ax.set_ylim(0.98, 0.99)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=12)

    fig.tight_layout()

    out = args.out or args.npz.parent / "auc_scores.pdf"
    fig.savefig(out)
    print(f"Saved figure to {out}")

    png_out = out.with_suffix(".png")
    fig.savefig(png_out, dpi=150)
    print(f"Saved figure to {png_out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
