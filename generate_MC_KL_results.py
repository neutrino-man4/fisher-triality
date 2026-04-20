#!/usr/bin/env python3

from __future__ import annotations

from importlib import metadata
import json
import os
import h5py
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/work/abal/triality/results')
FIGURES_DIR = ROOT / "figures" / "mSD_window" / "n75"
GENERATED_DIR = ROOT / "generated"
DATA_PATH = '/ceph/abal/QFIT/MC/joint_datasets/KL/data.h5'
WEB_DIR = Path("/web/abal/public_html/plots/triality/mSD_window/n75")
FEATURE_NAMES = ["EEC narrow", "EEC wide", r"$e_2^{(1)}$", r"$e_3^{(1)}$"]


@dataclass(frozen=True)
class EventModel:
    cores: np.ndarray
    core_probs: np.ndarray
    widths: np.ndarray
    group_weights: np.ndarray
    alpha: float


QCD_MODEL = EventModel(
    cores=np.array([[0.0, 0.0], [0.10, -0.05]]),
    core_probs=np.array([0.82, 0.18]),
    widths=np.array([0.045, 0.14]),
    group_weights=np.array([0.78, 0.22]),
    alpha=0.65,
)

TOP_MODEL = EventModel(
    cores=np.array([[0.32, 0.0], [-0.18, 0.27], [-0.18, -0.27]]),
    core_probs=np.array([0.38, 0.34, 0.28]),
    widths=np.array([0.05, 0.055, 0.055]),
    group_weights=np.array([0.38, 0.34, 0.28]),
    alpha=0.75,
)


def sample_event(rng: np.random.Generator, model: EventModel, n_constituents: int) -> tuple[np.ndarray, np.ndarray]:
    counts = rng.multinomial(n_constituents, model.core_probs)
    energies = []
    coords = []
    for idx, count in enumerate(counts):
        if count == 0:
            continue
        weights = rng.dirichlet(np.full(count, model.alpha)) * model.group_weights[idx]
        xy = model.cores[idx] + rng.normal(scale=model.widths[idx], size=(count, 2))
        energies.append(weights)
        coords.append(xy)
    z = np.concatenate(energies)
    z = z / z.sum()
    xy = np.concatenate(coords, axis=0)
    return z, xy


def event_features(z: np.ndarray, xy: np.ndarray) -> np.ndarray:
    delta = xy[:, None, :] - xy[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    iu = np.triu_indices(len(z), k=1)
    dij = distances[iu]
    zij = z[iu[0]] * z[iu[1]]

    eec_narrow = np.sum(zij * ((dij > 0.06) & (dij <= 0.20)))
    eec_wide = np.sum(zij * ((dij > 0.20) & (dij <= 0.80)))
    e2 = np.sum(zij * dij)

    e3 = 0.0
    for i, j, k in combinations(range(len(z)), 3):
        e3 += z[i] * z[j] * z[k] * distances[i, j] * distances[i, k] * distances[j, k]

    return np.array([eec_narrow, eec_wide, e2, e3], dtype=float)


def generate_sample(
    *,
    seed: int = 7,
    n_events: int = 12000,
    n_constituents: int = 18,
    top_fraction: float = 0.22,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = np.empty((n_events, 4), dtype=float)
    labels = np.empty(n_events, dtype=int)
    for idx in range(n_events):
        is_top = rng.random() < top_fraction
        model = TOP_MODEL if is_top else QCD_MODEL
        z, xy = sample_event(rng, model, n_constituents)
        data[idx] = event_features(z, xy)
        labels[idx] = int(is_top)
    return data, labels


def standardize(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std

def whiten(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0)
    centered = data - mean
    cov = np.cov(centered, rowvar=False, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    cov_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    zdata = centered @ cov_inv_sqrt
    return zdata

def summarize(data: np.ndarray, labels: np.ndarray) -> dict:
    qcd = data[labels == 0]
    top = data[labels == 1]
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    zdata = standardize(data)
    #zdata = whiten(data)
    zqcd = zdata[labels == 0]
    ztop = zdata[labels == 1]

    corr = np.cov(zdata, rowvar=False, bias=True)
    third = np.einsum("ni,nj,nk->ijk", zdata, zdata, zdata) / len(zdata)

    direction = ztop.mean(axis=0) - zqcd.mean(axis=0)
    direction = direction / np.linalg.norm(direction)
    projections = zdata @ direction
    cubic_coeff = np.einsum("i,j,k,ijk", direction, direction, direction, third)
    quad_coeff = direction @ corr @ direction

    t_values = np.linspace(0.0, 0.65, 65)
    exact = np.array([np.log(np.mean(np.exp(t * projections))) for t in t_values])
    quad = 0.5 * quad_coeff * t_values**2
    cubic = quad + (cubic_coeff / 6.0) * t_values**3

    selected_t = np.array([0.15, 0.30, 0.45, 0.60])
    exact_sel = np.array([np.log(np.mean(np.exp(t * projections))) for t in selected_t])
    quad_sel = 0.5 * quad_coeff * selected_t**2
    cubic_sel = quad_sel + (cubic_coeff / 6.0) * selected_t**3

    pairwise = []
    for i in range(4):
        for j in range(i + 1, 4):
            pairwise.append(
                {
                    "features": [FEATURE_NAMES[i], FEATURE_NAMES[j]],
                    "value": float(corr[i, j]),
                }
            )

    triplets = []
    for i, j, k in combinations(range(4), 3):
        triplets.append(
            {
                "features": [FEATURE_NAMES[i], FEATURE_NAMES[j], FEATURE_NAMES[k]],
                "value": float(third[i, j, k]),
            }
        )

    centrality2 = np.sum(np.abs(corr - np.eye(4)), axis=1)
    centrality3 = np.zeros(4)
    for i, j, k in combinations(range(4), 3):
        weight = abs(third[i, j, k])
        centrality3[[i, j, k]] += weight

    results = {
        "feature_names": FEATURE_NAMES,
        "raw_means_qcd": qcd.mean(axis=0).tolist(),
        "raw_means_top": top.mean(axis=0).tolist(),
        "raw_mean_all": mean.tolist(),
        "raw_std_all": std.tolist(),
        "direction": direction.tolist(),
        "pairwise_correlations": pairwise,
        "triplet_weights": triplets,
        "centrality_order2": centrality2.tolist(),
        "centrality_order3": centrality3.tolist(),
        "kl_curve": {
            "t": t_values.tolist(),
            "exact": exact.tolist(),
            "quadratic": quad.tolist(),
            "cubic": cubic.tolist(),
        },
        "kl_table": [
            {
                "t": float(t),
                "exact": float(e),
                "quadratic": float(q),
                "cubic": float(c),
                "quadratic_error": float(abs(e - q)),
                "cubic_error": float(abs(e - c)),
            }
            for t, e, q, c in zip(selected_t, exact_sel, quad_sel, cubic_sel, strict=True)
        ],
    }
    return results


def make_kl_figure(results: dict, metadata: dict) -> None:
    curve = results["kl_curve"]
    t = np.array(curve["t"])
    exact = np.array(curve["exact"])
    quad = np.array(curve["quadratic"])
    cubic = np.array(curve["cubic"])

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.plot(t, exact, color="#102542", linewidth=2.5, label="Exact KL", alpha=0.5)
    ax.plot(t, quad, color="#b24745", linewidth=2.0, linestyle="--", label="Quadratic")
    ax.plot(t, cubic, color="#4f7d39", linewidth=2.0, linestyle="-.", label="Quadratic + cubic")
    ax.set_xlabel(r"Shift magnitude $t$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p_{\theta}\Vert p_{\theta+t v})$")
    ax.set_title(f"100000 events (22% TTBar): local KL expansion")
    ax.grid(alpha=0.25)
    # Add annotation for pT cut and mass window
    ax.text(0.02, 0.65,
        rf"$p_T > {metadata['pt_cut']:.0f}$ GeV"
        "\n\n"
        rf"$m_{{\mathrm{{SD}}}} \in [{metadata['mass_win_lo']:.0f},\, {metadata['mass_win_hi']:.0f}]$ GeV""\n\n"
        rf"{metadata['frac_top']*100:.0f}% top jets"
        "\n\n"
        r"$N_\mathrm{constituents} = " + f"{metadata['num_particles']}, " + r"N_\mathrm{events} = " + f"{metadata['num_events']}" + "$",
        transform=ax.transAxes, fontsize=14)
    ax.legend(frameon=False, loc="lower right", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "MC_kl_comparison.pdf")
    fig.savefig(FIGURES_DIR / "MC_kl_comparison.png")
    fig.savefig(WEB_DIR / "MC_kl_comparison.png", dpi=500)
    plt.close(fig)


def make_hypergraph_figure(results: dict, metadata: dict) -> None:
    feature_names = results["feature_names"]
    centrality2 = np.array(results["centrality_order2"])
    centrality3 = np.array(results["centrality_order3"])
    triplets = results["triplet_weights"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))

    x = np.arange(len(feature_names))
    width = 0.36
    axes[0].bar(x - width / 2, centrality2, width=width, color="#102542", label="Pairwise graph")
    axes[0].bar(x + width / 2, centrality3, width=width, color="#d97925", label="3-hypergraph")
    axes[0].set_xticks(x, feature_names, rotation=20, ha="right")
    axes[0].set_ylabel("Centrality")
    axes[0].set_title("Feature centrality by order")
    axes[0].set_ylim(0.0, 1.35 * max(np.max(centrality2), np.max(centrality3)))
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    labels = [" / ".join(item["features"]) for item in triplets]
    values = [item["value"] for item in triplets]
    colors = ["#4f7d39" if value > 0 else "#b24745" for value in values]
    axes[1].barh(labels, values, color=colors)
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel(r"Standardized $I^{(3)}_{abc}$")
    axes[1].set_title("Distinct 3-hyperedge weights")
    axes[1].grid(axis="x", alpha=0.25)

    fig.text(0.51, 0.02,
        "---------------------------------------\n"
        rf"$p_T > {metadata['pt_cut']:.0f}$ GeV,"
        "\n"
        rf"$m_{{\mathrm{{SD}}}} \in [{metadata['mass_win_lo']:.0f},\, {metadata['mass_win_hi']:.0f}]$ GeV,"
        "\n"
        rf"{metadata['frac_top']*100:.0f}% top jets,"
        "\n"
        r"$N_\mathrm{constituents} = " + f"{metadata['num_particles']}, " + r"N_\mathrm{events} = " + f"{metadata['num_events']}" + "$",
        ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "MC_hypergraph_summary_standardized.pdf")
    fig.savefig(FIGURES_DIR / "MC_hypergraph_summary_standardized.png", dpi=500)
    fig.savefig(WEB_DIR / "MC_hypergraph_summary_standardized.png", dpi=500)

    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    GENERATED_DIR.mkdir(exist_ok=True, parents=True)
    WEB_DIR.mkdir(exist_ok=True, parents=True)
    #data, labels = generate_sample()
    with h5py.File(DATA_PATH, "r") as fh:
        data = fh["features"][()]
        labels = fh["labels"][()]
        pt_cut = fh.attrs["pt_cut_GeV"]
        mass_win_lo = fh.attrs["mass_win_lo"]
        mass_win_hi = fh.attrs["mass_win_hi"]
        frac_top = fh.attrs["frac_top"]
        num_particles = fh.attrs["num_particles_used"]
        print(f"Loaded data with shape {data.shape} and labels with shape {labels.shape}")
        print(f"pT cut: {pt_cut}, mass window: [{mass_win_lo}, {mass_win_hi}], TTBar fraction: {frac_top}")
    results = summarize(data, labels)
    metadata = {
        "num_events": len(data),
        "num_particles": num_particles,
        "pt_cut": pt_cut,
        "mass_win_lo": mass_win_lo,
        "mass_win_hi": mass_win_hi,
        "frac_top": frac_top,
    }

    with (GENERATED_DIR / "MC_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    make_kl_figure(results, metadata)
    make_hypergraph_figure(results, metadata)
    #import pdb;pdb.set_trace()
    
    #print(f"Wrote {(GENERATED_DIR / 'MC_results.json').relative_to(ROOT)}")
    #print(f"Wrote {(FIGURES_DIR / 'MC_kl_comparison.pdf').relative_to(ROOT)}")
    #print(f"Wrote {(FIGURES_DIR / 'MC_hypergraph_summary.pdf').relative_to(ROOT)}")
    

if __name__ == "__main__":
    main()
