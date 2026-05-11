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

PARTICLE_STR = "n25"
ROOT = Path('/work/abal/triality/results')
FIGURES_DIR = ROOT / "figures" / "mSD_window" / f"{PARTICLE_STR}"
GENERATED_DIR = ROOT / "generated"
DATA_PATH = f'/ceph/abal/QFIT/MC/joint_datasets/KL/data_{PARTICLE_STR}.h5'
WEB_DIR = Path(f"/web/abal/public_html/plots/triality/mSD_window/{PARTICLE_STR}")
FEATURE_NAMES = ["EEC narrow", "EEC wide", r"$e_2^{(1)}$", r"$e_3^{(1)}$"]
_PREPROCESSING_STR = "whitened"  # or "_{_PREPROCESSING_STR}"

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

def whiten(data: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = data.mean(axis=0)
    centered = data - mean
    cov = np.cov(centered, rowvar=False, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    cov_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    zdata = centered @ cov_inv_sqrt
    n_features = data.shape[1]
    col_norms = np.abs(cov_inv_sqrt).sum(axis=0)
    W_norm = cov_inv_sqrt / col_norms
    weights = {
        f'wf{j}': {f'f{i}': W_norm[i, j] for i in range(n_features)}
        for j in range(n_features)
    }
    return zdata, weights

def summarize(data: np.ndarray, labels: np.ndarray) -> dict:
    qcd = data[labels == 0]
    top = data[labels == 1]
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    if "standardize" in _PREPROCESSING_STR:
        print("Standardizing data...")
        zdata = standardize(data)
    elif "whiten" in _PREPROCESSING_STR:
        print("Whitening data...")
        zdata, weights = whiten(data)
        n_feat = len(FEATURE_NAMES)
        plain_names = ["EEC narrow", "EEC wide", "e2", "e3"]
        col_w = 10
        header = f"{'':14}" + "".join(f"{'wf'+str(j):>{col_w}}" for j in range(n_feat))
        print("\nWhitened feature composition (column-normalised mixing matrix):")
        print(header)
        for i, name in enumerate(plain_names):
            row = f"{name:<14}" + "".join(
                f"{weights['wf'+str(j)]['f'+str(i)]:>{col_w}.4f}" for j in range(n_feat)
            )
            print(row)

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

    t_values = np.linspace(0.0, 0.65, 66)
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

    t_refs = np.array([0.15, 0.30, 0.45, 0.60])
    t_ids = np.searchsorted(t, t_refs)
    print("Selected t values and errors:")
    
    print(f"{'t':<8} | {'Exact KL':<12} | {'Quadratic':<12} | {'Cubic':<20} | {'Error Reduction':<15}")
    print("-" * 80)
    for idx in range(len(t_refs)):
        quad_error = abs(exact[t_ids[idx]] - quad[t_ids[idx]])
        cubic_error = abs(exact[t_ids[idx]] - cubic[t_ids[idx]])
        error_reduction = quad_error / cubic_error if cubic_error > 0 else float('inf')
        # print(" \t t | \t Exact KL | \t Quadratic | \t Cubic (incl. Quadratic) | \t Cubic/Quadratic error reduction | \t")
        # print(f" \t {t_refs[idx]:.2f} | \t {exact[t_ids[idx]]:.4f} | \t {quad[t_ids[idx]]:.4f} | \t {cubic[t_ids[idx]]:.4f} | \t { (quad_error/(quad_error - cubic_error)):.1f}x")
        # Header
        # Data row
        print(f"{t_refs[idx]:<8.2f} | {exact[t_ids[idx]]:<12.6f} | {quad[t_ids[idx]]:<12.6f} | {cubic[t_ids[idx]]:<20.6f} | {error_reduction:<15.1f}x")
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.plot(t, exact, color="#102542", linewidth=2.5, label="Exact KL", alpha=0.5)
    ax.plot(t, quad, color="#b24745", linewidth=2.0, linestyle="--", label="Quadratic")
    ax.plot(t, cubic, color="#4f7d39", linewidth=2.0, linestyle="-.", label="Quadratic + cubic")
    ax.set_xlabel(r"Shift magnitude $t$", fontsize=17)
    ax.set_xscale("log")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p_{\theta}\Vert p_{\theta+t v})$",fontsize=17)
    ax.set_title(f"100000 events (22% TTBar): local KL expansion",fontsize=18.5)
    ax.grid(alpha=0.25)
    # Add annotation for pT cut and mass window
    ax.text(0.02, 0.65,
        rf"$p_T > {metadata['pt_cut']:.0f}$ GeV"
        "\n\n"
        rf"$m_{{\mathrm{{SD}}}} \in [{metadata['mass_win_lo']:.0f},\, {metadata['mass_win_hi']:.0f}]$ GeV""\n\n"
        rf"{metadata['frac_top']*100:.0f}% top jets"
        "\n\n"
        r"$N_\mathrm{constituents} = " + f"{metadata['num_particles']}, " + r"N_\mathrm{events} = " + f"{metadata['num_events']}" + "$",
        transform=ax.transAxes, fontsize=16)
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"MC_kl_comparison_{_PREPROCESSING_STR}.pdf")
    fig.savefig(FIGURES_DIR / f"MC_kl_comparison_{_PREPROCESSING_STR}.png")
    fig.savefig(WEB_DIR / f"MC_kl_comparison_{_PREPROCESSING_STR}.png", dpi=500)
    plt.close(fig)


def make_kl_residuals(results: dict, metadata: dict) -> None:
    curve = results["kl_curve"]
    t = np.array(curve["t"])
    exact = np.array(curve["exact"])
    quad = np.array(curve["quadratic"])
    cubic = np.array(curve["cubic"])

    res_quad = (quad - exact)
    res_cubic = (cubic - exact)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.plot(t, res_quad, color="#b24745", linewidth=2.0, linestyle="--", label=r"Quadratic $-$ Exact")
    ax.plot(t, res_cubic, color="#4f7d39", linewidth=2.0, linestyle="-.", label=r"(Quadratic + cubic) $-$ Exact")
    ax.axhline(0.0, color="#102542", linewidth=1.0, alpha=0.4)
    ax.set_xlabel(r"Shift $t$",fontsize=18)
    #ax.set_xscale("log")
    ax.set_ylabel(r"Residual: $\hat{D}_{\mathrm{KL}} - D_{\mathrm{KL}}$",fontsize=18)
    ax.set_yscale("symlog", linthresh=1e-5, linscale=0.5)
    ax.set_ylim(-1e-3, 1e-3)
    ax.set_xlim(0,0.45)
    ax.set_title("100000 events (22% TTBar): KL expansion residuals",fontsize=18)
    ax.grid(alpha=0.25)
    ax.text(0.54, 0.65,
        rf"$p_T > {metadata['pt_cut']:.0f}$ GeV"
        "\n\n"
        rf"$m_{{\mathrm{{SD}}}} \in [{metadata['mass_win_lo']:.0f},\, {metadata['mass_win_hi']:.0f}]$ GeV""\n\n"
        rf"{metadata['frac_top']*100:.0f}% top jets"
        "\n\n"
        r"$N_\mathrm{constituents} = " + f"{metadata['num_particles']}, " + r"N_\mathrm{events} = " + f"{metadata['num_events']}" + "$",
        transform=ax.transAxes, fontsize=13)
    ax.legend(frameon=False, loc="lower right", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"MC_kl_residuals_{_PREPROCESSING_STR}.pdf")
    fig.savefig(FIGURES_DIR / f"MC_kl_residuals_{_PREPROCESSING_STR}.png")
    fig.savefig(WEB_DIR / f"MC_kl_residuals_{_PREPROCESSING_STR}.png", dpi=500)
    plt.close(fig)


def make_hypergraph_figure(results: dict, metadata: dict, separate: bool = False) -> None:
    feature_names = results["feature_names"]
    centrality2 = np.array(results["centrality_order2"])
    centrality3 = np.array(results["centrality_order3"])
    triplets = results["triplet_weights"]

    if "whiten" in _PREPROCESSING_STR:
        display_names = [rf"$F_{{{i}}}$" for i in range(len(feature_names))]
    else:
        display_names = feature_names
    name_map = dict(zip(feature_names, display_names))

    triplet_labels = [" / ".join(name_map[f] for f in item["features"]) for item in triplets]
    values = [item["value"] for item in triplets]
    colors = ["#4f7d39" if v > 0 else "#b24745" for v in values]

    meta_text = (
        "---------------------------------------\n"
        rf"$p_T > {metadata['pt_cut']:.0f}$ GeV,"
        "\n"
        rf"$m_{{\mathrm{{SD}}}} \in [{metadata['mass_win_lo']:.0f},\, {metadata['mass_win_hi']:.0f}]$ GeV,"
        "\n"
        rf"{metadata['frac_top']*100:.0f}% top jets,"
        "\n"
        r"$N_\mathrm{constituents} = " + f"{metadata['num_particles']}, " + r"N_\mathrm{events} = " + f"{metadata['num_events']}" + "$"
    )

    def _draw_centrality(ax: plt.Axes) -> None:
        x = np.arange(len(display_names))
        width = 0.36
        if "standardize" in _PREPROCESSING_STR:
            ax.bar(x - width / 2, centrality2, width=width, color="#102542", label="Pairwise graph")
        #ax.bar(x - width / 2, centrality2, width=width, color="#102542", label="Pairwise graph")
        ax.bar(x + width / 2, centrality3, width=width, color="#d97925", label="3-hypergraph")
        ax.set_xticks(x, display_names, rotation=20, ha="right", fontsize=11)
        ax.set_ylabel("Centrality", fontsize=11)
        ax.set_title(f"{_PREPROCESSING_STR.capitalize()} feature centrality by order", fontsize=13)
        ax.set_ylim(0.0, 1.35 * max(np.max(centrality2), np.max(centrality3)))
        ax.legend(frameon=False, fontsize=11)
        ax.grid(axis="y", alpha=0.25)

    def _draw_triplets(ax: plt.Axes) -> None:
        ax.barh(triplet_labels, values, color=colors)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(r"$I^{(3)}_{abc}$" + f" from {_PREPROCESSING_STR} features", fontsize=13)
        ax.set_title("Distinct 3-hyperedge weights")
        ax.grid(axis="x", alpha=0.25)

    if separate:
        fig_c, ax_c = plt.subplots(figsize=(5.0, 4.2))
        _draw_centrality(ax_c)
        fig_c.text(0.5, 0.01, " ", ha="center", fontsize=7)
        fig_c.tight_layout()
        fig_c.savefig(FIGURES_DIR / f"MC_hypergraph_centrality_{_PREPROCESSING_STR}.pdf")
        fig_c.savefig(FIGURES_DIR / f"MC_hypergraph_centrality_{_PREPROCESSING_STR}.png", dpi=500)
        fig_c.savefig(WEB_DIR / f"MC_hypergraph_centrality_{_PREPROCESSING_STR}.png", dpi=500)
        plt.close(fig_c)

        fig_t, ax_t = plt.subplots(figsize=(5.5, 4.2))
        _draw_triplets(ax_t)
        fig_t.text(0.5, 0.01, " ", ha="center", fontsize=7)
        fig_t.tight_layout()
        fig_t.savefig(FIGURES_DIR / f"MC_hypergraph_triplets_{_PREPROCESSING_STR}.pdf")
        fig_t.savefig(FIGURES_DIR / f"MC_hypergraph_triplets_{_PREPROCESSING_STR}.png", dpi=500)
        fig_t.savefig(WEB_DIR / f"MC_hypergraph_triplets_{_PREPROCESSING_STR}.png", dpi=500)
        plt.close(fig_t)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
        _draw_centrality(axes[0])
        _draw_triplets(axes[1])
        fig.text(0.51, 0.02, meta_text, ha="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"MC_hypergraph_summary_{_PREPROCESSING_STR}.pdf")
        fig.savefig(FIGURES_DIR / f"MC_hypergraph_summary_{_PREPROCESSING_STR}.png", dpi=500)
        fig.savefig(WEB_DIR / f"MC_hypergraph_summary_{_PREPROCESSING_STR}.png", dpi=500)
        plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    GENERATED_DIR.mkdir(exist_ok=True, parents=True)
    WEB_DIR.mkdir(exist_ok=True, parents=True)
    #data, labels = generate_sample()
    print(f"Loading data from {DATA_PATH}...")
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
    make_kl_residuals(results, metadata)
    make_hypergraph_figure(results, metadata, separate=True)
    #import pdb;pdb.set_trace()
    
    #print(f"Wrote {(GENERATED_DIR / 'MC_results.json').relative_to(ROOT)}")
    #print(f"Wrote {(FIGURES_DIR / 'MC_kl_comparison.pdf').relative_to(ROOT)}")
    #print(f"Wrote {(FIGURES_DIR / 'MC_hypergraph_summary.pdf').relative_to(ROOT)}")
    

if __name__ == "__main__":
    main()
