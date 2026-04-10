#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import dataset_builder as db
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from configs.observables import FEATURE_LABELS, FEATURE_PLAIN, OBSERVABLES
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/work/abal/triality/results')
FIGURES_DIR = ROOT / "figures"
GENERATED_DIR = ROOT / "generated"
WEB_DIR = Path("/web/abal/public_html/plots/triality/EFT")


# FEATURE_LABELS = [
#     r"$\mathrm{EEC}_1$",
#     r"$\mathrm{EEC}_2$",
#     r"$\mathrm{EEC}_3$",
#     r"$\mathrm{EEC}_4$",
#     r"$e_2^{(1)}$",
#     r"$e_2^{(2)}$",
#     r"$e_3^{(1)}$",
#     r"$e_3^{(2)}$",
# ]
# FEATURE_PLAIN = [
#     "EEC_1",
#     "EEC_2",
#     "EEC_3",
#     "EEC_4",
#     "e2^(1)",
#     "e2^(2)",
#     "e3^(1)",
#     "e3^(2)",
# ]


@dataclass(frozen=True)
class EventModel:
    cores: np.ndarray
    core_probs: np.ndarray
    widths: np.ndarray
    group_weights: np.ndarray
    alpha: float


REFERENCE_MODEL = EventModel(
    cores=np.array(
        [
            [0.34, 0.00],
            [-0.17, 0.29],
            [-0.17, -0.29],
            [0.00, 0.00],
        ]
    ),
    core_probs=np.array([0.33, 0.29, 0.26, 0.12]),
    widths=np.array([0.045, 0.050, 0.052, 0.18]),
    group_weights=np.array([0.37, 0.31, 0.22, 0.10]),
    alpha=0.78,
)


BENCHMARK_DIRECTION = np.array(
    [-0.287, -0.221, 0.600, 0.499, 0.197, 0.133, 0.253, 0.373],
    dtype=float,
)
BENCHMARK_DIRECTION /= np.linalg.norm(BENCHMARK_DIRECTION)

DELTA_KAPPA = 0.05
LAMBDA_CUBIC = 3.0
REDUNDANCY_BETA = 1.0
COMMON_CORE_SIZE = 2

def whiten(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0)
    centered = data - mean
    cov = np.cov(centered, rowvar=False, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    cov_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    zdata = centered @ cov_inv_sqrt
    return zdata

def compute_benchmark_direction(
    data_sm: np.ndarray,
    data_eft: np.ndarray,
    regularisation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the natural parameter direction u from SM and EFT samples.

    The procedure is:
        1. Standardise both samples using SM means and standard deviations.
        2. Compute the standardised mean shift Delta = <z>_EFT - <z>_SM.
        3. Estimate the correlation matrix I^(2) from the SM sample.
        4. Solve I^(2) u = Delta for u.
        5. Normalise u to unit norm.

    Parameters
    ----------
    data_sm : np.ndarray
        SM sample of shape (N_sm, D), where D is the number of observables.
    data_eft : np.ndarray
        EFT sample of shape (N_eft, D), same observable ordering as data_sm.
    regularisation : float, optional
        Ridge parameter epsilon added to the diagonal of I^(2) before
        inversion, i.e. I^(2) -> I^(2) + epsilon * I. Use a small positive
        value (e.g. 1e-6) if the correlation matrix is near-singular.
        Default is 0.0 (no regularisation).

    Returns
    -------
    u : np.ndarray
        Unit-normalised natural parameter direction, shape (D,).
    delta : np.ndarray
        Standardised mean shift vector, shape (D,).
    fisher2 : np.ndarray
        Correlation matrix I^(2) from SM sample, shape (D, D).
    u_unnormalised : np.ndarray
        Raw solution (I^(2))^{-1} Delta before normalisation, shape (D,).

    Raises
    ------
    ValueError
        If the two samples have different numbers of observables.
    """
    if data_sm.shape[1] != data_eft.shape[1]:
        raise ValueError(
            f"Observable dimensions do not match: "
            f"SM has {data_sm.shape[1]}, EFT has {data_eft.shape[1]}"
        )

    # Step 1: standardise using SM statistics
    mean_sm: np.ndarray = data_sm.mean(axis=0)
    std_sm: np.ndarray = data_sm.std(axis=0)
    std_sm = np.where(std_sm > 0.0, std_sm, 1.0)  # guard against zero variance

    z_sm: np.ndarray = (data_sm - mean_sm) / std_sm
    z_eft: np.ndarray = (data_eft - mean_sm) / std_sm

    # Step 2: standardised mean shift
    delta: np.ndarray = z_eft.mean(axis=0) - z_sm.mean(axis=0)

    # Step 3: correlation matrix from SM (biased estimator, consistent with paper)
    fisher2: np.ndarray = np.cov(z_sm, rowvar=False, bias=True)

    # Step 4: solve I^(2) u = Delta, with optional ridge regularisation
    if regularisation > 0.0:
        fisher2_reg = fisher2 + regularisation * np.eye(fisher2.shape[0])
    else:
        fisher2_reg = fisher2

    u_unnormalised: np.ndarray = np.linalg.solve(fisher2_reg, delta)
    #u_unnormalised: np.ndarray = delta
    #u_unnormalised: np.ndarray = np.linalg.pinv(fisher2_reg) @ delta
    # Step 5: normalise to unit norm
    norm: float = float(np.linalg.norm(u_unnormalised))
    if norm < 1e-15:
        raise ValueError("Benchmark direction has near-zero norm; samples may be indistinguishable.")

    u: np.ndarray = u_unnormalised / norm

    return u, delta, fisher2, u_unnormalised

def sample_event(
    rng: np.random.Generator, model: EventModel, n_constituents: int
) -> tuple[np.ndarray, np.ndarray]:
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
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    iu = np.triu_indices(len(z), k=1)
    dij = distances[iu]
    zij = z[iu[0]] * z[iu[1]]

    bins = [(0.03, 0.08), (0.08, 0.16), (0.16, 0.30), (0.30, 0.80)]
    values = [np.sum(zij * ((dij > lo) & (dij <= hi))) for lo, hi in bins]
    values.append(np.sum(zij * dij))
    values.append(np.sum(zij * dij**2))

    e3_beta1 = 0.0
    e3_beta2 = 0.0
    for i, j, k in combinations(range(len(z)), 3):
        prod = distances[i, j] * distances[i, k] * distances[j, k]
        weight = z[i] * z[j] * z[k]
        e3_beta1 += weight * prod
        e3_beta2 += weight * prod**2

    values.extend([e3_beta1, e3_beta2])
    return np.array(values, dtype=float)


def generate_reference_sample(
    *, seed: int = 11, n_events: int = 12_000, n_constituents: int = 18
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.empty((n_events, len(FEATURE_LABELS)), dtype=float)
    for idx in range(n_events):
        data[idx] = event_features(*sample_event(rng, REFERENCE_MODEL, n_constituents))
    return data


def rank_labels(rank: np.ndarray) -> list[str]:
    return [FEATURE_PLAIN[idx] for idx in rank]


def quadratic_marginal_gain(
    candidate: int,
    selected: list[int],
    fisher2: np.ndarray,
    benchmark_direction: np.ndarray,
    redundancy_beta: float,
) -> float:
    if selected:
        redundancy = float(np.mean([abs(fisher2[candidate, idx]) for idx in selected]))
        cross_term = sum(fisher2[candidate, idx] * benchmark_direction[idx] for idx in selected)
    else:
        redundancy = 0.0
        cross_term = 0.0

    self_term = benchmark_direction[candidate] ** 2 * fisher2[candidate, candidate]
    return float(
        (abs(self_term) + abs(2.0 * benchmark_direction[candidate] * cross_term))
        / (1.0 + redundancy_beta * redundancy)
    )


def cubic_completion_gain(
    candidate: int,
    selected: list[int],
    fisher3: np.ndarray,
    benchmark_direction: np.ndarray,
) -> float:
    if len(selected) < 2:
        return 0.0

    gain = 0.0
    for idx_b, idx_c in combinations(selected, 2):
        gain += abs(
            fisher3[candidate, idx_b, idx_c]
            * benchmark_direction[candidate]
            * benchmark_direction[idx_b]
            * benchmark_direction[idx_c]
        )
    return float(gain)


def greedy_selection_order(
    *,
    initial_core: list[int],
    fisher2: np.ndarray,
    fisher3: np.ndarray,
    benchmark_direction: np.ndarray,
    redundancy_beta: float,
    lambda_cubic: float,
    cubic_rescale: float,
) -> np.ndarray:
    selected = list(initial_core)
    remaining = set(range(len(benchmark_direction))) - set(selected)

    while remaining:
        best_feature = None
        best_score = None
        for feature in sorted(remaining):
            qgain = quadratic_marginal_gain(
                feature,
                selected,
                fisher2,
                benchmark_direction,
                redundancy_beta,
            )
            score = qgain
            if lambda_cubic > 0.0:
                cgain = (
                    lambda_cubic
                    * cubic_rescale
                    * cubic_completion_gain(feature, selected, fisher3, benchmark_direction)
                )
                score += cgain
                print(feature, np.round(qgain,3), np.round(cgain,3))
            if (
                best_score is None
                or score > best_score + 1.0e-12
                or (abs(score - best_score) <= 1.0e-12 and feature < best_feature)
            ):
                best_feature = feature
                best_score = score

        selected.append(best_feature)
        remaining.remove(best_feature)

    return np.array(selected, dtype=int)


def summarize(data: np.ndarray, benchmark_direction: np.ndarray=None) -> dict:
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    zdata = (data - mean) / std

    fisher2 = np.cov(zdata, rowvar=False, bias=True)
    fisher3 = np.einsum("ni,nj,nk->ijk", zdata, zdata, zdata) / len(zdata)
    if benchmark_direction is None:
        print("No benchmark direction provided; using default physics-motivated values.")
        u = BENCHMARK_DIRECTION.copy()
    else:
        u = benchmark_direction
    
    quad_node = np.abs(u * (fisher2 @ u))
    cubic_contract = np.einsum("abc,b,c->a", fisher3, u, u)
    cubic_node = np.abs(u * cubic_contract)

    quad_node_norm = quad_node / quad_node.sum()
    cubic_node_norm = cubic_node / cubic_node.sum()
    multi_node = quad_node_norm + LAMBDA_CUBIC * cubic_node_norm
    common_quad_core = list(np.argsort(quad_node)[::-1][:COMMON_CORE_SIZE])
    core_remaining = sorted(set(range(len(FEATURE_LABELS))) - set(common_quad_core))
    core_quadratic_ref = max(
        quadratic_marginal_gain(idx, common_quad_core, fisher2, u, REDUNDANCY_BETA)
        for idx in core_remaining
    )
    # DEFINE CUBIC CORE
    common_cubic_core = list(np.argsort(multi_node)[::-1][:COMMON_CORE_SIZE])
    #common_cubic_core = common_quad_core
    core_remaining = sorted(set(range(len(FEATURE_LABELS))) - set(common_cubic_core))
    core_cubic_ref = max(
        cubic_completion_gain(idx, common_cubic_core, fisher3, u)
        for idx in core_remaining
    )
    cubic_rescale = core_quadratic_ref / core_cubic_ref if core_cubic_ref > 0.0 else 1.0
    import pdb;pdb.set_trace()
    
    pair_rank = greedy_selection_order(
        initial_core=common_quad_core,
        fisher2=fisher2,
        fisher3=fisher3,
        benchmark_direction=u,
        redundancy_beta=REDUNDANCY_BETA,
        lambda_cubic=0.0,
        cubic_rescale=cubic_rescale,
    )
    hyper_rank = greedy_selection_order(
        initial_core=common_cubic_core,
        fisher2=fisher2,
        fisher3=fisher3,
        benchmark_direction=u,
        redundancy_beta=REDUNDANCY_BETA,
        lambda_cubic=LAMBDA_CUBIC,
        cubic_rescale=cubic_rescale,
    )
    import pdb;pdb.set_trace()
    full_i2 = float(u @ fisher2 @ u)
    full_i3 = float(np.einsum("i,j,k,ijk", u, u, u, fisher3))
    full_proj = zdata @ u
    full_kl = float(np.log(np.mean(np.exp(DELTA_KAPPA * full_proj))))
    T = np.einsum("ijk,i,j,k->ijk", fisher3, u, u, u)
    r3_den = np.einsum("ijk,ijk->", T, T)
    curves = {"graph": [], "hypergraph": []}
    selections = {}
    for method, rank in [("graph", pair_rank), ("hypergraph", hyper_rank)]:
        for k in range(2, len(FEATURE_LABELS) + 1):
            mask = np.zeros(len(FEATURE_LABELS))
            mask[rank[:k]] = 1.0
            u_mask = u * mask
            r3_u = np.einsum("ijk,ijk,i,j,k->", T, T, mask, mask, mask) / r3_den if r3_den > 0.0 else 0.0

            i2 = float(u_mask @ fisher2 @ u_mask)
            i3 = float(np.einsum("i,j,k,ijk", u_mask, u_mask, u_mask, fisher3))
            kl = float(np.log(np.mean(np.exp(DELTA_KAPPA * (zdata @ u_mask)))))
            curves[method].append(
                {
                    "k": k,
                    "features": rank_labels(rank[:k]),
                    "retained_i2": i2 / full_i2,
                    "squared_i3": r3_u,
                    "retained_i3": i3 / full_i3,
                    "retained_kl": kl / full_kl,
                    "relative_limit": float(np.sqrt(full_i2 / i2)),
                }
            )
        k4 = curves[method][2]
        selections[method] = {
            "features": k4["features"],
            "retained_i2": k4["retained_i2"],
            "retained_i3": k4["retained_i3"],
            "retained_kl": k4["retained_kl"],
            "relative_limit": k4["relative_limit"],
        }

    triplets = []
    for i, j, k in combinations(range(len(FEATURE_LABELS)), 3):
        raw_weight = float(fisher3[i, j, k])
        aligned_score = float(abs(raw_weight * u[i] * u[j] * u[k]))
        triplets.append(
            {
                "features": [FEATURE_PLAIN[i], FEATURE_PLAIN[j], FEATURE_PLAIN[k]],
                "weight": raw_weight,
                "aligned_score": aligned_score,
            }
        )
    triplets.sort(key=lambda item: item["aligned_score"], reverse=True)

    results = {
        "feature_labels": FEATURE_LABELS,
        "feature_plain": FEATURE_PLAIN,
        "benchmark_direction": u.tolist(),
        "delta_kappa": DELTA_KAPPA,
        "lambda_cubic": LAMBDA_CUBIC,
        "redundancy_beta": REDUNDANCY_BETA,
        "common_quad_core_size": COMMON_CORE_SIZE,
        "common_quad_core": rank_labels(np.array(common_quad_core, dtype=int)),
        "cubic_rescale": cubic_rescale,
        "selection_rule": {
            "graph": "common quadratic core plus greedy quadratic marginal gain",
            "hypergraph": "common quadratic core plus greedy quadratic and triplet-completion gain",
        },
        "reference_mean": mean.tolist(),
        "reference_std": std.tolist(),
        "quadratic_node_score": quad_node.tolist(),
        "cubic_node_score": cubic_node.tolist(),
        "multi_node_score": multi_node.tolist(),
        "pair_rank": rank_labels(pair_rank),
        "hyper_rank": rank_labels(hyper_rank),
        "full_coefficients": {
            "i2": full_i2,
            "i3": full_i3,
            "exact_kl": full_kl,
        },
        "curves": curves,
        "k4_selections": selections,
        "top_triplets": triplets[:8],
    }
    return results


def make_constraint_figure(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    for method, color in [("graph", "#9b2c2c"), ("hypergraph", "#1f4f82")]:
        curve = results["curves"][method]
        ks = [item["k"] for item in curve]
        retained_i2 = [item["retained_i2"] for item in curve]
        retained_i3 = [item["squared_i3"] for item in curve]
        retained_kl = [item["retained_kl"] for item in curve]
        rel_limit = [item["relative_limit"] for item in curve]

        label = "Pairwise graph" if method == "graph" else "Fisher hypergraph"
        axes[0].plot(ks, retained_i2, color=color, linewidth=2.2, label=f"{label}: $R_2$")
        axes[0].plot(ks, retained_i3, color=color, linewidth=2.2, linestyle="-.", label=f"{label}: $R_3^{(2)}$")
        axes[0].plot(
            ks,
            retained_kl,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=f"{label}: $R_{{\\mathrm{{KL}}}}$",
        )
        axes[1].plot(ks, rel_limit, color=color, linewidth=2.2, label=label)

    axes[0].set_xlabel("Retained observables")
    axes[0].set_ylabel("Retained fraction")
    axes[0].set_xticks(range(2, len(FEATURE_LABELS) + 1))
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Information retained under compression")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Retained observables")
    axes[1].set_ylabel(r"$\Delta\kappa_{95}/\Delta\kappa_{95}^{\rm full}$")
    axes[1].set_xticks(range(2, len(FEATURE_LABELS) + 1))
    axes[1].set_ylim(0.95, 2.25)
    axes[1].set_title("Relative local EFT interval")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eft_constraint_summary.pdf")
    fig.savefig(WEB_DIR / "eft_constraint_summary.png", dpi=600)
    plt.close(fig)


def make_hypergraph_figure(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))

    x = np.arange(len(FEATURE_LABELS))
    width = 0.38
    quad = np.array(results["quadratic_node_score"])
    multi = np.array(results["multi_node_score"])

    axes[0].bar(x - width / 2, quad, width=width, color="#9b2c2c", label="Pairwise graph score")
    axes[0].bar(x + width / 2, multi, width=width, color="#1f4f82", label="Multi-order score")
    axes[0].set_xticks(x, FEATURE_LABELS, rotation=22, ha="right")
    axes[0].set_ylabel("Operator-aligned node score")
    axes[0].set_title("Feature ranking in the $O_{tG}$ benchmark")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    triplets = results["top_triplets"][:6]
    labels = [" / ".join(item["features"]) for item in triplets]
    values = [item["aligned_score"] for item in triplets]
    colors = ["#1f4f82" if item["weight"] > 0 else "#d97925" for item in triplets]
    axes[1].barh(labels, values, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"$|u_a u_b u_c\,I^{(3)}_{abc}|$")
    axes[1].set_title("Leading EFT-sensitive 3-hyperedges")
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eft_hypergraph_summary.pdf")
    fig.savefig(WEB_DIR / "eft_hypergraph_summary.png", dpi=600)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    GENERATED_DIR.mkdir(exist_ok=True)
    WEB_DIR.mkdir(exist_ok=True)
    observables={
        "EEC_1": [0.03, 0.08],
        "EEC_2":   [0.08, 0.16],
        "EEC_3":   [0.16, 0.30],
        "EEC_4":   [0.30, 0.80],
        "ECF_e2b1":   (2, 1),        # e_2^(1)
        "ECF_e2b2":   (2, 2),        # e_2^(2)
        "ECF_e3b1":   (3, 1),        # e_3^(1)
        "ECF_e3b2":   (3, 2),        # e_3^(2)   
    }
    data, _, _, _ = db.build_dataset(config_path="configs/data_sm.yaml",observables=OBSERVABLES, num_jets=200000, num_particles=50,seed=42) 
    data_eft, _, _, _ = db.build_dataset(config_path="configs/data_eft.yaml", observables=OBSERVABLES, num_jets=200000, num_particles=18,seed=42)
    # u_benchmark, delta, fisher2, _ = compute_benchmark_direction(data, data_eft, regularisation=0.0)
    fisher = np.cov(data, rowvar=False, bias=True)
    u_benchmark = (data_eft.mean(axis=0) - data.mean(axis=0))/ data.std(axis=0)
    u_whitened = np.linalg.solve(fisher, u_benchmark)
    u_benchmark /= np.linalg.norm(u_benchmark)
    u_whitened /= np.linalg.norm(u_whitened)
    results = summarize(data, benchmark_direction=u_whitened)

    with (GENERATED_DIR / "eft_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    make_constraint_figure(results)
    make_hypergraph_figure(results)

    print(f"Wrote {(GENERATED_DIR / 'eft_results.json')}")
    print(f"Wrote {(FIGURES_DIR / 'eft_constraint_summary.pdf')}")
    print(f"Wrote {(FIGURES_DIR / 'eft_hypergraph_summary.pdf')}")


def signal_decomposition(fisher2, delta, label):
    eigenvalues, eigenvectors = np.linalg.eigh(fisher2)
    projections = eigenvectors.T @ delta
    fi_components = projections**2 / eigenvalues
    fi_total = fi_components.sum()
    
    print(f"\n=== Signal decomposition: {label} ===")
    for i, (ev, proj, fi) in enumerate(
        zip(eigenvalues, projections, fi_components)
    ):
        print(f"  mode {i}: lambda={ev:.6f}, "
              f"|proj|={abs(proj):.6f}, "
              f"FI_contrib={fi:.6f}, "
              f"frac={fi/fi_total:.4f}")
    
    sorted_fracs = np.sort(fi_components / fi_total)[::-1]
    cumulative = np.cumsum(sorted_fracs)
    print(f"\n  Cumulative: {', '.join(f'{c:.3f}' for c in cumulative)}")

if __name__ == "__main__":
    main()


