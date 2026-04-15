#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import dataset_builder as db
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import argparse

import direction_definitions as dd
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import logging
from configs.observables import FEATURE_LABELS, FEATURE_PLAIN, OBSERVABLES

_RESULTS_ROOT = Path("/work/abal/triality/results")
_RUNS_ROOT    = Path("/work/abal/triality/RUNS")
_WEB_ROOT     = Path("/web/abal/public_html/plots/triality")


@dataclass(frozen=True)
class RunPaths:
    figures_dir: Path
    generated_dir: Path
    web_dir: Path
    run_dir: Path
    dataset_path: Path
    dataset_shifted_path: Path
    metapath: Path

    def makedirs(self) -> None:
        for d in (self.figures_dir, self.generated_dir, self.web_dir, self.run_dir):
            d.mkdir(exist_ok=True, parents=True)


def build_paths(comparison_type: str) -> RunPaths:
    if comparison_type == "ttbar_eft":
        run_dir = _RUNS_ROOT / "ttbar_eft"
        return RunPaths(
            figures_dir=_RESULTS_ROOT / "eft" / "figures",
            generated_dir=_RESULTS_ROOT / "eft" / "generated",
            web_dir=_WEB_ROOT / "eft",
            run_dir=run_dir,
            dataset_path=run_dir / "data_TTBar_SM.npy",
            dataset_shifted_path=run_dir / "data_TTBar_EFT.npy",
            metapath=run_dir / "metadata.npz",
        )
    elif comparison_type == "ttbar_vs_wqq":
        run_dir = _RUNS_ROOT / "ttbar_VS_wqq"
        return RunPaths(
            figures_dir=_RESULTS_ROOT / "w_vs_top" / "figures",
            generated_dir=_RESULTS_ROOT / "w_vs_top" / "generated",
            web_dir=_WEB_ROOT / "w_vs_top",
            run_dir=run_dir,
            dataset_path=run_dir / "data_WtoQQ.npy",
            dataset_shifted_path=run_dir / "data_TTBar_SM.npy",
            metapath=run_dir / "metadata.npz",
        )
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type!r}")

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

DELTA_KAPPA = 0.55
LAMBDA_CUBIC = 3.0
REDUNDANCY_BETA = 1.0
COMMON_CORE_SIZE = 3

def compute_benchmark_direction(
    dataset_base: np.ndarray,
    dataset_shifted: np.ndarray,
    regularisation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the natural parameter direction u from SM and EFT samples.

    The procedure is:
        1. Standardise both samples using SM means and standard deviations.
        2. Compute the standardised mean shift Delta = <z>_shifted - <z>_base.
        3. Estimate the correlation matrix I^(2) from the SM sample.
        4. Solve I^(2) u = Delta for u.
        5. Normalise u to unit norm.

    Parameters
    ----------
    dataset_base : np.ndarray
        SM sample of shape (N_base, D), where D is the number of observables.
    dataset_shifted : np.ndarray
        EFT sample of shape (N_shifted, D), same observable ordering as dataset_base.
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
    if dataset_base.shape[1] != dataset_shifted.shape[1]:
        raise ValueError(
            f"Observable dimensions do not match: "
            f"SM has {dataset_base.shape[1]}, EFT has {dataset_shifted.shape[1]}"
        )

    # Step 1: standardise using SM statistics
    mean_base: np.ndarray = dataset_base.mean(axis=0)
    std_base: np.ndarray = dataset_base.std(axis=0)
    std_base = np.where(std_base > 0.0, std_base, 1.0)  # guard against zero variance

    z_base: np.ndarray = (dataset_base - mean_base) / std_base
    z_shifted: np.ndarray = (dataset_shifted - mean_base) / std_base

    # Step 2: standardised mean shift
    delta: np.ndarray = z_shifted.mean(axis=0) - z_base.mean(axis=0)

    # Step 3: correlation matrix from SM (biased estimator, consistent with paper)
    fisher2: np.ndarray = np.cov(z_base, rowvar=False, bias=True)

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

def standardise_sample(sample: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None) -> np.ndarray:
    """Standardise a sample using its mean and standard deviation.
    Parameters:
    sample : np.ndarray
        Input sample of shape (N, D).
    mean : np.ndarray, optional
        Mean to use for standardisation.
    std : np.ndarray, optional
        Standard deviation to use for standardisation.
    If mean and std are not provided, they will be computed from the sample itself.
    Returns:
    np.ndarray
        Standardised sample of shape (N, D).
    """
    if mean is None and std is None:
        mean = sample.mean(axis=0)
        std = sample.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)  # guard against zero variance
    return (sample - mean) / std

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
            score = quadratic_marginal_gain(
                feature,
                selected,
                fisher2,
                benchmark_direction,
                redundancy_beta,
            )
            if lambda_cubic > 0.0:
                quad = score
                print("Quadratic gain for feature", FEATURE_PLAIN[feature], "is", quad)
                score += (
                    lambda_cubic
                    * cubic_rescale
                    * cubic_completion_gain(feature, selected, fisher3, benchmark_direction)
                )
                print("Cubic contribution for feature", FEATURE_PLAIN[feature], "is", score-quad)
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


def summarize(data: np.ndarray, benchmark_direction: np.ndarray=None, standardised: bool=False) -> dict:
    #mean = data.mean(axis=0)
    #std = data.std(axis=0)
    #zdata = (data - mean) / std
    if standardised:
        zdata = data
    else:
        zdata = standardise_sample(data)
    
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
    multi_node = LAMBDA_CUBIC * cubic_node_norm + quad_node_norm 
    common_quad_core = list(np.argsort(quad_node)[::-1][:COMMON_CORE_SIZE])
    core_remaining = sorted(set(range(len(FEATURE_LABELS))) - set(common_quad_core))
    common_cubic_core = list(np.argsort(multi_node)[::-1][:COMMON_CORE_SIZE])
    core_quadratic_ref = max(
        quadratic_marginal_gain(idx, common_quad_core, fisher2, u, REDUNDANCY_BETA)
        for idx in core_remaining
    )

    core_remaining = sorted(set(range(len(FEATURE_LABELS))) - set(common_cubic_core))
    core_cubic_ref = max(
        cubic_completion_gain(idx, common_cubic_core, fisher3, u)
        for idx in core_remaining
    )
    cubic_rescale = core_quadratic_ref / core_cubic_ref if core_cubic_ref > 0.0 else 1.0
    
    
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

    full_i2 = float(u @ (fisher2) @ u)
    full_i3 = float(np.einsum("i,j,k,ijk", u, u, u, fisher3))
    full_proj = zdata @ u
    full_kl = float(np.log(np.mean(np.exp(DELTA_KAPPA * full_proj))))
    curves = {"graph": [], "hypergraph": []}
    selections = {}
    
    for method, rank in [("graph", pair_rank), ("hypergraph", hyper_rank)]:
        for k in range(2, len(FEATURE_LABELS) + 1):
            mask = np.zeros(len(FEATURE_LABELS))
            mask[rank[:k]] = 1.0
            u_mask = u * mask
            #u_mask = np.abs(u_mask)

            i2 = float(u_mask @ (fisher2) @ u_mask)
            i3 = float(np.einsum("i,j,k,ijk", u_mask, u_mask, u_mask, fisher3))
            kl = float(np.log(np.mean(np.exp(DELTA_KAPPA * (zdata @ u_mask)))))
            curves[method].append(
                {
                    "k": k,
                    "features": rank_labels(rank[:k]),
                    "retained_i2": i2 / full_i2,
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
        #"reference_mean": mean.tolist(),
        #"reference_std": std.tolist(),
        "quadratic_node_score": quad_node_norm.tolist(),
        "cubic_node_score": cubic_node_norm.tolist(),
        "multi_node_score": multi_node.tolist(),
        "pair_rank": rank_labels(pair_rank),
        "hyper_rank": rank_labels(hyper_rank),
        "pair_rank_idx": pair_rank,
        "hyper_rank_idx": hyper_rank,
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


def make_constraint_figure(results: dict, paths: RunPaths) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    for method, color in [("graph", "#9b2c2c"), ("hypergraph", "#1f4f82")]:
        curve = results["curves"][method]
        ks = [item["k"] for item in curve]
        retained_i2 = [item["retained_i2"] for item in curve]
        retained_i3 = [item["retained_i3"] for item in curve]
        retained_kl = [item["retained_kl"] for item in curve]
        rel_limit = [item["relative_limit"] for item in curve]

        label = "Pairwise graph" if method == "graph" else "Fisher hypergraph"
        axes[0].plot(ks, retained_i2, color=color, linewidth=2.2, label=f"{label}: $R_2$")
        # axes[0].plot(
        #     ks,
        #     retained_kl,
        #     color=color,
        #     linewidth=2.0,
        #     linestyle="--",
        #     label=f"{label}: $R_{{\\mathrm{{KL}}}}$",
        # )
        axes[1].plot(ks, rel_limit, color=color, linewidth=2.2, label=label)

    axes[0].set_xlabel("Retained observables")
    axes[0].set_ylabel("Retained fraction")
    axes[0].set_xticks(range(2, len(FEATURE_LABELS) + 1))
    axes[0].set_ylim(0.0, 2.05)
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
    fig.savefig(paths.figures_dir / "eft_constraint_summary.pdf")
    fig.savefig(paths.web_dir / "eft_constraint_summary.png", dpi=600)
    plt.close(fig)


def make_hypergraph_figure(results: dict, paths: RunPaths) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17.4,8.3))

    x = np.arange(len(FEATURE_LABELS))
    width = 0.38
    quad = np.array(results["quadratic_node_score"])
    multi = np.array(results["multi_node_score"])
    cubic = np.array(results["cubic_node_score"])
    axes[0].bar(x - width / 2, quad, width=width, color="#9b2c2c", label="Pairwise graph score")
    axes[0].bar(x + width / 2, cubic, width=width, color="#1f4f82", label="Cubic-hypergraph score")
    axes[0].set_xticks(x, FEATURE_LABELS, rotation=22, ha="right")
    axes[0].set_ylabel("Operator-aligned node score")
    axes[0].set_title("Feature ranking")
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
    fig.savefig(paths.figures_dir / "eft_hypergraph_summary.pdf")
    fig.savefig(paths.web_dir / "eft_hypergraph_summary.png", dpi=600)
    plt.close(fig)


_YAML = {
    "ttbar_vs_wqq": ("configs/wqq.yaml",     "configs/ttbar_sm.yaml"),
    "ttbar_eft":    ("configs/ttbar_sm.yaml", "configs/ttbar_eft.yaml"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EFT constraint summary and hypergraph figures.")
    parser.add_argument("--load", action="store_true", help="Load precomputed datasets instead of building them from scratch.")
    parser.add_argument("--type", choices=["ttbar_vs_wqq", "ttbar_eft"], default="ttbar_vs_wqq", help="Type of comparison to perform.")
    args = parser.parse_args()

    paths = build_paths(args.type)
    paths.makedirs()
    data_yaml, shifted_yaml = _YAML[args.type]

    if args.load:
        logging.info(f"Loading datasets from {paths.dataset_path} and {paths.dataset_shifted_path}")
        dataset = np.load(paths.dataset_path)
        dataset_shifted = np.load(paths.dataset_shifted_path)
    else:
        logging.info("Building datasets from scratch.")
        dataset, _, _, _ = db.build_dataset(config_path=data_yaml, observables=OBSERVABLES, num_jets=100000, num_particles=25, seed=42)
        dataset_shifted, _, _, _ = db.build_dataset(config_path=shifted_yaml, observables=OBSERVABLES, num_jets=100000, num_particles=25, seed=42)
        np.save(paths.dataset_path, dataset)
        np.save(paths.dataset_shifted_path, dataset_shifted)
        logging.info(f"Saved datasets to {paths.dataset_path} and {paths.dataset_shifted_path} for future usage")

    joint_dataset = np.concatenate([dataset, dataset_shifted], axis=0)
    joint_mean = joint_dataset.mean(axis=0)
    joint_std = joint_dataset.std(axis=0)
    dataset = (dataset - joint_mean) / joint_std
    dataset_shifted = (dataset_shifted - joint_mean) / joint_std
    u_benchmark = dataset_shifted.mean(axis=0) - dataset.mean(axis=0)
    u_benchmark /= np.linalg.norm(u_benchmark)

    results = summarize(joint_dataset, benchmark_direction=u_benchmark, standardised=False)
    hyper_rank_idx = results["hyper_rank_idx"]
    pair_rank_idx = results["pair_rank_idx"]
    np.savez(paths.metapath, benchmark_direction=u_benchmark, hyper_rank_idx=hyper_rank_idx, pair_rank_idx=pair_rank_idx)
    logging.info(f"Saved benchmark direction and feature ranks to {paths.metapath} for N = {len(joint_dataset)} total samples and D = {len(FEATURE_LABELS)} features.")

    make_constraint_figure(results, paths)
    make_hypergraph_figure(results, paths)

    print(f"Wrote {paths.figures_dir / 'eft_constraint_summary.pdf'}")
    print(f"Wrote {paths.figures_dir / 'eft_hypergraph_summary.pdf'}")


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


