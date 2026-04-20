#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_eft_results import EventModel, sample_event

from config.observables import OBSERVABLES
import data.build_dataset as db

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures/basis_design"
GENERATED_DIR = ROOT / "generated/basis_design"


EEC_EDGES = [
    0.02,
    0.035,
    0.05,
    0.065,
    0.08,
    0.10,
    0.12,
    0.145,
    0.17,
    0.20,
    0.24,
    0.28,
    0.33,
    0.39,
    0.46,
    0.54,
    0.63,
    0.73,
    0.85,
]
EEC_BINS = list(zip(EEC_EDGES[:-1], EEC_EDGES[1:]))
E2_BETAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
E3_BETAS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

FEATURE_LABELS = [rf"$\mathrm{{EEC}}_{idx}$" for idx in range(1, len(EEC_BINS) + 1)]
FEATURE_LABELS.extend(rf"$e_2^{{({beta:g})}}$" for beta in E2_BETAS)
FEATURE_LABELS.extend(rf"$e_3^{{({beta:g})}}$" for beta in E3_BETAS)

FEATURE_PLAIN = [f"EEC_{idx}" for idx in range(1, len(EEC_BINS) + 1)]
FEATURE_PLAIN.extend(f"e2^({beta:g})" for beta in E2_BETAS)
FEATURE_PLAIN.extend(f"e3^({beta:g})" for beta in E3_BETAS)

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

DEFORMATION_MODELS = {
    "wide_angle": EventModel(
        cores=np.array(
            [
                [0.34, 0.00],
                [-0.17, 0.29],
                [-0.17, -0.29],
                [0.00, 0.00],
            ]
        ),
        core_probs=np.array([0.31, 0.27, 0.24, 0.18]),
        widths=np.array([0.048, 0.053, 0.055, 0.24]),
        group_weights=np.array([0.34, 0.28, 0.21, 0.17]),
        alpha=0.72,
    ),
    "collinear": EventModel(
        cores=np.array(
            [
                [0.34, 0.00],
                [-0.17, 0.29],
                [-0.17, -0.29],
                [0.00, 0.00],
            ]
        ),
        core_probs=np.array([0.35, 0.30, 0.27, 0.08]),
        widths=np.array([0.032, 0.036, 0.038, 0.14]),
        group_weights=np.array([0.39, 0.32, 0.23, 0.06]),
        alpha=0.88,
    ),
    "open_angle": EventModel(
        cores=np.array(
            [
                [0.38, 0.00],
                [-0.19, 0.33],
                [-0.19, -0.33],
                [0.00, 0.01],
            ]
        ),
        core_probs=np.array([0.33, 0.29, 0.26, 0.12]),
        widths=np.array([0.046, 0.051, 0.053, 0.18]),
        group_weights=np.array([0.36, 0.31, 0.22, 0.11]),
        alpha=0.78,
    ),
    "asymmetric": EventModel(
        cores=np.array(
            [
                [0.34, 0.00],
                [-0.17, 0.29],
                [-0.17, -0.29],
                [0.00, 0.00],
            ]
        ),
        core_probs=np.array([0.39, 0.23, 0.23, 0.15]),
        widths=np.array([0.044, 0.055, 0.055, 0.18]),
        group_weights=np.array([0.43, 0.24, 0.19, 0.14]),
        alpha=0.76,
    ),
    "bridge": EventModel(
        cores=np.array(
            [
                [0.34, 0.00],
                [-0.17, 0.29],
                [-0.17, -0.29],
                [0.08, 0.18],
                [0.00, 0.00],
            ]
        ),
        core_probs=np.array([0.30, 0.26, 0.24, 0.08, 0.12]),
        widths=np.array([0.046, 0.051, 0.053, 0.075, 0.18]),
        group_weights=np.array([0.34, 0.29, 0.21, 0.07, 0.09]),
        alpha=0.74,
    ),
    "fourth_prong": EventModel(
        cores=np.array(
            [
                [0.34, 0.00],
                [-0.17, 0.29],
                [-0.17, -0.29],
                [0.06, 0.36],
                [0.00, 0.00],
            ]
        ),
        core_probs=np.array([0.29, 0.25, 0.22, 0.12, 0.12]),
        widths=np.array([0.046, 0.051, 0.053, 0.06, 0.18]),
        group_weights=np.array([0.32, 0.28, 0.21, 0.09, 0.10]),
        alpha=0.75,
    ),
}

TASK_TITLES = {
    "wide_angle": "Wide-angle enhancement",
    "collinear": "Collinear hardening",
    "open_angle": "Opening-angle shift",
    "asymmetric": "Prong asymmetry",
    "bridge": "Inter-prong bridge",
    "fourth_prong": "Soft fourth prong",
}

REFERENCE_SEED = 11
TASK_SEED_BASE = 20
N_EVENTS_REFERENCE = 2_800
N_EVENTS_TASK = 2_800
N_CONSTITUENTS = 20
DELTA_PROBE = 0.55
LAMBDA_BASIS = 1.5
CURVE_KS = list(range(6, 21, 2))
SUMMARY_K = 16


def rank_labels(rank: np.ndarray | list[int]) -> list[str]:
    return [FEATURE_PLAIN[idx] for idx in rank]


def quadratic_subset_fraction(
    selected: set[int],
    quadratic_terms: np.ndarray,
    full_quadratic: np.ndarray,
) -> float:
    if not selected:
        return 0.0

    indices = np.array(sorted(selected), dtype=int)
    fractions = []
    for task_idx in range(len(full_quadratic)):
        subset_value = quadratic_terms[task_idx][np.ix_(indices, indices)].sum()
        fractions.append(subset_value / full_quadratic[task_idx])
    return float(np.mean(fractions))


def aligned_triplet_coverage(
    selected: set[int],
    aligned_triplets: list[dict[str, object]],
    total_weight: float,
) -> float:
    if total_weight <= 0.0:
        return 0.0

    covered_weight = 0.0
    for item in aligned_triplets:
        i, j, k = item["indices"]
        if i in selected and j in selected and k in selected:
            covered_weight += float(item["aligned_score"])
    return float(covered_weight / total_weight)


def exact_retained_kl(
    selected: set[int],
    zref: np.ndarray,
    task_directions: np.ndarray,
    full_exact_kl: np.ndarray,
) -> tuple[float, float]:
    if not selected:
        return 0.0, 0.0

    mask = np.zeros(zref.shape[1], dtype=float)
    mask[list(selected)] = 1.0

    retained = []
    for task_idx, direction in enumerate(task_directions):
        subset_direction = direction * mask
        subset_kl = float(np.log(np.mean(np.exp(DELTA_PROBE * (zref @ subset_direction)))))
        retained.append(subset_kl / full_exact_kl[task_idx])

    return float(np.mean(retained)), float(np.median(retained))


def greedy_selection(
    *,
    n_features: int,
    max_k: int,
    objective,
) -> list[int]:
    selected: list[int] = []
    remaining = set(range(n_features))

    for _ in range(max_k):
        best_feature = None
        best_score = None
        for feature in sorted(remaining):
            score = objective(set(selected) | {feature})
            if (
                best_score is None
                or score > best_score + 1.0e-12
                or (abs(score - best_score) <= 1.0e-12 and feature < best_feature)
            ):
                best_score = score
                best_feature = feature

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


def event_features(z: np.ndarray, xy: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    iu = np.triu_indices(len(z), k=1)
    dij = distances[iu]
    zij = z[iu[0]] * z[iu[1]]

    values = [np.sum(zij * ((dij > lo) & (dij <= hi))) for lo, hi in EEC_BINS]
    values.extend(np.sum(zij * dij**beta) for beta in E2_BETAS)

    e3_values = [0.0 for _ in E3_BETAS]
    for i, j, k in combinations(range(len(z)), 3):
        product = distances[i, j] * distances[i, k] * distances[j, k]
        weight = z[i] * z[j] * z[k]
        for idx, beta in enumerate(E3_BETAS):
            e3_values[idx] += weight * product**beta

    values.extend(e3_values)
    return np.array(values, dtype=float)


def generate_sample(model: EventModel, *, seed: int, n_events: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.empty((n_events, len(FEATURE_LABELS)), dtype=float)
    for idx in range(n_events):
        data[idx] = event_features(*sample_event(rng, model, N_CONSTITUENTS))
    return data


def summarize() -> dict:
    reference = generate_sample(
        REFERENCE_MODEL,
        seed=REFERENCE_SEED,
        n_events=N_EVENTS_REFERENCE,
    )
    reference, _, _, _ = db.build_dataset(config_path="./configs/ttbar_sm.yaml", observables=OBSERVABLES, num_jets=100000, num_particles=25, seed=42)
        
    reference_mean = reference.mean(axis=0)
    reference_std = reference.std(axis=0)
    zref = (reference - reference_mean) / reference_std

    fisher2 = np.cov(zref, rowvar=False, bias=True)
    fisher3 = np.einsum("ni,nj,nk->ijk", zref, zref, zref) / len(zref)

    task_directions = []
    task_direction_dict = {}
    for idx, (task_name, model) in enumerate(DEFORMATION_MODELS.items()):
        sample = generate_sample(model, seed=TASK_SEED_BASE + idx, n_events=N_EVENTS_TASK)
        delta = (sample.mean(axis=0) - reference_mean) / reference_std
        delta /= np.linalg.norm(delta)
        task_directions.append(delta)
        task_direction_dict[task_name] = delta.tolist()
    task_directions = np.array(task_directions)

    full_exact_kl = np.array(
        [float(np.log(np.mean(np.exp(DELTA_PROBE * (zref @ direction))))) for direction in task_directions]
    )

    quadratic_terms = np.empty((len(task_directions), len(FEATURE_LABELS), len(FEATURE_LABELS)), dtype=float)
    for task_idx, direction in enumerate(task_directions):
        quadratic_terms[task_idx] = np.outer(direction, direction) * fisher2
    full_quadratic = np.array([terms.sum() for terms in quadratic_terms])

    quadratic_node_score = np.mean(
        np.abs(task_directions * (task_directions @ fisher2)),
        axis=0,
    )
    cubic_contract = np.einsum("abc,mb,mc->ma", fisher3, task_directions, task_directions)
    cubic_node_score = np.mean(np.abs(task_directions * cubic_contract), axis=0)

    multi_node_score = (
        quadratic_node_score / quadratic_node_score.sum()
        + LAMBDA_BASIS * cubic_node_score / cubic_node_score.sum()
    )
    aligned_triplets = []
    for i, j, k in combinations(range(len(FEATURE_LABELS)), 3):
        aligned_score = float(
            np.sum(np.abs(task_directions[:, i] * task_directions[:, j] * task_directions[:, k] * fisher3[i, j, k]))
        )
        aligned_triplets.append(
            {
                "indices": [i, j, k],
                "features": [FEATURE_PLAIN[i], FEATURE_PLAIN[j], FEATURE_PLAIN[k]],
                "aligned_score": aligned_score,
                "weight": float(fisher3[i, j, k]),
            }
        )
    aligned_triplets.sort(key=lambda item: item["aligned_score"], reverse=True)
    total_aligned_triplet_weight = sum(item["aligned_score"] for item in aligned_triplets)

    max_k = max(CURVE_KS)
    graph_rank = greedy_selection(
        n_features=len(FEATURE_LABELS),
        max_k=max_k,
        objective=lambda selected: quadratic_subset_fraction(selected, quadratic_terms, full_quadratic),
    )
    hyper_rank = greedy_selection(
        n_features=len(FEATURE_LABELS),
        max_k=max_k,
        objective=lambda selected: (
            quadratic_subset_fraction(selected, quadratic_terms, full_quadratic)
            + LAMBDA_BASIS * aligned_triplet_coverage(selected, aligned_triplets, total_aligned_triplet_weight)
        ),
    )

    curves = {"graph": [], "hypergraph": []}
    summary_selection = {}
    for method, rank in [("graph", graph_rank), ("hypergraph", hyper_rank)]:
        for k in CURVE_KS:
            selected = set(rank[:k])
            mean_retained_kl, median_retained_kl = exact_retained_kl(
                selected,
                zref,
                task_directions,
                full_exact_kl,
            )
            aligned_coverage = aligned_triplet_coverage(
                selected,
                aligned_triplets,
                total_aligned_triplet_weight,
            )
            curves[method].append(
                {
                    "k": k,
                    "features": rank_labels(rank[:k]),
                    "quadratic_objective": quadratic_subset_fraction(selected, quadratic_terms, full_quadratic),
                    "selection_objective": (
                        quadratic_subset_fraction(selected, quadratic_terms, full_quadratic)
                        + (LAMBDA_BASIS * aligned_coverage if method == "hypergraph" else 0.0)
                    ),
                    "mean_retained_kl": mean_retained_kl,
                    "median_retained_kl": median_retained_kl,
                    "aligned_triplet_coverage": float(aligned_coverage),
                }
            )

        summary_selection[method] = next(item for item in curves[method] if item["k"] == SUMMARY_K)

    task_responses = {}
    for task_name, direction in zip(DEFORMATION_MODELS, task_directions):
        response = np.abs(direction)
        top_indices = np.argsort(response)[::-1][:8]
        task_responses[task_name] = {
            "title": TASK_TITLES[task_name],
            "top_features": [
                {"feature": FEATURE_PLAIN[idx], "response": float(response[idx])}
                for idx in top_indices
            ],
        }

    return {
        "feature_labels": FEATURE_LABELS,
        "feature_plain": FEATURE_PLAIN,
        "eec_bins": EEC_BINS,
        "e2_betas": E2_BETAS,
        "e3_betas": E3_BETAS,
        "reference_seed": REFERENCE_SEED,
        "task_seed_base": TASK_SEED_BASE,
        "n_events_reference": N_EVENTS_REFERENCE,
        "n_events_task": N_EVENTS_TASK,
        "n_constituents": N_CONSTITUENTS,
        "delta_probe": DELTA_PROBE,
        "lambda_basis": LAMBDA_BASIS,
        "curve_ks": CURVE_KS,
        "summary_k": SUMMARY_K,
        "selection_rule": {
            "graph": "greedy quadratic marginal gain",
            "hypergraph": "greedy quadratic-plus-triplet marginal gain",
        },
        "task_titles": TASK_TITLES,
        "task_directions": task_direction_dict,
        "task_responses": task_responses,
        "reference_mean": reference_mean.tolist(),
        "reference_std": reference_std.tolist(),
        "quadratic_node_score": quadratic_node_score.tolist(),
        "cubic_node_score": cubic_node_score.tolist(),
        "multi_node_score": multi_node_score.tolist(),
        "graph_rank": rank_labels(graph_rank),
        "hyper_rank": rank_labels(hyper_rank),
        "curves": curves,
        "summary_selection": summary_selection,
        "top_aligned_triplets": aligned_triplets[:10],
    }


def make_figure(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    for method, color in [("graph", "#9b2c2c"), ("hypergraph", "#1f4f82")]:
        curve = results["curves"][method]
        ks = [item["k"] for item in curve]
        mean_kl = [item["mean_retained_kl"] for item in curve]
        median_kl = [item["median_retained_kl"] for item in curve]
        trip_cov = [item["aligned_triplet_coverage"] for item in curve]

        label = "Pairwise graph" if method == "graph" else "Fisher hypergraph"
        axes[0].plot(ks, mean_kl, color=color, linewidth=2.2, label=f"{label}: mean")
        axes[0].plot(
            ks,
            median_kl,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=f"{label}: median",
        )
        axes[1].plot(ks, trip_cov, color=color, linewidth=2.2, label=label)

    axes[0].set_xlabel("Retained observables")
    axes[0].set_ylabel(r"Retained local KL fraction")
    axes[0].set_xticks(CURVE_KS)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Retention across six nearby deformations")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_xlabel("Retained observables")
    axes[1].set_ylabel("Aligned triplet coverage")
    axes[1].set_xticks(CURVE_KS)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Retained deformation-sensitive 3-hyperedges")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "basis_design_summary.pdf")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    GENERATED_DIR.mkdir(exist_ok=True)

    results = summarize()

    with (GENERATED_DIR / "basis_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    make_figure(results)

    print(f"Wrote {(GENERATED_DIR / 'basis_results.json').relative_to(ROOT)}")
    print(f"Wrote {(FIGURES_DIR / 'basis_design_summary.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
