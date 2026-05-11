#!/usr/bin/env python3
"""
Same basis-design study as generate_basis_results.py, but driven by real
HDF5 data from h5_subjet_finder.py output instead of toy EventModel samples.

Key differences from the toy version:
  - Reference sample: all jets loaded from HDF5 (constituent kinematics →
    observables computed on the fly via vectorized batch functions).
  - Task deformation directions: derived from the six deformationMasks stored
    in each HDF5 file, by computing (masked_mean - reference_mean) / ref_std
    for each mask column.  No separate deformed-model dataset is generated.
  - Config-driven: --config points to a YAML file with paths.signal_base and
    paths.signal_filename (same schema as dataset_builder.py).

Everything else — Fisher tensors, greedy selection, KL curves, figures — is
identical to generate_basis_results.py.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
import KL_compute as klc#precision_matrix_KL, precompute_full_fisher_info

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subjet_data_handler import (
    compute_features_batch,
    discover_signal_files,
    load_config,
    load_jets_and_masks,
)

ROOT = Path("/work/abal/triality/results")#Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures/basis_design_real"
GENERATED_DIR = ROOT / "generated/basis_design_real"
WEB_DIR = Path("/web/abal/public_html/plots/triality/basis_design_real/")

# ---------------------------------------------------------------------------
# Observable grid — identical to generate_basis_results.py
# ---------------------------------------------------------------------------

EEC_EDGES = [
    0.02, 0.035, 0.05, 0.065, 0.08, 0.10, 0.12, 0.145, 0.17,
    0.20, 0.24, 0.28, 0.33, 0.39, 0.46, 0.54, 0.63, 0.73, 0.85,
]
EEC_EDGES = [
    0.02, 0.035, 0.05, 0.065, 0.08, 0.10, 0.12, 0.145, 0.17,
    0.20, 0.24, 0.28, 0.33, 0.39, 0.46, 0.54, 0.63, 0.73, 0.85,
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

# ---------------------------------------------------------------------------
# Study hyper-parameters — identical to generate_basis_results.py
# ---------------------------------------------------------------------------

N_JETS       = 100_000   # jets to load from HDF5
N_PARTICLES  = 25        # leading constituents per jet
CHUNK_SIZE   = 500       # batch size for feature computation
DELTA_PROBE  = 0.55
LAMBDA_BASIS = 3.0
CURVE_KS     = list(range(6, 21, 2))
SUMMARY_K    = 16

# Human-readable titles keyed by the deformationNames stored in the HDF5.
# Falls back gracefully if the stored name is not in this map.
TASK_TITLE_MAP = {
    "wide_angle":          "Wide-angle enhancement",
    "collinear_hardening": "Collinear hardening",
    "opening_angle":       "Opening-angle shift",
    "prong_asymmetry":     "Prong asymmetry",
    "inter_prong_bridge":  "Inter-prong bridge",
    "soft_4th_prong":      "Soft fourth prong",
}
TASK_IDX = {name: idx for idx, name in enumerate(TASK_TITLE_MAP.keys())}


# ---------------------------------------------------------------------------
# Analysis helpers — identical to generate_basis_results.py
# ---------------------------------------------------------------------------

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
    aligned_triplets: list[dict],
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
) -> tuple[float, float, list[float]]:
    if not selected:
        n = len(task_directions)
        return 0.0, 0.0, [0.0] * n
    mask = np.zeros(zref.shape[1], dtype=float)
    mask[list(selected)] = 1.0
    retained = []
    for task_idx, direction in enumerate(task_directions):
        subset_direction = direction * mask
        subset_kl = float(np.log(np.mean(np.exp(DELTA_PROBE * (zref @ subset_direction)))))
        retained.append(subset_kl / full_exact_kl[task_idx])
    return float(np.mean(retained)), float(np.median(retained)), retained


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reference_data(config_path: str, num_jets: int, num_particles: int) -> tuple:
    """
    Discover HDF5 files, load constituent kinematics and deformation masks,
    compute observables, and return the feature matrix with masks.

    Returns
    -------
    features   : float64 (N, D)
    masks      : bool   (N, 6)
    mask_names : list[str] of length 6
    """
    cfg = load_config(config_path)
    cuts = cfg.get("cuts", {})

    file_paths = discover_signal_files(config_path)

    part_deta, part_dphi, part_pt, masks, mask_names = load_jets_and_masks(
        file_paths,
        num_jets=num_jets,
        num_particles=num_particles,
        pt_cut=float(cuts.get("pt_cut", 0.0)),
        mass_win_lo=float(cuts.get("mass_win_lo", 0.0)),
        mass_win_hi=float(cuts.get("mass_win_hi", float("inf"))),
        idx_pt=int(cuts.get("idx_pt", 0)),
        idx_sdmass=int(cuts.get("idx_sdmass", 5)),
    )
    
    print(f"Loaded {len(part_pt)} jets with {part_pt.shape[1]} constituents each.")
    print("Computing observables...")

    features = compute_features_batch(
        part_deta, part_dphi, part_pt,
        eec_bins=EEC_BINS,
        e2_betas=E2_BETAS,
        e3_betas=E3_BETAS,
        chunk_size=CHUNK_SIZE,
    )

    return features, masks, mask_names


# ---------------------------------------------------------------------------
# Core analysis — mirrors summarize() in generate_basis_results.py
# ---------------------------------------------------------------------------

def summarize(features: np.ndarray, masks: np.ndarray, mask_names: list[str], *, mean: bool = False) -> dict:
    """
    Run the full basis-design study on a pre-computed feature matrix.

    Parameters
    ----------
    features   : float64 (N, D) — observable features for all reference jets
    masks      : bool   (N, 6) — per-jet deformation-selection flags
    mask_names : list[str] of length 6 — names of the six deformation tasks
    """
    reference_mean = features.mean(axis=0)
    reference_std  = features.std(axis=0)
    reference_std  = np.where(reference_std > 0.0, reference_std, 1.0)
    zref = (features - reference_mean) / reference_std
    fisher2 = np.cov(zref, rowvar=False, bias=True)
    fisher3 = np.einsum("ni,nj,nk->ijk", zref, zref, zref) / len(zref)
    #Task directions from deformation masks
    task_directions = []
    task_direction_dict = {}
    task_titles = {}
    #mask_names = ['collinear_hardening', 'wide_angle', 'opening_angle']#, 'prong_asymmetry', 'inter_prong_bridge', 'soft_4th_prong']
    for task_name in mask_names:
        col_idx = TASK_IDX.get(task_name)
        print(f"Processing task '{task_name}' (column {col_idx})...")
        task_mask = masks[:, col_idx]
        if task_mask.sum() == 0:
            raise RuntimeError(f"Deformation mask '{task_name}' selects no jets.")
        masked_mean = features[task_mask].mean(axis=0)
        base_mean = features[~task_mask].mean(axis=0)
        base_std = features[~task_mask].std(axis=0)
        
        #import pdb; pdb.set_trace()
        delta = (masked_mean - reference_mean) / reference_std
        #delta = (masked_mean - base_mean) / base_std
        
        norm = np.linalg.norm(delta)
        if norm < 1e-15:
            raise RuntimeError(f"Zero-norm direction for task '{task_name}'.")
        delta /= norm

        task_directions.append(delta)
        task_direction_dict[task_name] = delta.tolist()
        task_titles[task_name] = TASK_TITLE_MAP.get(task_name, task_name)

    task_directions = np.array(task_directions)
    full_info = klc.precompute_full_fisher_info(task_directions, fisher2)
    full_cubic_kl = klc.precompute_full_cubic_kl(task_directions, fisher2, fisher3)
    full_exact_kl = np.array(
        [
            float(np.log(np.mean(np.exp(DELTA_PROBE * (zref @ direction)))))
            for direction in task_directions
        ]
    )

    # Quadratic Fisher projections
    quadratic_terms = np.empty(
        (len(task_directions), len(FEATURE_LABELS), len(FEATURE_LABELS)), dtype=float
    )
    for task_idx, direction in enumerate(task_directions):
        quadratic_terms[task_idx] = np.outer(direction, direction) * fisher2
    full_quadratic = np.array([terms.sum() for terms in quadratic_terms])

    quadratic_node_score = np.mean(
        np.abs(task_directions * (task_directions @ fisher2)), axis=0
    )
    cubic_contract = np.einsum("abc,mb,mc->ma", fisher3, task_directions, task_directions)
    cubic_node_score = np.mean(np.abs(task_directions * cubic_contract), axis=0)
    #print("Quadratic node scores:", quadratic_node_score/quadratic_node_score.sum())
    #print("Cubic node scores:", cubic_node_score/cubic_node_score.sum())
    c1 = cubic_node_score / cubic_node_score.sum()
    q1 = quadratic_node_score / quadratic_node_score.sum()
    print("Ratio of cubic to quadratic node scores (normalized):", c1 / q1)
    multi_node_score = (
        quadratic_node_score / quadratic_node_score.sum()
        + LAMBDA_BASIS * cubic_node_score / cubic_node_score.sum()
    )
    print("Multi-node scores:", multi_node_score)
    
    # Aligned triplets
    aligned_triplets = []
    for i, j, k in combinations(range(len(FEATURE_LABELS)), 3):
        aligned_score = float(
            np.sum(
                np.abs(
                    task_directions[:, i]
                    * task_directions[:, j]
                    * task_directions[:, k]
                    * fisher3[i, j, k]
                )
            )
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

    # Greedy feature selection
    max_k = max(CURVE_KS)

    if mean:
        graph_rank = greedy_selection(
            n_features=len(FEATURE_LABELS),
            max_k=max_k,
            objective=lambda sel: quadratic_subset_fraction(sel, quadratic_terms, full_quadratic),
        )
        hyper_rank = greedy_selection(
            n_features=len(FEATURE_LABELS),
            max_k=max_k,
            objective=lambda sel: (
                quadratic_subset_fraction(sel, quadratic_terms, full_quadratic)
                + LAMBDA_BASIS * aligned_triplet_coverage(sel, aligned_triplets, total_aligned_triplet_weight)
            ),
        )

        curves = {"graph": [], "hypergraph": []}
        summary_selection = {}
        for method, rank in [("graph", graph_rank), ("hypergraph", hyper_rank)]:
            for k in CURVE_KS:
                selected = set(rank[:k])
                mean_retained_kl, median_retained_kl, per_task_kl = exact_retained_kl(
                    selected, zref, task_directions, full_exact_kl
                )
                mean_r, median_r, per_task = klc.precision_matrix_KL(
                    selected, task_directions, fisher2, full_fisher_info=full_info
                )
                mean_r3, median_r3, per_task_r3 = klc.cubic_pitman_ratio(
                    selected, task_directions, fisher2, fisher3, full_cubic_kl=full_cubic_kl
                )
                aligned_coverage = aligned_triplet_coverage(
                    selected, aligned_triplets, total_aligned_triplet_weight
                )
                curves[method].append(
                    {
                        "k": k,
                        "features": rank_labels(rank[:k]),
                        "quadratic_objective": quadratic_subset_fraction(
                            selected, quadratic_terms, full_quadratic
                        ),
                        "selection_objective": (
                            quadratic_subset_fraction(selected, quadratic_terms, full_quadratic)
                            + (LAMBDA_BASIS * aligned_coverage if method == "hypergraph" else 0.0)
                        ),
                        "mean_retained_kl": mean_r3,#mean_retained_kl,
                        "median_retained_kl": median_r3,#median_retained_kl,
                        "per_task_retained_kl": per_task_r3,#per_task_kl,
                        "aligned_triplet_coverage": float(aligned_coverage),
                    }
                )
                #import pdb;pdb.set_trace()
            summary_selection[method] = next(item for item in curves[method] if item["k"] == SUMMARY_K)

    else:
        # Independent greedy selection per task — no averaging over tasks
        def _quad_single(sel: set, t: int) -> float:
            if not sel:
                return 0.0
            idx = np.array(sorted(sel), dtype=int)
            return quadratic_terms[t][np.ix_(idx, idx)].sum() / full_quadratic[t]

        per_task_triplets: list[list[dict]] = []
        per_task_triplet_weight: list[float] = []
        for t in range(len(task_directions)):
            triplets_t = []
            for i, j, k in combinations(range(len(FEATURE_LABELS)), 3):
                score = abs(
                    task_directions[t, i]
                    * task_directions[t, j]
                    * task_directions[t, k]
                    * fisher3[i, j, k]
                )
                triplets_t.append({"indices": [i, j, k], "aligned_score": score})
            triplets_t.sort(key=lambda x: x["aligned_score"], reverse=True)
            per_task_triplets.append(triplets_t)
            per_task_triplet_weight.append(sum(x["aligned_score"] for x in triplets_t))

        graph_ranks: dict[str, list[int]] = {}
        hyper_ranks: dict[str, list[int]] = {}
        for t, task_name in enumerate(mask_names):
            graph_ranks[task_name] = greedy_selection(
                n_features=len(FEATURE_LABELS),
                max_k=max_k,
                objective=lambda sel, t=t: _quad_single(sel, t),
            )
            hyper_ranks[task_name] = greedy_selection(
                n_features=len(FEATURE_LABELS),
                max_k=max_k,
                objective=lambda sel, t=t: (
                    _quad_single(sel, t)
                    + LAMBDA_BASIS * aligned_triplet_coverage(
                        sel, per_task_triplets[t], per_task_triplet_weight[t]
                    )
                ),
            )

        curves: dict[str, dict] = {"graph": {}, "hypergraph": {}}
        summary_selection: dict[str, dict] = {"graph": {}, "hypergraph": {}}
        for method, ranks in [("graph", graph_ranks), ("hypergraph", hyper_ranks)]:
            for t, task_name in enumerate(mask_names):
                rank = ranks[task_name]
                task_curve = []
                for k in CURVE_KS:
                    selected = set(rank[:k])
                    mask_vec = np.zeros(zref.shape[1])
                    mask_vec[list(selected)] = 1.0
                    subset_kl = float(
                        np.log(np.mean(np.exp(DELTA_PROBE * (zref @ (task_directions[t] * mask_vec)))))
                    )
                    task_curve.append(
                        {
                            "k": k,
                            "features": rank_labels(rank[:k]),
                            "retained_kl": subset_kl / full_exact_kl[t],
                        }
                    )
                curves[method][task_name] = task_curve
                summary_selection[method][task_name] = next(
                    item for item in task_curve if item["k"] == SUMMARY_K
                )

    # Per-task feature responses
    task_responses = {}
    for task_name, direction in zip(mask_names, task_directions):
        response = np.abs(direction)
        top_indices = np.argsort(response)[::-1][:8]
        task_responses[task_name] = {
            "title": task_titles[task_name],
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
        "n_jets": len(features),
        "n_particles": N_PARTICLES,
        "delta_probe": DELTA_PROBE,
        "lambda_basis": LAMBDA_BASIS,
        "curve_ks": CURVE_KS,
        "summary_k": SUMMARY_K,
        "selection_rule": {
            "graph": "greedy quadratic marginal gain",
            "hypergraph": "greedy quadratic-plus-triplet marginal gain",
        },
        "task_names": mask_names,
        "task_titles": task_titles,
        "task_directions": task_direction_dict,
        "task_responses": task_responses,
        "reference_mean": reference_mean.tolist(),
        "reference_std": reference_std.tolist(),
        "quadratic_node_score": quadratic_node_score.tolist(),
        "cubic_node_score": cubic_node_score.tolist(),
        "multi_node_score": multi_node_score.tolist(),
        "mean_mode": mean,
        **(
            {"graph_rank": rank_labels(graph_rank), "hyper_rank": rank_labels(hyper_rank)}
            if mean
            else {
                "graph_ranks": {n: rank_labels(graph_ranks[n]) for n in mask_names},
                "hyper_ranks": {n: rank_labels(hyper_ranks[n]) for n in mask_names},
            }
        ),
        "curves": curves,
        "summary_selection": summary_selection,
        "top_aligned_triplets": aligned_triplets[:10],
    }


# ---------------------------------------------------------------------------
# Figure — identical layout to generate_basis_results.py
# ---------------------------------------------------------------------------

def make_figure(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    mean_mode = results.get("mean_mode", True)
    for method, color in [("graph", "#9b2c2c"), ("hypergraph", "#1f4f82")]:
        label = "Pairwise graph" if method == "graph" else "Fisher hypergraph"
        if mean_mode:
            curve    = results["curves"][method]
            ks       = [item["k"]                        for item in curve]
            mean_kl  = [item["mean_retained_kl"]         for item in curve]
            median_kl = [item["median_retained_kl"]      for item in curve]
            trip_cov = [item["aligned_triplet_coverage"] for item in curve]
        else:
            task_names = results["task_names"]
            ks = [item["k"] for item in results["curves"][method][task_names[0]]]
            all_kls = np.array([
                [item["retained_kl"] for item in results["curves"][method][tn]]
                for tn in task_names
            ])
            mean_kl   = all_kls.mean(axis=0).tolist()
            median_kl = np.median(all_kls, axis=0).tolist()
            trip_cov  = [0.0] * len(ks)

        axes[0].plot(ks, mean_kl,   color=color, linewidth=2.2, label=f"{label}: mean")
        axes[0].plot(ks, median_kl, color=color, linewidth=2.0,
                     linestyle="--", label=f"{label}: median")
        axes[1].plot(ks, trip_cov,  color=color, linewidth=2.2, label=label)

    axes[0].set_xlabel("Retained observables", fontsize=14)
    axes[0].set_ylabel(r"Retained local $R_3$ fraction",fontsize=14)
    axes[0].set_xticks(CURVE_KS)
    axes[0].set_ylim(0.35, 1.02)
    axes[0].set_title("Retention across three nearby deformations",fontsize=14)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=13, loc="lower right")

    axes[1].set_xlabel("Retained observables",fontsize=14)
    axes[1].set_ylabel("Aligned triplet coverage",fontsize=14)
    axes[1].set_xticks(CURVE_KS)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Retained deformation-sensitive 3-hyperedges", fontsize=14)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=13)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "basis_design_real_summary.pdf")
    fig.savefig(WEB_DIR / "basis_design_real_summary.png", dpi=500)
    plt.close(fig)


def make_separate_figures(results: dict) -> None:
    """One PDF per deformation task showing per-task retained KL vs k."""
    task_names = results["task_names"]
    task_titles = results["task_titles"]
    mean_mode = results.get("mean_mode", True)

    if mean_mode:
        ks = [item["k"] for item in results["curves"]["graph"]]
    else:
        ks = [item["k"] for item in results["curves"]["graph"][task_names[0]]]

    for task_idx, task_name in enumerate(task_names):
        fig, ax = plt.subplots(figsize=(5.5, 4.2))

        for method, color in [("graph", "#9b2c2c"), ("hypergraph", "#1f4f82")]:
            if mean_mode:
                per_task = [
                    item["per_task_retained_kl"][task_idx]
                    for item in results["curves"][method]
                ]
            else:
                per_task = [item["retained_kl"] for item in results["curves"][method][task_name]]
            label = "Pairwise graph" if method == "graph" else "Fisher hypergraph"
            ax.plot(ks, per_task, color=color, linewidth=2.2, label=label)

        ax.set_xlabel("Retained observables")
        ax.set_ylabel(r"Retained local $R_3$ fraction")
        ax.set_xticks(ks)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(task_titles.get(task_name, task_name))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        out = FIGURES_DIR / f"basis_real_{task_name}.pdf"
        fig.savefig(out)
        fig.savefig(WEB_DIR / f"basis_real_{task_name}.png", dpi=500)
        plt.close(fig)
        print(f"Wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basis-design study on real HDF5 jet data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ttbar_sm.yaml",
        metavar="PATH",
        help="YAML config pointing to HDF5 files with deformationMasks (default: configs/ttbar_sm.yaml).",
    )
    parser.add_argument(
        "--num-jets",
        type=int,
        default=N_JETS,
        metavar="N",
        help=f"Number of jets to load (default: {N_JETS}).",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=N_PARTICLES,
        metavar="P",
        help=f"Leading constituents per jet (default: {N_PARTICLES}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        metavar="C",
        help=f"Batch size for observable computation (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--mean",
        action="store_true",
        help="Average task objectives in greedy selection (default: independent per-task selection).",
    )
    parser.add_argument(
        "--separate-plots",
        action="store_true",
        help="Save one PDF per deformation task instead of the combined mean/median figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    global CHUNK_SIZE
    CHUNK_SIZE = args.chunk_size

    features, masks, mask_names = load_reference_data(
        args.config, args.num_jets, args.num_particles
    )
    mask_names = ["opening_angle", "inter_prong_bridge", "soft_4th_prong"]
    print(f"Feature matrix: {features.shape}  |  masks: {masks.shape}")
    print(f"Deformation tasks: {mask_names}")
    print("Running basis-design analysis...")
    results = summarize(features, masks, mask_names, mean=args.mean)

    out_json = GENERATED_DIR / "basis_results_real.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    if args.separate_plots:
        make_separate_figures(results)
    else:
        make_figure(results)
        print(f"Wrote {(FIGURES_DIR / 'basis_design_real_summary.pdf').relative_to(ROOT)}")

    print(f"Wrote {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
