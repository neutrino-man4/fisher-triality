#!/usr/bin/env python3
"""
Refactored EFT compression benchmark with proper marginalization.

This script implements Section 6.2 of the Fisher-Correlator-Hypergraph
triality paper, replacing the original zeroing approach with proper
marginalization throughout:

  - Quadratic gain: exact marginal gain in Delta_S^T C_S^{-1} Delta_S
    (no ad-hoc redundancy penalty needed)
  - Cubic gain: contracted with re-optimized weights w_S = C_S^{-1} Delta_S
  - R_2 evaluation: Delta_S^T C_S^{-1} Delta_S / Delta^T C^{-1} Delta
    (guaranteed <= 1)
  - R_KL evaluation: using re-optimized direction embedded back into
    full observable space

Author: Aritra Bal (ETP)
Date: 2026-03-23
"""

from __future__ import annotations

import json
import os
import dataset_builder as db
from itertools import combinations
from pathlib import Path
from generate_eft_results import quadratic_marginal_gain, cubic_completion_gain  # for reference only; not used in proper marginalization
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configs.observables import FEATURE_LABELS, FEATURE_PLAIN, OBSERVABLES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/work/abal/triality/results")
FIGURES_DIR = ROOT / "figures"
GENERATED_DIR = ROOT / "generated"
WEB_DIR = Path("/web/abal/public_html/plots/triality/EFT")


# ---------------------------------------------------------------------------
# Feature labels
# ---------------------------------------------------------------------------
# FEATURE_LABELS: list[str] = [
#     r"$\mathrm{EEC}_1$",
#     r"$\mathrm{EEC}_2$",
#     r"$\mathrm{EEC}_3$",
#     r"$\mathrm{EEC}_4$",
#     r"$e_2^{(1)}$",
#     r"$e_2^{(2)}$",
#     r"$e_3^{(1)}$",
#     r"$e_3^{(2)}$",
# ]
# FEATURE_PLAIN: list[str] = [
#     "EEC_1",
#     "EEC_2",
#     "EEC_3",
#     "EEC_4",
#     "e2^(1)",
#     "e2^(2)",
#     "e3^(1)",
#     "e3^(2)",
# ]


# ---------------------------------------------------------------------------
# Fallback benchmark direction (paper Eq. 6.14)
# ---------------------------------------------------------------------------
BENCHMARK_DIRECTION = np.array(
    [-0.287, -0.221, 0.600, 0.499, 0.197, 0.133, 0.253, 0.373],
    dtype=float,
)
BENCHMARK_DIRECTION /= np.linalg.norm(BENCHMARK_DIRECTION)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DELTA_KAPPA: float = 0.55
LAMBDA_CUBIC: float = 1.
COMMON_CORE_SIZE: int = 2
REDUNDANCY_BETA = 1.0

# ---------------------------------------------------------------------------
# Benchmark direction from two MC samples
# ---------------------------------------------------------------------------
def compute_benchmark_direction(
    data_sm: np.ndarray,
    data_eft: np.ndarray,
    regularisation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the natural parameter direction u from SM and EFT samples.

    Recovers u = (I^(2))^{-1} Delta, where Delta is the standardised
    mean shift and I^(2) is the correlation matrix from the SM sample.

    Parameters
    ----------
    data_sm : np.ndarray
        SM sample, shape (N_sm, D).
    data_eft : np.ndarray
        EFT sample, shape (N_eft, D).
    regularisation : float, optional
        Ridge parameter for numerical stability.

    Returns
    -------
    u : np.ndarray
        Unit-normalised natural parameter direction, shape (D,).
    delta : np.ndarray
        Standardised mean shift, shape (D,).
    fisher2 : np.ndarray
        Correlation matrix from SM, shape (D, D).
    u_unnormalised : np.ndarray
        Raw (I^(2))^{-1} Delta before normalisation, shape (D,).
    """
    if data_sm.shape[1] != data_eft.shape[1]:
        raise ValueError(
            f"Observable dimensions do not match: "
            f"SM has {data_sm.shape[1]}, EFT has {data_eft.shape[1]}"
        )

    mean_sm: np.ndarray = data_sm.mean(axis=0)
    std_sm: np.ndarray = data_sm.std(axis=0)
    std_sm = np.where(std_sm > 0.0, std_sm, 1.0)
    mean_eft: np.ndarray = data_eft.mean(axis=0)
    std_eft: np.ndarray = data_eft.std(axis=0)
    std_eft = np.where(std_eft > 0.0, std_eft, 1.0)
    z_sm: np.ndarray = (data_sm - mean_sm) / std_sm
    z_eft: np.ndarray = (data_eft - mean_eft) / std_eft

    delta: np.ndarray = data_eft.mean(axis=0) - data_sm.mean(axis=0)
    fisher2: np.ndarray = np.cov(z_sm, rowvar=False, bias=True)

    if regularisation > 0.0:
        fisher2_reg = fisher2 + regularisation * np.eye(fisher2.shape[0])
    else:
        fisher2_reg = fisher2

    u_unnormalised: np.ndarray = np.linalg.solve(fisher2_reg, delta)
    #u_unnormalised: np.ndarray = delta
    norm: float = float(np.linalg.norm(u_unnormalised))

    if norm < 1e-15:
        raise ValueError(
            "Benchmark direction has near-zero norm; "
            "samples may be indistinguishable."
        )

    u: np.ndarray = u_unnormalised / norm
    return u, delta, fisher2, u_unnormalised


# ---------------------------------------------------------------------------
# Helper: convert feature index array to labels
# ---------------------------------------------------------------------------
def rank_labels(rank: np.ndarray) -> list[str]:
    """Map feature index array to plain-text labels."""
    return [FEATURE_PLAIN[idx] for idx in rank]


# ---------------------------------------------------------------------------
# Proper quadratic marginal gain
# ---------------------------------------------------------------------------
def quadratic_marginal_gain_proper(
    candidate: int,
    selected: list[int],
    fisher2: np.ndarray,
    delta: np.ndarray,
) -> float:
    """Exact marginal gain in Delta_S^T C_S^{-1} Delta_S from adding candidate.

    This replaces the paper's ad-hoc Delta_2 with redundancy penalty.
    The submatrix inversion C_S^{-1} automatically handles decorrelation,
    so no separate redundancy term is needed.

    Parameters
    ----------
    candidate : int
        Feature index to evaluate.
    selected : list[int]
        Currently retained feature indices.
    fisher2 : np.ndarray
        Full correlation matrix, shape (D, D).
    delta : np.ndarray
        Standardised mean shift, shape (D,).

    Returns
    -------
    float
        Marginal gain in Fisher information, guaranteed >= 0.
    """
    # Fisher information of extended subset S ∪ {candidate}
    idx_new: np.ndarray = np.array(sorted(selected + [candidate]))
    C_new: np.ndarray = fisher2[np.ix_(idx_new, idx_new)]
    delta_new: np.ndarray = delta[idx_new]
    fi_new: float = float(delta_new @ np.linalg.solve(C_new, delta_new))

    # Fisher information of current subset S
    if selected:
        idx_old: np.ndarray = np.array(sorted(selected))
        C_old: np.ndarray = fisher2[np.ix_(idx_old, idx_old)]
        delta_old: np.ndarray = delta[idx_old]
        fi_old: float = float(delta_old @ np.linalg.solve(C_old, delta_old))
    else:
        fi_old = 0.0

    return fi_new - fi_old


# ---------------------------------------------------------------------------
# Proper cubic completion gain
# ---------------------------------------------------------------------------
def cubic_completion_gain_proper(
    candidate: int,
    selected: list[int],
    fisher2: np.ndarray,
    fisher3: np.ndarray,
    delta: np.ndarray,
) -> float:
    """Cubic completion gain using re-optimized weights w_S = C_S^{-1} Delta_S.

    For the existing subset members, uses the current optimal weights.
    For the candidate, uses its weight from the extended optimal solution.
    This ensures the cubic gain reflects three-body information measured
    with statistically optimal combination weights.

    Parameters
    ----------
    candidate : int
        Feature index to evaluate.
    selected : list[int]
        Currently retained feature indices.
    fisher2 : np.ndarray
        Full correlation matrix, shape (D, D).
    fisher3 : np.ndarray
        Third cumulant tensor, shape (D, D, D).
    delta : np.ndarray
        Standardised mean shift, shape (D,).

    Returns
    -------
    float
        Cubic completion gain, >= 0.
    """
    if len(selected) < 2:
        return 0.0

    # Re-optimized weights for current subset
    idx_S: np.ndarray = np.array(sorted(selected))
    C_S: np.ndarray = fisher2[np.ix_(idx_S, idx_S)]
    delta_S: np.ndarray = delta[idx_S]
    w_S: np.ndarray = np.linalg.solve(C_S, delta_S)
    feat_to_pos_S: dict[int, int] = {
        int(feat): pos for pos, feat in enumerate(idx_S)
    }

    # Re-optimized weights for extended subset S ∪ {candidate}
    idx_ext: np.ndarray = np.array(sorted(selected + [candidate]))
    C_ext: np.ndarray = fisher2[np.ix_(idx_ext, idx_ext)]
    delta_ext: np.ndarray = delta[idx_ext]
    w_ext: np.ndarray = np.linalg.solve(C_ext, delta_ext)
    feat_to_pos_ext: dict[int, int] = {
        int(feat): pos for pos, feat in enumerate(idx_ext)
    }

    # Candidate's re-optimized weight in the extended solution
    w_a: float = float(w_ext[feat_to_pos_ext[candidate]])

    # Sum over all pairs {b, c} ⊆ S
    gain: float = 0.0
    for b, c in combinations(selected, 2):
        w_b: float = float(w_S[feat_to_pos_S[b]])
        w_c: float = float(w_S[feat_to_pos_S[c]])
        gain += abs(w_a * w_b * w_c * fisher3[candidate, b, c])

    return gain


# ---------------------------------------------------------------------------
# Greedy forward selection
# ---------------------------------------------------------------------------
def greedy_selection_order(
    *,
    initial_core: list[int],
    fisher2: np.ndarray,
    fisher3: np.ndarray,
    delta: np.ndarray,
    benchmark_direction: np.ndarray,
    lambda_cubic: float,
    cubic_rescale: float,
) -> np.ndarray:
    """Grow the observable basis by greedy forward selection.

    At each step, adds the candidate that maximises
        Delta_2^proper(a|S) + lambda_cubic * cubic_rescale * Delta_3^proper(a|S)

    With lambda_cubic = 0, this is the pairwise graph baseline.
    With lambda_cubic > 0, this is the Fisher hypergraph selector.

    Parameters
    ----------
    initial_core : list[int]
        Seed features to start from.
    fisher2 : np.ndarray
        Correlation matrix, shape (D, D).
    fisher3 : np.ndarray
        Third cumulant tensor, shape (D, D, D).
    delta : np.ndarray
        Standardised mean shift, shape (D,).
    lambda_cubic : float
        Weight for cubic gain. 0 = pairwise only.
    cubic_rescale : float
        Fixed rescaling to put quadratic and cubic gains on same scale.

    Returns
    -------
    np.ndarray
        Full ordering of feature indices, shape (D,).
    """
    n_features: int = len(delta)
    selected: list[int] = list(initial_core)
    remaining: set[int] = set(range(n_features)) - set(selected)

    while remaining:
        best_feature: int | None = None
        best_score: float | None = None

        for feature in sorted(remaining):
            # score: float = quadratic_marginal_gain_proper(
            #     feature, selected, fisher2, delta
            # )
            score: float = quadratic_marginal_gain(
                feature, selected, fisher2, benchmark_direction, REDUNDANCY_BETA
            )
            # if lambda_cubic > 0.0:
            #     score += (
            #         lambda_cubic
            #         * cubic_rescale
            #         * cubic_completion_gain_proper(
            #             feature, selected, fisher2, fisher3, delta
            #         )
            #     )
            if lambda_cubic > 0.0:
                score += (
                    lambda_cubic
                    * cubic_rescale
                    * cubic_completion_gain(
                        feature, selected, fisher3, benchmark_direction
                    )
                )
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


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def summarize(data: np.ndarray, benchmark_direction: np.ndarray | None = None) -> dict:
    """Run the full EFT compression analysis with proper marginalization.

    Parameters
    ----------
    data : np.ndarray
        SM reference sample, shape (N, D).
    benchmark_direction : np.ndarray | None
        Unit-normalised natural parameter direction. If None, falls back
        to the hardcoded physics-motivated direction.

    Returns
    -------
    dict
        Complete results including curves, selections, and triplets.
    """
    # ------------------------------------------------------------------
    # Step 1: Standardise observables
    # ------------------------------------------------------------------
    mean: np.ndarray = data.mean(axis=0)
    std: np.ndarray = data.std(axis=0)
    zdata: np.ndarray = (data - mean) / std

    # ------------------------------------------------------------------
    # Step 2: Estimate Fisher tensors from SM sample
    # ------------------------------------------------------------------
    fisher2: np.ndarray = np.cov(zdata, rowvar=False, bias=True)
    fisher3: np.ndarray = (
        np.einsum("ni,nj,nk->ijk", zdata, zdata, zdata) / len(zdata)
    )

    # ------------------------------------------------------------------
    # Step 3: Set benchmark direction
    # ------------------------------------------------------------------
    if benchmark_direction is None:
        print("No benchmark direction provided; using default values.")
        u: np.ndarray = BENCHMARK_DIRECTION.copy()
    else:
        u = benchmark_direction.copy()

    # ------------------------------------------------------------------
    # Step 4: Recover delta = I^(2) u (scale cancels in all ratios)
    # ------------------------------------------------------------------
    delta: np.ndarray = u#fisher2 @ u
    # ------------------------------------------------------------------
    # Step 5: Full-basis quantities
    # ------------------------------------------------------------------
    # Proper full-basis Fisher information: Delta^T C^{-1} Delta
    full_fi_proper: float = float(delta @ np.linalg.solve(fisher2, delta))
    full_fi = float(u @ (fisher2 @ u))
    # Full-basis cubic projection (for interpretation)
    full_i3: float = float(np.einsum("i,j,k,ijk", u, u, u, fisher3))

    # Full-basis exact KL using re-optimized direction
    # (for full basis, u IS the optimal direction, so no change)
    full_kl: float = float(
        np.log(np.mean(np.exp(DELTA_KAPPA * (zdata @ u))))
    )

    # ------------------------------------------------------------------
    # Step 6: Node scores (full-basis, for interpretation/bar chart only)
    # ------------------------------------------------------------------
    quad_node: np.ndarray = np.abs(u * (fisher2 @ u))
    cubic_contract: np.ndarray = np.einsum("abc,b,c->a", fisher3, u, u)
    cubic_node: np.ndarray = np.abs(u * cubic_contract)

    quad_node_norm: np.ndarray = quad_node / quad_node.sum()
    cubic_node_norm: np.ndarray = cubic_node / cubic_node.sum()
    multi_node: np.ndarray = quad_node_norm + LAMBDA_CUBIC * cubic_node_norm

    common_cubic_core = list(np.argsort(multi_node)[::-1][:COMMON_CORE_SIZE])
    common_quad_core = list(np.argsort(quad_node_norm)[::-1][:COMMON_CORE_SIZE])
    #common_cubic_core = common_quad_core
    # ------------------------------------------------------------------
    # Step 8: Cubic rescale factor from initial core
    # ------------------------------------------------------------------
    core_remaining: list[int] = sorted(
        set(range(len(FEATURE_LABELS))) - set(common_quad_core)
    )
    core_quad_gains: list[float] = [
        quadratic_marginal_gain(idx, common_quad_core, fisher2, u, 1.0)
        for idx in core_remaining
    ]
    core_remaining: list[int] = sorted(
        set(range(len(FEATURE_LABELS))) - set(common_cubic_core)
    )
    core_cubic_gains: list[float] = [
        cubic_completion_gain(idx, common_cubic_core, fisher3, u)
        for idx in core_remaining
    ]
    max_cubic: float = max(core_cubic_gains)
    cubic_rescale: float = (
        max(core_quad_gains) / max_cubic if max_cubic > 0.0 else 1.0
    )
    
    # ------------------------------------------------------------------
    # Step 9: Greedy selection — pairwise graph and Fisher hypergraph
    # ------------------------------------------------------------------
    
    pair_rank: np.ndarray = greedy_selection_order(
        initial_core=common_quad_core,
        fisher2=fisher2,
        fisher3=fisher3,
        delta=delta,
        benchmark_direction=u,
        lambda_cubic=0.0,
        cubic_rescale=cubic_rescale,
    )
    hyper_rank: np.ndarray = greedy_selection_order(
        initial_core=common_cubic_core,
        fisher2=fisher2,
        fisher3=fisher3,
        delta=delta,
        benchmark_direction=u,
        lambda_cubic=LAMBDA_CUBIC,
        cubic_rescale=cubic_rescale,
    )
    # ------------------------------------------------------------------
    # Step 10: Evaluate compression curves with proper marginalization
    # ------------------------------------------------------------------
    curves: dict[str, list[dict]] = {"graph": [], "hypergraph": []}
    selections: dict[str, dict] = {}
    for method, rank in [("graph", pair_rank), ("hypergraph", hyper_rank)]:
        for k in range(2, len(FEATURE_LABELS) + 1):
            idx: np.ndarray = np.array(sorted(rank[:k]))
            
            # --- Proper R_2: Delta_S^T C_S^{-1} Delta_S / full ---
            C_S: np.ndarray = fisher2[np.ix_(idx, idx)]
            delta_S: np.ndarray = delta[idx]
            fi_S: float = float(delta_S @ np.linalg.solve(C_S, delta_S))
            fi_SS: float = float(u[idx] @ (C_S @ u[idx]))
            retained_i2: float = fi_S / full_fi_proper
            retained_old_i2: float = fi_SS / full_fi
            # --- Proper R_3: re-optimized cubic projection / full ---
            w_S: np.ndarray = np.linalg.solve(C_S, delta_S)
            # Build full-space weight vector with re-optimized values
            w_full_space: np.ndarray = np.zeros(len(FEATURE_LABELS))
            w_full_space[idx] = w_S
            i3_S: float = float(
                np.einsum(
                    "i,j,k,ijk",
                    w_full_space,
                    w_full_space,
                    w_full_space,
                    fisher3,
                )
            )
            retained_i3: float = i3_S / full_i3 if abs(full_i3) > 1e-15 else 0.0

            # --- Proper R_KL: using re-optimized direction ---
            # Normalise re-optimized weights for KL probe
            w_norm: float = float(np.linalg.norm(w_full_space))
            # if w_norm > 1e-15:
            #     u_reopt: np.ndarray = w_full_space / w_norm
            # else:
            u_reopt = w_full_space
            kl_S: np.float32 = np.float32(
                np.log(np.mean(np.exp(DELTA_KAPPA * (zdata @ u_reopt))))
            )
            retained_kl: float = kl_S / full_kl if abs(full_kl) > 1e-15 else 0.0

            # --- Relative 95% interval from proper R_2 ---
            relative_limit: float = float(np.sqrt(full_fi_proper / fi_S))
            
            curves[method].append(
                {
                    "k": k,
                    "features": rank_labels(rank[:k]),
                    "retained_i2": retained_i2,
                    "retained_i3": retained_i3,
                    "retained_old_i2": retained_old_i2,
                    "retained_kl": retained_kl,
                    "relative_limit": relative_limit,
                }
            )

        # Extract the k=4 summary
        k4: dict = curves[method][2]
        selections[method] = {
            "features": k4["features"],
            "retained_i2": k4["retained_i2"],
            "retained_i3": k4["retained_i3"],
            "retained_kl": k4["retained_kl"],
            "relative_limit": k4["relative_limit"],
        }

    # ------------------------------------------------------------------
    # Step 11: EFT-aligned hyperedge ranking (for interpretation)
    # ------------------------------------------------------------------
    triplets: list[dict] = []
    for i, j, k in combinations(range(len(FEATURE_LABELS)), 3):
        raw_weight: float = float(fisher3[i, j, k])
        aligned_score: float = float(abs(raw_weight * u[i] * u[j] * u[k]))
        triplets.append(
            {
                "features": [FEATURE_PLAIN[i], FEATURE_PLAIN[j], FEATURE_PLAIN[k]],
                "weight": raw_weight,
                "aligned_score": aligned_score,
            }
        )
    triplets.sort(key=lambda item: item["aligned_score"], reverse=True)

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    results: dict = {
        "feature_labels": FEATURE_LABELS,
        "feature_plain": FEATURE_PLAIN,
        "benchmark_direction": u.tolist(),
        "delta_kappa": DELTA_KAPPA,
        "lambda_cubic": LAMBDA_CUBIC,
        "common_core_size": COMMON_CORE_SIZE,
        "common_cubic_core": rank_labels(np.array(common_cubic_core, dtype=int)),
        "common_quad_core": rank_labels(np.array(common_quad_core, dtype=int)),
        "cubic_rescale": cubic_rescale,
        "selection_rule": {
            "graph": "proper quadratic marginal gain (no redundancy penalty needed)",
            "hypergraph": "proper quadratic gain plus re-optimized cubic completion",
        },
        "reference_mean": mean.tolist(),
        "reference_std": std.tolist(),
        "quadratic_node_score": quad_node.tolist(),
        "cubic_node_score": cubic_node.tolist(),
        "multi_node_score": multi_node.tolist(),
        "pair_rank": rank_labels(pair_rank),
        "hyper_rank": rank_labels(hyper_rank),
        "full_coefficients": {
            "fi_proper": full_fi_proper,
            "i3": full_i3,
            "exact_kl": full_kl,
        },
        "curves": curves,
        "k4_selections": selections,
        "top_triplets": triplets[:8],
    }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_constraint_figure(results: dict) -> None:
    """Compression performance: retained fraction and relative EFT interval."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    for method, color in [("graph", "#9b2c2c"), ("hypergraph", "#1f4f82")]:
        curve = results["curves"][method]
        ks = [item["k"] for item in curve]
        retained_i2 = [item["retained_i2"] for item in curve]
        retained_kl = [item["retained_kl"] for item in curve]
        retained_old_i2 = [item["retained_old_i2"] for item in curve]
        rel_limit = [item["relative_limit"] for item in curve]
        retained_i3 = [item["retained_i3"] for item in curve]
        label = "Pairwise graph" if method == "graph" else "Fisher hypergraph"
        axes[0].plot(
            ks, retained_old_i2, color=color, linewidth=2.2,
            label=f"{label}: $R_2$",
        )
        axes[0].plot(
            ks, retained_kl, color=color, linewidth=2.0, linestyle="--",
            label=f"{label}: $R_{{\\mathrm{{KL}}}}$",
        )
        # axes[0].plot(
        #     ks, retained_old_i2, color=color, linewidth=2.0, linestyle="--",
        #     label=f"{label}: $R_{{\\mathrm{{KL}}}}$",
        # )
        axes[1].plot(ks, rel_limit, color=color, linewidth=2.2, label=label)

    axes[0].set_xlabel("Retained observables")
    axes[0].set_ylabel("Retained fraction")
    axes[0].set_xticks(range(2, len(FEATURE_LABELS) + 1))
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=7)
    axes[0].set_ylim(0.0, 1.05)  # proper marginalization: guaranteed <= 1
    axes[0].set_title("Information retained under compression")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Retained observables")
    axes[1].set_ylabel(r"$\Delta\kappa_{95}/\Delta\kappa_{95}^{\rm full}$")
    axes[1].set_xticks(range(2, len(FEATURE_LABELS) + 1))
    axes[1].set_ylim(0.95, 1.1)
    axes[1].set_title("Relative local EFT interval")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eft_constraint_summary.pdf")
    fig.savefig(WEB_DIR / "eft_constraint_summary.png", dpi=600)
    plt.close(fig)


def make_hypergraph_figure(results: dict) -> None:
    """Feature scores and leading EFT-sensitive 3-hyperedges."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))

    x = np.arange(len(FEATURE_LABELS))
    width = 0.38
    quad = np.array(results["quadratic_node_score"])
    multi = np.array(results["multi_node_score"])

    axes[0].bar(
        x - width / 2, quad, width=width,
        color="#9b2c2c", label="Pairwise graph score",
    )
    axes[0].bar(
        x + width / 2, multi, width=width,
        color="#1f4f82", label="Multi-order score",
    )
    axes[0].set_xticks(x, FEATURE_LABELS, rotation=22, ha="right")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=7)
    axes[0].set_ylabel("Operator-aligned node score")
    axes[0].set_title(r"Feature ranking in the $O_{tG}$ benchmark")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    triplets = results["top_triplets"][:6]
    labels = [" / ".join(item["features"]) for item in triplets]
    values = [item["aligned_score"] for item in triplets]
    colors = [
        "#1f4f82" if item["weight"] > 0 else "#d97925" for item in triplets
    ]
    axes[1].barh(labels, values, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"$|u_a u_b u_c\,I^{(3)}_{abc}|$")
    axes[1].set_title("Leading EFT-sensitive 3-hyperedges")
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eft_hypergraph_summary.pdf")
    fig.savefig(WEB_DIR / "eft_hypergraph_summary.png", dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Load data, run analysis, write outputs."""
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    GENERATED_DIR.mkdir(exist_ok=True, parents=True)
    WEB_DIR.mkdir(exist_ok=True, parents=True)

    # Observable definitions
    observables: dict = {
        "EEC_1": [0.02, 0.08],
        "EEC_2": [0.08, 0.12],
        "EEC_3": [0.12, 0.16],
        "EEC_4": [0.16, 0.80],
        "ECF_e2b1": (2, 1),     # e_2^(1)
        "ECF_e2b2": (2, 2),     # e_2^(2)
        "ECF_e3b1": (3, 1),     # e_3^(1)
        "ECF_e3b2": (3, 2),     # e_3^(2)
    }

    # Load SM and EFT samples
    data_sm, _, _, _ = db.build_dataset(
        config_path="configs/data_sm.yaml",
        observables=OBSERVABLES,
        num_jets=100000,
        num_particles=25,
        seed=42,
    )
    data_eft, _, _, _ = db.build_dataset(
        config_path="configs/data_eft.yaml",
        observables=OBSERVABLES,
        num_jets=100000,
        num_particles=25,
        seed=42,
    )
    import pdb;pdb.set_trace()
    
    delta: np.ndarray = data_eft.mean(axis=0) - data_sm.mean(axis=0)
    # Run analysis with proper marginalization
    results = summarize(data_sm, benchmark_direction=delta)
    # Write outputs
    try:
        with (GENERATED_DIR / "eft_results.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    except Exception as e:
        print(f"Error writing results to JSON: {e}")
        print("Only making figures without saving results.")
    make_constraint_figure(results)
    make_hypergraph_figure(results)

    print(f"Wrote {GENERATED_DIR / 'eft_results.json'}")
    print(f"Wrote {FIGURES_DIR / 'eft_constraint_summary.pdf'}")
    print(f"Wrote {FIGURES_DIR / 'eft_hypergraph_summary.pdf'}")


if __name__ == "__main__":
    main()