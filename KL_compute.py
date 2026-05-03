#!/usr/bin/env python3
"""
Precision-matrix retained Fisher information for feature subset evaluation.

Provides precision_matrix_KL(), a drop-in alternative to exact_retained_kl()
that measures the fraction of Neyman-Pearson test power retained by a feature
subset S:

    ratio_t = (d_S^T Σ_SS^{-1} d_S) / (d^T Σ^{-1} d)

where d is the task direction, Σ is the reference covariance, and Σ_SS is the
submatrix restricted to selected features. This is the Pitman efficiency of the
subset — it measures SNR for detecting the shift, not signal variance. Unlike
the log-MGF ratio, it correctly penalises redundant features (which inflate the
covariance-based score but contribute nothing to the precision-based one) and
does not depend on an arbitrary probe parameter δ.
"""

from __future__ import annotations

import numpy as np


def precompute_full_fisher_info(
    task_directions: np.ndarray,
    fisher2: np.ndarray,
    rcond: float = 1e-12,
) -> np.ndarray:
    """
    Precompute d^T Σ^{-1} d for each task direction.

    Call once and pass the result to precision_matrix_KL to avoid
    recomputing the full-matrix pseudoinverse on every subset evaluation.

    Parameters
    ----------
    task_directions : (n_tasks, D)
    fisher2         : (D, D) reference covariance
    rcond           : singular-value cutoff for pseudoinverse

    Returns
    -------
    full_fisher_info : (n_tasks,)
    """
    sigma_inv = np.linalg.pinv(fisher2, rcond=rcond)
    return np.einsum("ti,ij,tj->t", task_directions, sigma_inv, task_directions)

def precompute_full_cubic_kl(
    task_directions: np.ndarray,
    fisher2: np.ndarray,
    fisher3: np.ndarray,
    rcond: float = 1e-12,
) -> np.ndarray:
    """Full-set cubic local-KL per task: (1/2) d^T Σ^{-1} d + (1/6) F3[θ,θ,θ]."""
    sigma_inv = np.linalg.pinv(fisher2, rcond=rcond)
    score_full = task_directions @ sigma_inv
    quad = np.einsum("ti,ti->t", task_directions, score_full)
    cubic = np.einsum("abc,ta,tb,tc->t", fisher3, score_full, score_full, score_full)
    return 0.5 * quad + cubic / 6.0

def precision_matrix_KL(
    selected: set[int],
    task_directions: np.ndarray,
    fisher2: np.ndarray,
    full_fisher_info: np.ndarray | None = None,
    rcond: float = 1e-12,
) -> tuple[float, float, list[float]]:
    """
    Precision-matrix retained Fisher information ratio for a feature subset.

    For each task direction d_t, computes the fraction of Neyman-Pearson test
    power retained when observing only the features in `selected`:

        ratio_t = (d_S^T Σ_SS^{-1} d_S) / (d^T Σ^{-1} d)

    This is the Pitman efficiency of subset S for detecting deformation t.
    In the Gaussian shift model, it equals the ratio of non-centrality
    parameters of the optimal likelihood ratio test — i.e. the fraction of
    statistical power preserved.

    Contrast with the log-MGF ratio used in exact_retained_kl(), which:
      - depends on an arbitrary probe δ
      - reduces to the covariance-based quadratic_subset_fraction at small δ
        (measuring signal variance, not SNR)
      - does not penalise feature redundancy

    Parameters
    ----------
    selected         : indices of the selected feature subset S
    task_directions  : (n_tasks, D) unit task-deformation directions
    fisher2          : (D, D) reference covariance matrix (Fisher-2 tensor)
    full_fisher_info : (n_tasks,) precomputed d^T Σ^{-1} d per task;
                       computed here if None (expensive for large D — prefer
                       precompute_full_fisher_info() when calling in a loop)
    rcond            : singular-value cutoff for pseudoinverse

    Returns
    -------
    mean_ratio   : float  — mean over tasks
    median_ratio : float  — median over tasks
    per_task     : list[float] — one ratio per task, in task_directions order
    """
    if not selected:
        n = len(task_directions)
        return 0.0, 0.0, [0.0] * n

    indices = np.array(sorted(selected), dtype=int)

    if full_fisher_info is None:
        full_fisher_info = precompute_full_fisher_info(task_directions, fisher2, rcond=rcond)

    sigma_ss = fisher2[np.ix_(indices, indices)]
    sigma_ss_inv = np.linalg.pinv(sigma_ss, rcond=rcond)

    d_s = task_directions[:, indices]  # (n_tasks, |S|)
    subset_fisher_info = np.einsum("ti,ij,tj->t", d_s, sigma_ss_inv, d_s)

    per_task = (subset_fisher_info / full_fisher_info).tolist()
    return float(np.mean(per_task)), float(np.median(per_task)), per_task

def cubic_pitman_ratio(
    selected: set[int],
    task_directions: np.ndarray,
    fisher2: np.ndarray,
    fisher3: np.ndarray,
    full_cubic_kl: np.ndarray | None = None,
    rcond: float = 1e-12,
) -> tuple[float, float, list[float]]:
    """
    Cubic-corrected Pitman ratio: local KL retention through third order.

    For each task direction d_t, computes the fraction of cubic-order local
    KL divergence retained when observing only the features in `selected`:

        score_S      = Σ_SS^{-1} d_S
        score_full   = Σ^{-1} d
        KL_S^(3)     = (1/2) d_S^T score_S
                       + (1/6) F3_SSS[a,b,c] · score_S[a] score_S[b] score_S[c]
        KL_full^(3)  = (1/2) d^T score_full
                       + (1/6) F3[a,b,c] · score_full[a] score_full[b] score_full[c]
        ratio_t      = KL_S^(3) / KL_full^(3)

    The cubic term enters with its sign preserved (no |·|), so destructive
    interference between triplet contributions is physical and contributes
    information about subset redundancy in the third Fisher tensor.

    For an exponential family with sufficient statistics z, F3 is the third
    joint cumulant κ^(3)_{abc} = ⟨z_a z_b z_c⟩_c (estimable from the reference
    sample). Reduces to precision_matrix_KL when fisher3 → 0.

    Note: ratio can exceed 1 if the cubic correction makes the local-KL
    expansion saturate beyond its radius of convergence — this is a
    diagnostic of large natural-parameter shift, not a metric defect.

    Parameters
    ----------
    selected         : indices of the selected feature subset S
    task_directions  : (n_tasks, D) unit task-deformation directions (mean shifts)
    fisher2          : (D, D) reference covariance matrix Σ
    fisher3          : (D, D, D) third joint cumulant tensor of sufficient statistics
    full_cubic_kl    : (n_tasks,) precomputed full-set cubic KL per task;
                       computed here if None (prefer precomputation in loops)
    rcond            : singular-value cutoff for pseudoinverse

    Returns
    -------
    mean_ratio   : float  — mean over tasks
    median_ratio : float  — median over tasks
    per_task     : list[float] — one ratio per task, in task_directions order
    """
    if not selected:
        n = len(task_directions)
        return 0.0, 0.0, [0.0] * n

    indices = np.array(sorted(selected), dtype=int)

    # Subset score: θ_S = Σ_SS^{-1} d_S, vectorized over tasks
    sigma_ss = fisher2[np.ix_(indices, indices)]
    sigma_ss_inv = np.linalg.pinv(sigma_ss, rcond=rcond)
    d_s = task_directions[:, indices]                              # (n_tasks, |S|)
    score_s = d_s @ sigma_ss_inv                                   # (n_tasks, |S|)

    # Quadratic and cubic subset contributions
    quad_s = np.einsum("ti,ti->t", d_s, score_s)                   # (n_tasks,)
    fisher3_ss = fisher3[np.ix_(indices, indices, indices)]        # (|S|, |S|, |S|)
    cubic_s = np.einsum(
        "abc,ta,tb,tc->t", fisher3_ss, score_s, score_s, score_s
    )                                                               # (n_tasks,)
    subset_kl = 0.5 * quad_s + cubic_s / 6.0

    # Full-set cubic KL (precompute when calling in a loop)
    if full_cubic_kl is None:
        sigma_inv = np.linalg.pinv(fisher2, rcond=rcond)
        score_full = task_directions @ sigma_inv                   # (n_tasks, D)
        quad_full = np.einsum("ti,ti->t", task_directions, score_full)
        cubic_full = np.einsum(
            "abc,ta,tb,tc->t", fisher3, score_full, score_full, score_full
        )
        full_cubic_kl = 0.5 * quad_full + cubic_full / 6.0

    per_task = (subset_kl / full_cubic_kl).tolist()
    return float(np.mean(per_task)), float(np.median(per_task)), per_task