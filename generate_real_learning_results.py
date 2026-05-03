#!/usr/bin/env python3
"""
Same learning benchmark as generate_learning_results.py, but driven by real
HDF5 data (background: ttbar SM, signal: ttbar EFT) instead of toy EventModel
samples.

Data pipeline mirrors generate_real_basis_results.py: constituent kinematics
are loaded from HDF5 files and projected onto the same 33-dimensional observable
basis (18 EEC bins + 9 E2 ECF + 6 E3 ECF).  A sampling pool of num_jets per
class is loaded up front; each repetition draws N_EVENTS_PER_CLASS jets from
each pool with a different seed to give variation across repetitions.

All classifier training, evaluation, and plotting logic is identical to
generate_learning_results.py.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from subjet_data_handler import (
    compute_features_batch,
    discover_signal_files,
    load_config,
)

ROOT = Path("/work/abal/triality/results")
FIGURES_DIR = ROOT / "figures/learning_results_real"
GENERATED_DIR = ROOT / "generated/learning_results_real"
WEB_DIR = Path("/web/abal/public_html/plots/triality/learning_results_real/")

# ---------------------------------------------------------------------------
# Observable grid — identical to generate_real_basis_results.py
# ---------------------------------------------------------------------------

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
# Hyper-parameters
# ---------------------------------------------------------------------------

N_PARTICLES        = 25
CHUNK_SIZE         = 500
N_EVENTS_PER_CLASS = 18000
N_VAL_PER_CLASS    = 1000
N_TEST_PER_CLASS   = 3000
TRAIN_SIZES        = [3000, 6000, 9000]
N_REPETITIONS      = 5
N_EPOCHS           = 75
HIDDEN_DIM         = 8
MLP_DIM            = [16]
TOP_PAIR_EDGES     = 150
TOP_TRIPLET_EDGES  = 200
LEARNING_RATE      = 0.01
WEIGHT_DECAY       = 1.0e-4
TASK_NAME          = "ttbar_soft4prong_vs_sm"
TASK_TITLE         = r"Soft 4th prong deformation: $t \to bq\bar{q}$"

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
# Data loading — mask-free; works for both SM (augmented) and EFT files
# ---------------------------------------------------------------------------

def load_features(config_path: str, num_jets: int, num_particles: int, load_mask:str=None) -> np.ndarray:
    """
    Discover HDF5 files from config, load constituent kinematics (no masks
    required), apply kinematic cuts, compute 33-dimensional observables.

    Returns
    -------
    features : float64 (N, 33)
    """
    cfg = load_config(config_path)
    cuts = cfg.get("cuts", {})
    pt_cut      = float(cuts.get("pt_cut", 0.0))
    mass_win_lo = float(cuts.get("mass_win_lo", 0.0))
    mass_win_hi = float(cuts.get("mass_win_hi", float("inf")))
    idx_pt      = int(cuts.get("idx_pt", 0))
    idx_sdmass  = int(cuts.get("idx_sdmass", 5))

    file_paths = discover_signal_files(config_path)
    apply_cuts = (pt_cut > 0.0) or (mass_win_lo > 0.0) or (mass_win_hi < float("inf"))

    deta_chunks: List[np.ndarray] = []
    dphi_chunks: List[np.ndarray] = []
    pt_chunks:   List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    total = 0

    for path in file_paths:
        if total >= num_jets:
            break
        with h5py.File(path, "r") as fh:
            jet_features: np.ndarray = fh["jetFeatures"][:]
            if load_mask is not None:
                mask_idx = TASK_IDX[load_mask]
                task_mask = fh["deformationMasks"][:, mask_idx].astype(bool)
            if apply_cuts:
                cut_mask = (
                    (jet_features[:, idx_pt] > pt_cut)
                    & (jet_features[:, idx_sdmass] > mass_win_lo)
                    & (jet_features[:, idx_sdmass] < mass_win_hi)
                )
            else:
                cut_mask = np.ones(len(jet_features), dtype=bool)

            n_pass = int(cut_mask.sum())
            if n_pass == 0:
                continue

            pfc: np.ndarray = fh["jetConstituentsList"][:, :num_particles, :]
            pfc = pfc[cut_mask]  # (n_pass, P, 3) — [deta, dphi, pt]
            if load_mask is not None:
                task_mask = task_mask[cut_mask]
        deta_chunks.append(pfc[:, :, 0])
        dphi_chunks.append(pfc[:, :, 1])
        pt_chunks.append(pfc[:, :, 2])
        mask_chunks.append(task_mask) if load_mask is not None else None

        total += n_pass

    if total == 0:
        raise RuntimeError(f"No jets loaded from config {config_path}. Check paths and cuts.")
    if total < num_jets:
        print(f"  Warning: only {total} jets available; requested {num_jets}.")

    part_deta = np.concatenate(deta_chunks, axis=0)[:num_jets].astype(np.float32)
    part_dphi = np.concatenate(dphi_chunks, axis=0)[:num_jets].astype(np.float32)
    part_pt   = np.concatenate(pt_chunks,   axis=0)[:num_jets].astype(np.float32)
    part_mask = np.concatenate(mask_chunks, axis=0)[:num_jets] if load_mask is not None else None
    features= compute_features_batch(
        part_deta, part_dphi, part_pt,
        eec_bins=EEC_BINS,
        e2_betas=E2_BETAS,
        e3_betas=E3_BETAS,
        chunk_size=CHUNK_SIZE,
    )
    if load_mask is not None:
        return features, part_mask
    return features


def generate_dataset(
    bkg_pool: np.ndarray,
    sig_pool: np.ndarray,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample N_EVENTS_PER_CLASS jets from each pool; return (features, labels)."""
    rng = np.random.default_rng(seed)
    bkg_idx = rng.choice(
        len(bkg_pool), size=N_EVENTS_PER_CLASS,
        replace=(len(bkg_pool) < N_EVENTS_PER_CLASS),
    )
    sig_idx = rng.choice(
        len(sig_pool), size=N_EVENTS_PER_CLASS,
        replace=(len(sig_pool) < N_EVENTS_PER_CLASS),
    )
    features = np.vstack([bkg_pool[bkg_idx], sig_pool[sig_idx]])
    labels = np.concatenate([
        np.zeros(N_EVENTS_PER_CLASS, dtype=int),
        np.ones(N_EVENTS_PER_CLASS,  dtype=int),
    ])
    return features, labels


# ---------------------------------------------------------------------------
# Analysis — identical to generate_learning_results.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_mlp(in_dim: int, mlp_dims: list[int], out_dim: int) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    current = in_dim
    for h in mlp_dims:
        layers.append(torch.nn.Linear(current, h))
        layers.append(torch.nn.ReLU())
        current = h
    layers.append(torch.nn.Linear(current, out_dim))
    return torch.nn.Sequential(*layers)


class PhysicsInformedClassifier(torch.nn.Module):
    def __init__(self, operator2: np.ndarray, operator3: np.ndarray | None, hidden_dim: int) -> None:
        super().__init__()
        self.register_buffer("operator2", torch.tensor(operator2, dtype=torch.float32))
        self.register_buffer(
            "operator3",
            torch.tensor(operator3, dtype=torch.float32) if operator3 is not None else None,
        )
        n = operator2.shape[0]
        self.self_weight = torch.nn.Parameter(torch.randn(n, hidden_dim) * 0.1)
        self.graph_mlp   = make_mlp(n, MLP_DIM, hidden_dim)
        self.hyper_mlp   = make_mlp(n, MLP_DIM, hidden_dim)
        self.readout     = torch.nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        messages  = (features.unsqueeze(-1) * self.self_weight).mean(dim=1)
        messages += self.graph_mlp(features @ self.operator2.T)
        if self.operator3 is not None:
            messages += self.hyper_mlp(features @ self.operator3.T)
        return self.readout(torch.relu(messages)).squeeze(-1)


def make_split(labels: np.ndarray, *, train_size: int, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    n_train_per_class = train_size // 2

    train_indices: list[int] = []
    val_indices:   list[int] = []
    test_indices:  list[int] = []
    for cls in (0, 1):
        indices = np.flatnonzero(labels == cls)
        indices = rng.permutation(indices)
        train_indices.extend(indices[:n_train_per_class])
        val_start = n_train_per_class
        val_stop  = val_start + N_VAL_PER_CLASS
        val_indices.extend(indices[val_start:val_stop])
        test_start = val_stop
        test_stop  = test_start + N_TEST_PER_CLASS
        test_indices.extend(indices[test_start:test_stop])

    return Split(
        train=np.array(train_indices, dtype=int),
        val=np.array(val_indices,   dtype=int),
        test=np.array(test_indices,  dtype=int),
    )


def hypergraph_operator(n_features: int, hyperedges: list[tuple[float, ...]]) -> np.ndarray:
    incidence = np.zeros((n_features, len(hyperedges)), dtype=float)
    weights   = np.zeros(len(hyperedges), dtype=float)
    for edge_idx, edge in enumerate(hyperedges):
        weights[edge_idx] = edge[0]
        for node in edge[1:]:
            incidence[int(node), edge_idx] = 1.0

    node_degree = incidence @ weights
    edge_degree = incidence.sum(axis=0)

    inv_sqrt_node = np.diag(1.0 / np.sqrt(node_degree + 1.0e-12))
    inv_edge      = np.diag(1.0 / (edge_degree + 1.0e-12))
    return inv_sqrt_node @ incidence @ np.diag(weights) @ inv_edge @ incidence.T @ inv_sqrt_node


def build_operators(
    reference_features: np.ndarray,
    *,
    random_triplets: bool = False,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    fisher2 = np.cov(reference_features, rowvar=False, bias=True)
    fisher3 = np.einsum("ni,nj,nk->ijk", reference_features, reference_features, reference_features)
    fisher3 = fisher3 / len(reference_features)

    pair_edges: list[tuple[float, int, int]] = []
    for i in range(fisher2.shape[0]):
        for j in range(i + 1, fisher2.shape[0]):
            pair_edges.append((abs(float(fisher2[i, j])), i, j))
    pair_edges.sort(reverse=True)
    top_pair_edges = pair_edges[:TOP_PAIR_EDGES]
    operator2 = hypergraph_operator(fisher2.shape[0], top_pair_edges)

    triplet_edges: list[tuple[float, int, int, int, float]] = []
    for i, j, k in combinations(range(fisher2.shape[0]), 3):
        signed_weight = float(fisher3[i, j, k])
        triplet_edges.append((abs(signed_weight), i, j, k, signed_weight))
    triplet_edges.sort(reverse=True)
    top_triplets = triplet_edges[:TOP_TRIPLET_EDGES]

    if random_triplets:
        rng = np.random.default_rng(rng_seed)
        all_triplets = list(combinations(range(fisher2.shape[0]), 3))
        chosen = rng.choice(len(all_triplets), size=TOP_TRIPLET_EDGES, replace=False)
        triplets_for_operator = [
            (top_triplets[idx][0], *all_triplets[int(choice)])
            for idx, choice in enumerate(chosen)
        ]
    else:
        triplets_for_operator = [(edge[0], edge[1], edge[2], edge[3]) for edge in top_triplets]

    operator3 = hypergraph_operator(fisher2.shape[0], triplets_for_operator)
    return operator2, operator3, top_pair_edges, top_triplets


def evaluate_mode(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    split: Split,
    mode: str,
    repetition: int,
) -> dict[str, object]:
    reference_train = features[split.train][labels[split.train] == 0]
    mean = reference_train.mean(axis=0)
    std  = reference_train.std(axis=0) + 1.0e-9
    standardized = (features - mean) / std
    reference_standardized = standardized[split.train][labels[split.train] == 0]
    reference_standardized = standardized[labels == 0] # use all background for more statistics
    operator2, operator3, top_pairs, top_triplets = build_operators(
        reference_standardized,
        random_triplets=(mode == "random_hypergraph"),
        rng_seed=90_000 + repetition,
    )

    torch.manual_seed(80_000 + repetition)
    if mode == "graph":
        model = PhysicsInformedClassifier(operator2, None, HIDDEN_DIM)
    else:
        model = PhysicsInformedClassifier(operator2, operator3, HIDDEN_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn   = torch.nn.BCEWithLogitsLoss()

    x_train = torch.tensor(standardized[split.train], dtype=torch.float32)
    y_train = torch.tensor(labels[split.train],       dtype=torch.float32)
    x_val   = torch.tensor(standardized[split.val],   dtype=torch.float32)
    x_test  = torch.tensor(standardized[split.test],  dtype=torch.float32)
    y_val   = labels[split.val]
    y_test  = labels[split.test]

    val_auc_curve: list[float] = []
    best_val_auc = -np.inf
    best_epoch   = -1
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss_fn(model(x_train), y_train).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(x_val)).cpu().numpy()
        sample_weights = compute_sample_weight('balanced', y_val)
        val_auc = float(roc_auc_score(y_val, val_scores, sample_weight=sample_weights))
        val_auc_curve.append(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch + 1
            best_state   = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(model(x_test)).cpu().numpy()
    sample_weights = compute_sample_weight('balanced', y_test)
    test_auc = float(roc_auc_score(y_test, test_scores, sample_weight=sample_weights))

    return {
        "test_auc":      test_auc,
        "best_val_auc":  float(best_val_auc),
        "best_epoch":    best_epoch,
        "val_auc_curve": val_auc_curve,
        "top_pairs":     top_pairs,
        "top_triplets":  top_triplets,
    }


def summarize(bkg_pool: np.ndarray, sig_pool: np.ndarray) -> dict[str, object]:
    mode_titles = {
        "graph":            "Pairwise Fisher graph",
        "hypergraph":       "Triality-informed hypergraph",
        "random_hypergraph": "Random-triplet hypergraph control",
    }

    all_results: dict[int, dict[str, list[dict[str, object]]]] = {
        train_size: {mode: [] for mode in mode_titles}
        for train_size in TRAIN_SIZES
    }

    representative_pairs:   list | None = None
    representative_triplets: list | None = None

    for repetition in range(N_REPETITIONS):
        print(f"  Repetition {repetition + 1}/{N_REPETITIONS}...")
        features, labels = generate_dataset(bkg_pool, sig_pool, seed=repetition)
        for train_size in TRAIN_SIZES:
            split = make_split(labels, train_size=train_size, seed=70_000 + repetition)
            for mode in mode_titles:
                result = evaluate_mode(
                    features=features,
                    labels=labels,
                    split=split,
                    mode=mode,
                    repetition=10_000 * repetition + train_size,
                )
                all_results[train_size][mode].append(result)
                if (
                    representative_pairs is None
                    and representative_triplets is None
                    and train_size == TRAIN_SIZES[1]
                    and mode == "hypergraph"
                    and repetition == 0
                ):
                    representative_pairs   = result["top_pairs"]
                    representative_triplets = result["top_triplets"]

    if representative_pairs is None or representative_triplets is None:
        raise RuntimeError("Failed to record representative hypergraph structure.")

    train_size_curve = []
    for train_size in TRAIN_SIZES:
        summary = {"n_train_total": train_size}
        for mode in mode_titles:
            test_aucs   = [item["test_auc"]   for item in all_results[train_size][mode]]
            best_epochs = [item["best_epoch"] for item in all_results[train_size][mode]]
            summary[mode] = {
                "mean_test_auc":  float(np.mean(test_aucs)),
                "std_test_auc":   float(np.std(test_aucs)),
                "mean_best_epoch": float(np.mean(best_epochs)),
            }
        train_size_curve.append(summary)

    representative_train_size = TRAIN_SIZES[1]
    epoch_curve = {"n_train_total": representative_train_size, "epochs": list(range(1, N_EPOCHS + 1))}
    for mode in mode_titles:
        curves = np.array(
            [item["val_auc_curve"] for item in all_results[representative_train_size][mode]],
            dtype=float,
        )
        epoch_curve[mode] = {
            "mean_val_auc": curves.mean(axis=0).tolist(),
            "std_val_auc":  curves.std(axis=0).tolist(),
        }

    top_pairs = [
        {
            "features":       [FEATURE_PLAIN[i], FEATURE_PLAIN[j]],
            "latex_features": [FEATURE_LABELS[i], FEATURE_LABELS[j]],
            "weight":         float(weight),
        }
        for weight, i, j in representative_pairs[:8]
    ]
    top_triplets = [
        {
            "features":       [FEATURE_PLAIN[i], FEATURE_PLAIN[j], FEATURE_PLAIN[k]],
            "latex_features": [FEATURE_LABELS[i], FEATURE_LABELS[j], FEATURE_LABELS[k]],
            "abs_weight":    float(abs_weight),
            "signed_weight": float(signed_weight),
        }
        for abs_weight, i, j, k, signed_weight in representative_triplets[:8]
    ]

    return {
        "benchmark": {
            "task_name":         TASK_NAME,
            "task_title":        TASK_TITLE,
            "n_events_per_class": N_EVENTS_PER_CLASS,
            "n_val_per_class":   N_VAL_PER_CLASS,
            "n_test_per_class":  N_TEST_PER_CLASS,
            "n_particles":       N_PARTICLES,
            "train_sizes":       TRAIN_SIZES,
            "n_repetitions":     N_REPETITIONS,
            "n_epochs":          N_EPOCHS,
            "hidden_dim":        HIDDEN_DIM,
            "top_pair_edges":    TOP_PAIR_EDGES,
            "top_triplet_edges": TOP_TRIPLET_EDGES,
            "learning_rate":     LEARNING_RATE,
            "weight_decay":      WEIGHT_DECAY,
            "n_bkg_pool":        len(bkg_pool),
            "n_sig_pool":        len(sig_pool),
        },
        "mode_titles":            mode_titles,
        "train_size_curve":       train_size_curve,
        "epoch_curve":            epoch_curve,
        "top_reference_pairs":    top_pairs,
        "top_reference_triplets": top_triplets,
    }


# ---------------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------------

def save_metrics(results: dict, path: Path) -> None:
    modes = list(results["mode_titles"].keys())
    curve = results["train_size_curve"]
    epoch_curve = results["epoch_curve"]
    data = {
        "train_sizes":       np.array([item["n_train_total"] for item in curve]),
        "epochs":            np.array(epoch_curve["epochs"]),
        "epoch_train_size":  np.array(epoch_curve["n_train_total"]),
        "task_title":        np.array(results["benchmark"]["task_title"]),
        "n_epochs":          np.array(results["benchmark"]["n_epochs"]),
    }
    for mode in modes:
        data[f"{mode}_mean_auc"]     = np.array([item[mode]["mean_test_auc"] for item in curve])
        data[f"{mode}_std_auc"]      = np.array([item[mode]["std_test_auc"]  for item in curve])
        data[f"{mode}_mean_val_auc"] = np.array(epoch_curve[mode]["mean_val_auc"])
        data[f"{mode}_std_val_auc"]  = np.array(epoch_curve[mode]["std_val_auc"])
    np.savez(path, **data)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figure(results: dict[str, object]) -> None:
    colors = {
        "graph":             "#9b2c2c",
        "hypergraph":        "#1f4f82",
        "random_hypergraph": "#7a6f5a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    curve       = results["train_size_curve"]
    train_sizes = [item["n_train_total"] for item in curve]
    ylims=[]
    for mode, label in results["mode_titles"].items():
        if mode == "random_hypergraph":
            continue
        means = [item[mode]["mean_test_auc"] for item in curve]
        stds  = [item[mode]["std_test_auc"]  for item in curve]
        ylims.append(min(np.array(means)-np.array(stds)) - 0.02)
        ylims.append(max(np.array(means)+np.array(stds)) + 0.02)
        axes[0].errorbar(
            train_sizes, means, yerr=stds,
            color=colors[mode], linewidth=2.2, marker="o", capsize=3.5, label=label,
        )

    axes[0].set_xlabel("Labeled training events")
    axes[0].set_ylabel("Test AUC")
    axes[0].set_xticks(train_sizes)
    axes[0].set_ylim(min(ylims), max(ylims))
    #axes[0].set_xlim(0,75)
    axes[0].set_title(TASK_TITLE)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    epoch_curve = results["epoch_curve"]
    epochs      = np.array(epoch_curve["epochs"])
    for mode, label in results["mode_titles"].items():
        if mode == "random_hypergraph":
            continue
        mean_auc = np.array(epoch_curve[mode]["mean_val_auc"])
        std_auc  = np.array(epoch_curve[mode]["std_val_auc"])
        axes[1].plot(epochs, mean_auc, color=colors[mode], linewidth=2.0, label=label)
        axes[1].fill_between(
            epochs, mean_auc - std_auc, mean_auc + std_auc,
            color=colors[mode], alpha=0.15,
        )

    axes[1].set_xlabel("Training epoch")
    axes[1].set_ylabel("Validation AUC")
    axes[1].set_xlim(1, N_EPOCHS)
    axes[1].set_ylim((mean_auc-std_auc).min() - 0.03, (mean_auc+std_auc).max() + 0.03)
    axes[1].set_title(f"Convergence at {epoch_curve['n_train_total']} labeled events")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_real_summary.pdf")
    fig.savefig(WEB_DIR / "learning_real_summary.png", dpi=500)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learning benchmark on real HDF5 jet data (bkg: ttbar SM, sig: ttbar EFT)."
    )
    parser.add_argument(
        "--bkg-config",
        type=str,
        default="configs/ttbar_sm.yaml",
        metavar="PATH",
        help="YAML config for background (ttbar SM) HDF5 files (default: configs/ttbar_sm.yaml).",
    )
    parser.add_argument(
        "--sig-config",
        type=str,
        default="configs/ttbar_eft.yaml",
        metavar="PATH",
        help="YAML config for signal (ttbar EFT) HDF5 files (default: configs/ttbar_eft.yaml).",
    )
    parser.add_argument(
        "--num-jets",
        type=int,
        default=50000,
        metavar="N",
        help="Jets to load per class for the sampling pool (default: 10000).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    global CHUNK_SIZE, N_PARTICLES
    CHUNK_SIZE  = args.chunk_size
    N_PARTICLES = args.num_particles

    print(f"Loading background jets from {args.bkg_config} ...")
    bkg_pool,task_mask = load_features(args.bkg_config, args.num_jets, N_PARTICLES, load_mask = "soft_4th_prong")
    sig_pool = bkg_pool[task_mask]
    bkg_pool = bkg_pool[~task_mask]
    print(f"  -> {len(bkg_pool)} background jets, feature shape {bkg_pool.shape}")
    print(f"  -> {len(sig_pool)} signal jets, feature shape {sig_pool.shape}")
    
    #print(f"Loading signal jets from {args.sig_config} ...")
    #sig_pool = load_features(args.sig_config, args.num_jets, N_PARTICLES)
    #print(f"  -> {len(sig_pool)} signal jets, feature shape {sig_pool.shape}")

    print("Running learning benchmark...")
    results = summarize(bkg_pool, sig_pool)

    out_json = GENERATED_DIR / "learning_results_real.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    make_figure(results)
    save_metrics(results, FIGURES_DIR / "metrics.npz")

    print(f"Wrote {out_json.relative_to(ROOT)}")
    print(f"Wrote {(FIGURES_DIR / 'learning_real_summary.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
