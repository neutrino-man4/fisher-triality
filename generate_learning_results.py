#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import generate_basis_results as basis


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
GENERATED_DIR = ROOT / "generated"

TRAIN_SIZES = [100, 200, 400]
N_EVENTS_PER_CLASS = 1_200
N_VAL_PER_CLASS = 200
N_TEST_PER_CLASS = 400
N_REPETITIONS = 10
N_EPOCHS = 100
HIDDEN_DIM = 4
TOP_PAIR_EDGES = 60
TOP_TRIPLET_EDGES = 40
LEARNING_RATE = 0.03
WEIGHT_DECAY = 1.0e-4
TASK_NAME = "fourth_prong"
TASK_TITLE = basis.TASK_TITLES[TASK_NAME]


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class PhysicsInformedClassifier(torch.nn.Module):
    def __init__(self, operator2: np.ndarray, operator3: np.ndarray | None, hidden_dim: int) -> None:
        super().__init__()
        self.register_buffer("operator2", torch.tensor(operator2, dtype=torch.float32))
        self.register_buffer(
            "operator3",
            torch.tensor(operator3, dtype=torch.float32) if operator3 is not None else None,
        )

        self.self_weight = torch.nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.graph_weight = torch.nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.hyper_weight = torch.nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.readout = torch.nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        messages = features.unsqueeze(-1) * self.self_weight
        messages = messages + torch.matmul(features, self.operator2.T).unsqueeze(-1) * self.graph_weight
        if self.operator3 is not None:
            messages = messages + torch.matmul(features, self.operator3.T).unsqueeze(-1) * self.hyper_weight
        pooled = torch.relu(messages).mean(dim=1)
        return self.readout(pooled).squeeze(-1)


def generate_dataset(*, seed: int) -> tuple[np.ndarray, np.ndarray]:
    reference = basis.generate_sample(
        basis.REFERENCE_MODEL,
        seed=10_000 + seed,
        n_events=N_EVENTS_PER_CLASS,
    )
    signal = basis.generate_sample(
        basis.DEFORMATION_MODELS[TASK_NAME],
        seed=20_000 + seed,
        n_events=N_EVENTS_PER_CLASS,
    )
    features = np.vstack([reference, signal])
    labels = np.concatenate(
        [
            np.zeros(len(reference), dtype=int),
            np.ones(len(signal), dtype=int),
        ]
    )
    return features, labels


def make_split(labels: np.ndarray, *, train_size: int, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    n_train_per_class = train_size // 2

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for cls in (0, 1):
        indices = np.flatnonzero(labels == cls)
        indices = rng.permutation(indices)
        train_indices.extend(indices[:n_train_per_class])
        val_start = n_train_per_class
        val_stop = val_start + N_VAL_PER_CLASS
        val_indices.extend(indices[val_start:val_stop])
        test_start = val_stop
        test_stop = test_start + N_TEST_PER_CLASS
        test_indices.extend(indices[test_start:test_stop])

    return Split(
        train=np.array(train_indices, dtype=int),
        val=np.array(val_indices, dtype=int),
        test=np.array(test_indices, dtype=int),
    )


def hypergraph_operator(n_features: int, hyperedges: list[tuple[float, ...]]) -> np.ndarray:
    incidence = np.zeros((n_features, len(hyperedges)), dtype=float)
    weights = np.zeros(len(hyperedges), dtype=float)
    for edge_idx, edge in enumerate(hyperedges):
        weights[edge_idx] = edge[0]
        for node in edge[1:]:
            incidence[int(node), edge_idx] = 1.0

    node_degree = incidence @ weights
    edge_degree = incidence.sum(axis=0)

    inv_sqrt_node = np.diag(1.0 / np.sqrt(node_degree + 1.0e-12))
    inv_edge = np.diag(1.0 / (edge_degree + 1.0e-12))
    return inv_sqrt_node @ incidence @ np.diag(weights) @ inv_edge @ incidence.T @ inv_sqrt_node


def build_operators(
    reference_features: np.ndarray,
    *,
    random_triplets: bool = False,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, int, int]], list[tuple[float, int, int, int, float]]]:
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
    std = reference_train.std(axis=0) + 1.0e-9
    standardized = (features - mean) / std
    reference_standardized = standardized[split.train][labels[split.train] == 0]

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
    loss_fn = torch.nn.BCEWithLogitsLoss()

    x_train = torch.tensor(standardized[split.train], dtype=torch.float32)
    y_train = torch.tensor(labels[split.train], dtype=torch.float32)
    x_val = torch.tensor(standardized[split.val], dtype=torch.float32)
    x_test = torch.tensor(standardized[split.test], dtype=torch.float32)
    y_val = labels[split.val]
    y_test = labels[split.test]

    val_auc_curve: list[float] = []
    best_val_auc = -np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(x_train), y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(x_val)).cpu().numpy()
        val_auc = float(roc_auc_score(y_val, val_scores))
        val_auc_curve.append(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(model(x_test)).cpu().numpy()
    test_auc = float(roc_auc_score(y_test, test_scores))

    return {
        "test_auc": test_auc,
        "best_val_auc": float(best_val_auc),
        "best_epoch": best_epoch,
        "val_auc_curve": val_auc_curve,
        "top_pairs": top_pairs,
        "top_triplets": top_triplets,
    }


def summarize() -> dict[str, object]:
    mode_titles = {
        "graph": "Pairwise Fisher graph",
        "hypergraph": "Triality-informed hypergraph",
        "random_hypergraph": "Random-triplet hypergraph control",
    }

    all_results: dict[int, dict[str, list[dict[str, object]]]] = {
        train_size: {mode: [] for mode in mode_titles}
        for train_size in TRAIN_SIZES
    }

    representative_pairs: list[tuple[float, int, int]] | None = None
    representative_triplets: list[tuple[float, int, int, int, float]] | None = None

    for repetition in range(N_REPETITIONS):
        features, labels = generate_dataset(seed=repetition)
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
                    representative_pairs = result["top_pairs"]
                    representative_triplets = result["top_triplets"]

    if representative_pairs is None or representative_triplets is None:
        raise RuntimeError("Failed to record representative hypergraph structure.")

    train_size_curve = []
    for train_size in TRAIN_SIZES:
        summary = {"n_train_total": train_size}
        for mode in mode_titles:
            test_aucs = [item["test_auc"] for item in all_results[train_size][mode]]
            best_epochs = [item["best_epoch"] for item in all_results[train_size][mode]]
            summary[mode] = {
                "mean_test_auc": float(np.mean(test_aucs)),
                "std_test_auc": float(np.std(test_aucs)),
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
            "std_val_auc": curves.std(axis=0).tolist(),
        }

    top_pairs = [
        {
            "features": [basis.FEATURE_PLAIN[i], basis.FEATURE_PLAIN[j]],
            "latex_features": [basis.FEATURE_LABELS[i], basis.FEATURE_LABELS[j]],
            "weight": float(weight),
        }
        for weight, i, j in representative_pairs[:8]
    ]
    top_triplets = [
        {
            "features": [basis.FEATURE_PLAIN[i], basis.FEATURE_PLAIN[j], basis.FEATURE_PLAIN[k]],
            "latex_features": [basis.FEATURE_LABELS[i], basis.FEATURE_LABELS[j], basis.FEATURE_LABELS[k]],
            "abs_weight": float(abs_weight),
            "signed_weight": float(signed_weight),
        }
        for abs_weight, i, j, k, signed_weight in representative_triplets[:8]
    ]

    return {
        "benchmark": {
            "task_name": TASK_NAME,
            "task_title": TASK_TITLE,
            "n_events_per_class": N_EVENTS_PER_CLASS,
            "n_val_per_class": N_VAL_PER_CLASS,
            "n_test_per_class": N_TEST_PER_CLASS,
            "n_constituents": basis.N_CONSTITUENTS,
            "train_sizes": TRAIN_SIZES,
            "n_repetitions": N_REPETITIONS,
            "n_epochs": N_EPOCHS,
            "hidden_dim": HIDDEN_DIM,
            "top_pair_edges": TOP_PAIR_EDGES,
            "top_triplet_edges": TOP_TRIPLET_EDGES,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "mode_titles": mode_titles,
        "train_size_curve": train_size_curve,
        "epoch_curve": epoch_curve,
        "top_reference_pairs": top_pairs,
        "top_reference_triplets": top_triplets,
    }


def make_figure(results: dict[str, object]) -> None:
    colors = {
        "graph": "#9b2c2c",
        "hypergraph": "#1f4f82",
        "random_hypergraph": "#7a6f5a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    curve = results["train_size_curve"]
    train_sizes = [item["n_train_total"] for item in curve]
    for mode, label in results["mode_titles"].items():
        means = [item[mode]["mean_test_auc"] for item in curve]
        stds = [item[mode]["std_test_auc"] for item in curve]
        axes[0].errorbar(
            train_sizes,
            means,
            yerr=stds,
            color=colors[mode],
            linewidth=2.2,
            marker="o",
            capsize=3.5,
            label=label,
        )

    axes[0].set_xlabel("Labeled training events")
    axes[0].set_ylabel("Test AUC")
    axes[0].set_xticks(train_sizes)
    axes[0].set_ylim(0.60, 0.76)
    axes[0].set_title("Soft fourth-prong classification")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    epoch_curve = results["epoch_curve"]
    epochs = np.array(epoch_curve["epochs"])
    for mode, label in results["mode_titles"].items():
        mean_auc = np.array(epoch_curve[mode]["mean_val_auc"])
        std_auc = np.array(epoch_curve[mode]["std_val_auc"])
        axes[1].plot(epochs, mean_auc, color=colors[mode], linewidth=2.0, label=label)
        axes[1].fill_between(
            epochs,
            mean_auc - std_auc,
            mean_auc + std_auc,
            color=colors[mode],
            alpha=0.15,
        )

    axes[1].set_xlabel("Training epoch")
    axes[1].set_ylabel("Validation AUC")
    axes[1].set_xlim(1, N_EPOCHS)
    axes[1].set_ylim(0.55, 0.80)
    axes[1].set_title(
        f"Convergence at {epoch_curve['n_train_total']} labeled events"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_summary.pdf")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    GENERATED_DIR.mkdir(exist_ok=True)

    results = summarize()

    with (GENERATED_DIR / "learning_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    make_figure(results)

    print(f"Wrote {(GENERATED_DIR / 'learning_results.json').relative_to(ROOT)}")
    print(f"Wrote {(FIGURES_DIR / 'learning_summary.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
