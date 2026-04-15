#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from tqdm import tqdm

from dataclasses import dataclass

from configs.observables import FEATURE_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_RESULTS_ROOT = Path("/work/abal/triality/results")
_RUNS_ROOT = Path("/work/abal/triality/RUNS")
_WEB_ROOT = Path("/web/abal/public_html/plots/triality")

N_FEATURES = len(FEATURE_LABELS)


@dataclass(frozen=True)
class RunPaths:
    figures_dir: Path
    web_dir: Path
    run_dir: Path
    save_path: Path
    class0_data: Path
    class1_data: Path
    metadata: Path

    def makedirs(self) -> None:
        for d in (self.figures_dir, self.web_dir, self.run_dir):
            d.mkdir(exist_ok=True, parents=True)


def build_paths(comparison_type: str) -> RunPaths:
    if comparison_type == "ttbar_vs_wqq":
        run_dir = _RUNS_ROOT / "ttbar_VS_wqq"
        return RunPaths(
            figures_dir=_RESULTS_ROOT / "w_vs_top" / "figures",
            web_dir=_WEB_ROOT / "w_vs_top",
            run_dir=run_dir,
            save_path=run_dir / "auc.npz",
            class0_data=run_dir / "data_WtoQQ.npy",
            class1_data=run_dir / "data_TTBar_SM.npy",
            metadata=run_dir / "metadata.npz",
        )
    elif comparison_type == "ttbar_eft":
        run_dir = _RUNS_ROOT / "ttbar_eft"
        return RunPaths(
            figures_dir=_RESULTS_ROOT / "eft" / "figures",
            web_dir=_WEB_ROOT / "eft",
            run_dir=run_dir,
            save_path=run_dir / "auc.npz",
            class0_data=run_dir / "data_TTBar_SM.npy",
            class1_data=run_dir / "data_TTBar_EFT.npy",
            metadata=run_dir / "metadata.npz",
        )
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type!r}")


def load_data(paths: RunPaths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load datasets and metadata using the provided RunPaths.

    Returns
    -------
    X : np.ndarray, shape (N, F)
        Combined feature matrix (already standardised by generate_eft_results).
    y : np.ndarray, shape (N,)
        Binary labels.
    meta : dict
        Loaded metadata npz file.
    """
    log.info("Loading class-0 data from %s", paths.class0_data)
    X0 = np.load(paths.class0_data)
    log.info("Loading class-1 data from %s", paths.class1_data)
    X1 = np.load(paths.class1_data)
    log.info("Loading metadata from %s", paths.metadata)
    meta = np.load(paths.metadata)

    y0 = np.zeros(len(X0), dtype=int)
    y1 = np.ones(len(X1), dtype=int)

    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([y0, y1], axis=0)

    log.info(
        "Dataset: N=%d (class0=%d, class1=%d), F=%d",
        len(X), len(X0), len(X1), X.shape[1],
    )
    return X, y, meta


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    seed: int = 42,
    cv: int = 1,
) -> tuple[float, float]:
    """Train a GBT classifier on the selected features and return (mean AUC, std AUC).

    If cv == 1, a single 30/20/50 train/val/test split is used and std is 0.
    If cv > 1, stratified k-fold cross-validation is used across the full dataset.
    """
    X_sub = X[:, feature_indices]
    d = len(feature_indices)
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=min(3, d),
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )

    if cv > 1:
        scores = cross_val_score(
            clf, X_sub, y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
            scoring="roc_auc",
            n_jobs=cv,
        )
        return float(scores.mean()), float(scores.std())

    # single split: 30/20/50
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_sub, y, test_size=0.50, random_state=seed, stratify=y
    )
    X_train, _, y_train, _ = train_test_split(
        X_trainval, y_trainval, test_size=0.40, random_state=seed, stratify=y_trainval
    )
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    return float(auc), 0.0


def run_auc_scan(
    X: np.ndarray,
    y: np.ndarray,
    pair_rank_idx: np.ndarray,
    hyper_rank_idx: np.ndarray,
    seed: int = 42,
    cv: int = 1,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Scan D from 2 to N_FEATURES, returning (means, stds) for each selection."""
    n_features = X.shape[1]
    pair_means, pair_stds = [], []
    hyper_means, hyper_stds = [], []

    for d in tqdm(range(2, n_features + 1), desc="Feature subset size D"):
        p_mean, p_std = train_and_evaluate(X, y, pair_rank_idx[:d], seed=seed, cv=cv)
        h_mean, h_std = train_and_evaluate(X, y, hyper_rank_idx[:d], seed=seed, cv=cv)

        log.debug("D=%d  pair=%.4f±%.4f  hyper=%.4f±%.4f", d, p_mean, p_std, h_mean, h_std)
        pair_means.append(p_mean); pair_stds.append(p_std)
        hyper_means.append(h_mean); hyper_stds.append(h_std)

    return pair_means, pair_stds, hyper_means, hyper_stds


def make_auc_figure(
    pair_means: list[float],
    pair_stds: list[float],
    hyper_means: list[float],
    hyper_stds: list[float],
    comparison_type: str,
    paths: RunPaths,
) -> None:
    n_features = len(FEATURE_LABELS)
    ks = np.arange(2, n_features + 1)
    pair_means_a = np.array(pair_means)
    pair_stds_a = np.array(pair_stds)
    hyper_means_a = np.array(hyper_means)
    hyper_stds_a = np.array(hyper_stds)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for means, stds, color, marker, label in [
        (pair_means_a,  pair_stds_a,  "#9b2c2c", "o", "Pairwise graph selection"),
        (hyper_means_a, hyper_stds_a, "#1f4f82", "s", "Fisher hypergraph selection"),
    ]:
        ax.plot(ks, means, color=color, linewidth=2.2, marker=marker, markersize=5, label=label)
        if np.any(stds > 0):
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.18)

    ax.set_xlabel("Retained observables ($D$)")
    ax.set_ylabel("Test AUC")
    ax.set_xticks(ks)
    ax.set_ylim(0.98, 0.99)
    title_map = {
        "ttbar_vs_wqq": r"AUC vs. number of features: $t\bar{t}$ vs. $W\!\to\!qq$",
        "ttbar_eft": r"AUC vs. number of features: $t\bar{t}$ SM vs. EFT",
    }
    ax.set_title(title_map.get(comparison_type, "AUC vs. retained observables"))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()

    pdf_path = paths.figures_dir / "auc_scores.pdf"
    png_web = paths.web_dir / "auc_scores.png"

    fig.savefig(pdf_path)
    log.info("Saved figure to %s", pdf_path)
    fig.savefig(png_web, dpi=150)
    log.info("Saved figure to %s", png_web)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifier AUC scan over feature subsets defined by triality ranking."
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load precomputed datasets instead of building them from scratch.",
    )
    parser.add_argument(
        "--type",
        choices=["ttbar_vs_wqq", "ttbar_eft"],
        default="ttbar_vs_wqq",
        help="Type of comparison to perform.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--cross-validate", dest="cv", type=int, default=1,
        metavar="K",
        help="Number of CV folds. Use 1 (default) for a single train/test split, or K>1 for stratified k-fold.",
    )
    args = parser.parse_args()
    if args.cv < 1:
        parser.error("--cross-validate must be >= 1")

    paths = build_paths(args.type)
    paths.makedirs()
    log.info("Comparison type: %s", args.type)

    if args.load and paths.save_path.exists():
        log.info("Loading precomputed AUC results from %s", paths.save_path)
        auc_data = np.load(paths.save_path)
        pair_means  = auc_data["pair_auc_mean"].tolist()
        pair_stds   = auc_data["pair_auc_std"].tolist()
        hyper_means = auc_data["hyper_auc_mean"].tolist()
        hyper_stds  = auc_data["hyper_auc_std"].tolist()
    else:
        if args.load:
            log.warning("--load specified but %s not found; running AUC scan instead.", paths.save_path)
        X, y, meta = load_data(paths)

        pair_rank_idx: np.ndarray = meta["pair_rank_idx"].astype(int)
        hyper_rank_idx: np.ndarray = meta["hyper_rank_idx"].astype(int)

        log.info("pair_rank_idx : %s", pair_rank_idx)
        log.info("hyper_rank_idx: %s", hyper_rank_idx)

        log.info("Starting AUC scan from D=2 to D=%d (cv=%d)", X.shape[1], args.cv)
        pair_means, pair_stds, hyper_means, hyper_stds = run_auc_scan(
            X, y, pair_rank_idx, hyper_rank_idx, seed=args.seed, cv=args.cv
        )
        np.savez(
            paths.save_path,
            pair_auc_mean=pair_means, pair_auc_std=pair_stds,
            hyper_auc_mean=hyper_means, hyper_auc_std=hyper_stds,
        )

    log.info("AUC results:")
    for d, (pm, ps, hm, hs) in enumerate(zip(pair_means, pair_stds, hyper_means, hyper_stds), start=2):
        log.info("  D=%d  pair=%.4f±%.4f  hyper=%.4f±%.4f", d, pm, ps, hm, hs)

    make_auc_figure(pair_means, pair_stds, hyper_means, hyper_stds, args.type, paths)


if __name__ == "__main__":
    main()
