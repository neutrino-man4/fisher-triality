"""
Data handler for augmented HDF5 files produced by h5_subjet_finder.py.

Discovers run_*/filename files from a YAML config (same schema as dataset_builder),
loads constituent kinematics and deformation masks, and computes vectorized
observable features (EEC bins, e2^beta, e3^beta) in batches.
"""

from __future__ import annotations

import logging
import pathlib
from typing import List, Optional, Tuple

import h5py
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / file discovery
# ---------------------------------------------------------------------------

def load_config(config_path: str | pathlib.Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def discover_signal_files(config_path: str | pathlib.Path) -> List[pathlib.Path]:
    """Return sorted run_*/signal_filename paths found under signal_base."""
    cfg = load_config(config_path)
    signal_base = pathlib.Path(cfg["paths"]["signal_base"])
    signal_filename = cfg["paths"]["signal_filename"]
    paths = sorted(signal_base.glob(f"run_*/{signal_filename}"))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {signal_base}/run_*/{signal_filename}. "
            "Check signal_base and signal_filename in the config."
        )
    logger.info("Discovered %d HDF5 file(s) under %s", len(paths), signal_base)
    for p in paths:
        logger.info("  %s", p)
    return paths


# ---------------------------------------------------------------------------
# HDF5 loading
# ---------------------------------------------------------------------------

def load_jets_and_masks(
    file_paths: List[pathlib.Path],
    num_jets: int,
    num_particles: int = 25,
    pt_cut: float = 0.0,
    mass_win_lo: float = 0.0,
    mass_win_hi: float = float("inf"),
    idx_pt: int = 0,
    idx_sdmass: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load constituent kinematics and deformation masks from augmented HDF5 files.

    Applies optional kinematic cuts using jetFeatures columns.  Files are
    opened sequentially and reading stops as soon as num_jets passing jets
    are collected.

    Parameters
    ----------
    file_paths    : Ordered list of augmented HDF5 file paths.
    num_jets      : Maximum number of jets to return.
    num_particles : Leading-pT constituents to use per jet (truncated to P).
    pt_cut        : Minimum jet pT [GeV]; 0.0 disables the cut.
    mass_win_lo   : Soft-drop mass lower edge [GeV]; 0.0 disables.
    mass_win_hi   : Soft-drop mass upper edge [GeV]; inf disables.
    idx_pt        : Column index of jet pT in jetFeatures.
    idx_sdmass    : Column index of jet soft-drop mass in jetFeatures.

    Returns
    -------
    part_deta  : float32 (N, P) — constituent delta-eta relative to jet axis
    part_dphi  : float32 (N, P) — constituent delta-phi relative to jet axis
    part_pt    : float32 (N, P) — constituent pT [GeV]; padded entries are 0
    masks      : bool   (N, 6) — deformation-selection masks
    mask_names : list[str] of length 6
    """
    deta_chunks: List[np.ndarray] = []
    dphi_chunks: List[np.ndarray] = []
    pt_chunks:   List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    mask_names:  Optional[List[str]] = None
    total = 0

    apply_cuts = (pt_cut > 0.0) or (mass_win_lo > 0.0) or (mass_win_hi < float("inf"))

    for path in file_paths:
        if total >= num_jets:
            break

        logger.info("Opening %s", path)
        with h5py.File(path, "r") as fh:
            jet_features: np.ndarray = fh["jetFeatures"][:]

            if apply_cuts:
                jet_pt     = jet_features[:, idx_pt]
                jet_sdmass = jet_features[:, idx_sdmass]
                cut_mask   = (
                    (jet_pt > pt_cut)
                    & (jet_sdmass > mass_win_lo)
                    & (jet_sdmass < mass_win_hi)
                )
            else:
                cut_mask = np.ones(len(jet_features), dtype=bool)

            n_pass = int(cut_mask.sum())
            logger.info("  %d / %d jets pass kinematic cuts", n_pass, len(jet_features))
            if n_pass == 0:
                continue

            # jetConstituentsList: (N_file, 100, 3) = [deta, dphi, pt]
            pfc: np.ndarray = fh["jetConstituentsList"][:, :num_particles, :]
            pfc = pfc[cut_mask]  # (n_pass, P, 3)

            raw_masks: np.ndarray = fh["deformationMasks"][()]
            #import pdb; pdb.set_trace()
            raw_masks = raw_masks[cut_mask]  # (n_pass, 6)

            if mask_names is None:
                mask_names = [
                    n.decode() if isinstance(n, bytes) else n
                    for n in fh["deformationNames"][:]
                ]

        deta_chunks.append(pfc[:, :, 0])
        dphi_chunks.append(pfc[:, :, 1])
        pt_chunks.append(pfc[:, :, 2])
        mask_chunks.append(raw_masks)
        total += n_pass
        logger.info("  Cumulative jets: %d / %d requested", total, num_jets)

    if total == 0:
        raise RuntimeError("No jets loaded. Check file paths and kinematic cuts.")

    if total < num_jets:
        logger.warning("Only %d jets available; requested %d.", total, num_jets)

    part_deta = np.concatenate(deta_chunks, axis=0)[:num_jets]
    part_dphi = np.concatenate(dphi_chunks, axis=0)[:num_jets]
    part_pt   = np.concatenate(pt_chunks,   axis=0)[:num_jets]
    masks     = np.concatenate(mask_chunks,  axis=0)[:num_jets]

    return (
        part_deta.astype(np.float32),
        part_dphi.astype(np.float32),
        part_pt.astype(np.float32),
        masks.astype(bool),
        mask_names or [],
    )


# ---------------------------------------------------------------------------
# Vectorized observable computation
# ---------------------------------------------------------------------------

def _safe_power(arr: np.ndarray, beta: float) -> np.ndarray:
    """arr ** beta with 0 ** beta = 0 (avoids 0^0=1 for zero-padded entries)."""
    return np.where(
        arr > 0.0,
        np.power(arr, beta, where=arr > 0.0, out=np.zeros_like(arr)),
        0.0,
    )


def _eec_batch(dr_upper: np.ndarray, Z: np.ndarray, bin_lo: float, bin_hi: float) -> np.ndarray:
    mask = (dr_upper > bin_lo) & (dr_upper < bin_hi)
    return (Z * mask).sum(axis=(1, 2))


def _ecf2_batch(dr_upper: np.ndarray, Z: np.ndarray, beta: float) -> np.ndarray:
    return (Z * _safe_power(dr_upper, beta)).sum(axis=(1, 2))


def _ecf3_batch(dr_full: np.ndarray, z: np.ndarray, beta: float) -> np.ndarray:
    """O(P^2) factored three-point ECF (see dataset_builder.py for derivation)."""
    dr_beta = _safe_power(dr_full, beta)
    A = dr_beta * z[:, None, :]          # A[b,i,j] = z_j * DR_{ij}^beta
    F = np.matmul(A, dr_beta)            # F[b,i,k] = sum_j z_j DR_{ij}^beta DR_{jk}^beta
    Z_full = z[:, :, None] * z[:, None, :]
    return (Z_full * dr_beta * F).sum(axis=(1, 2)) / 6.0


def compute_features_batch(
    part_deta: np.ndarray,
    part_dphi: np.ndarray,
    part_pt:   np.ndarray,
    eec_bins:  List[Tuple[float, float]],
    e2_betas:  List[float],
    e3_betas:  List[float],
    chunk_size: int = 500,
) -> np.ndarray:
    """
    Compute observable features for all jets in vectorized chunks.

    Parameters
    ----------
    part_deta, part_dphi : (N, P) float32 — constituent angular coordinates
    part_pt              : (N, P) float32 — constituent pT; 0 for padded slots
    eec_bins             : list of (lo, hi) angular bin edges
    e2_betas             : list of angular exponents for 2-point ECF
    e3_betas             : list of angular exponents for 3-point ECF
    chunk_size           : jets per inner batch (memory / speed trade-off)

    Returns
    -------
    features : float64 (N, D) where D = len(eec_bins) + len(e2_betas) + len(e3_betas)
    """
    N = part_pt.shape[0]
    D = len(eec_bins) + len(e2_betas) + len(e3_betas)
    features = np.empty((N, D), dtype=np.float64)

    n_chunks = (N + chunk_size - 1) // chunk_size
    for chunk_idx in range(n_chunks):
        lo = chunk_idx * chunk_size
        hi = min(lo + chunk_size, N)

        deta_b = part_deta[lo:hi].astype(np.float64)  # (B, P)
        dphi_b = part_dphi[lo:hi].astype(np.float64)
        pt_b   = part_pt[lo:hi].astype(np.float64)

        # pT fractions — renormalised over real constituents only
        pt_sum = pt_b.sum(axis=1, keepdims=True)
        z = np.where(pt_sum > 0.0, pt_b / pt_sum, 0.0)  # (B, P)

        # pairwise Delta-R matrices
        deta_ij = deta_b[:, :, None] - deta_b[:, None, :]  # (B, P, P)
        dphi_ij = dphi_b[:, :, None] - dphi_b[:, None, :]
        dphi_ij = (dphi_ij + np.pi) % (2.0 * np.pi) - np.pi
        dr_full  = np.sqrt(deta_ij ** 2 + dphi_ij ** 2)    # (B, P, P)

        upper = np.triu(np.ones((dr_full.shape[1], dr_full.shape[1]), dtype=bool), k=1)
        dr_upper = dr_full * upper[np.newaxis, :, :]        # (B, P, P)

        Z_upper = z[:, :, None] * z[:, None, :]  # cached for EEC / ECF2

        col = 0
        for bin_lo, bin_hi in eec_bins:
            features[lo:hi, col] = _eec_batch(dr_upper, Z_upper, bin_lo, bin_hi)
            col += 1
        for beta in e2_betas:
            features[lo:hi, col] = _ecf2_batch(dr_upper, Z_upper, beta)
            col += 1
        for beta in e3_betas:
            features[lo:hi, col] = _ecf3_batch(dr_full, z, beta)
            col += 1

        if (chunk_idx + 1) % 20 == 0 or hi == N:
            logger.info("  Feature computation: %d / %d jets done", hi, N)

    return features
