"""
Joint QCD + TTBar observable dataset builder.

Discovers run_XY HDF5 files under the configured base paths, applies
kinematic selection (pT > 300 GeV, 140 < sdmass < 200 GeV), computes
four jet-level observables:

    EEC_narrow  = Σ_{i<j, 0.06 < ΔR < 0.20}  z_i z_j
    EEC_wide    = Σ_{i<j, 0.20 < ΔR < 0.80}  z_i z_j
    e_2^(1)     = Σ_{i<j}        z_i z_j ΔR_ij                  (2-point ECF, β=1)
    e_3^(1)     = Σ_{i<j<k}      z_i z_j z_k ΔR_ij ΔR_ik ΔR_jk (3-point ECF, β=1)

and draws a class-balanced random sample (22 % TTBar, 78 % QCD) of the
requested size.  The result is saved as an (N, 4) float32 array together
with integer class labels to:

    /ceph/abal/QFIT/MC/joint_datasets/KL/data.h5

A per-class mean statistics table is printed at the end.

Author: Aritra Bal (ETP)
Date  : 2026-03-18
"""

import argparse
from itertools import combinations
import logging
import pathlib
from typing import List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# NOTE: The ntuplizer (ntuples_to_h5.py) writes files as {jet_type}.h5, so the
# default QCD filename would be 'qcd_dijet.h5'.  Update QCD_FILENAME below if
# your run used a different --jet-type string.
QCD_BASE    : pathlib.Path = pathlib.Path("/ceph/abal/QFIT/MC/HDF5/qcd_dijet")
TOP_BASE    : pathlib.Path = pathlib.Path("/ceph/abal/QFIT/MC/HDF5/TTBar")
OUTPUT_PATH : pathlib.Path = pathlib.Path("/ceph/abal/QFIT/MC/joint_datasets/KL/data.h5")

QCD_FILENAME : str = "qcd_dijet.h5"
TOP_FILENAME : str = "TTBar.h5"

# ---------------------------------------------------------------------------
# Physics & binning constants
# ---------------------------------------------------------------------------
PT_CUT       : float = 300.0   # GeV  — minimum jet pT
MASS_WIN_LO  : float = 140.0   # GeV  — lower edge of sdmass window
MASS_WIN_HI  : float = 220.0   # GeV  — upper edge of sdmass window
FRAC_TOP     : float = 0.22    # TTBar fraction in the final dataset

DR_NARROW_LO : float = 0.06    # ΔR bin edges for EEC_narrow
DR_NARROW_HI : float = 0.20
DR_WIDE_LO   : float = 0.20    # ΔR bin edges for EEC_wide
DR_WIDE_HI   : float = 0.80

# Column indices in the 'jetFeatures' dataset (see JET_FEATURE_NAMES in ntuplizer)
IDX_PT     : int = 0   # jet_pt
IDX_SDMASS : int = 5   # jet_sdmass

# Jets processed per inner loop in compute_observables (memory ↔ speed trade-off)
# At 2 000 jets: peak RAM ≈ 5 × (2000 × 100 × 100 × 8 B) ≈ 800 MB
COMPUTE_CHUNK : int = 2000


# ---------------------------------------------------------------------------
# Observable computation
# ---------------------------------------------------------------------------
def compute_observables(
    pair_dr   : np.ndarray,   # (N, P, P)  upper-triangular ΔR (i < j only), P ≤ 100
    pt_weights: np.ndarray,   # (N, P)     normalised pT fractions over P particles
    desc      : str = "",
) -> np.ndarray:              # (N, 4)     float64
    """
    Compute [EEC_narrow, EEC_wide, e_2^(1), e_3^(1)] for a batch of jets.

    EEC bins
    --------
    The stored pair_delta_R contains ΔR_{ij} only for i < j (upper triangle).
    Summing Z_{ij} over this mask counts each pair exactly once:

        EEC_bin = Σ_{i<j, ΔR ∈ bin} z_i z_j

    e_2^(1)
    -------
        e_2 = Σ_{i<j} z_i z_j ΔR_{ij}

    e_3^(1) — factored formula
    --------------------------
    Direct evaluation is O(P³) per jet.  The following O(P²) identity is used:

        e_3 = (1/6) Σ_{i,k} z_i z_k ΔR_{ik} F_{ik}

    where:
        F_{ik} = Σ_j z_j ΔR_{ij} ΔR_{jk}   (computed as batched matmul)

    Using the full symmetric ΔR matrix (ΔR_full = DR + DR^T) ensures that
    all six permutations of distinct index triplets (i, j, k) are captured,
    while diagonal entries ΔR_{ii} = 0 suppress degenerate pairs automatically.

    Parameters
    ----------
    pair_dr    : Upper-triangular pairwise ΔR, shape (N, P, P), float32.
                 P is the number of particles used (≤ 100).
    pt_weights : Per-constituent pT fractions normalised over P particles,
                 shape (N, P), float32.
    desc       : Optional tqdm description prefix.

    Returns
    -------
    np.ndarray of shape (N, 4), dtype float64.
    """
    N : int = pair_dr.shape[0]
    out : np.ndarray = np.zeros((N, 4), dtype=np.float64)

    chunks = range(0, N, COMPUTE_CHUNK)
    for lo in tqdm(chunks, desc=desc or "  computing observables", unit="chunk", leave=False):
        hi : int = min(lo + COMPUTE_CHUNK, N)
        # Promote to float64 for numerical safety in triple products
        dr : np.ndarray = pair_dr[lo:hi].astype(np.float64)    # (B, P, P) upper tri
        z  : np.ndarray = pt_weights[lo:hi].astype(np.float64) # (B, P)

        # Full symmetric ΔR matrix (needed for the e_3 factorisation)
        dr_full : np.ndarray = dr + np.transpose(dr, (0, 2, 1)) # (B, P, P)

        # REMINDER AT THIS POINT !!!
        # dr: upper-triangular ΔR (i < j only), shape (B, P, P)
        # dr_full: full symmetric ΔR, shape (B, P, P)
        # z: normalised pT fractions, shape (B, P)
        
        # Outer weight product Z[n,i,j] = z[n,i] · z[n,j]
        Z_full : np.ndarray = z[:, :, None] * z[:, None, :]     # (B, P, P)

        # A[n,i,j] = ΔR_full[n,i,j] · z[n,j]   (broadcast z along axis 1)
        A : np.ndarray = dr_full * z[:, None, :]                # (B, P, P)
        # F[n,i,k] = Σ_j z[n,j] ΔR_full[n,i,j] ΔR_full[n,j,k]
        F : np.ndarray = np.matmul(A, dr_full)                  # (B, P, P)
        
        #import pdb;pdb.set_trace()
        # ---- EEC bins --------------------------------------------------------
        # Use upper-tri 'dr' so each pair (i < j) is counted exactly once.
        out[lo:hi, 0] = (Z_full * ((dr > DR_NARROW_LO) & (dr < DR_NARROW_HI))).sum(axis=(1, 2))
        out[lo:hi, 1] = (Z_full * ((dr > DR_WIDE_LO)   & (dr < DR_WIDE_HI)  )).sum(axis=(1, 2))

        # ---- e_2^(1) ---------------------------------------------------------
        # Upper-tri 'dr' guarantees the i < j ordering.
        out[lo:hi, 2] = (Z_full * dr).sum(axis=(1, 2))

        # ---- e_3^(1) via factored O(P²) formula ------------------------------
        # e3 = (1/6) Σ_{i,k} z_i z_k ΔR_full[i,k] F[i,k]
        out[lo:hi, 3] = (Z_full * dr_full * F).sum(axis=(1, 2)) / 6.0
        # ---- e_3^(1) via explicit triple loop ------------------------------------
        # B  : int        = hi - lo
        # P  : int        = dr_full.shape[1]
        # e3 : np.ndarray = np.zeros(B, dtype=np.float64)

        # for i, j, k in combinations(range(P), 3):
        #     e3 += (z[:, i] * z[:, j] * z[:, k]
        #         * dr_full[:, i, j] * dr_full[:, i, k] * dr_full[:, j, k])
        #out[lo:hi, 3] = e3
    return out


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def discover_files(base: pathlib.Path, filename: str) -> List[pathlib.Path]:
    """
    Return sorted run_XY/<filename> paths found under *base*.

    Parameters
    ----------
    base     : Root directory containing run_01/, run_02/, … sub-directories.
    filename : HDF5 leaf filename (e.g. 'TTBar.h5').

    Returns
    -------
    List of existing Path objects in ascending run order.

    Raises
    ------
    FileNotFoundError if no matching paths are found.
    """
    paths : List[pathlib.Path] = sorted(base.glob(f"run_*/{filename}"))
    if not paths:
        raise FileNotFoundError(
            f"No files found matching: {base / 'run_*' / filename}\n"
            f"Check that QCD_FILENAME / TOP_FILENAME match your ntuplizer output."
        )
    logger.info("Discovered %d file(s) under %s", len(paths), base)
    for p in paths:
        logger.info("  %s", p)
    return paths


# ---------------------------------------------------------------------------
# Per-class data collection
# ---------------------------------------------------------------------------
def collect_observables(
    file_paths   : List[pathlib.Path],
    n_needed     : int,
    label        : int,
    num_particles: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read HDF5 files sequentially, apply kinematic cuts, compute observables,
    and stop as soon as *n_needed* passing jets have been accumulated.

    Additional files beyond the first are opened only when the running total
    of passing jets falls short of *n_needed*.

    Particle truncation
    -------------------
    When *num_particles* > 0, only the first *num_particles* constituents are
    used (the ntuplizer stores them in pT-descending order).  pT fractions are
    recomputed from the raw constituent pT values of those P particles so that
    the normalisation denominator Σᵢ pTᵢ  (i = 0 … P-1) is self-consistent.
    The pairwise ΔR matrix is sliced to the [P × P] upper-left submatrix,
    which h5py reads directly as a hyperslab without loading the full 100×100.

    When *num_particles* ≤ 0, the pre-stored constituent_pt_weight array and
    the full 100×100 pair_delta_R are used (a warning is emitted at startup).

    Parameters
    ----------
    file_paths    : Ordered HDF5 files (run_01 first, …).
    n_needed      : Target number of passing jets to accumulate.
    label         : Integer class label (0 = QCD, 1 = TTBar).
    num_particles : Leading-pT particles to use; ≤ 0 means all 100.

    Returns
    -------
    obs    : ndarray of shape (≥n_needed, 4), float64 — raw observables.
    labels : ndarray of shape (≥n_needed,),  int8   — repeated *label*.

    Raises
    ------
    RuntimeError if all files are exhausted without reaching *n_needed* jets.
    """
    class_tag  : str              = "TTBar" if label == 1 else "QCD"
    use_full_p : bool             = num_particles <= 0
    P          : int              = num_particles if not use_full_p else 100
    obs_chunks : List[np.ndarray] = []
    total      : int              = 0

    file_bar = tqdm(file_paths, desc=f"[{class_tag}] files", unit="file", position=0)
    for path in file_bar:
        file_bar.set_postfix_str(path.parent.name)
        logger.info("[%s] Opening: %s", class_tag, path)

        with h5py.File(path, "r") as fh:
            # ---- Kinematic pre-filter (jet-level, lightweight) ---------------
            jet_features : np.ndarray = fh["jetFeatures"][:]      # (N_file, 13)
            jet_pt       : np.ndarray = jet_features[:, IDX_PT]
            jet_sdmass   : np.ndarray = jet_features[:, IDX_SDMASS]

            mask   : np.ndarray = (
                (jet_pt    >  PT_CUT)
                & (jet_sdmass > MASS_WIN_LO)
                & (jet_sdmass < MASS_WIN_HI)
            )
            idx    : np.ndarray = np.where(mask)[0]   # sorted integer indices
            n_pass : int        = idx.size
            n_total: int        = jet_features.shape[0]

            logger.info(
                "  %d / %d jets pass  (pT > %.0f GeV, %.0f < sdmass < %.0f GeV)",
                n_pass, n_total, PT_CUT, MASS_WIN_LO, MASS_WIN_HI,
            )

            if n_pass == 0:
                logger.warning("  No jets pass cuts in this file — skipping.")
                continue
            constituents : np.ndarray = fh["jetConstituentsList"][:, :P, :]  # (n_pass, P, 3)
            constituents = constituents[mask]
            # ---- Load EEC building blocks for passing jets only --------------
            # h5py supports combined fancy-index on axis-0 + slice on axes 1,2:
            #   dataset[sorted_int_array, :P, :P]
            # This reads only the P×P hyperslab from disk, avoiding the full
            # 100×100 allocation entirely.
            deta     : np.ndarray = constituents[..., 0].astype(np.float32)   # (n_pass, P)
            dphi     : np.ndarray = constituents[..., 1].astype(np.float32)
            part_pt  : np.ndarray = constituents[..., 2].astype(np.float32)

            # Compute upper-triangular pairwise ΔR in situ — O(P²), P=18 → trivial
            deta_diff : np.ndarray = deta[:, :, None] - deta[:, None, :]       # (n_pass, P, P)
            dphi_diff : np.ndarray = dphi[:, :, None] - dphi[:, None, :]
            pair_dr   : np.ndarray = np.sqrt(deta_diff**2 + dphi_diff**2)

            # Zero lower triangle + diagonal (replicate ntuplizer convention: i < j only)
            upper_mask : np.ndarray = np.triu(np.ones((P, P), dtype=bool), k=1)
            pair_dr   *= upper_mask[None, :, :]                                 # broadcast over batch
            if use_full_p:
                # Pre-computed weights normalised over all 100 constituents
                pt_weights : np.ndarray = fh["constituent_pt_weight"][idx]  # (n_pass, 100)
            else:
                # Read raw pT for the first P constituents only.
                # jetConstituentsList layout: (N, 100, 3) = [deta, dphi, pT]
                # Recompute pT fractions normalised over exactly P particles.
                # Padded constituents have pT = 0, so they contribute nothing.
                pt_sum     : np.ndarray = part_pt.sum(axis=1, keepdims=True)   # (n_pass, 1)
                pt_weights = np.where(pt_sum > 0.0, part_pt / pt_sum, 0.0).astype(np.float32)

        # Compute observables; discard large arrays immediately
        obs : np.ndarray = compute_observables(
            pair_dr, pt_weights,
            desc=f"  [{class_tag}] {path.parent.name}",
        )  # (n_pass, 4)
        del pair_dr, pt_weights

        obs_chunks.append(obs)
        total += n_pass
        logger.info(
            "  [%s] Cumulative passing jets: %d / %d needed",
            class_tag, total, n_needed,
        )

        if total >= n_needed:
            break   # do not open further files

    if total < n_needed:
        raise RuntimeError(
            f"Insufficient {class_tag} jets after all files: "
            f"collected {total}, needed {n_needed}.  "
            f"Try reducing --num_jets or widening the kinematic cuts."
        )

    all_obs    : np.ndarray = np.concatenate(obs_chunks, axis=0)          # (total, 4)
    all_labels : np.ndarray = np.full(total, fill_value=label, dtype=np.int8)
    return all_obs, all_labels


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a joint QCD+TTBar observable dataset "
            "(EEC_narrow, EEC_wide, e_2^(1), e_3^(1)) "
            "with configurable size and random seed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-jets",
        type=int,
        default=100000,
        metavar="N",
        help="Total jets in the output dataset.",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=18,
        metavar="P",
        help=(
            "Number of leading-pT constituents used to compute observables. "
            "pT fractions are renormalised over exactly these P particles. "
            "Set to ≤ 0 to use all 100 stored constituents (slow)."
        ),
    )
    # parser.add_argument(
    #     "--frac-top",
    #     type=float,
    #     default=0.22,
    #     metavar="FRAC",
    #     help=(
    #         "Fraction of TTBar jets in the final dataset. "
    #         "The remainder (1-F) will be QCD jets. " \
    #         "Set to 0.0 for pure QCD, 1.0 for pure TTBar."
    #     ),
    # )
    # parser.add_argument(
    #     "--pt-cut",
    #     type=float,
    #     default=0.0,
    #     metavar="PT",
    #     help="Minimum jet pT in GeV for kinematic selection.",
    # )
    # parser.add_argument(
    #     "--sdmass-window",
    #     type=float,
    #     nargs=2,
    #     default=[MASS_WIN_LO, MASS_WIN_HI],
    #     metavar=("LO", "HI"),
    #     help=(
    #         "Soft-drop mass window in GeV for kinematic selection as two floats: LO HI. "
    #         f"Default is {MASS_WIN_LO} {MASS_WIN_HI} GeV.  Set LO=0 and HI=1000 for no mass cut."
    #     ),
    # )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="NumPy random seed for reproducible sampling.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------
_COL_NAMES : List[str] = ["EEC_narrow", "EEC_wide", "e_2^(1)", "e_3^(1)"]
_COL_W     : List[int] = [12, 12, 12, 12, 14]  # [sample_col, c0, c1, c2, c3]


def _fmt(v: float) -> str:
    """Human-readable float: 4 decimal places, or scientific for |v| < 0.01."""
    return f"{v:.2e}" if abs(v) < 1e-2 else f"{v:.4f}"


def _table_row(label: str, means: np.ndarray) -> str:
    """Format one data row of the summary table."""
    return (
        f"{label:<{_COL_W[0]}}  "
        + "  ".join(f"{_fmt(means[i]):>{_COL_W[i + 1]}}" for i in range(4))
    )


def print_summary_table(features: np.ndarray, labels: np.ndarray) -> None:
    """
    Print per-class mean statistics for the four observables.

    Parameters
    ----------
    features : (N, 4) float array of computed observables.
    labels   : (N,)   int8 array of class labels (0=QCD, 1=TTBar).
    """
    qcd_means : np.ndarray = features[labels == 0].mean(axis=0)
    top_means : np.ndarray = features[labels == 1].mean(axis=0)

    header : str = (
        f"{'Sample':<{_COL_W[0]}}  "
        + "  ".join(f"{c:>{_COL_W[i + 1]}}" for i, c in enumerate(_COL_NAMES))
    )
    sep : str = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)
    print(_table_row("QCD-like",  qcd_means))
    print(_table_row("Top-like",  top_means))
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Main entry point: collect, sample, compute, save, summarise."""
    args           = parse_args()
    N              : int                = args.num_jets
    num_particles  : int                = args.num_particles
    rng            : np.random.Generator = np.random.default_rng(args.seed)
        # Override global constants with command-line arguments

    # Warn early if using all particles
    if num_particles <= 0:
        logger.warning(
            "--num-particles ≤ 0: ALL 100 stored constituents will be used. "
            "This may be very slow for large datasets."
        )
    else:
        logger.info("Using top-%d constituents (pT-ordered) per jet.", num_particles)

    n_top  : int = round(FRAC_TOP * N)   # 22 %
    n_qcd  : int = N - n_top             # 78 %

    logger.info(
        "Target dataset: N=%d  (TTBar=%d [%.0f%%], QCD=%d [%.0f%%])  seed=%d",
        N, n_top, 100 * FRAC_TOP, n_qcd, 100 * (1 - FRAC_TOP), args.seed,
    )

    # --- Discover run_XY files ------------------------------------------------
    qcd_files : List[pathlib.Path] = discover_files(QCD_BASE, QCD_FILENAME)
    top_files : List[pathlib.Path] = discover_files(TOP_BASE, TOP_FILENAME)

    # --- Collect observables for each class -----------------------------------
    logger.info("=== Collecting TTBar jets ===")
    obs_top, lbl_top = collect_observables(top_files, n_top, label=1,
                                           num_particles=num_particles)

    logger.info("=== Collecting QCD jets ===")
    obs_qcd, lbl_qcd = collect_observables(qcd_files, n_qcd, label=0,
                                           num_particles=num_particles)

    # --- Deterministic down-sampling to exactly n_top / n_qcd ----------------
    idx_top : np.ndarray = rng.choice(len(obs_top), size=n_top, replace=False)
    idx_qcd : np.ndarray = rng.choice(len(obs_qcd), size=n_qcd, replace=False)

    obs_top = obs_top[idx_top]
    obs_qcd = obs_qcd[idx_qcd]
    lbl_top = lbl_top[idx_top]
    lbl_qcd = lbl_qcd[idx_qcd]

    # --- Concatenate and globally shuffle ------------------------------------
    features : np.ndarray = np.concatenate([obs_top, obs_qcd], axis=0)  # (N, 4)
    labels   : np.ndarray = np.concatenate([lbl_top, lbl_qcd], axis=0)  # (N,)

    perm      : np.ndarray = rng.permutation(N)
    features  = features[perm]
    labels    = labels[perm]

    logger.info(
        "Final dataset: shape=%s  TTBar fraction=%.4f",
        features.shape, labels.mean(),
    )

    # --- Save to HDF5 --------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    feature_names_arr : np.ndarray = np.array(
        ["EEC_narrow", "EEC_wide", "e2_beta1", "e3_beta1"],
        dtype=h5py.special_dtype(vlen=str),
    )

    with h5py.File(OUTPUT_PATH, "w") as fh:
        # No compression on features — raw float32 for fast read-back
        fh.create_dataset("features",      data=features.astype(np.float32))
        fh.create_dataset("labels",        data=labels)
        fh.create_dataset("feature_names", data=feature_names_arr)
        # Provenance attributes
        fh.attrs["num_jets"]          = N
        fh.attrs["n_ttbar"]           = n_top
        fh.attrs["n_qcd"]             = n_qcd
        fh.attrs["seed"]              = args.seed
        fh.attrs["num_particles_used"] = num_particles if num_particles > 0 else 100
        fh.attrs["pt_cut_GeV"]        = PT_CUT
        fh.attrs["mass_win_lo"] = MASS_WIN_LO
        fh.attrs["mass_win_hi"] = MASS_WIN_HI
        fh.attrs["UNITS"] = "GeV"
        fh.attrs["frac_top"]          = FRAC_TOP

    logger.info("Saved: %s", OUTPUT_PATH)

    # --- Print summary table -------------------------------------------------
    print_summary_table(features, labels)


if __name__ == "__main__":
    main()