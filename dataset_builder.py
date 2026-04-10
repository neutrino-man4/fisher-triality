"""
Joint QCD + TTBar observable dataset builder — importable module.

Reads phase-space cuts and filesystem paths from a YAML config file.
Computes an arbitrary set of EEC and ECF observables (specified by the
caller) from raw constituent kinematics, applies kinematic selection, and
returns a numpy feature matrix together with truth labels, LaTeX-formatted
observable labels, and a metadata summary dict.

When executed as a standalone script it additionally saves the result to an
HDF5 file at the caller-supplied output path.

Author: Aritra Bal (ETP)
Date  : 2026-03-23
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py, sys
import numpy as np
import yaml
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
# Type aliases
# ---------------------------------------------------------------------------
ObservableSpec = Dict[str, Union[List[float], Tuple[int, float]]]
"""
Keys beginning with 'EEC' map to [bin_lo, bin_hi].
Keys beginning with 'ECF' map to (N_points, beta).
"""

ConfigDict = Dict[str, Any]

# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

_REQUIRED_CONFIG_KEYS: Dict[str, Sequence[str]] = {
    "paths": ("qcd_base", "qcd_filename", "top_base", "top_filename"),
    "cuts":  ("pt_cut", "mass_win_lo", "mass_win_hi"),
}


def load_config(config_path: Union[str, pathlib.Path]) -> ConfigDict:
    """
    Load and minimally validate a YAML configuration file.

    Expected top-level sections: ``paths``, ``cuts``, and optionally
    ``dataset`` (for ``frac_top``) and ``compute`` (for ``chunk_size``).

    Raises
    ------
    KeyError
        If a required key is absent from the config.
    """
    with open(config_path) as fh:
        cfg: ConfigDict = yaml.safe_load(fh)

    for section, keys in _REQUIRED_CONFIG_KEYS.items():
        for key in keys:
            if key not in cfg.get(section, {}):
                raise KeyError(f"Missing '{key}' under '{section}' in {config_path}")

    return cfg


# ---------------------------------------------------------------------------
# Observable computation — private batch functions
# ---------------------------------------------------------------------------

def _safe_power(arr: np.ndarray, beta: float) -> np.ndarray:
    """
    Element-wise ``arr ** beta``, returning 0 wherever ``arr == 0``.

    Avoids ``0 ** 0 = 1`` artefacts that would corrupt zero-padded entries
    in the upper-triangular ΔR matrix.
    """
    return np.where(arr > 0.0, np.power(arr, beta, where=arr > 0.0,
                                         out=np.zeros_like(arr)), 0.0)


def _compute_eec_batch(
    dr_upper: np.ndarray,
    z: np.ndarray,
    bin_lo: float,
    bin_hi: float,
) -> np.ndarray:
    """
    EEC in a single angular bin, summed over unique pairs i < j.

        EEC = Σ_{i<j,  bin_lo < ΔR_{ij} < bin_hi}  z_i z_j

    Parameters
    ----------
    dr_upper : Upper-triangular pairwise ΔR, shape (B, P, P). Lower triangle
               and diagonal must be zero (pairs i < j only).
    z        : Normalised pT fractions, shape (B, P).
    bin_lo, bin_hi : Angular bin edges (exclusive).

    Returns
    -------
    ndarray of shape (B,).
    """
    Z: np.ndarray = z[:, :, None] * z[:, None, :]
    mask: np.ndarray = (dr_upper > bin_lo) & (dr_upper < bin_hi)
    return (Z * mask).sum(axis=(1, 2))


def _compute_ecf2_batch(
    dr_upper: np.ndarray,
    z: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Two-point ECF:  e_2^β = Σ_{i<j}  z_i z_j  ΔR_{ij}^β.

    Parameters
    ----------
    dr_upper : Upper-triangular ΔR, shape (B, P, P).
    z        : Normalised pT fractions, shape (B, P).
    beta     : Angular exponent.

    Returns
    -------
    ndarray of shape (B,).
    """
    Z: np.ndarray = z[:, :, None] * z[:, None, :]
    return (Z * _safe_power(dr_upper, beta)).sum(axis=(1, 2))


def _compute_ecf3_batch(
    dr_full: np.ndarray,
    z: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Three-point ECF via an O(P²) factored identity:

        e_3^β = (1/6) Σ_{i,k} z_i z_k ΔR_{ik}^β  F_{ik}

    where  F_{ik} = Σ_j z_j  ΔR_{ij}^β  ΔR_{jk}^β,
    computed as a batched matrix multiplication.

    The factor 1/6 = 1/3! corrects for the 6 permutations of each distinct
    triplet (i, j, k) in the unsymmetrised double sum.  Diagonal entries
    ΔR_{ii} = 0 suppress degenerate two-index contributions automatically.

    Parameters
    ----------
    dr_full : Full symmetric ΔR (dr_upper + dr_upper^T), shape (B, P, P).
    z       : Normalised pT fractions, shape (B, P).
    beta    : Angular exponent.

    Returns
    -------
    ndarray of shape (B,).
    """
    Z_full: np.ndarray   = z[:, :, None] * z[:, None, :]
    dr_beta: np.ndarray  = _safe_power(dr_full, beta)               # (B, P, P)
    A: np.ndarray        = dr_beta * z[:, None, :]                  # A[i,j] = z_j DR_{ij}^β
    F: np.ndarray        = np.matmul(A, dr_beta)                    # F[i,k] = Σ_j z_j DR_{ij}^β DR_{jk}^β
    return (Z_full * dr_beta * F).sum(axis=(1, 2)) / 6.0


# ---------------------------------------------------------------------------
# Observable label generator
# ---------------------------------------------------------------------------

def _make_latex_label(key: str, params: Union[List[float], Tuple]) -> str:
    """
    Generate a LaTeX-formatted label string for a single observable.

    EEC keys  → ``$\\mathrm{EEC}_{[lo,\\,hi]}$``
    ECF keys  → ``$e_N^{(\\beta)}$``

    Parameters
    ----------
    key    : Observable identifier string starting with 'EEC' or 'ECF'.
    params : [bin_lo, bin_hi] for EEC; (N_points, beta) for ECF.
    """
    if key.upper().startswith("EEC"):
        lo, hi = params
        lo_s = f"{lo:g}"
        hi_s = f"{hi:g}"
        return rf"$\mathrm{{EEC}}_{{[{lo_s},\,{hi_s}]}}$"
    elif key.upper().startswith("ECF"):
        n_pts, beta = params
        beta_s = f"{beta:g}"
        return rf"$e_{{{n_pts}}}^{{({beta_s})}}$"
    else:
        raise ValueError(
            f"Observable key '{key}' must begin with 'EEC' or 'ECF'."
        )


# ---------------------------------------------------------------------------
# Batch observable dispatcher
# ---------------------------------------------------------------------------

def _compute_observables_batch(
    dr_upper: np.ndarray,
    dr_full: np.ndarray,
    z: np.ndarray,
    observables: ObservableSpec,
) -> np.ndarray:
    """
    Dispatch each observable in *observables* to the appropriate batch
    function and stack results column-wise.

    Parameters
    ----------
    dr_upper    : Upper-triangular ΔR, shape (B, P, P).
    dr_full     : Full symmetric ΔR (= dr_upper + dr_upper^T), shape (B, P, P).
    z           : Normalised pT fractions, shape (B, P).
    observables : Ordered dict of observable specs.

    Returns
    -------
    ndarray of shape (B, F) where F = len(observables).

    Raises
    ------
    ValueError  For unsupported ECF N-point orders.
    """
    columns: List[np.ndarray] = []

    for key, params in observables.items():
        key_up = key.upper()

        if key_up.startswith("EEC"):
            bin_lo, bin_hi = float(params[0]), float(params[1])
            col = _compute_eec_batch(dr_upper, z, bin_lo, bin_hi)

        elif key_up.startswith("ECF"):
            n_pts, beta = int(params[0]), float(params[1])
            if n_pts == 2:
                col = _compute_ecf2_batch(dr_upper, z, beta)
            elif n_pts == 3:
                col = _compute_ecf3_batch(dr_full, z, beta)
            else:
                raise ValueError(
                    f"ECF N={n_pts} is not supported. "
                    "Only N=2 (O(P²)) and N=3 (factored O(P²)) are implemented."
                )
        else:
            raise ValueError(
                f"Observable key '{key}' must start with 'EEC' or 'ECF'."
            )

        columns.append(col)

    return np.stack(columns, axis=1)   # (B, F)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_files(base: pathlib.Path, filename: str) -> List[pathlib.Path]:
    """
    Return sorted ``run_XY/<filename>`` paths found under *base*.

    Raises
    ------
    FileNotFoundError if no matching paths are found.
    """
    paths: List[pathlib.Path] = sorted(base.glob(f"run_*/{filename}"))
    if not paths:
        raise FileNotFoundError(
            f"No files found matching {base / 'run_*' / filename}. "
            "Check that qcd_filename / top_filename match the ntuplizer output."
        )
    logger.info("Discovered %d file(s) under %s", len(paths), base)
    for p in paths:
        logger.info("  %s", p)
    return paths


# ---------------------------------------------------------------------------
# Per-class data collection
# ---------------------------------------------------------------------------

def _collect_class_observables(
    file_paths: List[pathlib.Path],
    n_needed: int,
    label: int,
    num_particles: int,
    observables: ObservableSpec,
    pt_cut: float,
    mass_win_lo: float,
    mass_win_hi: float,
    idx_pt: int,
    idx_sdmass: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accumulate observable vectors for one jet class from HDF5 run files.

    Files are opened sequentially and closed as soon as *n_needed* passing
    jets have been collected, avoiding unnecessary I/O.  The ΔR matrix and
    pT fractions are computed on-the-fly from raw constituent kinematics
    (columns: deta, dphi, pT) rather than relying on pre-computed quantities.

    Parameters
    ----------
    file_paths   : Ordered list of run HDF5 files.
    n_needed     : Number of passing jets to collect.
    label        : Class integer (0 = QCD, 1 = TTBar).
    num_particles: Leading-pT constituents to use; all 100 if ≤ 0.
    observables  : Observable specification dict (see ``build_dataset``).
    pt_cut       : Minimum jet pT [GeV].
    mass_win_lo  : Lower soft-drop mass window edge [GeV].
    mass_win_hi  : Upper soft-drop mass window edge [GeV].
    idx_pt       : Column index of jet pT in ``jetFeatures``.
    idx_sdmass   : Column index of soft-drop mass in ``jetFeatures``.
    chunk_size   : Jets per inner compute chunk (memory/speed trade-off).

    Returns
    -------
    obs    : float64 array of shape (≥n_needed, F).
    labels : int8 array of shape (≥n_needed,) filled with *label*.

    Raises
    ------
    RuntimeError if all files are exhausted before reaching *n_needed*.
    """
    class_tag: str = "TTBar" if label == 1 else "QCD"
    use_all_p: bool = num_particles <= 0
    P: int = 100 if use_all_p else num_particles

    if use_all_p:
        logger.warning(
            "[%s] num_particles ≤ 0 — using all 100 stored constituents. "
            "This may be slow for large datasets.",
            class_tag,
        )

    obs_chunks: List[np.ndarray] = []
    total: int = 0

    file_bar = tqdm(file_paths, desc=f"[{class_tag}] files", unit="file", position=0)

    for path in file_bar:
        file_bar.set_postfix_str(path.parent.name)
        logger.info("[%s] Opening: %s", class_tag, path)

        with h5py.File(path, "r") as fh:
            jet_features: np.ndarray = fh["jetFeatures"][:]
            jet_pt: np.ndarray       = jet_features[:, idx_pt]
            jet_sdmass: np.ndarray   = jet_features[:, idx_sdmass]

            mask: np.ndarray = (
                (jet_pt > pt_cut)
                & (jet_sdmass > mass_win_lo)
                & (jet_sdmass < mass_win_hi)
            )
            n_pass: int  = mask.sum()
            n_total: int = len(jet_features)

            logger.info(
                "  %d / %d jets pass (pT > %.0f, %.0f < sdmass < %.0f GeV)",
                n_pass, n_total, pt_cut, mass_win_lo, mass_win_hi,
            )

            if n_pass == 0:
                logger.warning("  No jets pass cuts — skipping file.")
                continue

            # Read only P constituents (hyperslab on axes 1 and 2)
            constituents: np.ndarray = fh["jetConstituentsList"][:, :P, :]  # (n_jets, P, 3)
        
        constituents = constituents[mask]  # (n_pass, P, 3) after kinematic selection
        # Build ΔR and pT fractions from raw constituent kinematics
        deta: np.ndarray    = constituents[..., 0].astype(np.float32)   # (n_pass, P)
        dphi: np.ndarray    = constituents[..., 1].astype(np.float32)
        part_pt: np.ndarray = constituents[..., 2].astype(np.float32)

        deta_d: np.ndarray = deta[:, :, None] - deta[:, None, :]        # (n_pass, P, P)
        dphi_d: np.ndarray = dphi[:, :, None] - dphi[:, None, :]
        pair_dr_full: np.ndarray = np.sqrt(deta_d ** 2 + dphi_d ** 2).astype(np.float64)

        upper_mask: np.ndarray = np.triu(np.ones((P, P), dtype=bool), k=1)
        pair_dr_upper: np.ndarray = pair_dr_full * upper_mask[None, :, :]

        pt_sum: np.ndarray = part_pt.sum(axis=1, keepdims=True)
        z: np.ndarray = np.where(pt_sum > 0.0, part_pt / pt_sum, 0.0).astype(np.float64)

        # Chunk-wise observable computation
        chunk_obs: List[np.ndarray] = []
        for lo in tqdm(
            range(0, n_pass, chunk_size),
            desc=f"  [{class_tag}] {path.parent.name}",
            unit="chunk",
            leave=False,
        ):
            hi: int = min(lo + chunk_size, n_pass)
            dr_up: np.ndarray  = pair_dr_upper[lo:hi]
            dr_sym: np.ndarray = pair_dr_full[lo:hi]
            z_b: np.ndarray    = z[lo:hi]

            chunk_obs.append(
                _compute_observables_batch(dr_up, dr_sym, z_b, observables)
            )

        obs: np.ndarray = np.concatenate(chunk_obs, axis=0)  # (n_pass, F)
        obs_chunks.append(obs)
        total += n_pass

        logger.info(
            "  [%s] Cumulative passing jets: %d / %d needed",
            class_tag, total, n_needed,
        )

        if total >= n_needed:
            break

    if total < n_needed:
        raise RuntimeError(
            f"Insufficient {class_tag} jets after all files: "
            f"collected {total}, needed {n_needed}. "
            "Reduce --num-jets or widen the kinematic cuts."
        )

    all_obs: np.ndarray    = np.concatenate(obs_chunks, axis=0)
    all_labels: np.ndarray = np.full(total, fill_value=label, dtype=np.int8)
    return all_obs, all_labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(
    config_path: Union[str, pathlib.Path] = "configs/data.yaml",
    observables: Optional[ObservableSpec] = None,
    num_jets: int = 100_000,
    num_particles: int = 18,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str], Dict[str, Any]]:
    """
    Build a joint QCD + TTBar observable dataset.

    Reads filesystem paths and kinematic cuts from *config_path* (YAML).
    Computes the observables listed in *observables* from raw constituent
    kinematics, applies kinematic selection, and returns a balanced sample
    according to the ``frac_top`` entry in the config file.

    Parameters
    ----------
    config_path :
        Path to the YAML configuration file.  See ``load_config`` for the
        expected schema.  Defaults to ``configs/data.yaml``.
    observables :
        Ordered dict specifying which observables to compute.

        - **EEC** entries: ``"EEC_<name>": [bin_lo, bin_hi]``
          Sums  z_i z_j  over pairs with  bin_lo < ΔR_{ij} < bin_hi.
        - **ECF** entries: ``"ECF_<name>": (N_points, beta)``
          Computes the N-point ECF  e_N^β.  Supported: N ∈ {2, 3}.

        If *None*, a default four-observable set is used (narrow EEC,
        wide EEC, e_2^1, e_3^1) matching the original script.
    num_jets :
        Total jets in the output dataset (split between classes according
        to ``frac_top`` in the config).
    num_particles :
        Number of leading-pT constituents used to build observables.
        pT fractions are renormalised over exactly these P particles.
        Constituents beyond rank P are discarded.  Use ≤ 0 for all 100.
    seed :
        NumPy random seed for reproducible down-sampling and shuffling.

    Returns
    -------
    features : float32 array of shape (num_jets, F).
    labels   : int8 array of shape (num_jets,).  0 = QCD, 1 = TTBar.
    obs_labels : dict mapping each observable key to a LaTeX label string.
    metadata : Summary dict with all phase-space cuts and dataset parameters.
    """
    cfg: ConfigDict = load_config(config_path)

    # ---- Unpack config -------------------------------------------------------
    paths_cfg:   Dict[str, Any] = cfg["paths"]
    cuts_cfg:    Dict[str, Any] = cfg["cuts"]
    dataset_cfg: Dict[str, Any] = cfg.get("dataset", {})
    compute_cfg: Dict[str, Any] = cfg.get("compute", {})

    qcd_base: pathlib.Path = pathlib.Path(paths_cfg["qcd_base"])
    top_base: pathlib.Path = pathlib.Path(paths_cfg["top_base"])
    qcd_filename: str      = paths_cfg["qcd_filename"]
    top_filename: str      = paths_cfg["top_filename"]

    pt_cut:      float = float(cuts_cfg["pt_cut"])
    mass_win_lo: float = float(cuts_cfg["mass_win_lo"])
    mass_win_hi: float = float(cuts_cfg["mass_win_hi"])
    idx_pt:      int   = int(cuts_cfg.get("idx_pt",     0))
    idx_sdmass:  int   = int(cuts_cfg.get("idx_sdmass", 5))

    frac_top:   float = float(dataset_cfg.get("frac_top", 0.22))
    chunk_size: int   = int(compute_cfg.get("chunk_size", 2000))

    if not (0.0 <= frac_top <= 1.0):
        raise ValueError(f"frac_top must be in [0, 1]; got {frac_top}.")
    # Print logs
    logger.info("Configuration loaded from %s", config_path)
    logger.info("Paths: QCD base=%s  QCD filename=%s  TTBar base=%s  TTBar filename=%s",
                qcd_base, qcd_filename, top_base, top_filename)
    logger.info("Cuts: pT > %.0f GeV  mass window = [%.0f, %.0f] GeV  idx_pt=%d  idx_sdmass=%d",
                pt_cut, mass_win_lo, mass_win_hi, idx_pt, idx_sdmass)
    logger.info("Dataset: num_jets=%d  frac_top=%.2f  num_particles (per jet)=%d  seed=%d",
                num_jets, frac_top, num_particles, seed)
    
    # ---- Default observables -------------------------------------------------
    if observables is None:
        observables = {
            "EEC_narrow": [0.06, 0.20],
            "EEC_wide":   [0.20, 0.80],
            "ECF_e2b1":   (2, 1),
            "ECF_e3b1":   (3, 1),
        }

    # ---- Build LaTeX label map -----------------------------------------------
    obs_labels: Dict[str, str] = {
        key: _make_latex_label(key, params)
        for key, params in observables.items()
    }

    # ---- Compute class sizes -------------------------------------------------
    rng: np.random.Generator = np.random.default_rng(seed)
    n_top: int = round(frac_top * num_jets)
    n_qcd: int = num_jets - n_top

    logger.info(
        "Target: N=%d  TTBar=%d (%.0f%%)  QCD=%d (%.0f%%)  seed=%d",
        num_jets, n_top, 100 * frac_top, n_qcd, 100 * (1 - frac_top), seed,
    )

    # ---- Collect observables per class ---------------------------------------
    common_kw: Dict[str, Any] = dict(
        num_particles=num_particles,
        observables=observables,
        pt_cut=pt_cut,
        mass_win_lo=mass_win_lo,
        mass_win_hi=mass_win_hi,
        idx_pt=idx_pt,
        idx_sdmass=idx_sdmass,
        chunk_size=chunk_size,
    )

    all_obs_list:    List[np.ndarray] = []
    all_labels_list: List[np.ndarray] = []
    
    # Print warnings for zero-sample edge cases
    if n_qcd == 0 and n_top == 0:
        logger.warning("num_jets=0, exiting")
        sys.exit(0)
    elif n_qcd == 0:
        logger.warning("num_jets > 0 but frac_top=1.0 — no QCD jets will be collected.")
    elif n_top == 0:
        logger.warning("num_jets > 0 but frac_top=0.0 — no TTBar jets will be collected.")
    
    if n_top > 0:
        logger.info("=== Collecting TTBar jets ===")
        top_files = _discover_files(top_base, top_filename)
        obs_top, lbl_top = _collect_class_observables(
            top_files, n_top, label=1, **common_kw
        )
        idx_t = rng.choice(len(obs_top), size=n_top, replace=False)
        all_obs_list.append(obs_top[idx_t])
        all_labels_list.append(lbl_top[idx_t])

    if n_qcd > 0:
        logger.info("=== Collecting QCD jets ===")
        qcd_files = _discover_files(qcd_base, qcd_filename)
        obs_qcd, lbl_qcd = _collect_class_observables(
            qcd_files, n_qcd, label=0, **common_kw
        )
        idx_q = rng.choice(len(obs_qcd), size=n_qcd, replace=False)
        all_obs_list.append(obs_qcd[idx_q])
        all_labels_list.append(lbl_qcd[idx_q])

    # ---- Concatenate and shuffle globally -----------------------------------
    features: np.ndarray = np.concatenate(all_obs_list,    axis=0).astype(np.float32)
    labels:   np.ndarray = np.concatenate(all_labels_list, axis=0)

    perm: np.ndarray = rng.permutation(len(features))
    features = features[perm]
    labels   = labels[perm]

    logger.info(
        "Final dataset: shape=%s  TTBar fraction=%.4f",
        features.shape, labels.mean() if len(labels) else 0.0,
    )

    # ---- Metadata dict -------------------------------------------------------
    metadata: Dict[str, Any] = {
        "num_jets":      len(features),
        "num_particles": num_particles if num_particles > 0 else 100,
        "pt_cut":        pt_cut,
        "mass_win_lo":   mass_win_lo,
        "mass_win_hi":   mass_win_hi,
        "frac_top":      frac_top,
        "seed":          seed,
        "n_ttbar":       n_top,
        "n_qcd":         n_qcd,
        "observables":   list(observables.keys()),
        "obs_labels":    obs_labels,
    }

    return features, labels, obs_labels, metadata


# ---------------------------------------------------------------------------
# HDF5 serialisation (used only by standalone main)
# ---------------------------------------------------------------------------

def save_to_hdf5(
    output_path: Union[str, pathlib.Path],
    features: np.ndarray,
    labels: np.ndarray,
    obs_labels: Dict[str, str],
    metadata: Dict[str, Any],
) -> None:
    """
    Write features, labels, and metadata to an HDF5 file.

    The ``feature_names`` dataset stores the observable keys in column order.
    All scalar metadata entries are stored as HDF5 root attributes.

    Parameters
    ----------
    output_path : Destination HDF5 file path.
    features    : (N, F) float32 observable matrix.
    labels      : (N,)   int8 truth labels.
    obs_labels  : Dict mapping observable keys to LaTeX strings.
    metadata    : Dict of scalar provenance values.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    str_dtype = h5py.special_dtype(vlen=str)
    obs_keys = metadata["observables"]

    with h5py.File(output_path, "w") as fh:
        fh.create_dataset("features",       data=features)
        fh.create_dataset("labels",         data=labels)
        fh.create_dataset(
            "feature_names",
            data=np.array(obs_keys, dtype=str_dtype),
        )
        fh.create_dataset(
            "feature_latex_labels",
            data=np.array([obs_labels[k] for k in obs_keys], dtype=str_dtype),
        )

        for key, val in metadata.items():
            if key in ("observables", "obs_labels"):
                continue  # stored as dedicated datasets above
            if isinstance(val, (int, float, str, bool)):
                fh.attrs[key] = val

    logger.info("Saved dataset to %s", output_path)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point: build a dataset from a YAML config and save to HDF5.

    Accepts the YAML config path and output HDF5 path.  Observable set
    defaults to (EEC_narrow, EEC_wide, e_2^1, e_3^1).  Pass additional
    arguments via ``--num-jets``, ``--num-particles``, ``--seed``.
    """
    parser = argparse.ArgumentParser(
        description="Build a joint QCD+TTBar observable dataset and save to HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("configs/data.yaml"),
        metavar="PATH",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("/ceph/abal/QFIT/MC/joint_datasets/KL/data.h5"),
        metavar="PATH",
        help="Destination HDF5 output path.",
    )
    parser.add_argument(
        "--num-jets",
        type=int,
        default=100_000,
        metavar="N",
        help="Total jets in the output dataset.",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=18,
        metavar="P",
        help="Number of leading-pT constituents used per jet. ≤ 0 for all 100.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="NumPy random seed.",
    )
    args = parser.parse_args()

    features, labels, obs_labels, metadata = build_dataset(
        config_path=args.config,
        observables=None,           # use default four-observable set
        num_jets=args.num_jets,
        num_particles=args.num_particles,
        seed=args.seed,
    )

    save_to_hdf5(args.output, features, labels, obs_labels, metadata)

    # Per-class mean summary
    col_names = metadata["observables"]
    header = f"{'Sample':<12}  " + "  ".join(f"{c:>14}" for c in col_names)
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for cls_name, cls_label in (("QCD-like", 0), ("Top-like", 1)):
        mask = labels == cls_label
        if mask.any():
            means = features[mask].mean(axis=0)
            row = f"{cls_name:<12}  " + "  ".join(
                f"{(f'{v:.2e}' if abs(v) < 1e-2 else f'{v:.4f}'):>14}"
                for v in means
            )
            print(row)
    print(f"{sep}\n")


if __name__ == "__main__":
    main()