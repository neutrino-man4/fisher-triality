import numpy as np


def benchmark_direction(
    XW,
    XT,
    standardize=True,
    regularization=1e-6,
    normalize=True,
    return_stats=False,
):
    """
    Compute the Fisher/LDA-style benchmark direction

        u ∝ Sigma_within^{-1} (mu_W - mu_T)

    for two datasets XW and XT.

    Parameters
    ----------
    XW : array-like, shape (nW, d)
        Dataset for class W.
    XT : array-like, shape (nT, d)
        Dataset for class T.
    standardize : bool, default=True
        If True, apply one common standardization using the concatenated
        datasets before computing the direction.
    regularization : float, default=1e-6
        Ridge term added to Sigma_within for numerical stability.
    normalize : bool, default=True
        If True, return u / ||u||.
    return_stats : bool, default=False
        If True, also return a dictionary with intermediate quantities.

    Returns
    -------
    u : ndarray, shape (d,)
        Benchmark direction.
    stats : dict, optional
        Returned only if return_stats=True. Contains means, covariance,
        and standardization parameters.
    """
    XW = np.asarray(XW, dtype=float)
    XT = np.asarray(XT, dtype=float)

    if XW.ndim != 2 or XT.ndim != 2:
        raise ValueError("XW and XT must both be 2D arrays of shape (n_samples, n_features).")
    if XW.shape[1] != XT.shape[1]:
        raise ValueError("XW and XT must have the same number of features.")
    if XW.shape[0] < 2 or XT.shape[0] < 2:
        raise ValueError("Each dataset must contain at least 2 samples.")

    nW, d = XW.shape
    nT, _ = XT.shape

    mu_comb = None
    sigma_comb = None

    if standardize:
        X = np.vstack([XW, XT])
        mu_comb = X.mean(axis=0)
        sigma_comb = X.std(axis=0, ddof=1)

        if np.any(sigma_comb == 0):
            bad = np.where(sigma_comb == 0)[0]
            raise ValueError(f"Zero standard deviation for feature indices {bad.tolist()}.")

        XW_use = (XW - mu_comb) / sigma_comb
        XT_use = (XT - mu_comb) / sigma_comb
    else:
        XW_use = XW.copy()
        XT_use = XT.copy()

    muW = XW_use.mean(axis=0)
    muT = XT_use.mean(axis=0)

    XWc = XW_use - muW
    XTc = XT_use - muT

    SigmaW = (XWc.T @ XWc) / (nW - 1)
    SigmaT = (XTc.T @ XTc) / (nT - 1)

    Sigma_within = ((nW - 1) * SigmaW + (nT - 1) * SigmaT) / (nW + nT - 2)
    Sigma_within_reg = Sigma_within + regularization * np.eye(d)

    delta_mu = muW - muT
    u = np.linalg.solve(Sigma_within_reg, delta_mu)

    if normalize:
        norm = np.linalg.norm(u)
        if norm == 0:
            raise ValueError("Computed benchmark direction has zero norm.")
        u = u / norm

    if return_stats:
        stats = {
            "muW": muW,
            "muT": muT,
            "delta_mu": delta_mu,
            "SigmaW": SigmaW,
            "SigmaT": SigmaT,
            "Sigma_within": Sigma_within,
            "Sigma_within_reg": Sigma_within_reg,
            "mu_comb": mu_comb,
            "sigma_comb": sigma_comb,
            "standardized": standardize,
        }
        return u, stats

    return u