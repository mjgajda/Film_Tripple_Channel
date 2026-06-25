"""
Multichannel dosimetry for EBT3 radiochromic films.
Implements Pérez Azorín et al., Medical Physics 41(6), 2014.

Calibration curve (Eq. 1):
    D = (p_{1,k} - p_{2,k} * X_k) / (X_k - p_{3,k})

This module implements:
  - alpha_k  via closed-form Eq. (A6)
  - beta_k   via iterative scheme Eqs. (13) & (14)
  - D, Delta via linearised closed-form Eqs. (9), (10), (11), (12)
"""

import json
import numpy as np


# ---------------------------------------------------------------------------
# Load calibration parameters from JSON
# ---------------------------------------------------------------------------

def load_calibration_params(json_path):
    """
    Load calibration parameters from a JSON file saved by save_fit_params().

    Parameters
    ----------
    json_path : str — path to the JSON file

    Returns
    -------
    p1, p2, p3 : ndarray (3,) — calibration parameters for [R, G, B]
    cov        : ndarray (3, 3, 3) — covariance matrices per channel
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    channel_names = ('R', 'G', 'B')
    p1  = np.array([data[ch]['p1'] for ch in channel_names])
    p2  = np.array([data[ch]['p2'] for ch in channel_names])
    p3  = np.array([data[ch]['p3'] for ch in channel_names])
    cov = np.array([data[ch]['covariance'] for ch in channel_names])

    return p1, p2, p3, cov


# ---------------------------------------------------------------------------
# alpha_k  — Appendix Eq. (A6)
# ---------------------------------------------------------------------------

def compute_alpha(D_k, p1_k, p2_k, p3_k):
    """
    Compute alpha_k(i,j) from Eq. (A6).

    Parameters
    ----------
    D_k               : ndarray (H, W) — per-channel dose map for channel k
    p1_k, p2_k, p3_k  : float          — calibration parameters for channel k

    Returns
    -------
    alpha_k : ndarray (H, W)
    """
    numerator   = (p2_k + D_k) ** 2
    denominator = p2_k * p3_k - p1_k
    return numerator / denominator


# ---------------------------------------------------------------------------
# W  — Eq. (12)
# ---------------------------------------------------------------------------

def compute_W(alpha):
    """
    Parameters
    ----------
    alpha : ndarray (3, H, W)

    Returns
    -------
    W : ndarray (H, W)
    """
    return alpha.sum(axis=0) / (alpha ** 2).sum(axis=0)


# ---------------------------------------------------------------------------
# D, Delta — linearised closed-form  Eqs. (9) & (10)
# ---------------------------------------------------------------------------

def solve_linearised(D_scan, alpha, beta):
    """
    Closed-form solution for D and Delta (Eqs. 9, 10, 11, 12).

    Parameters
    ----------
    D_scan : ndarray (3, H, W) — per-channel doses from calibration curve
    alpha  : ndarray (3, H, W) — alpha_k  (from compute_alpha)
    beta   : ndarray (3, H, W) — beta_k   (from compute_beta_iterative)

    Returns
    -------
    D     : ndarray (H, W)
    Delta : ndarray (H, W)
    """
    D_scan_prime = D_scan + alpha * beta                         # (3, H, W)

    W          = compute_W(alpha)                                # (H, W)
    sum_ak     = alpha.sum(axis=0)                               # (H, W)
    sum_ak2    = (alpha ** 2).sum(axis=0)                        # (H, W)
    sum_Dp     = D_scan_prime.sum(axis=0)                        # (H, W)
    sum_ak_Dp  = (alpha * D_scan_prime).sum(axis=0)              # (H, W)

    D     = (sum_Dp - W * sum_ak_Dp) / (3.0 - W * sum_ak)
    Delta = (alpha * (D[np.newaxis] - D_scan_prime)).sum(axis=0) / sum_ak2

    return D, Delta


# ---------------------------------------------------------------------------
# beta_k — iterative scheme  Eqs. (13) & (14)
# ---------------------------------------------------------------------------

def compute_beta_iterative(X_unexp, p1, p2, p3,
                           max_iter=100, tol=1e-6):
    """
    Iterative calculation of beta_k from the unexposed film scan.

    Parameters
    ----------
    X_unexp          : ndarray (3, H, W) — RGB pixels of the unexposed film
    p1, p2, p3       : array-like (3,)   — calibration params per channel
    max_iter         : int               — maximum iterations
    tol              : float             — convergence threshold on |Delta|

    Returns
    -------
    beta   : ndarray (3, H, W) — converged beta_k maps
    n_iter : int               — number of iterations taken
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    n_ch = X_unexp.shape[0]

    D_unexp = np.empty_like(X_unexp, dtype=float)
    for k in range(n_ch):
        D_unexp[k] = (p1[k] - p2[k] * X_unexp[k]) / (X_unexp[k] - p3[k])

    alpha_unexp = np.empty_like(X_unexp, dtype=float)
    for k in range(n_ch):
        alpha_unexp[k] = compute_alpha(D_unexp[k], p1[k], p2[k], p3[k])

    beta       = -D_unexp / alpha_unexp
    Delta_prev = np.zeros(X_unexp.shape[1:])

    for n in range(1, max_iter + 1):
        _, Delta_new = solve_linearised(D_unexp, alpha_unexp, beta)
        beta         = -D_unexp / alpha_unexp + Delta_new[np.newaxis]

        change = np.abs(Delta_new - Delta_prev).max()
        if change < tol:
            return beta, n

        Delta_prev = Delta_new

    import warnings
    warnings.warn(f"beta iteration did not converge after {max_iter} iterations "
                  f"(max |ΔDelta| = {change:.2e})")
    return beta, max_iter


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def single_channel_dose(
    X_k,
    channel,
    json_path=None,
    p1_k=None, p2_k=None, p3_k=None,
):
    """
    Compute dose from a single channel via direct inversion of Eq. (1).
    No beta/Delta correction is applied.

    Parameters
    ----------
    X_k       : array-like     — OD values for a single channel
    channel   : str            — 'R', 'G', or 'B'
    json_path : str or None    — path to calibration JSON; takes precedence
                                 over p1_k/p2_k/p3_k if provided
    p1_k, p2_k, p3_k : float  — calibration params (used if no json_path)

    Returns
    -------
    D     : ndarray — dose values (same shape as X_k)
    D_err : ndarray — propagated 1-sigma uncertainty (zeros if no covariance)
    """
    X_k = np.asarray(X_k, dtype=float)

    # --- Load parameters ---
    cov_k = None
    if json_path is not None:
        p1_all, p2_all, p3_all, cov_all = load_calibration_params(json_path)
        idx   = {'R': 0, 'G': 1, 'B': 2}[channel.upper()]
        p1_k  = p1_all[idx]
        p2_k  = p2_all[idx]
        p3_k  = p3_all[idx]
        cov_k = cov_all[idx]                  # (3, 3)
    else:
        if any(p is None for p in (p1_k, p2_k, p3_k)):
            raise ValueError("Provide either json_path or all of p1_k, p2_k, p3_k.")

    # --- Dose via Eq. (1) ---
    denom = X_k - p3_k
    D     = (p1_k - p2_k * X_k) / denom

    # --- Propagate parameter uncertainty ---
    if cov_k is not None:
        dD_dp1 =  1.0 / denom
        dD_dp2 = -X_k  / denom
        dD_dp3 =  D    / denom          # equivalent to (p1-p2*X)/(X-p3)^2
        J      = np.stack([dD_dp1, dD_dp2, dD_dp3], axis=-1)   # (..., 3)
        D_var  = np.einsum('...i,ij,...j->...', J, cov_k, J)
        D_err  = np.sqrt(np.abs(D_var))
    else:
        D_err = np.zeros_like(D)

    return D, D_err

def multichannel_dose(
    X,
    X_unexp,
    p1=None, p2=None, p3=None,    # pass directly, or use json_path
    json_path=None,                # path to saved calibration JSON
    D_scan=None,
    beta_max_iter=100,
    beta_tol=1e-3,
):
    """
    Full multichannel dosimetry pipeline.

    Calibration parameters can be supplied either directly as p1/p2/p3
    arrays or loaded from a JSON file produced by save_fit_params().
    Providing json_path takes precedence over p1/p2/p3.

    Parameters
    ----------
    X          : ndarray (3, H, W) — RGB pixel values of exposed film
    X_unexp    : ndarray (3, H, W) — RGB pixel values of unexposed film
    p1, p2, p3 : array-like (3,)   — calibration params per channel
    json_path  : str or None       — path to calibration JSON file
    D_scan     : ndarray (3, H, W) — pre-computed per-channel doses (optional)
    beta_max_iter : int
    beta_tol      : float

    Returns
    -------
    D          : ndarray (H, W)    — corrected dose map
    Delta      : ndarray (H, W)    — perturbation map
    alpha      : ndarray (3, H, W)
    beta       : ndarray (3, H, W)
    beta_iters : int
    cov        : ndarray (3, 3, 3) or None — per-channel covariance matrices
                                             (only available when json_path used)
    """
    # --- Load parameters ---
    cov = None
    if json_path is not None:
        p1, p2, p3, cov = load_calibration_params(json_path)
    else:
        if p1 is None or p2 is None or p3 is None:
            raise ValueError("Provide either json_path or all of p1, p2, p3.")
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        p3 = np.asarray(p3, dtype=float)

    n_ch = X.shape[0]

    # --- Step 1: per-channel dose ---
    if D_scan is None:
        D_scan = np.empty_like(X, dtype=float)
        for k in range(n_ch):
            D_scan[k] = (p1[k] - p2[k] * X[k]) / (X[k] - p3[k])

    # --- Step 2: alpha_k via Eq. (A6) ---
    alpha = np.empty_like(X, dtype=float)
    for k in range(n_ch):
        alpha[k] = compute_alpha(D_scan[k], p1[k], p2[k], p3[k])

    # --- Step 3: beta_k via iterative scheme Eqs. (13) & (14) ---
    beta, beta_iters = compute_beta_iterative(
        X_unexp, p1, p2, p3,
        max_iter=beta_max_iter, tol=1e-6,
    )

    # --- Step 4: linearised solve Eqs. (9) & (10) ---
    D, Delta = solve_linearised(D_scan, alpha, beta)

    return D, Delta, alpha, beta, beta_iters, cov


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def dose_calculation(calibration_json='calibration_params_EBT4.json', X=None, X_unexp=None,):

    p1, p2, p3, cov = load_calibration_params(calibration_json)


    D, Delta, alpha, beta, n_iter, _ = multichannel_dose(
        X, X_unexp, json_path=calibration_json
    )

    return D, Delta, alpha, beta, n_iter, cov