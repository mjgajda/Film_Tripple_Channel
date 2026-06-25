# utils/helpers.py
import numpy as np

def polyfit_linear(x: np.ndarray, y: np.ndarray):
    p = np.polyfit(x, y, 1)
    yfit = np.polyval(p, x)
    resid = y - yfit
    S = {"resid": resid, "yfit": yfit}
    return p, S

def poly_apply(p: np.ndarray, x: np.ndarray):
    return np.polyval(p, x), None

def scanner_to_OD(pv: np.ndarray, max_val: float = (2**16 - 1)) -> np.ndarray:
    """Scanner OD = log10((max) / PV). Avoid log of zero."""
    pv = pv.astype(np.float64)
    pv = np.clip(pv, 1, max_val)
    return np.log10(max_val / pv)
import numpy as np

import numpy as np

def compute_calibrated_OD_values(
    film_masks_pre: list[np.ndarray],
    film_masks_post: list[np.ndarray],
    full_img_pre: np.ndarray,
    full_img_post: np.ndarray,
    P_pre: np.ndarray,
    P_post: np.ndarray,
    scanner_to_OD,
    poly_apply
) -> tuple[
    list[np.ndarray],  # 1D pre values
    list[np.ndarray],  # 1D post values
    list[np.ndarray],  # 2D pre maps (masked)
    list[np.ndarray]   # 2D post maps (masked)
]:
    """
    Compute calibrated OD values for each ROI.

    Returns:
        - OD_pre_vals_1d: list of 1D OD arrays (ROI-only values)
        - OD_post_vals_1d: list of 1D OD arrays
        - OD_pre_maps_2d: list of 2D OD maps with NaN outside ROI
        - OD_post_maps_2d: list of 2D OD maps with NaN outside ROI
    """

    OD_pre_vals = []
    OD_post_vals = []
    OD_pre_maps = []
    OD_post_maps = []

    for i, (mask_pre, mask_post) in enumerate(zip(film_masks_pre, film_masks_post)):
        # Use common mask to enforce matching ROI
        common_mask = mask_pre & mask_post

        if not np.any(common_mask):
            print(f"[Warning] Empty ROI at index {i} — skipping.")
            continue

        # Get pixel values inside ROI only (flattened)
        pre_vals = full_img_pre[common_mask]
        post_vals = full_img_post[common_mask]

        # Convert pixel values to OD
        OD_pre = scanner_to_OD(pre_vals)
        OD_post = scanner_to_OD(post_vals)

        # Apply calibration polynomials
        OD_pre_cal, _ = poly_apply(P_pre, OD_pre)
        OD_post_cal, _ = poly_apply(P_post, OD_post)

        # Filter out low OD values
        valid_mask = (OD_pre_cal >= 0.15) & (OD_post_cal >= 0.15)
        if not np.any(valid_mask):
            print(f"[Info] Skipped ROI {i} — all OD values < 0.15.")
            continue

        # Store 1D calibrated OD values
        OD_pre_vals.append(OD_pre_cal[valid_mask])
        OD_post_vals.append(OD_post_cal[valid_mask])

        # --- Create full-size 2D maps masked with NaN outside ROI ---

        # Convert full image to OD and calibrate
        full_OD_pre = scanner_to_OD(full_img_pre)
        full_OD_post = scanner_to_OD(full_img_post)

        full_OD_pre_cal, _ = poly_apply(P_pre, full_OD_pre)
        full_OD_post_cal, _ = poly_apply(P_post, full_OD_post)

        # Create masked 2D OD maps (NaN outside mask)
        OD_pre_map = np.full_like(full_OD_pre_cal, np.nan, dtype=np.float64)
        OD_post_map = np.full_like(full_OD_post_cal, np.nan, dtype=np.float64)

        OD_pre_map[common_mask] = full_OD_pre_cal[common_mask]
        OD_post_map[common_mask] = full_OD_post_cal[common_mask]

        OD_pre_maps.append(OD_pre_map)
        OD_post_maps.append(OD_post_map)

    return OD_pre_vals, OD_post_vals, OD_pre_maps, OD_post_maps
def calibrate_od_values_only(od_arrays: list[np.ndarray], P: np.ndarray, poly_apply):
    """
    Apply calibration polynomial P to each OD array and return calibrated values only.

    Parameters
    ----------
    od_arrays : list of 1D np.ndarray
        Raw OD values for each ROI.
    P : np.ndarray
        Calibration polynomial coefficients.
    poly_apply : callable
        Function: calibrated, _ = poly_apply(P, od_array)

    Returns
    -------
    calibrated_od_arrays : list[np.ndarray]
        List of calibrated OD arrays, one per ROI.
    """

    calibrated = []

    for arr in od_arrays:
        cal, _ = poly_apply(P, arr)
        if arr is None or arr.size == 0 or np.mean(cal) > 0.09:
            calibrated.append(cal)
            continue

        
        

    return calibrated
