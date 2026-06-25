import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import matplotlib.ticker as ticker

# This function defines what the LABEL looks like
def to_um(pixel_value, pos):
    return f'{(pixel_value - 1500) * 4:.0f}'


# Equation to calculate the Dose from the OD values
def dose_Calculation_from_OD_diff(OD_diff):
    """
    Calculate dose from optical density difference using a calibration curve.

    OD_diff: Optical density difference (float or np.ndarray)

    Returns:
        dose: Calculated dose (same shape as OD_diff)
    """
    # Calibration parameters (example values, replace with actual calibration data)
    a = 8.7705
    b = -8.0102
    c= 9.3871
    d= -.5839

    # Calculate dose using the cubic calibration curve
    dose = a * OD_diff**3 + b * OD_diff**2 + c * OD_diff + d
    return dose

# Equations to apply equations and converting the scanner to OD values
def polyfit_linear(x: np.ndarray, y: np.ndarray):
    p = np.polyfit(x, y, 1)
    yfit = np.polyval(p, x)
    resid = y - yfit
    S = {"resid": resid, "yfit": yfit}
    return p, S

def poly_apply(p: np.ndarray, x: np.ndarray):
    """Apply polynomial calibration."""
    return np.polyval(p, x), None

def scanner_to_OD(pv: np.ndarray, max_val: float = (2**16 - 1)) -> np.ndarray:
    """Scanner OD = log10(max / PV)."""
    pv = pv.astype(np.float64)
    pv = np.clip(pv, 1, max_val)
    return np.log10(max_val / pv)

def get_clean_aligned_data(pre, post, M_rigid):
    """
    Loads clean (Blue) channels and aligns the Post image to the Pre image.
    """
    # Load Pre (Channel 0 = Blue to avoid Red markers)
    img_pre = cv2.imread(pre, cv2.IMREAD_UNCHANGED)
    pre_clean = img_pre[..., 2].astype(np.float64) 

    # Load Post (Channel 0)
    img_post = cv2.imread(post, cv2.IMREAD_UNCHANGED)
    post_clean = img_post[..., 2].astype(np.float64) 

    # Apply M_rigid to align Post to Pre
    h, w = pre_clean.shape
    aligned_post = cv2.warpAffine(post_clean, M_rigid, (w, h), flags=cv2.INTER_LINEAR)
    
    return pre_clean, aligned_post
