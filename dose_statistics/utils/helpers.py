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
