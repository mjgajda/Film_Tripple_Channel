# analysis/dose_math.py
import numpy as np
import math
from statsmodels.nonparametric.smoothers_lowess import lowess

def _dose_from_netOD(netOD: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    a, b, c, d = coeffs
    dose = a * netOD**3 + b * netOD**2 + c * netOD + d
    dose[dose < 0] = 0
    return dose

def _lowess(y: np.ndarray, frac: float = 0.05) -> np.ndarray:
    x = np.arange(len(y))
    sm = lowess(y, x, frac=frac, return_sorted=False)
    return sm

def Find_p50(line_dose: np.ndarray) -> tuple[np.ndarray, float]:
    ind1 = np.where(line_dose >= 0.98 * np.max(line_dose))[0]
    max_val_ref = float(np.median(line_dose[ind1])) if len(ind1) else float(np.max(line_dose))
    min_val_ref = float(np.min(line_dose))
    p50max = 0.5 * (max_val_ref + min_val_ref)
    inds = np.where(line_dose <= p50max)[0]
    if len(inds) == 0:
        return np.array([0, len(line_dose)//2, len(line_dose)-1]), max_val_ref
    left = inds[inds < len(line_dose)//2]
    right = inds[inds > len(line_dose)//2]
    i1 = int(left[-1]) if len(left) else 0
    i2 = int(right[0]) if len(right) else len(line_dose) - 1
    p = np.array([i1, int(round(0.5*(i1+i2))), i2])
    return p, max_val_ref
