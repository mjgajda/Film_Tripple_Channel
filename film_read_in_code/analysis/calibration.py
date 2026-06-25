# analysis/calibration.py
import numpy as np
import matplotlib.pyplot as plt
from film_read_in_code.utils.helpers import scanner_to_OD
from film_read_in_code.gui.selectors import EllipseCollector
from film_read_in_code.utils.helpers import polyfit_linear

def NISTfilter_cal(masks, scan, refODs: np.ndarray):
    P_values, S_values, refOD_us = [], [], []

    n = len(refODs)
    refOD_vals = np.zeros(n)
    refOD_u = np.zeros(n)

    for j, mask in enumerate(masks):
        vals = scan[mask].astype(np.float64)
        vals = vals[vals > 0]
        vals_OD = scanner_to_OD(vals)
        refOD_vals[j] = np.median(vals_OD)
        refOD_u[j] = np.std(vals_OD) / max(refOD_vals[j], 1e-12) * 100.0

    # Linear fit
    P, S = polyfit_linear(refOD_vals, refODs)
    y_pred = np.polyval(P, refOD_vals)

    # --- Compute R² ---
    ss_res = np.sum((refODs - y_pred) ** 2)
    ss_tot = np.sum((refODs - np.mean(refODs)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # --- Plot calibration ---
    xs = np.linspace(refOD_vals.min(), refOD_vals.max(), 200)
    plt.plot(xs, np.polyval(P, xs), '--', label=f'Fit (R² = {r_squared:.4f})')
    plt.plot(refOD_vals, refODs, 'o', label='Measured')
    plt.xlabel('Scanner-Measured OD')
    plt.ylabel('Reference OD')
    plt.legend()
    plt.title('NIST Calibration Fit')
    plt.grid(True)
    plt.show()

    return P, S, refOD_us