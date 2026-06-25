# analysis/calibration.py
import numpy as np
import matplotlib.pyplot as plt
from analysis.roi_analysis import scanner_to_OD
from gui.selectors import EllipseCollector
from utils.helpers import polyfit_linear

def NISTfilter_cal(file, path, refODs: np.ndarray, colorSpectrum: int):
    P_values, S_values, refOD_us = [], [], []

    
    from input_output.image_loader import load_image_channel
    scan = load_image_channel(path, file, colorSpectrum)
    n = len(refODs)
    collector = EllipseCollector(scan, n, "Outline each NIST reference OD (press Enter to confirm each ellipse).")
    masks = collector.run()

    refOD_vals = np.zeros(n)
    refOD_u = np.zeros(n)

    for j, mask in enumerate(masks):
        vals = scan[mask].astype(np.float64)
        vals = vals[vals > 0]
        vals_OD = scanner_to_OD(vals)
        refOD_vals[j] = np.median(vals_OD)
        refOD_u[j] = np.std(vals_OD) / max(refOD_vals[j], 1e-12) * 100.0

    P, S = polyfit_linear(refOD_vals, refODs)

    xs = np.linspace(refOD_vals.min(), refOD_vals.max(), 200)
    plt.plot(xs, np.polyval(P, xs), '--', label='Fit')
    plt.plot(refOD_vals, refODs, 'o', label='Measured')
    plt.xlabel('Scanner-Measured OD'); plt.ylabel('Reference OD')
    plt.legend(); plt.title('NIST calibration fit')
    plt.show()

   

    return P, S, refOD_us
