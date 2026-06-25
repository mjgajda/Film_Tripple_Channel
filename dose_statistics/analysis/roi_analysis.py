# analysis/roi_analysis.py
import numpy as np
import math
from dose_statistics.gui.dialogs import ask_text, ask_yesno
from dose_statistics.gui.selectors import EllipseCollector
from dose_statistics.utils.helpers import scanner_to_OD
from dose_statistics.utils.helpers import poly_apply



def BatchROIanalysis_OD(OD_pre, OD_post, colorSpectrum):
    """
    Perform ROI OD analysis using the same ROI mask for both pre and post-irradiation images.
    file_pair, path_pair, P_pair, S_pair, refOD_u_pair: np.array([pre, post])
    """
   
    # Load both pre- and post-irradiation images
    scan_pre = OD_pre
    scan_post = OD_post

    numrois = int(ask_text("ROIs", "How many ROIs do you need? (0 = continuous)", "0"))

    values, uncertainties, raw, pv_unc = [], [], [], []

    def do_pair_ellipse():
        

        OD_pre = scan_pre
        OD_post = scan_post

        # Median + uncertainty for both
        OD_pre_med = np.median(OD_pre)
        OD_post_med = np.median(OD_post)

        OD_pre_std = np.std(OD_pre)
        OD_post_std = np.std(OD_post)

        # Compute net OD
        netOD = OD_post_med - OD_pre_med
        netOD_u = math.sqrt(OD_pre_std**2 + OD_post_std**2)

       
        OD_u = netOD_u

        # Record data
        raw.append(np.median(netOD))      # raw post pixel value
        pv_unc.append(np.std(netOD_u))      # pixel std

    # Loop behavior
    if numrois == 0:
        while True:
            do_pair_ellipse()
            if not ask_yesno("Continue?", "Continue taking ROI data?", True):
                break
    else:
        for _ in range(numrois):
            do_pair_ellipse()

    # Combine results
    return np.column_stack([
        np.array(raw), 
        np.array(pv_unc)
    ])



def BatchROIanalysis_Dose(dose):    

    values, uncertainty = [], []

    Dose_Median = np.median(dose)

    Dose_mean = np.mean(dose)

    values.append(Dose_Median)
    uncertainty.append(np.std(dose))

    return np.column_stack([np.array(values), np.array(uncertainty)])