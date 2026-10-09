import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import re
from dose_statistics.Dose_calculation.correction_to_film import  calculate_od, readInRawODFromPicture
from pathlib import Path
import json
from dose_statistics.analysis.histogram import BatchHistogramDose, BatchHistogram_OD
from dose_statistics.analysis.profile import Batch2Dprofile_Dose, Batch2Dprofile_OD
from dose_statistics.analysis.roi_analysis import BatchROIanalysis_Dose, BatchROIanalysis_OD
from dose_statistics.multichannel_dosimetry import dose_calculation, single_channel_dose
from dose_statistics.Constants import constants
from dose_statistics.gui.dialogs import ask_text, ask_float, ask_yesno, ask_open_file
channel_map = {'r': 2, 'g': 1, 'b': 0}

def match_array_sizes(arr1, arr2):
    # Get the minimum size for every dimension
    # This handles cases where one array might be 10x5 and the other 8x12
    min_shape = tuple(min(s1, s2) for s1, s2 in zip(arr1.shape, arr2.shape))
    
    # Create slices for each dimension: e.g., (slice(0, 8), slice(0, 5))
    selector = tuple(slice(0, dim) for dim in min_shape)
    
    # Return both arrays shrunk to the overlapping dimensions
    return arr1[selector], arr2[selector]

def analyzeODs(preOD, postOD, i, chan, saveDirectory):
    preOD_match, postOD_match = match_array_sizes(preOD, postOD)

    stats_hist_OD = BatchHistogram_OD(preOD_match, postOD_match, i, chan, saveDirectory)
    stats_prof_OD = Batch2Dprofile_OD(preOD_match, postOD_match, i, chan, saveDirectory)

    return postOD_match - preOD_match

def save_dose_and_od_to_csv(
    i,
    OD_r, OD_g, OD_b,
    D,
    D_r=None, D_g=None, D_b=None,
    D_r_err=None, D_g_err=None, D_b_err=None,
    save_dir='',
):
    """
    Save per-pixel OD values, single-channel doses, and multichannel dose
    for image pair index i to a CSV file.

    Parameters
    ----------
    i                    : int         — image pair index
    OD_r, OD_g, OD_b    : ndarray (H, W) — OD maps per channel
    D                    : ndarray (H, W) — multichannel corrected dose
    D_r, D_g, D_b        : ndarray (H, W) or None — single-channel doses
    D_r_err, D_g_err, D_b_err : ndarray (H, W) or None — dose uncertainties
    save_dir             : str         — directory to save CSV files
    """
    os.makedirs(save_dir, exist_ok=True)

    # Flatten all arrays to 1D
    H, W   = D.shape
    n_pix  = H * W
    pixels = np.arange(n_pix)
    rows, cols = np.divmod(pixels, W)

    data = {
        'index':    i,
        'row':      rows,
        'col':      cols,
        'OD_r':     OD_r.ravel(),
        'OD_g':     OD_g.ravel(),
        'OD_b':     OD_b.ravel(),
        'D_multichannel': D.ravel(),
    }


    

    if D_r is not None: data['D_r'] = D_r.ravel()
    if D_g is not None: data['D_g'] = D_g.ravel()
    if D_b is not None: data['D_b'] = D_b.ravel()

    if D_r_err is not None: data['D_r_err'] = D_r_err.ravel()
    if D_g_err is not None: data['D_g_err'] = D_g_err.ravel()
    if D_b_err is not None: data['D_b_err'] = D_b_err.ravel()

    none_keys = [k for k, v in data.items() if v is None]
    if none_keys:
        raise ValueError(f"Found None values for keys: {none_keys}")
    
    for name, value in data.items():
        length = len(value) if hasattr(value, '__len__') else 1
        print(f"{name}: length={length}")
        if hasattr(value, '__len__') and length != n_pix:
            print(f"  ⚠️  mismatch — expected {n_pix}")

    df       = pd.DataFrame(data)
    csv_path = os.path.join(save_dir, f'dose_od_pair_{i:03d}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved pair {i:03d} → {csv_path}  ({n_pix} pixels)")
    return save_dir


def main(postParentDir=None, preParentDir=None, parentDir=None, doseCurveJson='', film_labels=[]):
    ODs_per_channel_post = {'r': [], 'g': [], 'b': []}
    ODs_per_channel_pre  = {'r': [], 'g': [], 'b': []}

    postParentDir, snapshotsPost, preParentDir, snapshotsPre, NISTPost, NISTPre, doseCurveJson, saveDirectory = \
        constants(postParentDir, preParentDir, parentDir, doseCurveJson)

    def label_for(pos):
        return film_labels[pos] if pos < len(film_labels) else pos
    
    for channel in ['r', 'g', 'b']:
        for pos, (filePre, filePost) in enumerate(zip(snapshotsPre, snapshotsPost)):
            label = label_for(pos)

            pre_data  = np.load(NISTPre[channel],  allow_pickle=True)
            post_data = np.load(NISTPost[channel], allow_pickle=True)
            P_pre     = pre_data.item()['P']
            P_post    = post_data.item()['P']

            preOD, postOD = readInRawODFromPicture(filePre, filePost, channel_map[channel], P_pre, P_post)
            preOD, postOD = calculate_od(preOD, postOD, P_pre, P_post, channel_map[channel])

            analyzeODs(preOD, postOD, label, channel, saveDirectory)

            preOD, postOD = match_array_sizes(preOD, postOD)
            ODs_per_channel_pre[channel].append(preOD)
            ODs_per_channel_post[channel].append(postOD)

    found = True
    n_pairs = len(snapshotsPre)
    summary_rows = []                      # <-- add before the loop

    for pos in range(n_pairs):
        label = label_for(pos)

        csv_path = os.path.join(saveDirectory, f'dose_od_pair_{label:03d}.csv')
        if os.path.exists(csv_path) and found:
            if ask_yesno("File exists",
                         f"CSV for pair {label:03d} already exists. "
                         f"Do you want to continue analyzing dose csvs?",
                         default_yes=True):
                break
            else:
                found = False

        OD_r = ODs_per_channel_post['r'][pos]
        OD_g = ODs_per_channel_post['g'][pos]
        OD_b = ODs_per_channel_post['b'][pos]

        Unexp_r = ODs_per_channel_pre['r'][pos]
        Unexp_g = ODs_per_channel_pre['g'][pos]
        Unexp_b = ODs_per_channel_pre['b'][pos]

        X       = np.stack([OD_r, OD_g, OD_b], axis=0)
        X_unexp = np.stack([Unexp_r, Unexp_g, Unexp_b], axis=0)

        D, Delta, alpha, beta, n_iter, cov = dose_calculation(
            calibration_json=doseCurveJson,
            X=X,
            X_unexp=X_unexp,
        )
        print(f"Pair {label:02d} | multichannel — mean dose: {D.mean():.4f}  "
              f"std: {D.std():.4f}  iters: {n_iter}")

        D_r, D_r_err = single_channel_dose(OD_r, channel='R', json_path=doseCurveJson)
        D_g, D_g_err = single_channel_dose(OD_g, channel='G', json_path=doseCurveJson)
        D_b, D_b_err = single_channel_dose(OD_b, channel='B', json_path=doseCurveJson)

        save_dir = save_dose_and_od_to_csv(
            i=label,
            OD_r=OD_r, OD_g=OD_g, OD_b=OD_b,
            D=D,
            D_r=D_r, D_g=D_g, D_b=D_b,
            D_r_err=D_r_err, D_g_err=D_g_err, D_b_err=D_b_err,
            save_dir=saveDirectory,
        )
        summary_rows.append(                # <-- add
            summarize_film_doses(
                label,
                OD_r, OD_g, OD_b, D,
                D_r, D_g, D_b,
                D_r_err, D_g_err, D_b_err,
            )
        )

    return saveDirectory

def summarize_film_doses(label, OD_r, OD_g, OD_b, D,
                         D_r, D_g, D_b,
                         D_r_err=None, D_g_err=None, D_b_err=None):
    """
    Collapse the per-pixel maps for one film into a single summary row:
    mean/std OD and dose per channel, plus mean uncertainty per channel.
    Returns a dict (one row).
    """
    def stats(arr):
        if arr is None:
            return np.nan, np.nan
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan, np.nan
        return float(a.mean()), float(a.std())

    OD_r_mean, OD_r_std = stats(OD_r)
    OD_g_mean, OD_g_std = stats(OD_g)
    OD_b_mean, OD_b_std = stats(OD_b)

    D_multi_mean, D_multi_std = stats(D)
    D_r_mean, D_r_std = stats(D_r)
    D_g_mean, D_g_std = stats(D_g)
    D_b_mean, D_b_std = stats(D_b)

    # mean of the per-pixel uncertainty maps (if provided)
    D_r_err_mean, _ = stats(D_r_err)
    D_g_err_mean, _ = stats(D_g_err)
    D_b_err_mean, _ = stats(D_b_err)

    n_pix = int(np.isfinite(np.asarray(D, dtype=float).ravel()).sum())

    return {
        'film':            label,
        'n_pixels':        n_pix,
        'OD_r_mean':       OD_r_mean,  'OD_r_std': OD_r_std,
        'OD_g_mean':       OD_g_mean,  'OD_g_std': OD_g_std,
        'OD_b_mean':       OD_b_mean,  'OD_b_std': OD_b_std,
        'D_multi_mean':    D_multi_mean, 'D_multi_std': D_multi_std,
        'D_r_mean':        D_r_mean,   'D_r_std':  D_r_std,   'D_r_err_mean': D_r_err_mean,
        'D_g_mean':        D_g_mean,   'D_g_std':  D_g_std,   'D_g_err_mean': D_g_err_mean,
        'D_b_mean':        D_b_mean,   'D_b_std':  D_b_std,   'D_b_err_mean': D_b_err_mean,
    }


def save_film_dose_summary(summary_rows, save_dir):
    """Write the accumulated per-film summary rows to one CSV."""
    if not summary_rows:
        print("No summary rows to save.")
        return None
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(summary_rows).sort_values('film').reset_index(drop=True)
    out_path = os.path.join(save_dir, 'film_dose_summary.csv')
    df.to_csv('/home/michaelgajda/cal_Lab_Work/film/film_triple_channel_dosimetry/ebtxD_lot/film_dose_summary.csv', index=False)
    print(f"Saved film dose summary → {out_path}  ({len(df)} films)")
    return out_path

