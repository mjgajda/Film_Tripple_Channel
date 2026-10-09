# main.py
import glob
import os

import numpy as np

from film_read_in_code.gui.dialogs import ask_yesno
from film_read_in_code.make_masks_high_res import Load_Image_and_Run_Clustering
from film_read_in_code.Get_analysis_areas import get_analysis_areas, get_nist_analysis_areas
from film_read_in_code.input_output.getUserInfo import get_user_inputs, getFile
from film_read_in_code.input_output.od_json_export import save_od_json
from film_read_in_code.analysis.calibration import NISTfilter_cal
from film_read_in_code.utils.helpers import poly_apply

channel_map = {'r': 2, 'g': 1, 'b': 0}
channel_map_inv = {v: k for k, v in channel_map.items()}


def _run_and_save_channels(file, path, stem, nist_info, film_info, refODs, labels,
                           STATS_DIR, tag=None):
    """Run R, G, B for one set of film masks and save stats + OD JSON per channel."""
    name = f"{stem}_{tag}" if tag else stem
    for chan in [2, 1, 0]:  # R, G, B — same masks for every channel
        color = channel_map_inv[chan]
        stats = Load_Image_and_Run_Clustering(file, path, color, nist_info, film_info, refODs, labels)
        save_stats_dict(stats, name, color=color, STATS_DIR=STATS_DIR)
        save_od_json(stats, STATS_DIR, calib_fn=NISTfilter_cal, apply_fn=poly_apply, tag=tag)


def load_or_run_clustering(role: str, load_fn, film_labels, multiLabels=False):
    """
    Use cached OD JSONs for this role ('pre' or 'post') if present, otherwise run
    clustering. For each set of film masks and each channel it saves
        <name>_<c>_stats.npy   stats dictionary
        <name>_<c>_od.json     mask-filtered OD summary + calibration + provenance
        <name>_<c>_od.npz      per-pixel OD and pixel coordinates
    where <name> = <stem> (one label list) or <stem>_set<k> (multiLabels).

    Returns
    -------
    STATS_DIR : str   folder holding the files above (<stem>_tiles)
    path      : str   folder of the scan TIFF
    """
    file, path = getFile()
    stem = os.path.splitext(os.path.basename(file))[0]

    STATS_DIR = os.path.join(path, f"{stem}_tiles")
    os.makedirs(STATS_DIR, exist_ok=True)

    # Cache = OD JSONs for every channel
    print(f"[INFO] Looking for OD JSONs in: {STATS_DIR}")
    cached = {c: glob.glob(os.path.join(STATS_DIR, f"{stem}*_{c}_od.json")) for c in channel_map}
    if all(cached.values()):
        if ask_yesno("Load Cached OD?",
                     f"Saved {role} OD JSONs found for this file. Do you want to load them?", True):
            return STATS_DIR, path
    elif glob.glob(os.path.join(STATS_DIR, f"{stem}*_stats.npy")):
        print(f"[INFO] Found {role} stats .npy files but no OD JSONs (older run); "
              f"clustering again to create them.")

    print(f"[INFO] Running clustering for {role}...")

    max_mb, nist_clusters, top_n, n_background, od_thresh_factor, refODs = get_user_inputs()

    nist_info = get_nist_analysis_areas(file, path, 2, max_mb, nist_clusters, top_n,
                                        n_background, od_thresh_factor)

    if multiLabels:
        # one file set per label list, so no set overwrites another
        for k, labels in enumerate(film_labels):
            film_info = get_analysis_areas(file, path, 2, max_mb, len(labels),
                                           n_background, od_thresh_factor)
            _run_and_save_channels(file, path, stem, nist_info, film_info, refODs, labels,
                                   STATS_DIR, tag=f"set{k}")
    else:
        film_info = get_analysis_areas(file, path, 2, max_mb, len(film_labels),
                                       n_background, od_thresh_factor)
        _run_and_save_channels(file, path, stem, nist_info, film_info, refODs, film_labels,
                               STATS_DIR)

    print(f"[INFO] Saved {role} stats and OD JSONs to: {STATS_DIR}")
    return STATS_DIR, path


def save_stats_dict(stats: dict, role: str, color: str, STATS_DIR: str = "stats"):
    """Save a stats dictionary to <STATS_DIR>/<role>_<color>_stats.npy (overwrites)."""
    os.makedirs(STATS_DIR, exist_ok=True)
    out_path = os.path.join(STATS_DIR, f"{role}_{color}_stats.npy")
    np.save(out_path, stats, allow_pickle=True)  # type: ignore
    print(f"[INFO] Saved {role} stats to: {out_path}")


def run_clustering(film_labels, multiLabels=False):
    path_pre, parent_path_pre = load_or_run_clustering(
        "pre", Load_Image_and_Run_Clustering, film_labels, multiLabels=multiLabels)
    path_post, parent_path_post = load_or_run_clustering(
        "post", Load_Image_and_Run_Clustering, film_labels, multiLabels=multiLabels)

    return path_pre, parent_path_pre, path_post, parent_path_post
