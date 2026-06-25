# main.py
from film_read_in_code.gui.dialogs import ask_text, ask_yesno, ask_open_file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from film_read_in_code.make_masks_high_res import Load_Image_and_Run_Clustering
import os
# from clustering.save_images import save_cropped_rois
from film_read_in_code.Get_analysis_areas import get_analysis_areas
from film_read_in_code.input_output.getUserInfo import get_user_inputs, getFile

channel_map = {'r': 2, 'g': 1, 'b':0}
channel_map_inv = {v: k for k, v in channel_map.items()}


def load_or_run_clustering(role: str, load_fn, film_labels):
    """
    Checks if the stats dictionary already exists for this role ('pre' or 'post').
    If yes → load it from disk.
    If no  → run the clustering function and save the dictionary.

    Parameters
    ----------
    role : str
        'pre' or 'post'
    load_fn : callable
        Function that runs Load_Image_and_Run_Clustering()

    Returns
    -------
    stats : dict
        Loaded or newly generated stats dictionary.
    """
    file, path = getFile()
    # Extract filename WITHOUT extension
    stem = os.path.splitext(os.path.basename(file))[0]
    temp_dir = path

    # Join the path with the new folder name    
    STATS_DIR = os.path.join(temp_dir, f"{file[:-4]}_tiles")

    if not os.path.exists(STATS_DIR):
        os.makedirs(STATS_DIR, exist_ok=True)

    # Stats path is based on stem, NOT full filename
    stats_path = os.path.join(STATS_DIR, f"{stem}_r_stats.npy")
    print(f"[INFO] Looking for stats at: {stats_path}")
    # Load if exists
    if os.path.exists(stats_path):
        if ask_yesno("Load Cached Stats?", f"Saved {role} stats found for this file. Do you want to load them?", True):
            return STATS_DIR, path

    max_mb, n_clusters, nist_clusters, top_n, n_background, od_thresh_factor, refODs = \
        get_user_inputs()
    

    # Otherwise run clustering
    print(f"[INFO] No cached {role} stats found. Running clustering...")

    info = get_analysis_areas(file, path, 2, max_mb, n_clusters, nist_clusters, top_n, n_background, od_thresh_factor)

    for chan in [2,1,0]: # loop through all channels to save stats for each
        stats = Load_Image_and_Run_Clustering(file, path, channel_map_inv[chan], info, refODs, film_labels)

    # Save result
        save_stats_dict(stats, stem, color=channel_map_inv[chan], STATS_DIR=STATS_DIR)
        print(f"[INFO] Saved new {role} stats to:", stats_path)

    return  STATS_DIR, path


# Directory where stats dictionaries will be stored inside the current folder



def save_stats_dict(stats: dict, role: str, color: str, STATS_DIR: str = "stats"):
    """
    Save a stats dictionary to disk based on the file name inside the dict.

    Parameters
    ----------
    stats : dict
        Dictionary returned by Load_Image_and_Run_Clustering().
        Ideally contains a key like 'file', 'file_name', or 'filename'.
    role : str
        'pre' or 'post' (used to distinguish the two).
    """
    # Try to infer a base name from multiple possible keys
    

    if not os.path.exists(STATS_DIR):
        os.makedirs(STATS_DIR, exist_ok=True)

    out_path = os.path.join(STATS_DIR, f"{role}_{color}_stats.npy")

    if os.path.exists(out_path):
        print(f"[INFO] Stats file already exists, not overwriting: {out_path}")
        return

    np.save(out_path, stats, allow_pickle=True)
    print(f"[INFO] Saved {role} stats to: {out_path}")


def run_clustering(film_labels):
    path_pre, parent_path = load_or_run_clustering("pre", Load_Image_and_Run_Clustering, film_labels)

    path_post, parent_path = load_or_run_clustering("post", Load_Image_and_Run_Clustering, film_labels)

    return path_pre, parent_path, path_post, parent_path

