import numpy as np
from film_read_in_code.gui.dialogs import ask_text, ask_yesno, ask_open_file
from film_read_in_code.gui.selectors import RectSelectorOnce
from film_read_in_code.input_output.image_loader import load_image_channel
from film_read_in_code.clustering.k_means_cluturing import (
    collect_background_mask,
    compute_od_threshold,
    filter_valid_coords,
    run_kmeans_on_coords,
    reorder_labels_by_top_left_pixel,
    generate_masks_from_top_labels_by_x,
    generate_masks_raster_order,
    sort_and_relabel_masks_by_median,
)

from film_read_in_code.clustering.kmeans_testing_utils import test_cluster_labels_visual
# ----------------------------------------------

# ------------------------------------------------------
# 2. LOAD AND SELECT REGION (scaled only)
# ------------------------------------------------------
def load_and_select_region(path, file, channel, max_mb):

    selected_full, scaled, red_mask_scaled, scale_x, scale_y, Dimensions = load_image_channel(path=path, file=file, channel=channel, max_mb=max_mb)

    print("[INFO] Draw a rectangle to define region and vertical cut.")
    _, (x, y, w, h) = RectSelectorOnce(scaled, "Define rectangular region").run()

    upper_limit = min(y + h + 30, scaled.shape[0])

    return scaled, red_mask_scaled, (x, y, w, h), upper_limit, scale_x, scale_y, Dimensions



# ------------------------------------------------------
# 3. REGION / CUT CLUSTERING
# ------------------------------------------------------
def cluster_region_once(region_img, n_background, od_thresh_factor, nist_clusters, top_n):

    background_mask_region = collect_background_mask(region_img, n_background, title="Select Background (Region)")
    od_thresh_region = compute_od_threshold(region_img, background_mask_region, od_thresh_factor)

    coords_r, coords_valid_r, valid_mask_r = filter_valid_coords(
        region_img, od_thresh_region, background_mask_region)

    labels_r = run_kmeans_on_coords(coords_valid_r, nist_clusters)
    labels_r = reorder_labels_by_top_left_pixel(
        coords_r, coords_valid_r, valid_mask_r, labels_r, region_img.shape[:2])

    masks_cal = generate_masks_from_top_labels_by_x(
        coords_r, coords_valid_r, valid_mask_r, labels_r, region_img.shape, top_n)

    masks_cal, labels_r = sort_and_relabel_masks_by_median(masks_cal)
    return masks_cal, labels_r


def confirm_masks_with_user(image, masks, labels, title):
    """Display mask overlay and ask user whether to accept or rerun."""
    accepted = False
    while not accepted:
        test_cluster_labels_visual(image, masks, f"{title} — Review Result")
        accepted = ask_yesno("Accept Clustering?", f"Do you accept this clustering for {title}?", True)
        if not accepted:
            print(f"[INFO] Rerunning clustering for {title}...")
            return False  # user rejected
    return True


# ------------------------------------------------------
# 4. MASK SHRINKING (scaled only)
# ------------------------------------------------------
def shrink_masks_to_median_window(masks, fraction=0.2):

    shrunk = []
    for m in masks:
        ys, xs = np.where(m)
        if ys.size == 0:
            shrunk.append(m.copy())
            continue

        y_med, x_med = np.median(ys), np.median(xs)

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        y_range = (y_max - y_min) * fraction / 2
        x_range = (x_max - x_min) * fraction / 2

        y0 = int(max(y_med - y_range, 0))
        y1 = int(min(y_med + y_range, m.shape[0] - 1))
        x0 = int(max(x_med - x_range, 0))
        x1 = int(min(x_med + x_range, m.shape[1] - 1))

        new_mask = np.zeros_like(m, bool)
        new_mask[y0:y1+1, x0:x1+1] = m[y0:y1+1, x0:x1+1]
        shrunk.append(new_mask)

    return shrunk
def cluster_cut_once(scaled_img_cut, n_background, od_thresh_factor, n_clusters):

    background_mask_cut = collect_background_mask(scaled_img_cut, n_background, title="Select Background (Cut)")
    od_thresh_cut = compute_od_threshold(scaled_img_cut, background_mask_cut, od_thresh_factor)

    coords_c, coords_valid_c, valid_mask_c = filter_valid_coords(
        scaled_img_cut, od_thresh_cut, background_mask_cut)

    labels_c = run_kmeans_on_coords(coords_valid_c, n_clusters)

    masks_fils = generate_masks_raster_order(
        coords_c, coords_valid_c, valid_mask_c, labels_c, scaled_img_cut.shape)

    masks_fils, labels_c = sort_and_relabel_masks_by_median(masks_fils)
    return masks_fils, labels_c

def get_analysis_areas(file, path, channel, max_mb, n_clusters, nist_clusters, top_n, n_background, od_thresh_factor):
    

    while True:
        scaled_img_nist, red_mask_scaled, (x, y, w, h), upper_limit, scale_x, scale_y, full_shape = load_and_select_region(path, file, channel, max_mb)

        # Convenience handles
        Hs, Ws = scaled_img_nist.shape
        Hf, Wf = full_shape

        # -------------------- 1) REGION CLUSTERING (NIST REGION) --------------------
        region_img = scaled_img_nist[y:y+h, x:x+w]
        masks_cal, labels_r = cluster_region_once(
            region_img, n_background, od_thresh_factor, nist_clusters, top_n
        )
        if confirm_masks_with_user(region_img, masks_cal, labels_r, "NIST Region"):
            break

    # Shrink NIST masks around their median
    masks_cal = shrink_masks_to_median_window(masks_cal, fraction=0.05)

    # Embed NIST masks back into full scaled image
    nist_masks_scaled = []
    for m in masks_cal:
        mm = np.zeros_like(scaled_img_nist, dtype=bool)
        mm[y:y+h, x:x+w] = m
        nist_masks_scaled.append(mm)
    # Vertical cut image (everything below the NIST region + margin)
    

    while True:
        scaled_img_film, red_mask_scaled, (x, y, w, h), upper_limit, scale_x, scale_y, full_shape = load_and_select_region(path, file, channel, max_mb)
        cut_img = scaled_img_film[y:y+h, x:x+w]
        masks_fils, labels_c = cluster_cut_once(
            cut_img, n_background, od_thresh_factor, n_clusters
        )
        if confirm_masks_with_user(cut_img, masks_fils, labels_c, "Vertical Cut"):
            break

    # Shrink vertical-cut masks
    masks_fils = shrink_masks_to_median_window(masks_fils, fraction=.5)

    # Embed vertical-cut masks back into full scaled image
    cut_masks_scaled = []
    for m in masks_fils:
        mm = np.zeros_like(scaled_img_film, dtype=bool)
        mm[y:y+h, x:x+w] = m
        cut_masks_scaled.append(mm)
    

    return nist_masks_scaled, cut_masks_scaled, scaled_img_film, scaled_img_nist, full_shape, scale_x, scale_y
