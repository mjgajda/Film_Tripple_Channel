import numpy as np
from gui.dialogs import ask_text, ask_yesno, ask_open_file
from gui.selectors import RectSelectorOnce
from input_output.image_loader import load_image_channel, upscale_mask
from clustering.k_means_cluturing import (
    collect_background_mask,
    compute_od_threshold,
    filter_valid_coords,
    run_kmeans_on_coords,
    reorder_labels_by_top_left_pixel,
    generate_masks_from_top_labels_by_x,
    generate_masks_raster_order,
    sort_and_relabel_masks_by_median,
)
from clustering.kmeans_testing_utils import test_cluster_labels_visual


# ------------------------------------------------------
# 1. USER INPUT AND IMAGE LOADING
# ------------------------------------------------------
def get_user_inputs():
    file, path = ask_open_file("Select an image file")

    color_str = ask_text("Color Channel", "Select channel (r/g/b)", "r").strip().lower()
    channel_map = {'r': 2, 'g': 1, 'b': 0}
    if color_str not in channel_map:
        raise ValueError("Channel must be 'r', 'g', or 'b'")
    channel = channel_map[color_str]

    max_mb = float(ask_text("Downscale Limit", "Max memory (MB) for downscale", "5.0"))
    n_clusters = int(ask_text("Clustering", "Number of K-means clusters", "21"))
    nist_clusters = 7
    top_n = int(ask_text("Clustering", "Top-N X clusters to include", "21"))
    n_background = int(ask_text("Background", "How many background ellipses?", "1"))
    od_thresh_factor = float(ask_text("Background Filter", "OD threshold factor", ".7"))

    return file, path, channel, max_mb, n_clusters, nist_clusters, top_n, n_background, od_thresh_factor


def load_and_select_region(path, file, channel, max_mb):
    full_img, scaled_img,_ ,img= load_image_channel(path=path, file=file, channel=channel, max_mb=max_mb)
    print("[INFO] Draw a rectangle to define region and vertical cut (press Enter to confirm).")
    _, (x, y, w, h) = RectSelectorOnce(scaled_img, "Define rectangular region for clustering").run()
    upper_limit = min(y + h + 30, scaled_img.shape[0])
    return full_img, scaled_img, img,(x, y, w, h), upper_limit


# ------------------------------------------------------
# 2. REGION / CUT CLUSTERING HELPERS
# ------------------------------------------------------
def cluster_region_once(region_img, n_background, od_thresh_factor, nist_clusters, top_n):
    """Perform one run of clustering for the rectangular region."""
    background_mask_region = collect_background_mask(region_img, n_background, title="Select Background (Region)")
    od_thresh_region = compute_od_threshold(region_img, background_mask_region, od_thresh_factor)
    coords_r, coords_valid_r, valid_mask_r = filter_valid_coords(region_img, od_thresh_region, background_mask_region)

    labels_r = run_kmeans_on_coords(coords_valid_r, nist_clusters)
    labels_r = reorder_labels_by_top_left_pixel(coords_r, coords_valid_r, valid_mask_r, labels_r, region_img.shape[:2])

    masks_cal = generate_masks_from_top_labels_by_x(coords_r, coords_valid_r, valid_mask_r, labels_r, region_img.shape, top_n)
    masks_cal, labels_r = sort_and_relabel_masks_by_median(masks_cal)
    return masks_cal, labels_r


def cluster_cut_once(scaled_img_cut, n_background, od_thresh_factor, n_clusters):
    """Perform one run of clustering for the vertical cut section."""
    background_mask_cut = collect_background_mask(scaled_img_cut, n_background, title="Select Background (Vertical Cut)")
    od_thresh_cut = compute_od_threshold(scaled_img_cut, background_mask_cut, od_thresh_factor)
    coords_c, coords_valid_c, valid_mask_c = filter_valid_coords(scaled_img_cut, od_thresh_cut, background_mask_cut)
    labels_c = run_kmeans_on_coords(coords_valid_c, n_clusters)

    masks_fils = generate_masks_raster_order(coords_c, coords_valid_c, valid_mask_c, labels_c, scaled_img_cut.shape)
    masks_fils, labels_c = sort_and_relabel_masks_by_median(masks_fils)
    return masks_fils, labels_c


# ------------------------------------------------------
# 3. ACCEPTANCE LOOP LOGIC
# ------------------------------------------------------
def confirm_masks_with_user(image, masks, labels, title):
    """Display mask overlay and ask user whether to accept or rerun."""
    accepted = False
    while not accepted:
        test_cluster_labels_visual(image, masks, labels, title=f"{title} — Review Result")
        accepted = ask_yesno("Accept Clustering?", f"Do you accept this clustering for {title}?", True)
        if not accepted:
            print(f"[INFO] Rerunning clustering for {title}...")
            return False  # user rejected
    return True


# ------------------------------------------------------
# 4. UPSCALING MASKS
# ------------------------------------------------------
def upscale_masks_to_full(scaled_img, full_img, masks, region_bounds=None, upper_limit=None):
    """Upscale masks back to full image space, either for a region or a vertical cut."""
    mask_full_list = []
    for m in masks:
        padded_scaled = np.zeros_like(scaled_img, dtype=bool)
        if region_bounds:
            x, y, w, h = region_bounds
            padded_scaled[y:y+h, x:x+w] = m
        elif upper_limit is not None:
            padded_scaled[upper_limit:, :] = m
        mask_full = upscale_mask(padded_scaled, full_img.shape[:2])
        mask_full_list.append(mask_full)
    return mask_full_list

## Shrinking the masks
def shrink_masks_to_median_window(masks: list[np.ndarray], fraction: float = 0.2) -> list[np.ndarray]:
    """
    Shrink each 2D boolean mask to include only the central fraction (e.g., 20%)
    of pixels around its median position.

    Parameters:
    - masks: list of 2D boolean arrays (same shape)
    - fraction: fraction of width/height to keep (0.2 = ±20% around median)

    Returns:
    - List of modified boolean masks of same shape.
    """
    import numpy as np

    shrunk_masks = []
    for mask in masks:
        ys, xs = np.where(mask)
        if ys.size == 0:
            shrunk_masks.append(mask.copy())
            continue

        # Median center of cluster
        y_med, x_med = np.median(ys), np.median(xs)

        # Full cluster span
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        # 20% window around median relative to total span
        y_range = (y_max - y_min) * fraction / 2
        x_range = (x_max - x_min) * fraction / 2

        # Compute bounds
        y_low, y_high = int(max(y_med - y_range, 0)), int(min(y_med + y_range, mask.shape[0] - 1))
        x_low, x_high = int(max(x_med - x_range, 0)), int(min(x_med + x_range, mask.shape[1] - 1))

        # Build shrunk mask
        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[y_low:y_high + 1, x_low:x_high + 1] = mask[y_low:y_high + 1, x_low:x_high + 1]
        shrunk_masks.append(new_mask)

    return shrunk_masks


# ------------------------------------------------------
# 5. MAIN PIPELINE
# ------------------------------------------------------
def Load_Image_and_Run_Clustering():
    file, path, channel, max_mb, n_clusters, nist_clusters, top_n, n_background, od_thresh_factor = get_user_inputs()
    full_img, scaled_img,img, (x, y, w, h), upper_limit = load_and_select_region(path, file, channel, max_mb)
    # upper_limit = 0
    # # --- REGION CLUSTERING WITH USER APPROVAL ---
    while True:
        masks_cal, labels_r = cluster_region_once(scaled_img[y:y+h, x:x+w], n_background, od_thresh_factor, nist_clusters, top_n)
        if confirm_masks_with_user(scaled_img[y:y+h, x:x+w], masks_cal, labels_r, "Rectangular Region"):
            break

    mask_full_region_list = upscale_masks_to_full(scaled_img, full_img, masks_cal, region_bounds=(x, y, w, h))
    # --- CUT CLUSTERING WITH USER APPROVAL ---
    while True:
        masks_fils, labels_c = cluster_cut_once(scaled_img[upper_limit:, :], n_background, od_thresh_factor, n_clusters)
        if confirm_masks_with_user(scaled_img[upper_limit:, :], masks_fils, labels_c, "Vertical Cut"):
            break

    mask_full_cut_list = upscale_masks_to_full(scaled_img, full_img, masks_fils, upper_limit=upper_limit)

    mask_full_region_list = shrink_masks_to_median_window(mask_full_region_list, fraction=0.4)
    mask_full_cut_list = shrink_masks_to_median_window(mask_full_cut_list, fraction=0.4)
    # --- FINAL COMBINATION VISUALIZATION ---
    all_masks_full = mask_full_region_list + mask_full_cut_list
    test_cluster_labels_visual(full_img, all_masks_full, labels=None, title="Full-Resolution Combined Masks")
    return mask_full_region_list, mask_full_cut_list, full_img

if __name__ == "__main__":
    Load_Image_and_Run_Clustering()