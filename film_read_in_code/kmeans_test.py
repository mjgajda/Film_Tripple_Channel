from gui.dialogs import ask_text, ask_yesno, ask_open_file
from clustering.k_means_cluturing import (
    collect_background_mask,
    compute_od_threshold,
    filter_valid_coords,
    run_kmeans_on_coords,
    generate_mask_from_labels,
    upscale_mask,
    generate_masks_raster_order,
    generate_masks_from_top_labels_by_x,
    remap_labels_by_raster_order,
    reorder_labels_by_top_left_pixel,
    scanner_to_OD,
    sort_and_relabel_masks_by_median
)
from input_output.image_loader import load_image_channel
from clustering.kmeans_testing_utils import (
    show_kmeans_labels,
    test_cluster_labels_visual,
    show_mask_overlay
)
import numpy as np


import os
import matplotlib.pyplot as plt
from matplotlib import cm

from gui.selectors import RectSelectorOnce
def color_mask_overlay(image: np.ndarray, masks: list[np.ndarray], title: str = "", alpha: float = 0.4):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')


    # Use tab20 colormap with 21 colors
    cmap = cm.get_cmap('tab20', 21)


    for i, mask in enumerate(masks):
        rgba = np.zeros((*mask.shape, 4))
        rgba[..., :3] = cmap(i % 21)[:3]
        rgba[..., 3] = mask.astype(float) * alpha
    ax.imshow(rgba)


    ax.set_title(title)
    plt.show()

def Load_Image_and_Run_Clustering():
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

    # Load full and scaled image
    full_img, scaled_img = load_image_channel(
        path=path,
        file=file,
        channel=channel,
        max_mb=max_mb
    )

    # Select rectangular region
    print("[INFO] Draw a rectangle to define region and vertical cut (press Enter to confirm).")
    _, (x, y, w, h) = RectSelectorOnce(scaled_img, "Define rectangular region for clustering").run()
    upper_limit = min(y + h + 30, scaled_img.shape[0])

    # --- CLUSTERING IN RECTANGULAR REGION ---
    region_img = scaled_img[y:y+h, x:x+w]
    background_mask_region = collect_background_mask(region_img, n_background, title="Select Background (Region)")
    od_thresh_region = compute_od_threshold(region_img, background_mask_region, od_thresh_factor)
    coords_r, coords_valid_r, valid_mask_r = filter_valid_coords(region_img, od_thresh_region, background_mask_region)
    labels_r = run_kmeans_on_coords(coords_valid_r, nist_clusters)
    labels_r = reorder_labels_by_top_left_pixel(
    coords=coords_r,
    coords_valid=coords_valid_r,
    valid_mask=valid_mask_r,
    labels=labels_r,
    shape=region_img.shape[:2]
)
    
    masks_cal = generate_masks_from_top_labels_by_x(
        coords_r,
        coords_valid_r,
        valid_mask_r,
        labels_r,
        region_img.shape,
        top_n,
    )
    masks_cal, labels_r = sort_and_relabel_masks_by_median(masks_cal)
    # Expand masks back to scaled image space
    mask_scaled_region_list = []
    for m in masks_cal:
        padded = np.zeros_like(scaled_img, dtype=bool)
        padded[y:y+h, x:x+w] = m
        mask_scaled_region_list.append(padded)

    # --- CLUSTERING IN VERTICAL CUT ---
    scaled_img_cut = scaled_img[upper_limit:, :]
    full_img_cut = full_img[int(full_img.shape[0] * (upper_limit / scaled_img.shape[0])):, :]
    background_mask_cut = collect_background_mask(scaled_img_cut, n_background, title="Select Background (Vertical Cut)")
    od_thresh_cut = compute_od_threshold(scaled_img_cut, background_mask_cut, od_thresh_factor)
    coords_c, coords_valid_c, valid_mask_c = filter_valid_coords(scaled_img_cut, od_thresh_cut, background_mask_cut)
    labels_c = run_kmeans_on_coords(coords_valid_c, n_clusters)
    # labels_c = remap_labels_by_raster_order(
    #     coords_valid_c,
    #     labels_c)
    masks_fils = generate_masks_raster_order(
        coords_c,
        coords_valid_c,
        valid_mask_c,
        labels_c,
        scaled_img_cut.shape
    )

    masks_fils, labels_c = sort_and_relabel_masks_by_median(masks_fils)
    # --- Visualize both results with 21-color overlay ---
    # --- after rectangular region clustering ---
    test_cluster_labels_visual(region_img, masks_cal, labels_r, title="Region Clusters (Sorted Masks)")

    # --- after vertical cut clustering ---
    test_cluster_labels_visual(scaled_img_cut, masks_fils, labels_c, title="Vertical Cut Clusters (Masks)")


   
     




def main():
    Load_Image_and_Run_Clustering()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except Exception as e:
        print("Error:", e)
        raise