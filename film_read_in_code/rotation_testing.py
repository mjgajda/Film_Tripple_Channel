import cv2
import numpy as np
from clustering.make_masks import (get_user_inputs)
import os
def load_image_channel(
    path: str, file: str, channel: int, max_mb: float = None,
    detect_red: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load image, extract one color channel, optionally detect red markers.
    Returns:
        selected: Full-resolution selected channel
        scaled: Downscaled selected channel
        red_mask: Binary mask of red regions (if detect_red=True)
    """
    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {file}")

    # --- Handle grayscale or color ---
    if img.ndim == 2:
        selected = img.astype(np.float64)
    else:
        selected = img[..., channel].astype(np.float64)

    # --- Downscale if needed ---
    scaled = selected
    if max_mb is not None:
        h, w = selected.shape
        total_mb = h * w * selected.itemsize / (1024 * 1024)
        if total_mb > max_mb:
            scale = (max_mb / total_mb) ** 0.5
            new_size = (int(w * scale), int(h * scale))
            scaled = cv2.resize(selected, new_size, interpolation=cv2.INTER_AREA)

    # --- Optional red marker detection ---
    red_mask = None
    if detect_red and img.ndim == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

    return selected, scaled, red_mask

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_red_mask(image_bgr, sensitivity=1.0):
    """
    Detect red regions in a BGR image with tunable sensitivity.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low_sat = max(40, int(70 / sensitivity))
    high_val = 255
    hue_range = int(10 * sensitivity)

    lower_red1 = np.array([0, low_sat, 50])
    upper_red1 = np.array([hue_range, high_val, high_val])
    lower_red2 = np.array([180 - hue_range, low_sat, 50])
    upper_red2 = np.array([180, high_val, high_val])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    return mask

import cv2
import numpy as np

def get_centroids(mask, min_area=10, max_points=4):
    """
    Detect centroids of red markers from a binary mask.
    Keeps only the largest 'max_points' markers by area.

    Parameters:
        mask : np.ndarray
            Binary mask of red markers
        min_area : int
            Minimum area for a contour to be considered
        max_points : int
            Maximum number of centroids to return (default=4)
    Returns:
        np.ndarray of shape (N,2)
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)

    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centers.append([cx, cy])
        if len(centers) >= max_points:
            break  # stop once we have enough

    return np.array(centers, dtype=np.float32)


def align_using_red_markers_auto_retry(path_pre, file_pre, path_post, file_post, max_retries=1000):
    """
    Align two film images (pre, post) using red marker centroids.
    Automatically retries detection until both have the same number of markers.
    Returns aligned_pre, aligned_post, M, pts_ref, pts_mov
    """

    # --- Load both images ---
    img_pre = cv2.imread(os.path.join(path_pre, file_pre))
    img_post = cv2.imread(os.path.join(path_post, file_post))
    if img_pre is None or img_post is None:
        raise IOError("Could not read one or both images.")

    # --- Resize post to match pre if dimensions differ ---
    h_pre, w_pre = img_pre.shape[:2]
    h_post, w_post = img_post.shape[:2]
    if (h_pre, w_pre) != (h_post, w_post):
        print(f"[INFO] Resizing post image from ({w_post}x{h_post}) → ({w_pre}x{h_pre})")
        img_post = cv2.resize(img_post, (w_pre, h_pre), interpolation=cv2.INTER_AREA)

    sens_pre, sens_post = 1.0, 1.0

    for attempt in range(max_retries):
        mask_pre  = detect_red_mask(img_pre,  sensitivity=sens_pre)
        mask_post = detect_red_mask(img_post, sensitivity=sens_post)

        pts_pre  = get_centroids(mask_pre)
        pts_post = get_centroids(mask_post)

        print(f"[Attempt {attempt+1}] pre={len(pts_pre)}, post={len(pts_post)} markers")

        if len(pts_pre) >= 3 and len(pts_post) >= 3 and len(pts_pre) == len(pts_post):
            break  # good match
        if len(pts_pre) < len(pts_post):
            sens_pre *= 1.01
        elif len(pts_post) < len(pts_pre):
            sens_post *= 1.01
        else:
            sens_pre *= 1.1
            sens_post *= 1.1

    # --- Final sanity check ---
    if len(pts_pre) < 3 or len(pts_post) < 3 or len(pts_pre) != len(pts_post):
        raise RuntimeError(f"Failed to find equal markers after {max_retries} retries "
                           f"(pre={len(pts_pre)}, post={len(pts_post)}).")

    # --- Compute affine transform (post → pre) ---
    pts_pre  = pts_pre.reshape(-1, 2).astype(np.float32)
    pts_post = pts_post.reshape(-1, 2).astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(pts_post, pts_pre, method=cv2.RANSAC)
    if M is None:
        raise RuntimeError("Affine estimation failed — check marker positions or detection quality.")

    # --- Warp both images to a common space ---
    h, w = img_pre.shape[:2]
    aligned_pre  = cv2.warpAffine(img_pre,  np.eye(2,3, dtype=np.float32), (w, h))  # identity
    aligned_post = cv2.warpAffine(img_post, M, (w, h))

    # --- Visual Debug ---
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(cv2.cvtColor(mask_pre, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Pre Mask ({len(pts_pre)} markers)")
    ax[1].imshow(cv2.cvtColor(mask_post, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"Post Mask ({len(pts_post)} markers)")
    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"[INFO] Alignment successful with {len(pts_pre)} markers.")

    return aligned_pre, aligned_post, M, pts_pre, pts_post



import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def show_alignment_overlay(imgA, imgB_aligned, ptsA, ptsB):
    overlay = cv2.addWeighted(imgA, 0.5, imgB_aligned, 0.5, 0)
    for (x, y) in ptsA:
        cv2.circle(overlay, (int(x), int(y)), 8, (0,255,0), 2)  # green: ref
    for (x, y) in ptsB:
        cv2.circle(overlay, (int(x), int(y)), 8, (255,0,0), 2)  # blue: moving
    plt.figure(figsize=(8,8))
    plt.imshow(overlay[..., ::-1])  # BGR→RGB
    plt.title("Alignment check: green=ref, blue=moving")
    plt.axis('off')
    plt.show()

import cv2
import numpy as np

def get_marker_centroids(red_mask, min_area=5):
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centers.append((cx, cy))
    return np.array(centers, dtype=np.float32)


import cv2
import matplotlib.pyplot as plt

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_red_filtered_image(path, file, mask, title="Red-Only Image"):
    """
    Load an image from (path, file), and display only the red regions
    defined by 'mask' (everything else blacked out).
    Converts float64 images to uint8 for visualization.

    Parameters:
        path : str
            Directory containing the image file.
        file : str
            Image filename.
        mask : np.ndarray
            Binary mask of red regions (same shape as image).
        title : str
            Title for the display window.
    """

    # --- Load full image ---
    img_path = os.path.join(path, file)
    image_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise IOError(f"Could not read image from: {img_path}")

    # --- Ensure 3-channel BGR ---
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    # --- Convert float64 → uint8 safely for visualization ---
    if image_bgr.dtype != np.uint8:
        image_bgr_disp = cv2.convertScaleAbs(image_bgr)
    else:
        image_bgr_disp = image_bgr.copy()

    # --- Ensure mask is binary uint8 ---
    mask = (mask > 0).astype("uint8") * 255

    # --- Apply mask ---
    red_only = cv2.bitwise_and(image_bgr_disp, image_bgr_disp, mask=mask)

    # --- Convert to RGB for matplotlib ---
    red_only_rgb = cv2.cvtColor(red_only, cv2.COLOR_BGR2RGB)

    # --- Display ---
    plt.figure(figsize=(7,7))
    plt.imshow(red_only_rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import cv2

def test_alignment_with_histogram(img_ref, img_aligned, title="Alignment Difference Histogram"):
    """
    Compute pixel-by-pixel difference between reference and aligned images.
    Display histogram of differences and return key statistics.

    Parameters:
        img_ref (np.ndarray): Reference image
        img_aligned (np.ndarray): Aligned moving image
    """
    # --- Step 1: ensure same shape and dtype ---
    if img_ref.shape != img_aligned.shape:
        raise ValueError(f"Image shape mismatch: ref={img_ref.shape}, aligned={img_aligned.shape}")

    # Convert both to float for subtraction
    ref = img_ref.astype(np.float64)
    aligned = img_aligned.astype(np.float64)

    # --- Step 2: compute difference ---
    diff = aligned - ref
    diff_flat = diff.flatten()

    # --- Step 3: basic statistics ---
    mean_diff = np.mean(diff_flat)
    std_diff = np.std(diff_flat)
    rmse = np.sqrt(np.mean(diff_flat**2))

    print(f"📊 Mean difference: {mean_diff:.4f}")
    print(f"📉 Std deviation:  {std_diff:.4f}")
    print(f"📈 RMSE:           {rmse:.4f}")

    # --- Step 4: plot histogram ---
    plt.figure(figsize=(8,5))
    plt.hist(diff_flat, bins=100, color='steelblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Pixel Difference (Aligned - Reference)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # --- Optional: visualize difference image ---
    diff_vis = np.clip((diff - diff.min()) / (diff.max() - diff.min()), 0, 1)
    plt.figure(figsize=(6,6))
    plt.imshow(diff_vis, cmap='coolwarm')
    plt.title("Normalized Difference Image")
    plt.axis("off")
    plt.show()

    return mean_diff, std_diff, rmse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import cv2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def compare_alignment_gaussian(img_ref_before, img_mov_before,
                               img_ref_after, img_mov_after,
                               zero_range_sigma=2.0, bins=100):
    """
    Compare alignment quality before and after using pixel-difference histograms.
    The 'after' histogram is computed only from ±Nσ inliers and shows a Gaussian fit.
    """


    def compute_diff_stats(imgA, imgB):
        """
        Compute pixel-wise difference statistics (mean, std) between two images.
        Automatically resizes imgB to match imgA if their dimensions differ.

        Parameters:
            imgA, imgB : np.ndarray
                Input images (grayscale or color).

        Returns:
            diff_flat : np.ndarray
                Flattened array of pixel differences (B - A)
            mu : float
                Mean difference
            sigma : float
                Standard deviation of differences
        """

        # --- Ensure both are grayscale if needed ---
        if imgA.ndim == 3 and imgA.shape[2] == 3:
            imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        if imgB.ndim == 3 and imgB.shape[2] == 3:
            imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

        # --- Match dimensions if needed ---
        if imgA.shape != imgB.shape:
            print(f"[INFO] Resizing imgB from {imgB.shape[::-1]} → {imgA.shape[::-1]} to match shapes.")
            imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]), interpolation=cv2.INTER_AREA)

        # --- Compute difference ---
        A = imgA.astype(np.float64)
        B = imgB.astype(np.float64)
        diff = B - A
        diff_flat = diff.flatten()

        mu = np.mean(diff_flat)
        sigma = np.std(diff_flat)

        return diff_flat, mu, sigma

    # --- Compute stats before & after alignment ---
    diff_before, mu_before, sigma_before = compute_diff_stats(img_ref_before, img_mov_before)
    diff_after,  mu_after,  sigma_after  = compute_diff_stats(img_ref_after,  img_mov_after)

    # --- Filter after-alignment data to ±Nσ ---
    keep_after = diff_after 
    diff_after_inliers = diff_after
    mu_after_inlier, sigma_after_inlier = np.mean(diff_after_inliers), np.std(diff_after_inliers)

    # --- Plot side-by-side histograms ---
    plt.figure(figsize=(12,5))

    # BEFORE alignment: full data, no Gaussian
    plt.subplot(1,2,1)
    plt.hist(diff_before, bins=bins, color='orange', edgecolor='black', alpha=0.7)
    plt.title("Before Alignment")
    plt.xlabel("Pixel Difference (Post – Pre)")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)

    # AFTER alignment: only ±2σ data + Gaussian fit
    plt.subplot(1,2,2)
    plt.hist(diff_after_inliers, bins=bins, density=True,
             color='steelblue', edgecolor='black', alpha=0.6, label='Inlier Pixels (±2σ)')
    x = np.linspace(mu_after_inlier - 4*sigma_after_inlier,
                    mu_after_inlier + 4*sigma_after_inlier, 400)
    gauss = norm.pdf(x, mu_after_inlier, sigma_after_inlier)
    plt.plot(x, gauss, 'r-', linewidth=2, label=f'Gaussian Fit\nμ={mu_after_inlier:.4f}, σ={sigma_after_inlier:.4f}')
    plt.title(f"After Alignment (±{zero_range_sigma}σ)")
    plt.xlabel("Pixel Difference (Post – Pre)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    print(f"Before alignment: μ={mu_before:.4f}, σ={sigma_before:.4f}")
    print(f"After  alignment: μ={mu_after_inlier:.4f}, σ={sigma_after_inlier:.4f}")
    print(f"Improvement Δσ = {sigma_before - sigma_after_inlier:.4f}")
    print(f"Inliers retained after alignment: {keep_after.sum()/len(diff_after)*100:.1f}%")

    return {
        "mu_before": mu_before,
        "sigma_before": sigma_before,
        "mu_after": mu_after_inlier,
        "sigma_after": sigma_after_inlier
    }

def main():
    # --- Get inputs ---
    file_pre, path_pre, channel, max_mb, n_clusters, nist_clusters, top_n, n_background, od_thresh_factor = get_user_inputs()
    file_post, path_post, channel, max_mb, n_clusters, nist_clusters, top_n, n_background, od_thresh_factor = get_user_inputs()

    # --- Load images with red marker detection ---
    selected_pre, _, red_pre = load_image_channel(path_pre, file_pre, channel=2, detect_red=True)
    selected_post, _, red_post = load_image_channel(path_post, file_post, channel=2, detect_red=True)

    allignedPre, aligned_post, M, pts_ref, pts_mov = align_using_red_markers_auto_retry(
    path_pre, file_pre,
    path_post, file_post,
    max_retries=10000
)
    # --- Compare alignment before/after using Gaussian-fitted histograms ---
    print("\n=== Histogram Comparison: Before vs After Alignment ===")
    stats = compare_alignment_gaussian(
        img_ref_before=selected_pre,
        img_mov_before=selected_post,
        img_ref_after=allignedPre,
        img_mov_after=aligned_post,
        zero_range_sigma=2.0,   # keep ±2σ inliers near zero
        bins=100
    )

    print("\n=== Alignment Quality Summary ===")
    print(f"Before alignment: μ = {stats['mu_before']:.4f}, σ = {stats['sigma_before']:.4f}")
    print(f"After  alignment: μ = {stats['mu_after']:.4f}, σ = {stats['sigma_after']:.4f}")
    print(f"Improvement Δσ = {stats['sigma_before'] - stats['sigma_after']:.4f}")

    # --- Show difference map and centroid overlay ---
    # mean_diff, std_diff, rmse = test_alignment_with_histogram(selected_pre, aligned_post)
    show_alignment_overlay(allignedPre, aligned_post, pts_ref, pts_mov)

    # print("\n=== Pixelwise Stats ===")
    # print(f"Mean difference: {mean_diff:.4f}")
    # print(f"Std deviation:  {std_diff:.4f}")
    # print(f"RMSE:           {rmse:.4f}")


if __name__ == "__main__":
    main()