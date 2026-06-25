import cv2
import numpy as np
from clustering.make_masks import (get_user_inputs)

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

def detect_red_points(image, min_area=5):
    """
    Detect red blobs in an image and return their centroids.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red range (handles wrap-around)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Threshold both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Find connected components
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    return np.array(centers, dtype=np.float32), mask

def align_using_red_markers(img_ref, img_to_align, mask_ref, mask_to_align):
    # --- Extract marker coordinates ---
    pts_ref = get_marker_centroids(mask_ref)
    pts_mov = get_marker_centroids(mask_to_align)

    if len(pts_ref) < 3 or len(pts_mov) < 3:
        raise ValueError("Need ≥3 red markers in both images for reliable rotation/scale alignment")

    # --- Sort by X position (optional, simple matching) ---
    pts_ref = pts_ref[np.argsort(pts_ref[:, 0])]
    pts_mov = pts_mov[np.argsort(pts_mov[:, 0])]

    # --- Compute affine transform (rotation + translation + scale) ---
    M = cv2.estimateAffinePartial2D(pts_mov, pts_ref, method=cv2.RANSAC)[0]

    # --- Warp the moving image ---
    aligned = cv2.warpAffine(img_to_align, M, (img_ref.shape[1], img_ref.shape[0]))

    return aligned, M, pts_ref, pts_mov


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

from input_output.image_loader import load_image_channel

def main():
    file_pre, path_pre, channel, max_mb, n_clusters, nist_clusters, top_n, n_background, od_thresh_factor = get_user_inputs()
    file_post, path_post, channel, max_mb, n_clusters, nist_clusters, top_n, n_background, od_thresh_factor = get_user_inputs()
    selected_pre, _, red_pre = load_image_channel(path_pre, file_pre, channel=2, detect_red=True)
    selected_post, _, red_post = load_image_channel(path_post, file_post, channel=2, detect_red=True)

    aligned_post, M, pts_ref, pts_mov = align_using_red_markers(selected_pre, selected_post, red_pre, red_post)
    show_alignment_overlay(selected_pre, aligned_post, pts_ref, pts_mov)


if __name__ == "__main__":
    main()