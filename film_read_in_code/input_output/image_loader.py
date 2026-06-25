# io/image_loader.py
import os
import numpy as np
import cv2

import os
from typing import Optional, Tuple


def load_image_channel(
    path: str,
    file: str,
    channel: int,
    max_mb: Optional[float] = None,
    detect_red: bool = False,
) -> tuple[
    np.ndarray,          # selected_full
    np.ndarray,          # scaled
    Optional[np.ndarray],# red_mask_scaled (bool) or None
    float,               # scale_x
    float,               # scale_y
    Tuple[int, int],     # full_shape (H_full, W_full)
]:
    """
    Load image using OpenCV, extract a channel as float64, downscale based on
    max_mb, and optionally detect red markers.

    Returns
    -------
    selected_full : 2D float64
        Full-resolution selected channel.
    scaled : 2D float64
        Downscaled selected channel (possibly same size as full if no scaling).
    red_mask_scaled : 2D bool or None
        Red-marker mask in scaled coordinates (None if detect_red=False).
    scale_x : float
        scaled_width / full_width
    scale_y : float
        scaled_height / full_height
    full_shape : (H_full, W_full)
        Shape of the full-resolution selected channel.
    """
    img_path = os.path.join(path, file)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {img_path}")

    # Handle grayscale vs color
    if img.ndim == 2:
        selected_full = img.astype(np.float64)
        color_img = None  # no color info
    else:
        # OpenCV loads as BGR; channel index is in this BGR space
        selected_full = img[..., channel].astype(np.float64)
        color_img = img

    H_full, W_full = selected_full.shape

    # --- Optional red-marker detection on FULL resolution ---
    red_mask_full = None
    if detect_red and color_img is not None:
        # Simple heuristic: strong red, weak G/B
        b = color_img[..., 0].astype(np.float32)
        g = color_img[..., 1].astype(np.float32)
        r = color_img[..., 2].astype(np.float32)

        # Tune thresholds as needed
        red_mask_full = (
            (r > 100) &               # bright enough
            (r > g + 20) &
            (r > b + 20)
        )

    # --- Downscale selected channel (and red mask) as needed ---
    scaled = selected_full
    red_mask_scaled = red_mask_full

    if max_mb is not None:
        total_mb = H_full * W_full * selected_full.itemsize / (1024 * 1024)
        if total_mb > max_mb:
            scale = (max_mb / total_mb) ** 0.5
            new_W = max(int(W_full * scale), 1)
            new_H = max(int(H_full * scale), 1)

            # downscale selected channel
            scaled = cv2.resize(
                selected_full,
                (new_W, new_H),
                interpolation=cv2.INTER_AREA,
            )

            # downscale red mask to match scaled image
            if red_mask_full is not None:
                red_mask_scaled = cv2.resize(
                    red_mask_full.astype(np.uint8),
                    (new_W, new_H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

    H_scaled, W_scaled = scaled.shape
    scale_x = W_scaled / W_full
    scale_y = H_scaled / H_full

    return selected_full, scaled, red_mask_scaled, scale_x, scale_y, (H_full, W_full)


def upscale_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Upscale a binary mask to match the target image shape.
    
    Parameters:
    - mask: 2D boolean array from downscaled selection
    - target_shape: Shape of full-resolution image (height, width)

    Returns:
    - Resized mask with shape matching full-res image
    """
    resized = cv2.resize(mask.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)
