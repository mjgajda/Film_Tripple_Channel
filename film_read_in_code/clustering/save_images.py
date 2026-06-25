import os
from typing import List, Dict, Any

import cv2
import numpy as np


def save_paired_mask_square_crops(
    masks_pre: List[np.ndarray],
    masks_post: List[np.ndarray],
    img_pre: np.ndarray,
    img_post: np.ndarray,
    out_dir_pre: str = ".",
    out_dir_post: str = ".",
    base_name_pre: str = "pre",
    base_name_post: str = "post",
    max_val: float = 65535.0,
    expand_factor: float = 1.0,  # e.g. 1.1 to pad a bit
) -> List[Dict[str, Any]]:
    """
    For each index i:
      - Compute minimal enclosing circle for masks_pre[i] and masks_post[i] separately.
      - Use the midpoint of the two centers as the crop center.
      - Use the *smaller* of the two diameters (optionally scaled by expand_factor) as
        the square side length.
      - Crop that square from img_pre and img_post using the SAME bbox.
      - Save both crops as PNGs.

    Returns
    -------
    info_list : list of dict
        One dict per ROI with fields:
          - 'index'
          - 'x_min', 'y_min', 'x_max', 'y_max'
          - 'side'
          - 'file_pre', 'file_post'
          - 'pre_center', 'post_center', 'pre_radius', 'post_radius'
          - 'center_distance'
    """

    def _prepare_for_write(arr: np.ndarray) -> np.ndarray:
        """Convert image array to something OpenCV can write."""
        if arr.dtype in (np.float32, np.float64):
            arr_clipped = np.clip(arr, 0, max_val)
            arr_scaled = (arr_clipped / max_val) * 65535.0
            return arr_scaled.astype(np.uint16)
        elif arr.dtype == np.uint16 or arr.dtype == np.uint8:
            return arr
        else:
            arr_min = float(arr.min())
            arr_max = float(arr.max())
            if arr_max <= arr_min:
                return np.zeros_like(arr, dtype=np.uint8)
            norm = (arr - arr_min) / (arr_max - arr_min)
            return (norm * 255).astype(np.uint8)

    def _enclosing_circle(mask_bool: np.ndarray):
        """Return (cx, cy, radius) of minimal enclosing circle, or None if empty."""
        ys, xs = np.where(mask_bool)
        if ys.size == 0:
            return None
        pts = np.column_stack((xs, ys)).astype(np.float32)
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        if radius <= 0:
            return None
        return float(cx), float(cy), float(radius)

    def _compute_square_bbox_two_masks(
        m_pre_bool: np.ndarray,
        m_post_bool: np.ndarray,
        H: int,
        W: int,
    ):
        """
        Compute square bbox:
          - center = midpoint(pre_center, post_center)
          - side length = expand_factor * min(pre_diameter, post_diameter)
        """
        circle_pre = _enclosing_circle(m_pre_bool)
        circle_post = _enclosing_circle(m_post_bool)

        if circle_pre is None or circle_post is None:
            return None, None

        cx_pre, cy_pre, r_pre = circle_pre
        cx_post, cy_post, r_post = circle_post

        # Diameters
        d_pre = 2.0 * r_pre
        d_post = 2.0 * r_post

        # Use mid-center and smaller diameter
        cx = 0.5 * (cx_pre + cx_post)
        cy = 0.5 * (cy_pre + cy_post)
        side = int(np.ceil(min(d_pre, d_post) * expand_factor))

        side = max(1, min(side, H, W))
        half = side / 2.0

        x_min = int(round(cx - half))
        y_min = int(round(cy - half))
        x_max = x_min + side
        y_max = y_min + side

        # keep inside image
        if x_min < 0:
            x_min = 0
            x_max = side
        if y_min < 0:
            y_min = 0
            y_max = side
        if x_max > W:
            x_max = W
            x_min = W - side
        if y_max > H:
            y_max = H
            y_min = H - side

        # final safety clamp
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(W, x_max)
        y_max = min(H, y_max)

        if x_max <= x_min or y_max <= y_min:
            return None, None

        # enforce square (may shrink a bit)
        side = min(x_max - x_min, y_max - y_min)
        x_max = x_min + side
        y_max = y_min + side

        if side <= 0:
            return None, None

        # diagnostics
        info = {
            "pre_center": (cx_pre, cy_pre),
            "post_center": (cx_post, cy_post),
            "pre_radius": r_pre,
            "post_radius": r_post,
            "center_distance": float(np.hypot(cx_pre - cx_post, cy_pre - cy_post)),
        }
        return (x_min, y_min, x_max, y_max, side), info

    # --- sanity checks ---
    if len(masks_pre) != len(masks_post):
        raise ValueError("masks_pre and masks_post must have the same length.")

    if img_pre.ndim == 2:
        H_pre, W_pre = img_pre.shape
    else:
        H_pre, W_pre = img_pre.shape[:2]

    if img_post.ndim == 2:
        H_post, W_post = img_post.shape
    else:
        H_post, W_post = img_post.shape[:2]

    if (H_pre, W_pre) != (H_post, W_post):
        raise ValueError("img_pre and img_post must have the same spatial dimensions.")

    os.makedirs(out_dir_pre, exist_ok=True)
    os.makedirs(out_dir_post, exist_ok=True)

    info_list: List[Dict[str, Any]] = []

    for i, (m_pre, m_post) in enumerate(zip(masks_pre, masks_post), start=1):
        # Ensure 2D
        m_pre_2d = m_pre[..., 0] if m_pre.ndim > 2 else m_pre
        m_post_2d = m_post[..., 0] if m_post.ndim > 2 else m_post

        m_pre_bool = m_pre_2d.astype(bool)
        m_post_bool = m_post_2d.astype(bool)

        bbox, circ_info = _compute_square_bbox_two_masks(m_pre_bool, m_post_bool, H_pre, W_pre)
        if bbox is None:
            print(f"[WARN] ROI {i}: could not compute bbox; skipping.")
            continue

        x_min, y_min, x_max, y_max, side = bbox

        # print diagnostics so you can see what's going on
        print(
            f"ROI {i}: bbox=({x_min},{y_min})-({x_max},{y_max}), side={side}, "
            f"pre_center={circ_info['pre_center']}, post_center={circ_info['post_center']}, "
            f"pre_r={circ_info['pre_radius']:.2f}, post_r={circ_info['post_radius']:.2f}, "
            f"center_dist={circ_info['center_distance']:.2f}"
        )

        # Crop pre
        if img_pre.ndim == 2:
            crop_pre = img_pre[y_min:y_max, x_min:x_max]
        else:
            crop_pre = img_pre[y_min:y_max, x_min:x_max, :]

        crop_pre_to_save = _prepare_for_write(crop_pre)
        fname_pre = os.path.join(out_dir_pre, f"{base_name_pre}_roi{i:02d}.png")
        cv2.imwrite(fname_pre, crop_pre_to_save)

        # Crop post with same bbox
        if img_post.ndim == 2:
            crop_post = img_post[y_min:y_max, x_min:x_max]
        else:
            crop_post = img_post[y_min:y_max, x_min:x_max, :]

        crop_post_to_save = _prepare_for_write(crop_post)
        fname_post = os.path.join(out_dir_post, f"{base_name_post}_roi{i:02d}.png")
        cv2.imwrite(fname_post, crop_post_to_save)

        record = {
            "index": i,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "side": side,
            "file_pre": fname_pre,
            "file_post": fname_post,
        }
        record.update(circ_info)
        info_list.append(record)

    return info_list
