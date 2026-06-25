import os
import cv2
import numpy as np
from film_read_in_code.utils.helpers import scanner_to_OD
from pathlib import Path


def extract_channel(tile_bgr: np.ndarray, channel: str) -> np.ndarray:
    ch = channel.lower()
    scale = 65535.0 if tile_bgr.dtype == np.uint16 else 255.0

    if ch in ("r", "g", "b"):
        idx = {"b": 0, "g": 1, "r": 2}[ch]
        return np.clip(tile_bgr[..., idx], 0, scale).astype(np.float64)

    # Normalize for CMYK math
    b = np.clip(tile_bgr[..., 0], 0, scale).astype(np.float64) / scale
    g = np.clip(tile_bgr[..., 1], 0, scale).astype(np.float64) / scale
    r = np.clip(tile_bgr[..., 2], 0, scale).astype(np.float64) / scale

    k = 1.0 - np.maximum(np.maximum(r, g), b)
    denom = np.where(k < 1.0, 1.0 - k, 1.0)

    cmyk = {
        "c": (1.0 - r - k) / denom,
        "m": (1.0 - g - k) / denom,
        "y": (1.0 - b - k) / denom,
        "k": k,
    }
    return cmyk[ch] * scale


def compute_mask_od_from_tiff_tile(
    path: str,
    file: str,
    channel: str,                   # 'r','g','b','c','m','y','k'
    mask_scaled: np.ndarray,
    scaled_shape: tuple[int, int],
    full_shape: tuple[int, int],
    tile_id: str = "tile0",
    margin_factor: float = 1.2,
    temp_dir: str | None = None,
) -> np.ndarray:
    """
    cv2-only version of TIFF tile extraction.

    - Saves tile as full-color if the input TIFF is color.
    - Uses only the specified channel for OD computation.
    channel: one of 'r', 'g', 'b', 'c', 'm', 'y', 'k' (case-insensitive)
    """
    VALID = {"r", "g", "b", "c", "m", "y", "k"}
    if channel.lower() not in VALID:
        raise ValueError(f"channel must be one of {VALID}, got {channel!r}")

    # ----------------------------------------
    # 1) Build square bbox in scaled space
    # ----------------------------------------
    def square_bbox_around_mask(mask: np.ndarray, margin_factor_local: float = 1.2):
        ys, xs = np.where(mask)
        if ys.size == 0:
            raise ValueError("Mask is empty; cannot build square bbox.")

        y_med = float(np.median(ys))
        x_med = float(np.median(xs))

        span_y = ys.max() - ys.min() + 1
        span_x = xs.max() - xs.min() + 1
        side = int(max(span_x, span_y) * margin_factor_local)
        side = max(side, 1)

        half = side // 2
        y0 = int(round(y_med)) - half
        y1 = y0 + side
        x0 = int(round(x_med)) - half
        x1 = x0 + side

        Hs, Ws = mask.shape
        y0 = max(y0, 0)
        x0 = max(x0, 0)
        y1 = min(y1, Hs)
        x1 = min(x1, Ws)

        return y0, y1, x0, x1

    Hs, Ws = scaled_shape
    Hf, Wf = full_shape

    y0s, y1s, x0s, x1s = square_bbox_around_mask(
        mask_scaled, margin_factor_local=margin_factor
    )
    mask_crop_scaled = mask_scaled[y0s:y1s, x0s:x1s]

    # ----------------------------------------
    # 2) Map scaled bbox → full-res bbox
    # ----------------------------------------
    scale_y = Hf / Hs
    scale_x = Wf / Ws

    fy0 = int(np.floor(y0s * scale_y))
    fy1 = int(np.ceil(y1s * scale_y))
    fx0 = int(np.floor(x0s * scale_x))
    fx1 = int(np.ceil(x1s * scale_x))

    fy0 = max(fy0, 0)
    fx0 = max(fx0, 0)
    fy1 = min(fy1, Hf)
    fx1 = min(fx1, Wf)

    # ----------------------------------------
    # 3) Read TIFF using cv2 ONLY
    # ----------------------------------------
    tiff_path = os.path.join(path, file)
    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read TIFF: {tiff_path}")

    # Extract tile via slicing
    if img.ndim == 2:
        # Grayscale image → tile is 2D
        tile_full = img[fy0:fy1, fx0:fx1]
    else:
        # Color image → keep ALL channels for saving
        tile_full = img[fy0:fy1, fx0:fx1, :]

    # ----------------------------------------
    # 4) SAVE TILE in full color (if available)
    # ----------------------------------------
    # Your initial logic
    if temp_dir is None:
        temp_dir = path

    # Join the path with the new folder name
    save_folder = os.path.join(temp_dir, f"{file[:-4]}_tiles")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print(f"Files will be saved to: {save_folder}")
    base = os.path.splitext(os.path.basename(file))[0]
    out_tile_name = f"{base}_{tile_id}.tif"
    out_tile_path = os.path.join(save_folder, out_tile_name)

    # clip and cast; works for both grayscale (H,W) and color (H,W,C)
    tile_uint16 = np.clip(tile_full, 0, 65535).astype(np.uint16)
    cv2.imwrite(out_tile_path, tile_uint16)

    # ----------------------------------------
    # 5) Prepare data for OD (use in-memory tile_full, no need to reload)
    # ----------------------------------------
    if tile_full.ndim == 2:
        selected = tile_full.astype(np.float64)
    elif channel.lower() in ("c", "m", "y", "k"):
        # CMYK channels require conversion via extract_channel
        selected = extract_channel(tile_full, channel)
    else:
        # R, G, B — use original direct index
        idx = {"b": 0, "g": 1, "r": 2}[channel.lower()]
        selected = tile_full[..., idx].astype(np.float64)

    Ht, Wt = selected.shape

    # Resize mask to tile size
    mask_crop_full = cv2.resize(
        mask_crop_scaled.astype(np.uint8),
        (Wt, Ht),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    vals = selected[mask_crop_full]
    vals = vals[vals > 0]

    if vals.size == 0:
        return np.zeros(1)

    # ----------------------------------------
    # 6) Compute OD
    # ----------------------------------------
    od = scanner_to_OD(vals)
    return od