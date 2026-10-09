import numpy as np
from film_read_in_code.input_output.tiff_tile_od_CYM import compute_mask_od_from_tiff_tile

from film_read_in_code.utils.helpers import poly_apply

from film_read_in_code.analysis.calibration import NISTfilter_cal

MARGIN_FACTOR = 1.2


def analyze_nist_films(file, path, channel, info, refODs):

    nist_masks_scaled, scaled_img_nist, full_shape = info

    # -------------------- NIST MASK OD VIA TIFF TILES --------------------
    nist_od_arrays = []
    nist_od_mean = []
    nist_od_median = []

    for i, m in enumerate(nist_masks_scaled):
        od_vals = compute_mask_od_from_tiff_tile(
            path=path,
            file=file,
            channel=channel,
            mask_scaled=m,
            scaled_shape=scaled_img_nist.shape,
            full_shape=full_shape,
            tile_id=f"nist_{i}",
            margin_factor=MARGIN_FACTOR,
        )
        nist_od_arrays.append(od_vals)
        if np.isfinite(od_vals).any():
            nist_od_mean.append(float(np.nanmean(od_vals)))
            nist_od_median.append(float(np.nanmedian(od_vals)))
        else:
            nist_od_mean.append(0.0)
            nist_od_median.append(0.0)

    # Calibration fit on the scaled image + NIST masks
    P, S, refOD = NISTfilter_cal(nist_masks_scaled, scaled_img_nist, refODs)

    return P, S, refOD, nist_od_arrays, nist_od_mean, nist_od_median


def calibrate_region(od_vals: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Apply the NIST calibration polynomial to one region's OD values.
    NaN pixels stay NaN and every pixel keeps its position, so the result
    lines up with the region's pixel coordinates and with the other channels.
    """
    od_vals = np.asarray(od_vals, dtype=np.float64).ravel()
    cal = np.full(od_vals.shape, np.nan)
    finite = np.isfinite(od_vals)
    if finite.any():
        cal[finite], _ = poly_apply(P, od_vals[finite])
    return cal


def analyze_cut_regions(file, path, channel, info, P, film_labels):

    cut_masks_scaled, scaled_img_film, full_shape, scale_x, scale_y = info

    if len(cut_masks_scaled) != len(film_labels):
        print(f"[WARN] {len(cut_masks_scaled)} film masks but {len(film_labels)} film labels; "
              f"using the first {min(len(cut_masks_scaled), len(film_labels))}.")

    cut_labels, cut_od_raw, cut_od_cal, cut_rows, cut_cols = [], [], [], [], []

    for m, label in zip(cut_masks_scaled, film_labels):
        od_vals, rows, cols = compute_mask_od_from_tiff_tile(
            path=path,
            file=file,
            channel=channel,
            mask_scaled=m,
            scaled_shape=scaled_img_film.shape,
            full_shape=full_shape,
            tile_id=f"cut_{label}",
            margin_factor=MARGIN_FACTOR,
            return_coords=True,
        )
        cal = calibrate_region(od_vals, P)
        if np.isfinite(cal).any() and np.nanmean(cal) <= 0.09:
            print(f"[INFO] Film {label} ({channel}): mean calibrated OD {np.nanmean(cal):.3f} "
                  f"(unexposed or background); kept.")

        cut_labels.append(label)
        cut_od_raw.append(od_vals)
        cut_od_cal.append(cal)
        cut_rows.append(rows)
        cut_cols.append(cols)

    return (cut_labels, cut_od_raw, cut_od_cal, cut_rows, cut_cols,
            scaled_img_film, cut_masks_scaled, scale_x, scale_y, full_shape)


def Load_Image_and_Run_Clustering(file, path, channel, nist_info, film_info, refODs, film_labels):
    P, S, refOD, nist_od_arrays, nist_od_mean, nist_od_median = analyze_nist_films(
        file, path, channel, nist_info, refODs
    )

    # =====================================================
    #  2) SECOND ROUND: VERTICAL CUT CLUSTERING & ODs
    # ====================================================
    (cut_labels, cut_od_raw, cut_od_cal, cut_rows, cut_cols,
     scaled_img_film, cut_masks_scaled, scale_x, scale_y, full_shape) = analyze_cut_regions(
        file, path, channel, film_info, P, film_labels
    )

    return {
        # Provenance
        "file": file,
        "path": path,
        "channel": channel,
        "margin_factor": MARGIN_FACTOR,

        # NIST calibration
        "refODs_input": refODs,
        "reference_ODs": refOD,                 # from NISTfilter_cal
        "P": P,
        "S": S,
        "nist_od_arrays": nist_od_arrays,
        "nist_od_mean": nist_od_mean,
        "nist_od_median": nist_od_median,

        # Film regions (one entry per film label, all pixel-aligned)
        "cut_labels": cut_labels,
        "cut_od_raw": cut_od_raw,               # mask-filtered scanner OD
        "cut_od_arrays": cut_od_cal,            # NIST-calibrated OD
        "cut_rows": cut_rows,                   # full-res row of each value
        "cut_cols": cut_cols,                   # full-res col of each value
        "cut_region_image": scaled_img_film,
        "cut_masks_scaled": cut_masks_scaled,

        # Geometry / scaling info
        "scale_x": scale_x,
        "scale_y": scale_y,
        "scaled_image": scaled_img_film,
        "full_shape": full_shape,
    }
