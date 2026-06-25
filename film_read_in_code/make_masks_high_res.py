import numpy as np
from film_read_in_code.gui.dialogs import ask_text, ask_yesno, ask_open_file
from film_read_in_code.input_output.tiff_tile_od_CYM import compute_mask_od_from_tiff_tile


from film_read_in_code.utils.helpers import calibrate_od_values_only
from film_read_in_code.utils.helpers import poly_apply


from film_read_in_code.analysis.calibration import NISTfilter_cal

def analyze_nist_films(file, path, channel, info,refODs):

    nist_masks_scaled, cut_masks_scaled, scaled_img_film, scaled_img_nist, full_shape, scale_x, scale_y = info

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
            margin_factor=1.2,
        )
        # assume od_vals is a 1D array of OD values from that tile region
        nist_od_arrays.append(od_vals)
        if len(od_vals)> 0:
            nist_od_mean.append(float(od_vals.mean()))
            nist_od_median.append(float(np.median(od_vals)))
        else:
            nist_od_mean.append(0.0)
            nist_od_median.append(0.0)

    # Use your existing NISTfilter_cal implementation on the scaled image + NIST masks
    P, S, refOD = NISTfilter_cal(nist_masks_scaled, scaled_img_nist, refODs)

    return P, S, refOD, nist_od_arrays, nist_od_mean, nist_od_median

def analyze_cut_regions(file, path, channel, info, P, poly_apply, film_labels):

    nist_masks_scaled, cut_masks_scaled, scaled_img_film, scaled_img_nist, full_shape, scale_x, scale_y= info

    cut_od_arrays = []
    cut_od_mean = []
    cut_od_median = []

    for i, m in enumerate(cut_masks_scaled):
        od_vals = compute_mask_od_from_tiff_tile(
            path=path,
            file=file,
            channel=channel,
            mask_scaled=m,
            scaled_shape=scaled_img_film.shape,
            full_shape=full_shape,
            tile_id=f"cut_{film_labels[i]}",
            margin_factor=1.2,
        )

        cut_od_arrays.append(od_vals)
       
    cut_od_arrays = calibrate_od_values_only(cut_od_arrays, P, poly_apply)
    return cut_od_arrays, scaled_img_film, cut_masks_scaled, scale_x, scale_y, full_shape

def Load_Image_and_Run_Clustering(file, path, channel, info, refODs, film_labels):

    P, S, refOD, nist_od_arrays, nist_od_mean, nist_od_median = analyze_nist_films(file, path, channel, info,refODs)

    # =====================================================
    #  2) SECOND ROUND: VERTICAL CUT CLUSTERING & ODs
    # ====================================================

    cut_od_arrays, scaled_img_film, cut_masks_scaled, scale_x, scale_y, full_shape = analyze_cut_regions(file, path, channel, info, P, poly_apply, film_labels)
    # ------------------------------------------------------
    # RETURN: CALIBRATION + BOTH ROUNDS + REFERENCE SUBIMAGES
    # ------------------------------------------------------
    return {
        # NIST region results (first clustering round)
        "reference_ODs": refOD,                           # from NISTfilter_cal
        "P": P,
        "S": S,

        # Vertical cut results (second clustering round)
        "cut_region_image": scaled_img_film,                      # reference subimage
        "cut_masks_scaled": cut_masks_scaled,
        "cut_od_arrays": cut_od_arrays,
        

        # Geometry / scaling info
        "scale_x": scale_x,
        "scale_y": scale_y,
        "scaled_image": scaled_img_film,
        "full_shape": full_shape,
    }





