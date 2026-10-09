"""
Dose from the mask-filtered OD JSONs written by film_read_in_code.

For each film label found in both the pre and post scans:
  1. read the R, G, B calibrated OD (and pixel coordinates) from every
     <name>_<c>_od.json / .npz in each scan's <stem>_tiles folder
     (<name> = <stem>, or <stem>_set<k> when film_labels is a list of lists)
  2. drop pixels that are NaN in any channel, so R, G, B stay aligned
  3. unexposed OD = per-channel median of the pre film (unexp_mode='median')
  4. multichannel + single-channel dose per pixel
  5. write dose_od_pair_<label>.csv, film_dose_summary.csv, dose_provenance.json
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from dose_statistics.analysis.histogram import BatchHistogram_OD
from dose_statistics.multichannel_dosimetry import dose_calculation, single_channel_dose
from dose_statistics.gui.dialogs import ask_yesno
from film_read_in_code.input_output.od_json_export import load_od_json

CHANNELS = ("r", "g", "b")


# =====================================================================
# Reading the OD JSONs
# =====================================================================
def find_od_json_sets(tiles_dir: str) -> list:
    """
    All OD JSON sets in a scan's tiles folder: [{'r': path, 'g': path, 'b': path}, ...],
    one per <name>_r_od.json that has matching _g_ and _b_ files.
    """
    sets = []
    for r_path in sorted(glob.glob(os.path.join(tiles_dir, "*_r_od.json"))):
        base = r_path[: -len("_r_od.json")]
        paths = {c: f"{base}_{c}_od.json" for c in CHANNELS}
        missing = [c for c, p in paths.items() if not os.path.exists(p)]
        if missing:
            print(f"[WARN] {os.path.basename(base)}: no {missing} OD JSON; set skipped.")
            continue
        sets.append(paths)
    if not sets:
        raise FileNotFoundError(
            f"No OD JSONs (*_r_od.json) in {tiles_dir}. This scan was read in before OD JSONs "
            f"existed; rerun the read-in for it (answer no to loading cached data)."
        )
    return sets


def _label_key(label):
    """JSON labels come back as int/str; keep them hashable and comparable."""
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, str) and label.isdigit():
        return int(label)
    return label if isinstance(label, str) else str(label)


def load_channel_regions(tiles_dir: str, value_key: str = "calibrated"):
    """
    Load every r/g/b OD JSON set of one scan and merge them by film label.

    tiles_dir : the scan's <stem>_tiles folder
    value_key : 'calibrated' (NIST-calibrated OD) or 'od' (raw scanner OD)

    Returns
    -------
    regions : {label: {'r': 1D, 'g': 1D, 'b': 1D, 'rows': 1D|None, 'cols': 1D|None}}
    records : [{'r': json record, 'g': ..., 'b': ...}, ...]   one per set
    """
    regions, records = {}, []
    for paths in find_od_json_sets(tiles_dir):
        set_records, set_regions = {}, {}
        for ch in CHANNELS:
            record, arrays = load_od_json(paths[ch])
            record["_json_path"] = os.path.abspath(paths[ch])
            set_records[ch] = record

            for reg in record["regions"]:
                vals = reg["values"]
                entry = set_regions.setdefault(_label_key(reg["label"]), {"rows": None, "cols": None})
                entry[ch] = arrays[vals[f"{value_key}_key"]]
                # same mask for every channel, so coordinates from any channel will do
                if entry["rows"] is None and "rows_key" in vals:
                    entry["rows"] = arrays[vals["rows_key"]]
                    entry["cols"] = arrays[vals["cols_key"]]

        for label, entry in set_regions.items():
            if label in regions:
                print(f"[WARN] Film {label} appears in more than one label set in {tiles_dir}; "
                      f"using the first.")
                continue
            regions[label] = entry
        records.append(set_records)

    return regions, records


def stack_channels(entry: dict, label, tag: str):
    """
    Stack r/g/b pixel arrays into (3, N), keeping only pixels finite in every
    channel. Returns (X, rows, cols); rows/cols are None if not stored.
    """
    arrs = [np.asarray(entry[c], dtype=np.float64).ravel() for c in CHANNELS]
    rows, cols = entry.get("rows"), entry.get("cols")

    lengths = [a.size for a in arrs]
    if len(set(lengths)) > 1:
        raise ValueError(f"{tag} film {label}: channel lengths differ "
                         f"{dict(zip(CHANNELS, lengths))}; the R, G, B pixels do not line up. "
                         f"Rerun the read-in for this scan.")

    X = np.stack(arrs, axis=0)
    keep = np.all(np.isfinite(X), axis=0)
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"[INFO] {tag} film {label}: {n_drop} of {keep.size} pixels NaN in a channel; dropped.")

    if rows is not None and len(rows) == X.shape[1]:
        return X[:, keep], np.asarray(rows)[keep], np.asarray(cols)[keep]
    return X[:, keep], None, None


def build_unexposed(X_post: np.ndarray, X_pre: np.ndarray, mode: str = "median"):
    """
    Unexposed (pre) OD paired with each post pixel.

    'median' : per-channel median of the pre film, used for every post pixel
    'pixel'  : pixel i of pre with pixel i of post, truncated to the shorter
               (only meaningful if the two scans are registered)
    """
    if mode == "median":
        base = np.median(X_pre, axis=1, keepdims=True)
        return X_post, np.broadcast_to(base, X_post.shape).copy()
    if mode == "pixel":
        n = min(X_post.shape[1], X_pre.shape[1])
        return X_post[:, :n], X_pre[:, :n]
    raise ValueError(f"unexp_mode must be 'median' or 'pixel', got {mode!r}")


def _fmt_label(label) -> str:
    return f"{label:03d}" if isinstance(label, (int, np.integer)) else str(label)


# =====================================================================
# Output
# =====================================================================
def save_dose_and_od_to_csv(
    i,
    OD_r, OD_g, OD_b,
    D,
    D_r=None, D_g=None, D_b=None,
    D_r_err=None, D_g_err=None, D_b_err=None,
    save_dir='',
    rows=None, cols=None,
):
    """
    Save per-pixel OD, single-channel doses and multichannel dose for film i.
    rows/cols = full-resolution scan coordinates of each pixel (if available).
    """
    os.makedirs(save_dir, exist_ok=True)
    n_pix = np.size(D)

    data = {"index": i, "pixel": np.arange(n_pix)}
    if rows is not None:
        data["row"] = np.asarray(rows)
        data["col"] = np.asarray(cols)
    data.update({
        "OD_r": np.ravel(OD_r),
        "OD_g": np.ravel(OD_g),
        "OD_b": np.ravel(OD_b),
        "D_multichannel": np.ravel(D),
    })
    for name, arr in (("D_r", D_r), ("D_g", D_g), ("D_b", D_b),
                      ("D_r_err", D_r_err), ("D_g_err", D_g_err), ("D_b_err", D_b_err)):
        if arr is not None:
            data[name] = np.ravel(arr)

    bad = {k: np.size(v) for k, v in data.items() if np.ndim(v) and np.size(v) != n_pix}
    if bad:
        raise ValueError(f"Length mismatch (expected {n_pix}): {bad}")

    csv_path = os.path.join(save_dir, f"dose_od_pair_{_fmt_label(i)}.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f"Saved film {_fmt_label(i)} → {csv_path}  ({n_pix} pixels)")
    return save_dir


def summarize_film_doses(label, OD_r, OD_g, OD_b, D, D_r, D_g, D_b,
                         D_r_err=None, D_g_err=None, D_b_err=None) -> dict:
    """One summary row per film: mean/std of OD and dose per channel."""
    def stats(arr):
        if arr is None:
            return np.nan, np.nan
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        return (float(a.mean()), float(a.std())) if a.size else (np.nan, np.nan)

    row = {"film": label, "n_pixels": int(np.isfinite(np.asarray(D, float)).sum())}
    for name, arr in (("OD_r", OD_r), ("OD_g", OD_g), ("OD_b", OD_b), ("D_multi", D),
                      ("D_r", D_r), ("D_g", D_g), ("D_b", D_b)):
        row[f"{name}_mean"], row[f"{name}_std"] = stats(arr)
    for name, arr in (("D_r_err", D_r_err), ("D_g_err", D_g_err), ("D_b_err", D_b_err)):
        row[f"{name}_mean"] = stats(arr)[0]
    return row


def save_film_dose_summary(summary_rows, save_dir):
    if not summary_rows:
        print("No summary rows to save.")
        return None
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    df = df.iloc[df["film"].astype(str).argsort(kind="stable")].reset_index(drop=True)
    out_path = os.path.join(save_dir, "film_dose_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved film dose summary → {out_path}  ({len(df)} films)")
    return out_path


def save_provenance(save_dir, records_pre, records_post, doseCurveJson, value_key, unexp_mode):
    """Which OD JSONs, scans and calibrations produced these doses."""
    def strip(rec):
        return {"od_json": rec.get("_json_path"),
                "source_file": rec.get("source_file"),
                "calibration": rec.get("calibration")}

    prov = {
        "dose_curve_json": os.path.abspath(doseCurveJson) if doseCurveJson else None,
        "od_values_used": value_key,
        "unexposed_mode": unexp_mode,
        "pre": [{ch: strip(rec[ch]) for ch in CHANNELS} for rec in records_pre],
        "post": [{ch: strip(rec[ch]) for ch in CHANNELS} for rec in records_post],
    }
    out_path = os.path.join(save_dir, "dose_provenance.json")
    with open(out_path, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"Saved provenance → {out_path}")


# =====================================================================
# Main
# =====================================================================
def main(postParentDir=None, preParentDir=None, parentDir=None, doseCurveJson='',
         film_labels=None, value_key="calibrated", unexp_mode="median"):
    """
    value_key  : 'calibrated' (NIST-calibrated OD) or 'od' (raw scanner OD)
    unexp_mode : 'median' or 'pixel' — how the pre film's OD is paired with the post pixels
    Returns the results folder.
    """
    # pre/postParentDir = each scan's <stem>_tiles folder (from run_clustering)
    saveDirectory = os.path.join(parentDir, "dose_statistics_results")
    os.makedirs(saveDirectory, exist_ok=True)

    pre_regions, records_pre = load_channel_regions(preParentDir, value_key)
    post_regions, records_post = load_channel_regions(postParentDir, value_key)

    labels = [_label_key(l) for l in film_labels] if film_labels else \
        sorted(set(pre_regions) & set(post_regions), key=str)
    missing = [l for l in labels if l not in pre_regions or l not in post_regions]
    if missing:
        print(f"[WARN] Skipping films not in both pre and post OD JSONs: {missing}")
    labels = [l for l in labels if l not in missing]

    overwrite = None
    summary_rows = []

    for label in labels:
        csv_path = os.path.join(saveDirectory, f"dose_od_pair_{_fmt_label(label)}.csv")
        if os.path.exists(csv_path):
            if overwrite is None:
                overwrite = ask_yesno("Files exist", "Dose CSVs already exist. Overwrite them?",
                                      default_yes=True)
            if not overwrite:
                print(f"[INFO] Skipping film {_fmt_label(label)} (CSV exists)")
                continue

        X_pre, _, _ = stack_channels(pre_regions[label], label, "pre")
        X_post, rows, cols = stack_channels(post_regions[label], label, "post")
        X, X_unexp = build_unexposed(X_post, X_pre, unexp_mode)
        if rows is not None:
            rows, cols = rows[:X.shape[1]], cols[:X.shape[1]]

        for k, ch in enumerate(CHANNELS):
            # the real pre-film distribution, not the single median value used for dose
            pre_hist = X_unexp[k] if unexp_mode == "pixel" else X_pre[k]
            BatchHistogram_OD(pre_hist, X[k], label, ch, saveDirectory)

        # X, X_unexp: (3, N) pixel lists; the dose maths works per pixel along axis 0
        OD_r, OD_g, OD_b = X

        D, Delta, alpha, beta, n_iter, cov = dose_calculation(
            calibration_json=doseCurveJson, X=X, X_unexp=X_unexp,
        )
        print(f"Film {_fmt_label(label)} | multichannel — mean dose: {np.nanmean(D):.4f}  "
              f"std: {np.nanstd(D):.4f}  iters: {n_iter}")

        D_r, D_r_err = single_channel_dose(OD_r, channel='R', json_path=doseCurveJson)
        D_g, D_g_err = single_channel_dose(OD_g, channel='G', json_path=doseCurveJson)
        D_b, D_b_err = single_channel_dose(OD_b, channel='B', json_path=doseCurveJson)

        save_dose_and_od_to_csv(
            i=label,
            OD_r=OD_r, OD_g=OD_g, OD_b=OD_b,
            D=D,
            D_r=D_r, D_g=D_g, D_b=D_b,
            D_r_err=D_r_err, D_g_err=D_g_err, D_b_err=D_b_err,
            save_dir=saveDirectory,
            rows=rows, cols=cols,
        )
        summary_rows.append(summarize_film_doses(label, OD_r, OD_g, OD_b, D, D_r, D_g, D_b,
                                                 D_r_err, D_g_err, D_b_err))

    save_film_dose_summary(summary_rows, saveDirectory)
    save_provenance(saveDirectory, records_pre, records_post, doseCurveJson, value_key, unexp_mode)
    return saveDirectory


if __name__ == "__main__":
    main()
