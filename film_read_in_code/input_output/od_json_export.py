"""
Save mask-filtered OD to JSON with calibration + source provenance.

Layout written to out_dir (one pair per scan and channel):
    <stem>_<channel>_od.json   region summaries, NIST calibration (coefficients and the
                               fit/apply functions' source), source TIFF path/size/SHA-256,
                               geometry
    <stem>_<channel>_od.npz    per-pixel arrays referenced from the JSON:
                               cut_<i>_od       raw scanner OD inside the film mask
                               cut_<i>_od_cal   NIST-calibrated OD (NaN where invalid)
                               cut_<i>_rows/cols full-resolution pixel coordinates
                               nist_<i>_od      raw OD inside each NIST filter mask
"""

import datetime as _dt
import hashlib
import inspect
import json
import os

import numpy as np


def _to_jsonable(obj):
    """Recursively convert numpy / tuple / misc objects into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _file_info(file_path: str, hash_file: bool = True) -> dict:
    st = os.stat(file_path)
    info = {
        "path": os.path.abspath(file_path),
        "name": os.path.basename(file_path),
        "size_bytes": st.st_size,
        "modified": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(),
    }
    if hash_file:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()
    return info


def _summary(vals) -> dict:
    vals = np.asarray(vals, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def _func_ref(fn) -> dict:
    if fn is None:
        return {}
    ref = {"name": f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"}
    try:
        ref["source"] = inspect.getsource(fn)
    except (OSError, TypeError):
        pass
    return ref


def save_od_json(
    stats: dict,
    out_dir: str,
    calib_fn=None,          # e.g. NISTfilter_cal
    apply_fn=None,          # e.g. poly_apply
    hash_file: bool = True,
    tag: str | None = None, # optional suffix, e.g. label-set name
) -> str:
    """
    Write <stem>[_<tag>]_<channel>_od.json (+ .npz) from a Load_Image_and_Run_Clustering stats dict.
    Returns the JSON path.
    """
    os.makedirs(out_dir, exist_ok=True)

    src_path = os.path.join(stats["path"], stats["file"])
    stem = os.path.splitext(os.path.basename(stats["file"]))[0]
    if tag:
        stem = f"{stem}_{tag}"
    channel = str(stats["channel"])
    base = f"{stem}_{channel}_od"
    json_path = os.path.join(out_dir, base + ".json")
    npz_path = os.path.join(out_dir, base + ".npz")

    arrays = {}

    # ---- cut regions ----
    n_lab, n_raw, n_cal = (len(stats["cut_labels"]), len(stats["cut_od_raw"]),
                           len(stats["cut_od_arrays"]))
    if not n_lab == n_raw == n_cal:
        raise ValueError(f"Region count mismatch: {n_lab} labels, {n_raw} raw OD arrays, "
                         f"{n_cal} calibrated arrays")

    regions = []
    for i, label in enumerate(stats["cut_labels"]):
        od = np.asarray(stats["cut_od_raw"][i])
        cal = np.asarray(stats["cut_od_arrays"][i])
        od_key, cal_key = f"cut_{i}_od", f"cut_{i}_od_cal"
        arrays[od_key], arrays[cal_key] = od, cal
        values = {"file": os.path.basename(npz_path), "od_key": od_key, "calibrated_key": cal_key}
        if "cut_rows" in stats and "cut_cols" in stats:
            rows_key, cols_key = f"cut_{i}_rows", f"cut_{i}_cols"
            arrays[rows_key] = np.asarray(stats["cut_rows"][i], dtype=np.int32)
            arrays[cols_key] = np.asarray(stats["cut_cols"][i], dtype=np.int32)
            values.update(rows_key=rows_key, cols_key=cols_key)
        regions.append({
            "index": i,
            "label": _to_jsonable(label),
            "od": _summary(od),
            "calibrated": _summary(cal),
            "values": values,
        })

    # ---- NIST regions (calibration inputs) ----
    nist_regions = []
    for i, od in enumerate(stats.get("nist_od_arrays", [])):
        key = f"nist_{i}_od"
        arrays[key] = np.asarray(od)
        nist_regions.append({
            "index": i,
            "od": _summary(od),
            "values": {"file": os.path.basename(npz_path), "od_key": key},
        })

    record = {
        "schema_version": 1,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_file": _file_info(src_path, hash_file=hash_file),
        "channel": channel,
        "calibration": {
            "derived_from": os.path.basename(src_path),
            "fit_function": _func_ref(calib_fn),
            "apply_function": _func_ref(apply_fn),
            "coefficients_P": _to_jsonable(stats["P"]),
            "fit_stats_S": _to_jsonable(stats["S"]),
            "reference_ODs_input": _to_jsonable(stats.get("refODs_input")),
            "reference_ODs_measured": _to_jsonable(stats["reference_ODs"]),
        },
        "geometry": {
            "full_shape": _to_jsonable(stats["full_shape"]),
            "scaled_shape": _to_jsonable(np.shape(stats["scaled_image"])),
            "scale_x": _to_jsonable(stats["scale_x"]),
            "scale_y": _to_jsonable(stats["scale_y"]),
            "margin_factor": stats.get("margin_factor"),
        },
        "regions": regions,
        "nist_regions": nist_regions,
    }

    np.savez_compressed(npz_path, **arrays)
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[INFO] Saved OD JSON: {json_path}")
    return json_path


def load_od_json(json_path: str) -> tuple[dict, dict]:
    """Load the JSON record and its referenced arrays. Returns (record, arrays)."""
    with open(json_path) as f:
        record = json.load(f)
    npz_path = os.path.splitext(json_path)[0] + ".npz"
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    return record, arrays
