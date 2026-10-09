import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import re
from pathlib import Path
from analyze_Dose_CSVs.utillity import CHANNEL_KEY_MAP, CHANNEL_CFG,  load_doses_from_csv,channel_title_label, channel_save_label, fit_gaussian,extract_film_number

def plot_film_histogram(csv_path, film_type=None, channels=None,
                        n_bins=100, output_dir=None,
                        filmLabel=None):        # <-- add these
    """
    Parameters
    ----------
    x_min : float or None  — lower dose bound for display (Gy)
    x_max : float or None  — upper dose bound for display (Gy)
    """
    if channels is not None:
        invalid = [ch for ch in channels if ch not in CHANNEL_KEY_MAP]
        if invalid:
            raise ValueError(
                f"Invalid channel(s): {invalid}. "
                f"Choose from {list(CHANNEL_KEY_MAP.keys())}"
            )

    csv_path   = Path(csv_path)
    type_label = f" — {film_type}"                     if film_type else ""
    ch_label   = f" — {channel_title_label(channels)}"
    title      = f"Film {filmLabel}{type_label}{ch_label}"
    save_name  = (
        f"dose_histogram_film_{filmLabel}"
        f"{'_' + str(film_type) if film_type else ''}"
        f"_{channel_save_label(channels)}.png"
    )

    data = load_doses_from_csv(csv_path, channels)

    fig, ax = plt.subplots(figsize=(10, 6))

    cols_to_plot = (
        [CHANNEL_KEY_MAP[ch] for ch in channels]
        if channels else list(CHANNEL_CFG.keys())
    )

    current_lower_bound = 10000000
    current_upper_bound = 0
    gaussian_fits = pd.DataFrame(columns=['channel', 'mu', 'sigma'])
    for col in cols_to_plot:
        if col not in data:
            continue

        cfg    = CHANNEL_CFG[col]
        values = data[col]

       # trim lower 5% and upper 5% instead of fixed x_min/x_max
        lower_bound = np.percentile(values, 2)
        upper_bound = np.percentile(values, 98)
        values = values[(values >= lower_bound) & (values <= upper_bound)]

        if len(values) == 0:
            print(f"Warning: no data remaining after percentile trim "
                f"for channel {col}")
            continue

        ax.hist(
            values,
            bins=n_bins,
            range=(lower_bound, upper_bound),
            color=cfg['color'],
            alpha=cfg['alpha'],
            label=f"{cfg['label']} (n={len(values):,})",
            density=False,
        )

        x_fit, y_fit, mu, sigma = fit_gaussian(values, n_bins=n_bins)

        ax.plot(
            x_fit, y_fit,
            color=cfg['color'],
            linewidth=2.0,
            linestyle='--',
            label=f"{cfg['label']} fit  μ={mu:.3f} Gy, σ={sigma:.3f} Gy",
        )

        gaussian_fits = pd.concat(
            [gaussian_fits, pd.DataFrame({'channel': [col], 'mu': [mu], 'sigma': [sigma]})],
            ignore_index=True
        )
    # set axis limits if specified
    if lower_bound < current_lower_bound:
        current_lower_bound = lower_bound
    if upper_bound > current_upper_bound:
        current_upper_bound = upper_bound   
    ax.set_xlim(current_lower_bound,
                current_upper_bound)

    ax.set_xlabel('Dose (Gy)', fontsize=13)
    ax.set_ylabel('Counts',    fontsize=13)
    ax.set_title(title,        fontsize=14, fontweight='bold')
    ax.legend(fontsize=9,      loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if output_dir:
        # make a subdirectory named after the parent folder of the CSVs
        csv_parent_name = csv_path.parent.name
        save_dir = Path(output_dir) / csv_parent_name
        os.makedirs(save_dir, exist_ok=True)
        out_path = save_dir / save_name
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        gaussian_fits_path = save_dir / f"gaussian_fits_film_{filmLabel}.csv"
        gaussian_fits.to_csv(gaussian_fits_path, index=False)
        print(f"Saved → {out_path}")



def plot_all_film_histograms(csv_dir, film_types=None, channels=None,
                             n_bins=100, output_dir=None,
                             x_min=None, x_max=None, film_label=None):
    csv_files = sorted(
        Path(csv_dir).glob('dose_od_pair_*.csv'),   # not the summary CSV
        key=lambda f: extract_film_number(f)
    )

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    # normalize film_types to one entry per file
    if film_types is None:
        film_types = [None] * len(csv_files)
    elif isinstance(film_types, str):
        film_types = [film_types] * len(csv_files)   # same type for every film
    elif len(film_types) != len(csv_files):
        raise ValueError(
            f"film_types has {len(film_types)} entries but found "
            f"{len(csv_files)} CSV files."
        )

    for idx, csv_path in enumerate(csv_files):
        # label comes straight from the filename — the one source of truth,
        # so it never drifts if film_labels isn't sorted ascending.
        label = extract_film_number(csv_path)
        film_type = film_types[idx]

        print(f"Processing {csv_path.name}  "
              f"(film={label}, film_type={film_type}, channels={channels})")
        plot_film_histogram(
            csv_path=csv_path,
            film_type=film_type,
            channels=channels,
            n_bins=n_bins,
            output_dir=output_dir,
            filmLabel=label,
        )
        

def plot_histograms_for_all_films(film_type=None, csv_dir='', output_dir='',
                                  x_min=None, x_max=None, film_labels=None):
    plot_all_film_histograms(
        csv_dir=csv_dir,
        film_types=film_type,
        channels=['multi', 'r', 'g', 'b'],
        n_bins=100,
        output_dir=output_dir,
        x_min=x_min,
        x_max=x_max,
        film_label=film_labels,
    )
