import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
from pathlib import Path

CHANNEL_CFG = {
    'D_multichannel': {'label': 'Multichannel',  'color': '#333333', 'alpha': 0.35},
    'D_r':            {'label': 'Red channel',   'color': '#E24B4A', 'alpha': 0.35},
    'D_g':            {'label': 'Green channel', 'color': '#1D9E75', 'alpha': 0.35},
    'D_b':            {'label': 'Blue channel',  'color': '#378ADD', 'alpha': 0.35},
}

CHANNEL_KEY_MAP = {
    'multi': 'D_multichannel',
    'r':     'D_r',
    'g':     'D_g',
    'b':     'D_b',
}

CHANNEL_NAMES = {
    'multi': 'Multichannel',
    'r':     'Red',
    'g':     'Green',
    'b':     'Blue',
}


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def load_doses_from_csv(csv_path, channels=None):
    """
    Load dose columns from a single CSV file.

    Parameters
    ----------
    csv_path : str or Path
    channels : list of str or None — subset of ['multi','r','g','b'];
                                     loads all four if None
    """
    cols_to_load = (
        [CHANNEL_KEY_MAP[ch] for ch in channels]
        if channels else list(CHANNEL_CFG.keys())
    )

    data = {}
    df   = pd.read_csv(csv_path)

    for col in cols_to_load:
        if col in df.columns:
            vals      = df[col].dropna().values
            data[col] = vals[np.isfinite(vals)]
        else:
            print(f"Warning: column '{col}' not found in {Path(csv_path).name}")

    return data


def fit_gaussian(values, n_bins=100):
    """Fit a Gaussian to a histogram of values."""
    counts, bin_edges = np.histogram(values, bins=n_bins, density=False)
    bin_centres       = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    p0 = (counts.max(), values.mean(), values.std())

    try:
        params, _ = curve_fit(gaussian, bin_centres, counts, p0=p0, maxfev=5000)
        amp, mu, sigma = params
        sigma = abs(sigma)
    except RuntimeError:
        print("Warning: Gaussian fit did not converge, using mean/std estimates.")
        mu, sigma = values.mean(), values.std()
        amp       = counts.max()

    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
    y_fit = gaussian(x_fit, amp, mu, sigma)

    return x_fit, y_fit, mu, sigma


def extract_film_number(csv_path):
    """Extract trailing number from filename e.g. dose_od_pair_003.csv -> 3."""
    match = re.search(r'_(\d+)\.csv$', Path(csv_path).name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract film number from filename: {csv_path}")


def channel_title_label(channels):
    """Build a readable string of selected channels for use in titles."""
    if channels is None or set(channels) == set(CHANNEL_KEY_MAP.keys()):
        return 'All channels'
    return ', '.join(CHANNEL_NAMES[ch] for ch in channels)


def channel_save_label(channels):
    """Build a compact string of selected channels for use in filenames."""
    if channels is None or set(channels) == set(CHANNEL_KEY_MAP.keys()):
        return 'all'
    return '_'.join(channels)