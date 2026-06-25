import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# This function defines what the LABEL looks like
def to_um_half(pixel_value, data_length, scale=.004):
    offset = data_length / 2
    return f'{(pixel_value - offset) * scale:.0f}'
def to_um(pixel_value, pos):
    return f'{(pixel_value-1500) * .004:.0f}'



def gaussian(x, a, x0, sigma, offset):
    """Gaussian model for marker profile fitting."""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset


def get_marker_metrology(od_diff, marker_mask, num_profiles=20):
    """Fits Gaussians to marker profiles to find Offset and Jitter (Sigma)."""
    h, w = od_diff.shape
    offsets = []
    sigmas = []

    # Find rows that contain marker pixels
    marker_rows = np.where(np.any(marker_mask, axis=1))[0]
    if len(marker_rows) < num_profiles: return 0.0, 1.0

    # Sample profiles across the markers
    row_indices = np.linspace(0, len(marker_rows)-1, num_profiles, dtype=int)
    
    for r_idx in marker_rows[row_indices]:
        profile = od_diff[r_idx, :]
        # Focus the fit only on the part of the profile where the mask is active
        cols = np.where(marker_mask[r_idx, :])[0]
        if len(cols) < 5: continue
        
        # Center the search window around the marker
        c_min, c_max = max(0, cols.min()-10), min(w, cols.max()+10)
        x = np.arange(c_min, c_max)
        y = profile[c_min:c_max]
        
        # Initial Guess [Amp, Center, Sigma, Offset]
        p0 = [np.max(y)-np.min(y), x[np.argmax(y)], 2.0, np.min(y)]
        try:
            popt, _ = curve_fit(gaussian, x, y, p0=p0, maxfev=800)
            sigmas.append(abs(popt[2]))
            offsets.append(popt[3])
        except:
            continue

    avg_offset = np.median(offsets) if offsets else 0.0
    avg_sigma = np.median(sigmas) if sigmas else 1.0
    return avg_offset, avg_sigma

def dose_2d_heatmap(dose_map, film_num, saveDirectory):
    """
    INPUT - dose_map- 2D numpy matrix of dose values
            film_num- Film number for saving the plot
    
    OUTPUT - Saved 2D histogram showing the center dose map of a film
    
    first sets up the axes and then plots the dose map
    
    """
    fig, ax = plt.subplots(figsize=(5, 4))
     # Get data length from the first data set
    data_length = len(dose_map[0]) 
    # Constants for conversion
    scale = .004         # 4 um per pixel
    step_um = .75        # Tick every 750um
    pixel_step = step_um / scale # 187.5 pixels
    center_pixel = data_length / 2
    im = ax.imshow(dose_map, cmap='inferno')
    ax.set_title(f"Corrected Dose Map (Gy)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Dose (Gy)')
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    left_ticks = np.arange(center_pixel, 0, -pixel_step)
    right_ticks = np.arange(center_pixel + pixel_step, data_length, pixel_step)
    all_ticks = np.sort(np.concatenate([left_ticks, right_ticks]))

    ax.set_xticks(all_ticks)

    # Use a lambda to pass the data_length into your to_um function
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda val, pos: to_um_half(val, data_length))
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda val, pos: to_um_half(val, data_length))
    )
    ax.set_yticks(all_ticks)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout()
    plt.savefig(f'{saveDirectory}/Film_{film_num}/Corrected_dose_map')