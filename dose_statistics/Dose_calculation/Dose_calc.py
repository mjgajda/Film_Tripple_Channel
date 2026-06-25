

from Dose_calculation.setup.Dose_setup_functions import get_clean_aligned_data, scanner_to_OD, dose_Calculation_from_OD_diff
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
from constants import path, plot_save_path

def to_um(pixel_value, data_length, scale=.004):
    offset = data_length / 2
    return f'{(pixel_value - offset) * scale:.0f}'

def gaussian(x, a, mu, sigma, c):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def analyze_dose_combined(dose_map, film_num, num_samples=5, window_px=50):
    rows, cols = dose_map.shape
    data_length = rows
    
    # 1. Setup Figure with two subplots: Map (Small) and Profile (Large)
    fig, (ax_map, ax_prof) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 4]})
    
    # Generate 5 colors
    x_positions = np.linspace(cols * 0.1, cols * 0.9, num_samples).astype(int)
    y_axis = np.arange(rows)

    # --- Miniature Dose Map (Left Panel) ---
    ax_map.imshow(dose_map, cmap='magma', aspect='auto')
    ax_map.set_title("Data Extraction Locations")
    
    print(f"--- Combined Multi-Location Results ---")
    profiles = []
    for i in range(num_samples):
        x_pos = x_positions[i]
        
        # Draw location on the map
        ax_map.axvline(x_pos, linestyle='--', lw=1.5, alpha=0.8)

        # 2. Derive averaged Y-profile
        x_start_idx =  x_pos 
        x_end_idx = cols 
        profile = dose_map[:, x_start_idx]
        print(f"Profile {i+1} at X={to_um(x_pos, cols)}: Max Dose = {np.max(profile):.2f} Gy")
        profiles.append(profile)
    data = np.mean(profiles, axis=0)
    # 3. Plot Raw Data (Regular Alpha, Solid)
    # We only label the first raw data to keep legend clean, or label by X
    ax_prof.plot(y_axis, data, color='black', lw=1, alpha=0.8, label=f'Averaged Data')

    # 4. Find and Fit Peaks
    peaks, _ = find_peaks(data, height=np.max(data)*0.3, distance=50)
    profile = data
    fwhms = []
    for j, peak in enumerate(peaks):
        window = 45
        start, end = max(0, peak-window), min(rows, peak+window)
        x_fit, y_fit = y_axis[start:end], profile[start:end]
        
        try:
            p0 = [profile[peak], peak, 10, np.min(profile)]
            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
            a, mu_fit, sigma, c = popt
            fwhm = 2.35482 * abs(sigma)
            
            # Plot Gaussian Fit (Faint, Dashed)
            # Using a slightly different style to distinguish fits from raw data
            x_high = np.linspace(start, end, 100)
            y_high = gaussian(x_high, *popt)
            ax_prof.plot(x_high, y_high,  lw=2, linestyle='--', alpha=0.4,
                            label=f'Stripe {j+1} FWHM: {fwhm * 3.969:.2f} µm')
            
            # Fill under the fit
            ax_prof.fill_between(x_high, y_high, c,  alpha=0.05)
            fwhms.append(fwhm * 3.969)
        except Exception:
            continue
    print(f"Mean FWHM: {np.mean(fwhms):.2f} µm ± {np.std(fwhms):.2f} µm")
    # --- Formatting Left Map ---
    ax_map.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: to_um(val, cols)))
    ax_map.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: to_um(val, rows)))
    
    # --- Formatting Right Profile ---
    ax_prof.set_title("Combined Profiles & Gaussian Fits", fontsize=14)
    ax_prof.set_xlabel("Vertical Position (mm)")
    ax_prof.set_ylabel("Dose (Gy)")
    ax_prof.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: to_um(val, rows)))
    
    # Tick management for profile
    scale = .004
    step_um = .75
    pixel_step = step_um / scale
    center_pixel = data_length / 2
    all_ticks = np.sort(np.concatenate([
        np.arange(center_pixel, 0, -pixel_step),
        np.arange(center_pixel + pixel_step, data_length, pixel_step)
    ]))
    ax_prof.set_xticks(all_ticks)

    # Legend outside to the right
    ax_prof.legend(loc='upper left',   ncol=1)
    ax_prof.grid(True, linestyle=':', alpha=0.6)

    ax_map.set_xticks(all_ticks)
    ax_map.set_yticks(all_ticks)

    plt.tight_layout()
    plt.savefig(f"{plot_save_path}/Film_{film_num}/Combined_Profile_and_Fits.png")
    return fwhms