# analysis/histogram.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from gui.dialogs import ask_text, ask_yesno
from gui.selectors import EllipseCollector
from utils.helpers import scanner_to_OD
from utils.helpers import poly_apply

def BatchHistogram(OD_pre_vals, OD_post_vals):
    
    """
    Perform Histogram OD analysis using the same ROI mask for both pre and post-irradiation images.
    file_pair, path_pair, P_pair, S_pair, refOD_u_pair: np.array([pre, post])
    """
    bins = int(ask_text("Histogram Bins", "Number of histogram bins", "10000"))

    numrois = len(OD_pre_vals)

    hist_data = []

    def do_one_histogram(OD_pre_vals, OD_post_vals,index=None):
    # Make sure OD_pre_vals and OD_post_vals are NumPy arrays
        OD_pre_vals = np.asarray(OD_pre_vals)
        OD_post_vals = np.asarray(OD_post_vals)
        
        OD_diff_vals = OD_post_vals - OD_pre_vals

        calibrated = []
        histograms = []

        for film_OD_Calibrated in [OD_pre_vals, OD_post_vals, OD_diff_vals]:
            counts, bin_edges = np.histogram(film_OD_Calibrated, bins=bins, range=(-0.1, 1.1))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            calibrated.append(film_OD_Calibrated)
            histograms.append(counts)

        plot_graph(
            x=bin_centers,
            y=[histograms[0], histograms[1], histograms[2]],
            title="OD Histogram Comparison",
            xlabel="Optical Density (OD)",
            ylabel="Counts",
            legend=["Pre", "Post", "Post - Pre"],
            index=index
        )
        hist_data.append((bin_centers, histograms[2]))  # Store difference histogram


    for i in range(numrois):
        do_one_histogram(OD_pre_vals[i], OD_post_vals[i],index=i)

    return hist_data



def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def plot_graph(x, y, title="", xlabel="", ylabel="", legend=None,index=None):
    """
    Plots a histogram and overlays a Gaussian fit with mean and std in the legend.

    Parameters:
    - x: 1D array (bin centers)
    - y: 1D array (histogram counts)
    - title: Plot title
    - xlabel: X-axis label
    - ylabel: Y-axis label
    - legend: List with one label for histogram (optional)
    """
    plt.figure(figsize=(8, 5))

    is_multi_series = isinstance(y, (list, tuple))
    y_series_list = y if is_multi_series else [y]
    legend = legend if legend else [None] * len(y_series_list)

    for i, y_series in enumerate(y_series_list):
        norm_y = y_series / np.max(y_series)
        hist_label = legend[i] if legend[i] else f"Series {i+1}"
        plt.bar(x, norm_y, width=(x[1] - x[0]), alpha=0.4, edgecolor='black', label=hist_label)

        # Gaussian fit for each histogram
        try:
            popt, _ = curve_fit(gaussian, x,norm_y)
            a, mu, sigma = popt
            gauss_y = gaussian(x, *popt) / np.max(gaussian(x, *popt))  # Normalize for plotting
            gauss_label = f"Gaussian Fit {i+1}\nμ = {mu:.3f}, σ = {sigma:.3f}"
            plt.plot(x, gauss_y, linewidth=2, label=gauss_label)
        except RuntimeError:
            print(f"Gaussian fit failed on series {i+1}.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"/home/michaelgajda/cal_Lab_Work/hisograms/{title.replace(' ', '_')}_{index}.png")

