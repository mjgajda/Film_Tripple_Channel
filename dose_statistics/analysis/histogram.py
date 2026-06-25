# analysis/histogram.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

def checkForSaveDirectory(sub_folder_name="histograms", saveDirectory="./Saves"):
    # 1. Define the base save directory
    base_path = Path(saveDirectory)
    
    # 2. Define the inner directory
    # This creates a path like: ./Saves/processed_data/
    target_path = base_path / sub_folder_name
    
    # 3. Create both (parents=True ensures the base saveDirectory is also created)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Files will be saved to: {target_path.resolve()}")
    
    return target_path

def BatchHistogram_OD(OD_pre, OD_post, iteration, chan, saveDirectory):
    savePath = checkForSaveDirectory(saveDirectory)
   
    numrois = int(0)
    hist_data = []

    def do_one_histogram():
        calibrated = []
        histograms = []
        bins = 100  # Number of bins for the histogram
        for film_OD_Calibrated in [OD_pre, OD_post]:
            counts, bin_edges = np.histogram(film_OD_Calibrated, bins=bins, range=(-0.1, 1.1))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            calibrated.append(film_OD_Calibrated)
            histograms.append(counts)

        

        # Subtract histograms: post - pre
        diff_OD = calibrated[1] - calibrated[0]
        counts, bin_edges = np.histogram(diff_OD, bins=bins, range=(-0.1, 1.1))
        hist_data.append((bin_centers, counts))

        plot_graph(
                        x=bin_centers,
                        y=[histograms[0], histograms[1], counts],
                        title="OD Histogram Comparison",
                        xlabel="Optical Density (OD)",
                        ylabel="Counts",
                        legend=["Pre", "Post", "Post - Pre"],
                        savePath=savePath,
                        labels=f"OD_{chan}_Film_{iteration}"
                        )

        
    do_one_histogram()

    return hist_data

def BatchHistogramDose (Dose, iteration, chan):

    savePath = checkForSaveDirectory()
    numrois = int(0)
    hist_data = []

    def do_one_histogram():
        calibrated = []
        histograms = []
        bins = 100 # Number of bins for the histogram
        
        counts, bin_edges = np.histogram(Dose, bins=bins, range=(-0.1, 40.1))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        calibrated.append(Dose)
        histograms.append(counts)

        # Subtract histograms: post - pre

        hist_data.append((bin_centers, counts))

        plot_graph(
                        x=bin_centers,
                        y=[histograms[0]],
                        title="OD Histogram Comparison",
                        xlabel="Optical Density (OD)",
                        ylabel="Counts",
                        legend=["Pre", "Post", "Post - Pre"],
                        savePath=savePath,
                        labels=f"Dose_{chan}_Film_{iteration}"
                        )

    do_one_histogram()

    return hist_data


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def plot_graph(x, y, title="", xlabel="", ylabel="", legend=None,savePath=Path(''), labels="a"):
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
            popt, _ = curve_fit(gaussian, x, y_series)
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
    plt.savefig(str(savePath / f"Histogram_{labels}.png"))

