from film_read_in_code.main import get_user_inputs, run_clustering
from dose_statistics.main import main as dose_main
import os
from Constants import type, doseCurveJsonPath, film_labels
from analyze_Dose_CSVs.doseStatistics import plot_histograms_for_all_films

VALID_TYPES = {'EBT4', 'EBTXD'}

try:
    if type not in VALID_TYPES:
        raise ValueError(f"Invalid film type '{type}'. Must be one of: {VALID_TYPES}")
except ValueError as e:
    raise ValueError(f"Film type validation error: {e}")
except Exception as e:
    raise RuntimeError(f"Failed to set film type: {e}")

try:
    if not os.path.exists(doseCurveJsonPath):
        raise FileNotFoundError(f"Calibration file not found: {doseCurveJsonPath}")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Calibration file error: {e}")
except Exception as e:
    raise RuntimeError(f"Failed to construct calibration file path: {e}")

try:
    if not all(isinstance(label, int) for label in film_labels):
        raise TypeError("All film labels must be integers")
except TypeError as e:
    raise TypeError(f"Invalid film labels: {e}")
except Exception as e:
    raise RuntimeError(f"Failed to set film labels: {e}")

def main():
    
    # Run clustering
    path_pre, parentDir, path_post, parentDir = run_clustering(film_labels)
    histSaveDir = parentDir
    # Run dose statistics
    csvSaveDir = dose_main(postParentDir=path_post, preParentDir=path_pre, parentDir=parentDir, doseCurveJson=doseCurveJsonPath, film_labels=film_labels)

    output_dir = os.path.join(parentDir, "dose_statistics_results", "histograms")
    plot_histograms_for_all_films(film_type=type, csv_dir=csvSaveDir, output_dir=output_dir, x_min=0, x_max=30, film_labels=film_labels)

if __name__ == "__main__":
    main()