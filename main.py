from film_read_in_code.main import get_user_inputs, run_clustering
from dose_statistics.main import main as dose_main
import os
from Constants import typeFilm, doseCurveJsonPath, film_labels
from analyze_Dose_CSVs.doseStatistics import plot_histograms_for_all_films
from itertools import chain

VALID_TYPES = {'EBT4', 'EBTXD'}

try:
    if typeFilm not in VALID_TYPES:
        raise ValueError(f"Invalid film type '{typeFilm}'. Must be one of: {VALID_TYPES}")
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
def is_list_of_lists(var):
    return isinstance(var, list) and len(var) > 0 and all(isinstance(item, list) for item in var)


def main():
    
    # Run clustering
    if type( film_labels[0]) == list:
        path_pre, parentDir, path_post, parentDir = run_clustering( film_labels, multiLabels=True)
        new_labels = list(chain.from_iterable( film_labels))
    else:
        path_pre, parentDir, path_post, parentDir = run_clustering( film_labels)
        new_labels = film_labels
    histSaveDir = parentDir
    # # Run dose statistics
    csvSaveDir = dose_main(postParentDir=path_post, preParentDir=path_pre, parentDir=parentDir, doseCurveJson=doseCurveJsonPath, film_labels=new_labels)

    output_dir = os.path.join(parentDir, "dose_statistics_results", "histograms")
    plot_histograms_for_all_films(film_type=typeFilm, csv_dir=csvSaveDir, output_dir=output_dir,  film_labels=new_labels)

if __name__ == "__main__":
    main()