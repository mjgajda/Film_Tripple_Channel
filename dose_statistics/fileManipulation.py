from pathlib import Path
import re

def extract_snapshots_from_directories(directories):
    """
    Given a list of directory paths, extract all files matching the pattern "*_cut_*.tif",
    sort them by the numeric value in the filename, and return a list of file paths.

    Parameters
    ----------
    directories : list of str
        List of directory paths to search for files.
    Returns
    -------
    list of str
        Sorted list of file paths matching the pattern.
    """
    all_files = []
    for temp_dir in directories:
        print(f"Processing directory: {temp_dir}")
        folder_path = Path(temp_dir)

        # 1. Find all potential matches
        extracted_files = list(folder_path.glob("*_cut_*.tif"))
        
        filtered_files = extracted_files

        # 4. Sort the filtered subset
        filtered_files.sort(key=lambda f: int(re.search(r'_cut_(\d+)', f.name).group(1)))
        
        print(f"Found {len(filtered_files)} files ")
        all_files.extend(str(file) for file in filtered_files)
    return all_files

def extract_NIST_files(directory_path):
    """
    Given a directory path, find all *_stats.npy files and return them
    organized by channel (r, g, b).
    
    Parameters
    ----------
    directory_path : str
        Path to the directory containing *_stats.npy files.
    
    Returns
    -------
    dict
        Dictionary with keys 'r', 'g', 'b' mapping to their stats file paths.
    """
    folder_path = Path(directory_path)
    stats_files = {}
    
    for channel in ['r', 'g', 'b']:
        # Match files ending in _{channel}_stats.npy
        matches = list(folder_path.glob(f"*_{channel}_stats.npy"))
        if len(matches) == 1:
            stats_files[channel] = str(matches[0])
        elif len(matches) == 0:
            print(f"Warning: No {channel}_stats.npy found in {directory_path}")
            stats_files[channel] = None
        else:
            print(f"Warning: Multiple {channel}_stats.npy found in {directory_path}, using first.")
            stats_files[channel] = str(matches[0])
    
    return stats_files

