from film_read_in_code.gui.dialogs import ask_text, ask_yesno, ask_open_file
import numpy as np
def getFile():
    try:
        file, path = ask_open_file("Select an image file")
        if not file or not path:
            raise ValueError("No file selected.")
        return file, path
    except Exception as e:
        raise RuntimeError(f"File selection failed: {e}")
def get_user_inputs():
    # File selection

    # Max memory
    while True:
        try:
            max_mb = float(ask_text("Downscale Limit", "Max memory (MB) for downscale", "5.0"))
            if max_mb <= 0:
                raise ValueError("Must be positive.")
            break
        except ValueError as e:
            print(f"[ERROR] Invalid max_mb input: {e}. Please try again.")

    # Number of clusters
    while True:
        try:
            n_clusters = int(ask_text("Clustering", "How many films are being analyzed?", "7"))
            if n_clusters <= 0:
                raise ValueError("Must be a positive integer.")
            break
        except ValueError as e:
            print(f"[ERROR] Invalid n_clusters input: {e}. Please try again.")

    nist_clusters = 7

    # Top N filters
    while True:
        try:
            top_n = int(ask_text("Clustering", "How many NIST reference filters are there?", "7"))
            if top_n <= 0:
                raise ValueError("Must be a positive integer.")
            break
        except ValueError as e:
            print(f"[ERROR] Invalid top_n input: {e}. Please try again.")

    # NIST OD values
    while True:
        try:
            ref_str = ask_text("NIST Calibration", "Enter NIST OD values", "0.3 0.4 0.5 0.6 0.7 0.9 1.0")
            refODs = np.array([float(s) for s in ref_str.split()])
            if len(refODs) == 0:
                raise ValueError("Must enter at least one OD value.")
            break
        except ValueError as e:
            print(f"[ERROR] Invalid NIST OD input please enter numbers with just spaces between them: {e}. Please try again.")

    # Number of background ellipses
    while True:
        try:
            n_background = int(ask_text("Background", "How many background ellipses?", "1"))
            if n_background <= 0:
                raise ValueError("Must be a positive integer.")
            break
        except ValueError as e:
            print(f"[ERROR] Invalid n_background input: {e}. Please try again.")

    # OD threshold factor
    while True:
        try:
            od_thresh_factor = float(ask_text("Background Filter", "OD threshold factor", ".7"))
            if not (0.0 < od_thresh_factor <= 1.0):
                raise ValueError("Should be between 0 and 1.")
            break
        except ValueError as e:
            print(f"[ERROR] Invalid od_thresh_factor input: {e}. Please try again.")

    return  max_mb, n_clusters, nist_clusters, top_n, n_background, od_thresh_factor, refODs