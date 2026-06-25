# analysis/profile.py
import numpy as np
import matplotlib.pyplot as plt
from dose_statistics.gui.dialogs import ask_text, ask_yesno
from dose_statistics.gui.selectors import RectSelectorOnce
from dose_statistics.utils.helpers import scanner_to_OD
from dose_statistics.utils.helpers import poly_apply
from pathlib import Path

def checkForSaveDirectory(sub_folder_name="Profiles", saveDirectory="./Saves"):
    # 1. Define the base save directory
    base_path = Path(saveDirectory)
    
    # 2. Define the inner directory
    # This creates a path like: ./Saves/processed_data/
    target_path = base_path / sub_folder_name
    
    # 3. Create both (parents=True ensures the base saveDirectory is also created)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Files will be saved to: {target_path.resolve()}")
    
    return target_path

def plot_2d_profile(profile, index, directory, label, measurement_type="OD"):
    plt.figure(figsize=(6,5))
    im = plt.imshow(profile, cmap="inferno")
    plt.title(f"2D Dose Profile {label} {index + 1} ({measurement_type})")
    plt.colorbar(im, label=measurement_type)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(str(directory / f"Profile_{label}"))


def plot_2d_profile_difference(profile_diff, index, directory, label):
    plt.figure(figsize=(6,5))
    im = plt.imshow(profile_diff, cmap="inferno", vmax=1.1, vmin=-.1)
    plt.title(f"2D Dose Profile Difference {index + 1} (Post - Pre)")
    plt.colorbar(im, label='OD Difference')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(str(directory / f"Profile_Difference_{label}"))

def Batch2Dprofile_OD(OD_pre, OD_post, index,channel,saveDirectory):
    directory = checkForSaveDirectory(saveDirectory)

    profile_data = []


    def do_one_rectangle(i):
    

        # Calibrate both to OD
        od_maps = [OD_pre, OD_post]

        # Difference = post - pre
        od_diff = od_maps[1] - od_maps[0]
        profile_data.append(od_diff)

        # Plot all 3 maps
        plot_2d_profile(od_maps[0], index,  directory, label=f"Pre_{index + 1}_{channel}")
        plot_2d_profile(od_maps[1], index, directory, label=f"Post_{index + 1}_{channel}")
        plot_2d_profile(od_diff,     index, directory, label=f"Difference_{index + 1}_{channel}")

    do_one_rectangle(0)


    return profile_data

def Batch2Dprofile_Dose(Dose, index, channel):
    directory = checkForSaveDirectory()
    profile_data = []

    def do_one_rectangle(i):
        # Calibrate both to Dose
        dose_maps = [Dose]

        # Plot the dose map
        plot_2d_profile(dose_maps[0], i, directory, label=f"Dose_{i + 1}_{channel}", measurement_type="Dose (Gy)")

    do_one_rectangle(0)

    return profile_data