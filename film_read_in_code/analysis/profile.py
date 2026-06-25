import numpy as np
import matplotlib.pyplot as plt
from gui.dialogs import ask_yesno
import os

def get_non_nan_bounds(img: np.ndarray):
    """
    Return (x_min, x_max, y_min, y_max) of the non-NaN region in a 2D array.
    """
    valid = ~np.isnan(img)
    if not np.any(valid):
        return None

    ys, xs = np.where(valid)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    return x_min, x_max, y_min, y_max

def Batch2Dprofile(OD_pre_maps, OD_post_maps):
    """
    Generate 2D dose profile plots (pre, post, diff), and zoom into the region
    where the OD maps have valid (non-NaN) values. Does NOT modify the data,
    only the view.
    """
    profile_data = []
    numerois = len(OD_post_maps)

    show_plots = ask_yesno("Show Plots", "Display plots interactively after saving?")

    for i in range(numerois):
        pre_map = OD_pre_maps[i]
        post_map = OD_post_maps[i]

        # Difference map (same size, same NaN pattern if maps were made the same way)
        diff_map = post_map - pre_map
        profile_data.append(diff_map)

        # Compute zoom bounds from non-NaN region (use post_map as reference)
        bounds = get_non_nan_bounds(post_map)
        if bounds is None:
            print(f"[Warning] ROI {i+1}: no valid OD values, skipping.")
            continue

        x_min, x_max, y_min, y_max = bounds

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        titles = [f"Pre (ROI {i+1})", f"Post (ROI {i+1})", f"Difference (Post - Pre)"]
        data = [pre_map, post_map, diff_map]
        cmaps = ["inferno", "inferno", "inferno"]
        vmins = [0, 0, -0.1]
        vmaxs = [1.1, 1.1, 1.1]

        for ax, img, title, cmap, vmin, vmax in zip(axs, data, titles, cmaps, vmins, vmaxs):
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='OD')

            # 👇 This is the “zoom” – no data change, only axis limits
            ax.set_xlim(x_min, x_max)
            # imshow uses image coords: y increases downward, so invert
            ax.set_ylim(y_max, y_min)

        plt.suptitle(f"2D Dose Profile Comparison — ROI {i + 1}", fontsize=14)

        save_dir = "/home/michaelgajda/cal_Lab_Work/cmaps"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"2D_profile_ROI_{i + 1}.png")
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] {save_path}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    return profile_data
