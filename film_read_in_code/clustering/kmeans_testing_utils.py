import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from typing import Optional

def show_kmeans_labels(label_map: np.ndarray, image: Optional[np.ndarray] = None, title: str = "K-means Clusters"):
    """
    Display K-means cluster labels as an image overlay or standalone.

    Parameters:
    - label_map: 2D array with integer cluster labels or -1
    - image: optional grayscale image to overlay labels on
    - title: title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if image is not None:
        ax.imshow(image, cmap='gray')
    unique_labels = np.unique(label_map[label_map >= 0])
    n_labels = unique_labels.size
    base_cmap = cm.get_cmap("tab10", n_labels)
    colors = [base_cmap(i) for i in range(n_labels)]
    cmap = mcolors.ListedColormap(colors)

    masked_labels = np.ma.masked_where(label_map < 0, label_map)
    im = ax.imshow(masked_labels, cmap=cmap, alpha=0.5 if image is not None else 1.0)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, ticks=range(n_labels))
    plt.show()

def test_cluster_labels_visual(
    scaled_img: np.ndarray,
    masks: list[np.ndarray],
    labels: np.ndarray,
    title: str = "Cluster Labels (from Masks)"
):
    """
    Visualize precomputed cluster masks and labels on top of an image.

    Parameters:
    - scaled_img: 2D grayscale image to overlay
    - masks: list of boolean masks (each mask corresponds to one cluster)
    - labels: array of cluster labels (used for color indexing / legend)
    - title: plot title
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import cm
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(scaled_img, cmap="gray")

    n_masks = len(masks)
    cmap = cm.get_cmap("tab20", n_masks)
    colors = [cmap(i) for i in range(n_masks)]
    cmap_list = mcolors.ListedColormap(colors)

    # Combine all masks into one label map
    label_map = np.full(scaled_img.shape, -1, dtype=int)
    for i, mask in enumerate(masks):
        label_map[mask] = i

    # Plot with alpha overlay
    masked_labels = np.ma.masked_where(label_map < 0, label_map)
    im = ax.imshow(masked_labels, cmap=cmap_list, alpha=0.5)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_masks))
    cbar.set_label("Cluster Index (Raster Sorted)")

    plt.show()



def show_mask_overlay(image: np.ndarray, mask: np.ndarray, title: str = "Mask Overlay", alpha: float = 0.4):
    """
    Overlay a binary mask on a grayscale image.

    Parameters:
    - image: grayscale 2D image
    - mask: 2D boolean or binary mask
    - title: title for the plot
    - alpha: transparency of the overlay
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    ax.imshow(mask, cmap='Reds', alpha=alpha)
    ax.set_title(title)
    plt.show()
