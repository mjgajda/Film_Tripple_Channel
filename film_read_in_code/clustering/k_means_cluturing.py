import numpy as np
from film_read_in_code.gui.selectors import EllipseCollector
from film_read_in_code.utils.helpers import scanner_to_OD
from sklearn.cluster import KMeans
from film_read_in_code.input_output.image_loader import upscale_mask

def collect_background_mask(image: np.ndarray, n: int, title: str) -> np.ndarray:
    """Run EllipseCollector and return combined background mask."""
    collector = EllipseCollector(image, n, title)
    masks = collector.run()
    return np.logical_or.reduce(masks)

def compute_od_threshold(image: np.ndarray, background_mask: np.ndarray, factor: float = 1.0) -> float:
    """Compute OD threshold based on background region and scaling factor."""
    background_od = scanner_to_OD(image[background_mask])
    return background_od.mean() * factor

def filter_valid_coords(image: np.ndarray, od_thresh: float, background_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid coordinates and mask for filtering before KMeans."""
    h, w = image.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    od_flat = scanner_to_OD(image.ravel())
    
    exclude_mask =  (od_flat <= od_thresh)
    valid_mask = exclude_mask
    coords_valid = coords[valid_mask]

    return coords, coords_valid, valid_mask
def reorder_labels_by_top_left_pixel(
    coords: np.ndarray,
    coords_valid: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, int]
) -> np.ndarray:
    """
    Reorder cluster labels by their top-left pixel position (raster order).

    Assumes coords and coords_valid are in (y, x) order, matching shape.
    """
    import numpy as np

    # Reconstruct full-size label map
    label_map_flat = np.full(coords.shape[0], -1, dtype=int)
    label_map_flat[valid_mask] = labels
    label_map = label_map_flat.reshape(shape)

    cluster_top_left = []
    for cluster_id in np.unique(labels):
        coords_in_cluster = np.argwhere(label_map == cluster_id)
        if coords_in_cluster.size == 0:
            continue
        y_min, x_min = coords_in_cluster[:, 0].min(), coords_in_cluster[:, 1].min()
        cluster_top_left.append((cluster_id, y_min, x_min))

    # Sort by raster order: top to bottom, then left to right
    sorted_ids = [
        cid for cid, _, _ in sorted(cluster_top_left, key=lambda tup: (tup[1], tup[2]))
    ]

    # Remap cluster labels
    id_map = {old: new for new, old in enumerate(sorted_ids)}
    labels_remapped = np.array([id_map[label] for label in labels])

    return labels_remapped

def run_kmeans_on_coords(coords_valid: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """Run KMeans clustering on (x, y) coordinates."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coords_valid)

    # Get (y, x) centroids by flipping from (x, y)
    centroids = kmeans.cluster_centers_[:, [1, 0]]

    # Sort cluster IDs spatially: first by Y, then by X
    sorted_ids = sorted(range(n_clusters), key=lambda i: (centroids[i][0], centroids[i][1]))

    # Build mapping: old_label → new_spatial_order_label
    label_map = {old: new for new, old in enumerate(sorted_ids)}

    # Apply remapping
    labels_remapped = np.array([label_map[lbl] for lbl in labels])

    return labels_remapped
def sort_and_relabel_masks_by_median(masks: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Sorts 2D boolean masks by their median (center) position in raster order,
    rounded to the nearest 100 pixels, and updates their labels to match the new order.

    Parameters:
    - masks: list of 2D boolean arrays of equal shape

    Returns:
    - sorted_masks: list of masks reordered in raster order
    - label_map: 2D array (same shape as masks) with relabeled indices 0..N-1
    """
    import numpy as np

    indexed = []
    for i, mask in enumerate(masks):
        ys, xs = np.where(mask)
        if ys.size == 0:
            median_y, median_x = np.inf, np.inf
        else:
            median_y = round(np.median(ys) / 100) * 100
            median_x = round(np.median(xs) / 100) * 100
        indexed.append((i, median_y, median_x))

    # Sort by Y first (top to bottom), then X (left to right)
    sorted_indices = [i for i, _, _ in sorted(indexed, key=lambda t: (t[1], t[2]))]
    sorted_masks = [masks[i] for i in sorted_indices]

    # Build a relabeled map
    label_map = np.full_like(sorted_masks[0], -1, dtype=int)
    for new_label, mask in enumerate(sorted_masks):
        label_map[mask] = new_label

    return sorted_masks, label_map




def generate_mask_from_labels(
    coords: np.ndarray,
    coords_valid: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, ...],
    top_n: int = 2
) -> np.ndarray:
    """Build label map and return a mask for the top-N clusters (by Y position)."""
    label_map_flat = np.full(coords.shape[0], -1, dtype=int)
    label_map_flat[valid_mask] = labels
    label_map = label_map_flat.reshape(shape)

    # Get top-N cluster IDs by average Y
    cluster_y_means = []
    for cluster_id in np.unique(labels):
        cluster_y = coords_valid[labels == cluster_id][:, 1]
        cluster_y_means.append((cluster_id, cluster_y.mean()))

    top_ids = {cid for cid, _ in sorted(cluster_y_means, key=lambda x: x[1], reverse=True)[:top_n]}
    mask = np.isin(label_map, list(top_ids))

    return mask
import numpy as np

def generate_masks_from_top_labels_by_x(
    coords: np.ndarray,
    coords_valid: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, ...],
    top_n: int = 2
) -> list[np.ndarray]:
    """
    Build a label map and return a list of masks for the top-N clusters 
    with the lowest average X position (i.e., sorted ascending).

    Parameters:
    - coords: Full (y, x) coordinates of the image as flat array
    - coords_valid: Subset of coords used in clustering
    - valid_mask: Boolean mask of valid pixels in coords
    - labels: Labels assigned to coords_valid
    - shape: Shape of the target image (H, W)
    - top_n: Number of clusters to return masks for (lowest mean X)

    Returns:
    - List of 2D boolean masks, one per selected cluster
    """
    # Create label map for the full image
    label_map_flat = np.full(coords.shape[0], -1, dtype=int)
    label_map_flat[valid_mask] = labels
    label_map = label_map_flat.reshape(shape)

    # Compute mean X for each cluster
    cluster_x_means = []
    for cluster_id in np.unique(labels):
        cluster_x = coords_valid[labels == cluster_id][:, 0]  # X coordinate
        cluster_x_means.append((cluster_id, cluster_x.mean()))

    # Sort by mean X ascending, and take top-N cluster IDs
    top_ids = [cid for cid, _ in sorted(cluster_x_means, key=lambda x: x[1])[:top_n]]

    # Generate one mask per selected cluster
    masks = [(label_map == cluster_id) for cluster_id in top_ids]

    return masks

import numpy as np

def generate_masks_raster_order(
    coords: np.ndarray,
    coords_valid: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, ...]
) -> list[np.ndarray]:
    """
    Generate cluster masks sorted from top-left to bottom-right in raster scan order.

    Sorting is based on cluster centroid:
    - First by Y (top to bottom)
    - Then by X (left to right)

    Parameters:
    - coords: Full (y, x) coordinates of the image as flat array
    - coords_valid: Subset of coords used in clustering
    - valid_mask: Boolean mask of valid pixels in coords
    - labels: Labels assigned to coords_valid
    - shape: Shape of the target image (H, W)

    Returns:
    - List of 2D boolean masks (one per cluster), sorted spatially
    """
    # Create label map for the full image
    label_map_flat = np.full(coords.shape[0], -1, dtype=int)
    label_map_flat[valid_mask] = labels
    label_map = label_map_flat.reshape(shape)

    # Compute centroids (X, Y) for each cluster
    centroids = {}
    for cluster_id in np.unique(labels):
        cluster_coords = coords_valid[labels == cluster_id]
        cx = cluster_coords[:, 0].mean()
        cy = cluster_coords[:, 1].mean()
        centroids[cluster_id] = (cy, cx)  # Notice: Y first

    # Sort by Y ascending, then X ascending (top to bottom, left to right)
    sorted_ids = sorted(centroids, key=lambda cid: (centroids[cid][0], centroids[cid][1]))

    # Generate masks in sorted order
    masks = [(label_map == cid) for cid in sorted_ids]

    return masks

import numpy as np

def remap_labels_by_raster_order(
    coords_valid: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Remap cluster labels based on raster order (top-to-bottom, left-to-right)
    using the centroids of each cluster.

    Parameters:
    - coords_valid: (N, 2) array of (y, x) coordinates (NumPy-style)
    - labels: (N,) array of KMeans cluster labels

    Returns:
    - labels_remapped: (N,) array with spatially sorted label IDs
    """
    centroids = {}
    for cid in np.unique(labels):
        cluster_coords = coords_valid[labels == cid]
        y_mean = cluster_coords[:, 0].mean()  # Y first
        x_mean = cluster_coords[:, 1].mean()  # X second
        centroids[cid] = (y_mean, x_mean)

    # Sort by Y (row), then X (column)
    sorted_ids = sorted(centroids, key=lambda cid: (centroids[cid][0], centroids[cid][1]))

    # Map old cluster IDs to new spatially sorted IDs
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
    labels_remapped = np.array([id_map[label] for label in labels])

    return labels_remapped



def generate_mask_from_label(
    coords: np.ndarray,
    coords_valid: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, ...],
    target_label: int
) -> np.ndarray:
    """
    Generate a binary mask where pixels belonging to a specific cluster label are True.

    Parameters:
    - coords: Full set of (y, x) coordinate positions (from np.indices().reshape(-1, 2))
    - coords_valid: Subset of coords used in clustering (after OD/background filtering)
    - valid_mask: Boolean mask marking valid coords in the full array
    - labels: Cluster labels (1D array) corresponding to coords_valid
    - shape: Shape of the original image (height, width)
    - target_label: The cluster label to isolate

    Returns:
    - mask: Boolean 2D array where pixels belonging to the target cluster are True
    """

    # Fill label map: full array with -1, insert labels at valid positions
    label_map_flat = np.full(coords.shape[0], -1, dtype=int)
    label_map_flat[valid_mask] = labels

    # Reshape back to 2D spatial label map
    label_map = label_map_flat.reshape(shape)

    # Create binary mask where label matches the target
    mask = (label_map == target_label)

    return mask

def cluster_and_mask_od_filtered(
    scaled_img: np.ndarray,
    full_img: np.ndarray,
    n_clusters: int = 5,
    top_n: int = 2,
    use_background_filter: bool = True,
    title: str = "Select Background Region(s)",
    n_background: int = 1,
    od_thresh_factor: float = 1.0
) -> np.ndarray:
    """Pipeline that filters by OD and returns an upscaled mask from clustering."""

    if use_background_filter:
        background_mask = collect_background_mask(scaled_img, n_background, title)
        od_thresh = compute_od_threshold(scaled_img, background_mask, od_thresh_factor)
    else:
        background_mask = np.zeros_like(scaled_img, dtype=bool)
        od_thresh = -np.inf  # keep all pixels

    coords, coords_valid, valid_mask = filter_valid_coords(scaled_img, od_thresh, background_mask)
    labels = run_kmeans_on_coords(coords_valid, n_clusters)
    mask_scaled = generate_mask_from_labels(coords, coords_valid, valid_mask, labels, tuple(scaled_img.shape[:2]), top_n)
    mask_full = upscale_mask(mask_scaled, full_img.shape[:2])

    return mask_full
