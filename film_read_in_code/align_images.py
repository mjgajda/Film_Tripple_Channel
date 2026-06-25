import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.stats import norm
from typing import Optional, List, Tuple
from read_images import load_image_channel
from read_images import detect_red_mask
from read_images import get_marker_centers

def build_circle_mask(shape, radius_factor=0.4):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return ((xx - w/2)**2 + (yy - h/2)**2) <= (radius_factor * min(h, w))**2

def plot_analysis(pre, post, mask, title="Alignment Analysis"):
    pre_v, post_v = pre[mask], post[mask]
    diff = post_v - pre_v
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1D Histogram
    axes[0].hist(diff, bins=80, density=True, color='skyblue', alpha=0.7)
    mu, std = np.mean(diff), np.std(diff)
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    axes[0].plot(x, norm.pdf(x, mu, std), 'r', lw=2, label=f'μ={mu:.2f}\nσ={std:.2f}')
    axes[0].set_title("1D Difference Histogram")
    axes[0].legend()

    # 2D Joint Histogram
    h2d = axes[1].hist2d(pre_v, post_v, bins=80, cmap='hot', cmin=1)
    axes[1].plot([pre_v.min(), pre_v.max()], [pre_v.min(), pre_v.max()], 'c--', alpha=0.5)
    axes[1].set_title("2D Joint Intensity (Pre vs Aligned)")
    plt.colorbar(h2d[3], ax=axes[1])
    plt.show()

def run_full_alignment(path_pre, file_pre, path_post, file_post):
    # 1. Load Data
    pre_chan, pre_col = load_image_channel(path_pre, file_pre, 2)
    post_chan, post_col = load_image_channel(path_post, file_post, 2)
    
    # Resize post to pre
    if pre_chan.shape != post_chan.shape:
        post_col = cv2.resize(post_col, (pre_chan.shape[1], pre_chan.shape[0]))
        post_chan = cv2.resize(post_chan, (pre_chan.shape[1], pre_chan.shape[0]))

    # 2. Detect & Match Markers
    mask_pre = detect_red_mask(pre_col)
    mask_post = detect_red_mask(post_col)
    
    pts_pre = get_marker_centers(mask_pre, 4)
    pts_post = get_marker_centers(mask_post, 4)
    
    # Ensure equal number of points
    n = min(len(pts_pre), len(pts_post))
    pts_pre, pts_post = pts_pre[:n], pts_post[:n]
    pts_post_matched = match_points(pts_post, pts_pre)

    # 3. Calculate M_rigid (The Transform Matrix)
    M_rigid = estimate_rigid_transform(pts_post_matched, pts_pre)
    print(f"Transform Matrix:\n{M_rigid}")

    # 4. Apply Translation & Rotation (Fixes your current issue)
    h, w = pre_chan.shape
    aligned_post = cv2.warpAffine(post_chan, M_rigid, (w, h), flags=cv2.INTER_LINEAR)
    
    # 5. Composite and Error Check
    roi_mask = build_circle_mask(pre_chan.shape, 0.4)
    
    # Create spatial display
    composite = pre_chan.copy()
    composite[roi_mask] = aligned_post[roi_mask]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121); plt.imshow(composite, cmap='viridis'); plt.title("Aligned Composite")
    plt.subplot(122); plt.imshow(aligned_post - pre_chan, cmap='coolwarm'); plt.title("Spatial Diff")
    plt.show()

    # Run Histograms
    plot_analysis(pre_chan, aligned_post, roi_mask)
    return M_rigid