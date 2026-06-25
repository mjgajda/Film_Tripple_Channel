from dose_statistics.Dose_calculation.setup.Dose_setup_functions import get_clean_aligned_data, scanner_to_OD, dose_Calculation_from_OD_diff
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm



def load_and_align_images(pre, post, M_rigid):
    """Load pre and post images from disk and align post to pre using a rigid transform."""
    img_pre  = cv2.imread(pre,  cv2.IMREAD_UNCHANGED)
    img_post = cv2.imread(post, cv2.IMREAD_UNCHANGED)
    h, w = img_pre.shape[:2]
    img_post_aligned = cv2.warpAffine(img_post, M_rigid, (w, h), flags=cv2.INTER_LINEAR)
    return img_pre, img_post_aligned
def readInRawODFromPicture(pre, post, channel, P_pre, P_post):
    img_pre  = cv2.imread(pre,  cv2.IMREAD_UNCHANGED)
    img_post = cv2.imread(post, cv2.IMREAD_UNCHANGED)

    return img_pre, img_post

def calculate_od(img_pre, img_post_aligned, P_pre, P_post,channel=0):
    """Compute the raw OD difference map using the blue channel and calibration polynomials."""
    pre_channel  = img_pre[..., channel].astype(np.float64)
    post_channel = img_post_aligned[..., channel].astype(np.float64)
    od_pre  = np.polyval(P_pre,  scanner_to_OD(pre_channel))
    od_post = np.polyval(P_post, scanner_to_OD(post_channel))

    return od_pre, od_post

def calculate_od_difference(img_pre, img_post_aligned, P_pre, P_post,channel=0):
    """Compute the raw OD difference map using the blue channel and calibration polynomials."""
    pre_channel  = img_pre[..., channel].astype(np.float64)
    post_channel = img_post_aligned[..., channel].astype(np.float64)
    od_pre  = np.polyval(P_pre,  scanner_to_OD(pre_channel))
    od_post = np.polyval(P_post, scanner_to_OD(post_channel))

    return np.mean(od_post) - np.mean(od_pre), np.std(od_pre) + np.std(od_post)





def compute_corrected_od(pre, post, M_rigid, P_pre, P_post, subtraction_offset, jitter_sigma):
    """
    Reload clean image data and apply local alignment correction.
    Uses a uniform filter sized to 2*sigma to smooth out jitter spikes in the pre OD.
    """
    from scipy.ndimage import uniform_filter

    pre_data, post_aligned = get_clean_aligned_data(pre, post, M_rigid)

    od_pre  = np.polyval(P_pre,  scanner_to_OD(pre_data))
    od_post = np.polyval(P_post, scanner_to_OD(post_aligned))

    # Shift pre OD by background offset then smooth over the jitter search window
    search_size = max(1, int(np.ceil(jitter_sigma * 2)))
    od_pre_corrected = uniform_filter(od_pre + subtraction_offset, size=search_size * 2 + 1)

    return od_pre_corrected, od_post


def extract_science_dose(od_pre, od_post, subtraction_offset, roi=(1000, 2000)):
    """
    Crop to the science ROI, compute the net OD difference, and convert to dose.
    ROI is applied equally to both axes.
    """
    r0, r1 = roi
    od_pre_sci  = od_pre [r0 :r1, r0:r1]
    od_post_sci = od_post[r0 :r1, r0:r1]
    # --- 1. Slice your region ---

    # --- 2. Flatten to 1D ---
    data = od_pre_sci.flatten()

    # --- 3. Fit Gaussian ---
    mu, std = norm.fit(data)
    print(f"Original mean: {mu:.4f}, std: {std:.4f}")

    # --- 4. Shift mean to 0.2 ---
    target_mean = 0.2
    shifted_data = data + (target_mean - mu)

    # --- 5. Verify new distribution ---
    new_mu, new_std = norm.fit(shifted_data)
    print(f"New mean: {new_mu:.4f}, std: {new_std:.4f}")

    # --- 6. Reshape back to 2D ---
    shifted_2d = shifted_data.reshape(od_pre_sci.shape)

    # --- 7. Plot histogram + Gaussian fit ---
    plt.figure()

    # Histogram
    plt.hist(shifted_data, bins=50,  alpha=0.6, label='Shifted Data')
    plt.hist(data, bins=50, alpha=0.6, label='Original Data')

    # Gaussian curve
    # x = np.linspace(shifted_data.min(), shifted_data.max(), 200)
    # pdf = norm.pdf(x, new_mu, new_std)
    # plt.plot(x, pdf, linewidth=2)

    plt.title("Pre OD Distribution Shifted to Target Mean")
    plt.xlabel("OD Value")
    plt.ylabel("Counts")

    plt.savefig("gaussian_fit_shifted.png")
    plt.close()

    od_diff = (od_post_sci - subtraction_offset) - shifted_2d
    return dose_Calculation_from_OD_diff(od_diff)


