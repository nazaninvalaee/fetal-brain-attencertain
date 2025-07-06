import numpy as np
import cv2 as cv
from scipy.ndimage import map_coordinates, gaussian_filter

def reduce_2d(data1, data2, n):
    i = 0
    c = 0

    if n == 2:
        while True:
            x = np.reshape(data1[:, :, i], (-1))
            m1, m2 = max(x), min(x)

            if m1 == m2:
                if c == 0:
                    data1 = data1[:, :, 1:]
                    data2 = data2[:, :, 1:]
                else:
                    data1 = data1[:, :, :-1]
                    data2 = data2[:, :, :-1]
            elif c == 0:
                i = -1
                c = 1
            else:
                break

    elif n == 1:
        while True:
            x = np.reshape(data1[:, i, :], (-1))
            m1, m2 = max(x), min(x)

            if m1 == m2:
                if c == 0:
                    data1 = data1[:, 1:, :]
                    data2 = data2[:, 1:, :]
                else:
                    data1 = data1[:, :-1, :]
                    data2 = data2[:, :-1, :]
            elif c == 0:
                i = -1
                c = 1
            else:
                break

    else: # n == 0
        while True:
            x = np.reshape(data1[i, :, :], (-1))
            m1, m2 = max(x), min(x)

            if m1 == m2:
                if c == 0:
                    data1 = data1[1:, :, :]
                    data2 = data2[1:, :, :]
                else:
                    data1 = data1[:-1, :, :]
                    data2 = data2[:-1, :, :]
            elif c == 0:
                i = -1
                c = 1
            else:
                break

    return data1, data2

# Flipping the images for data augmentation
# MODIFIED to handle (H, W, 1) image input from preprocess_slice
def flip(d1_img, d2_label, flip_code):
    """
    Flips a 2D image slice and its corresponding label horizontally or vertically.

    Args:
        d1_img (np.ndarray): The 2D image slice, expected shape (H, W, 1).
        d2_label (np.ndarray): The 2D label slice, expected shape (H, W).
        flip_code (int): 0 for vertical flip, 1 for horizontal flip, -1 for both.

    Returns:
        tuple: (flipped_image, flipped_label)
    """
    # Squeeze the image to (H, W) for cv.flip, then re-expand to (H, W, 1)
    flipped_img = np.expand_dims(cv.flip(d1_img.squeeze(), flip_code), axis=-1)
    flipped_label = cv.flip(d2_label, flip_code)
    return flipped_img, flipped_label

# Blurring the images to account for fetal movement artifacts
# MODIFIED to handle (H, W, 1) image input from preprocess_slice
def blur(x_img, apply_blur=True):
    """
    Applies Gaussian blur to a 2D image slice.

    Args:
        x_img (np.ndarray): The 2D image slice, expected shape (H, W, 1).
        apply_blur (bool): If False, returns the image unchanged.

    Returns:
        np.ndarray: The blurred image slice, shape (H, W, 1).
    """
    if not apply_blur:
        return x_img
    else:
        # Randomly choose one of 3 blur kernels for slightly more distinct blur
        f = np.random.randint(3)
        # Squeeze the image to (H, W) for cv.GaussianBlur, then re-expand to (H, W, 1)
        if f == 0:
            blurred_img = cv.GaussianBlur(x_img.squeeze(), (5, 5), 0)
        elif f == 1:
            blurred_img = cv.GaussianBlur(x_img.squeeze(), (9, 1), 0)
        else: # f == 2
            blurred_img = cv.GaussianBlur(x_img.squeeze(), (1, 9), 0)
        return np.expand_dims(blurred_img, axis=-1)

# NEW: Advanced Intensity Variations
def random_brightness_contrast(img, brightness_factor_range=(0.7, 1.3), contrast_factor_range=(0.7, 1.3)):
    """
    Applies random brightness and contrast adjustments.
    Args:
        img (np.ndarray): The 2D image slice, expected shape (H, W, 1).
        brightness_factor_range (tuple): Range for brightness factor.
        contrast_factor_range (tuple): Range for contrast factor.
    Returns:
        np.ndarray: Adjusted image.
    """
    img_float = img.astype(np.float32)

    # Apply contrast
    contrast_factor = np.random.uniform(contrast_factor_range[0], contrast_factor_range[1])
    adjusted_img = img_float * contrast_factor
    
    # Apply brightness (additive, relative to mean for better preservation of black/white points)
    brightness_factor = np.random.uniform(brightness_factor_range[0], brightness_factor_range[1])
    adjusted_img = adjusted_img + brightness_factor * (img_float.mean() - adjusted_img) # This is a common way to apply brightness that's more robust

    # Clamp values to original data type range
    adjusted_img = np.clip(adjusted_img, np.min(img), np.max(img)).astype(img.dtype)
    return np.expand_dims(adjusted_img, axis=-1) # Ensure (H,W,1) output

def random_gamma_correction(img, gamma_range=(0.7, 1.5)):
    """
    Applies random gamma correction.
    Args:
        img (np.ndarray): The 2D image slice, expected shape (H, W, 1).
        gamma_range (tuple): Range for gamma value.
    Returns:
        np.ndarray: Gamma-corrected image.
    """
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    
    # Normalize to [0, 1] for gamma correction, then scale back to original range
    img_min = np.min(img).astype(np.float32)
    img_max = np.max(img).astype(np.float32)
    
    # Avoid division by zero if img_max == img_min (e.g., for all zero images)
    if img_max == img_min:
        return img # Return original if no intensity variation

    img_norm = (img.astype(np.float32) - img_min) / (img_max - img_min)
    gamma_corrected_img_norm = np.power(img_norm, gamma)
    gamma_corrected_img = gamma_corrected_img_norm * (img_max - img_min) + img_min
    
    return np.expand_dims(np.clip(gamma_corrected_img, img_min, img_max).astype(img.dtype), axis=-1)

# NEW: Random Affine Transformations (Translation, Rotation)
def random_affine_transform(img, label, max_translation_pixels=10, max_rotation_deg=10):
    """
    Applies random affine transformations (translation, rotation) to image and label.
    Args:
        img (np.ndarray): The 2D image slice, expected shape (H, W, 1).
        label (np.ndarray): The 2D label slice, expected shape (H, W).
        max_translation_pixels (int): Max pixels for translation in x/y.
        max_rotation_deg (float): Max degrees for rotation.
    Returns:
        tuple: (transformed_image, transformed_label)
    """
    rows, cols = img.shape[0], img.shape[1]
    
    # Random parameters
    tx = np.random.uniform(-max_translation_pixels, max_translation_pixels)
    ty = np.random.uniform(-max_translation_pixels, max_translation_pixels)
    angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)

    # Get rotation matrix around center
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    # Add translation to the rotation matrix
    M[0, 2] += tx
    M[1, 2] += ty

    transformed_img = cv.warpAffine(img.squeeze(), M, (cols, rows), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    transformed_label = cv.warpAffine(label, M, (cols, rows), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0) # Labels use INTER_NEAREST for preserving class IDs, background 0

    return np.expand_dims(transformed_img, axis=-1), transformed_label

# NEW: Elastic Deformation
def elastic_transform(image, label, alpha=34, sigma=4, random_state=None):
    """
    Elastic deformation of images as described in [Simard2003].
    This function adapted from https://gist.github.com/chsasank/4d8f68caf01f041a645329f21b5693d7
    Args:
        image (np.ndarray): The 2D image slice, expected shape (H, W, 1).
        label (np.ndarray): The 2D label slice, expected shape (H, W).
        alpha (float): Scaling factor for the displacement field (controls intensity of deformation).
        sigma (float): Standard deviation of the Gaussian filter (controls smoothness of deformation).
        random_state (np.random.RandomState): Random state for reproducibility.
    Returns:
        tuple: (transformed_image, transformed_label)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.squeeze().shape # (H, W)
    
    # Generate random displacement fields
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    # Apply transformation using map_coordinates
    transformed_image = map_coordinates(image.squeeze(), indices, order=1, mode='reflect').reshape(shape)
    transformed_label = map_coordinates(label, indices, order=0, mode='constant', cval=0).reshape(shape) # Label uses order 0 (nearest neighbor) to preserve class IDs, with 0 for out-of-bounds

    return np.expand_dims(transformed_image, axis=-1), transformed_label

# Edge detection to enhance segmentation labels for edge-aware loss function
# Note: This function is an independent utility and is not currently integrated
# into the model's training or direct output pipeline.
def detect_edges(label):
    """
    Detects edges in a 2D label mask using Canny edge detection.

    Args:
        label (np.ndarray): The 2D label mask (H, W), expected to be uint8.

    Returns:
        np.ndarray: A 2D numpy array with detected edges (binary mask).
    """
    # Use the Canny edge detection
    edges = cv.Canny(label.astype(np.uint8), threshold1=100, threshold2=200)
    return edges
