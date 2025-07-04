import numpy as np
import cv2 as cv

# Function to remove black slices (all 0s) along a particular axis
# Note: This function is designed for 3D volumes.
# It is currently not used in the slice-by-slice data_generator in create_dataset.py
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
        f = np.random.randint(3) # Randomly choose one of 3 blur kernels
        # Squeeze the image to (H, W) for cv.GaussianBlur, then re-expand to (H, W, 1)
        if f == 0:
            blurred_img = cv.GaussianBlur(x_img.squeeze(), (11, 11), 0)
        elif f == 1:
            blurred_img = cv.GaussianBlur(x_img.squeeze(), (15, 1), 0)
        else: # f == 2
            blurred_img = cv.GaussianBlur(x_img.squeeze(), (1, 15), 0)
        return np.expand_dims(blurred_img, axis=-1)

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
