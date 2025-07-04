import numpy as np
import cv2 as cv

# Function to remove black slices (all 0s) along a particular axis
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

    else:
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
def flip(d1, d2, i):
    if i == 2:
        return d1, d2  
    else:
        return cv.flip(d1, i), cv.flip(d2, i)  

# Blurring the images to account for fetal movement artifacts
def blur(x_img, strength=1): # Renamed 'i' to 'strength' to indicate it controls if blur happens
    if strength == 0: # No blur
        return x_img
    else:
        f = np.random.randint(3)
        # Squeeze to (H, W), blur, then re-expand
        if f == 0:
            blurred_img = cv.GaussianBlur(x_img.squeeze(), (11, 11), 0)
        elif f == 1:
            blurred_img = cv.GaussianBlur(x_img.squeeze(), (15, 1), 0)
        else:
            blurred_img = cv.GaussianBlur(x_img.squeeze(), (1, 15), 0)
        return np.expand_dims(blurred_img, axis=-1)

# Edge detection to enhance segmentation labels for edge-aware loss function
def detect_edges(label):
    # Use the Canny edge detection
    edges = cv.Canny(label.astype(np.uint8), threshold1=100, threshold2=200)
    return edges
