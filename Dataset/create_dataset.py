import numpy as np
import os
import nibabel as nib
import time
from tqdm import tqdm
import gc # Import garbage collection for explicit memory management

from sklearn.model_selection import train_test_split
from Dataset.preprocessing import reduce_2d, flip, blur # Assuming these are compatible with 2D slices
from skimage.transform import resize
import tensorflow as tf # For tf.data.Dataset

# --- Helper function to preprocess a single slice ---
def preprocess_slice(img_slice_2d, label_slice_2d):
    """
    Applies resizing, normalization, and adds channel dimension to a single 2D slice.

    Args:
        img_slice_2d (np.ndarray): A 2D numpy array for the image slice.
        label_slice_2d (np.ndarray): A 2D numpy array for the label slice.

    Returns:
        tuple: (preprocessed_image, preprocessed_label)
    """
    # Ensure all images are the same shape (256x256)
    # preserve_range=True for image to maintain intensity scale for normalization later
    # order=0 for label (nearest-neighbor) to maintain integer class labels
    img_resized = resize(img_slice_2d, (256, 256), preserve_range=True, anti_aliasing=True)
    label_resized = resize(label_slice_2d, (256, 256), order=0, anti_aliasing=False, preserve_range=True)

    # Normalize input image to 0-1 range (standard for neural networks)
    # Check if max value is 0 to avoid division by zero
    max_val = np.max(img_resized)
    if max_val > 0:
        img_normalized = img_resized.astype(np.float32) / max_val # Using float32 for model input
    else:
        img_normalized = img_resized.astype(np.float32) # Keep as float even if all zeros

    # Add channel dimension (H, W, 1) for the image
    img_final = np.expand_dims(img_normalized, axis=-1)

    # Ensure label data type is correct for SparseCategoricalCrossentropy (uint8 for integer labels)
    label_final = label_resized.astype(np.uint8)

    return img_final, label_final

# --- This function now only handles file paths, not image data ---
def prepare_filepaths(path1, path2, n=40):
    """
    Prepares lists of (input_nii_path, output_nii_path) tuples.

    Args:
        path1 (str): Path to the folder containing 3D input MRI volumes (.nii.gz).
        path2 (str): Path to the folder containing 3D output segmentation masks (.nii.gz).
        n (int): Number of volumes to consider from the dataset.

    Returns:
        list: A list of (input_nii_path, output_nii_path) tuples for all selected volumes.
    """
    if not path1.endswith('/'):
        path1 += '/'
    if not path2.endswith('/'):
        path2 += '/'

    input_filenames = sorted([f for f in os.listdir(path1) if f.endswith('.nii.gz')])
    output_filenames = sorted([f for f in os.listdir(path2) if f.endswith('.nii.gz')])

    matched_filepaths = []
    # Assuming input and output filenames correspond after sorting (e.g., sub-001.nii.gz and sub-001_dseg.nii.gz)
    # You might want to add a more robust matching logic if filenames are very different.
    for i in range(min(n, len(input_filenames))): # Limit to 'n' volumes
        f1_name = input_filenames[i]
        f2_name = f1_name[:-10] + 'dseg' + f1_name[-7:] # Reconstruct output filename

        input_path = os.path.join(path1, f1_name)
        output_path = os.path.join(path2, f2_name)

        if os.path.exists(output_path): # Check if the corresponding output file exists
            matched_filepaths.append((input_path, output_path))
        else:
            print(f"Warning: Corresponding output file not found for {f1_name}. Skipping.")

    if not matched_filepaths:
        raise ValueError("No matching NIfTI files found in the specified paths.")

    print(f"Prepared {len(matched_filepaths)} volume pairs for processing.")
    return matched_filepaths


# --- Data Generator Function ---
def data_generator(filepaths_list, slices_per_volume=None):
    """
    A Python generator that yields preprocessed 2D slices from 3D NIfTI volumes.
    tqdm is *removed* from this generator because it runs in a TensorFlow graph context.

    Args:
        filepaths_list (list): A list of (input_nii_path, output_nii_path) tuples.
        slices_per_volume (int or None): If an integer, samples this many slices
                                         evenly from each volume along all three axes.
                                         If None, takes all slices. This can help
                                         further reduce memory if slicing every single
                                         slice is still too much.

    Yields:
        tuple: (preprocessed_image_slice, preprocessed_label_slice)
    """
    # IMPORTANT: Do NOT use tqdm directly within this generator function
    # when it's passed to tf.data.Dataset.from_generator().
    # TensorFlow runs this in a graph context, often with multiple workers,
    # which leads to jumbled tqdm output and potential errors.
    # The progress during training/validation will be shown by Keras's own progress bar.

    for img_path, label_path in filepaths_list: # Removed tqdm wrapper here
        try:
            # Load 3D volumes
            img_volume = nib.load(img_path).get_fdata()
            label_volume = nib.load(label_path).get_fdata()

            # Ensure data types are suitable to reduce memory footprint early
            img_volume = img_volume.astype(np.float32)
            label_volume = label_volume.astype(np.uint8)

            # Define axes and how to slice them
            axes = [0, 1, 2] # Saggital, Coronal, Axial

            for axis in axes:
                slice_shape = img_volume.shape[axis]

                # Determine slices to extract
                if slices_per_volume is not None and slices_per_volume > 0:
                    # Select slices evenly, ensuring at least one slice
                    selected_slice_indices = np.linspace(0, slice_shape - 1, slices_per_volume, dtype=int)
                else:
                    selected_slice_indices = range(slice_shape) # Take all slices

                for j in selected_slice_indices:
                    if axis == 0:
                        d1_slice, d2_slice = img_volume[j, :, :], label_volume[j, :, :]
                    elif axis == 1:
                        d1_slice, d2_slice = img_volume[:, j, :], label_volume[:, j, :]
                    else: # axis == 2
                        d1_slice, d2_slice = img_volume[:, :, j], label_volume[:, :, j]

                    # Preprocess the 2D slice
                    preprocessed_img, preprocessed_label = preprocess_slice(d1_slice, d2_slice)

                    # --- Apply optional augmentations (flip, blur) here if desired ---
                    # Ensure your 'flip' and 'blur' functions from Dataset.preprocessing
                    # are designed to work on (H, W, 1) image and (H, W) label.
                    # This should be done on the preprocessed slices before yielding.
                    # For example:
                    # if np.random.rand() < 0.5: # 50% chance to flip
                    #     preprocessed_img, preprocessed_label = flip(preprocessed_img, preprocessed_label)
                    # if np.random.rand() < 0.3: # 30% chance to blur
                    #     preprocessed_img = blur(preprocessed_img)
                    # -----------------------------------------------------------------

                    yield preprocessed_img, preprocessed_label

            # Explicitly delete volumes after processing all their slices
            del img_volume, label_volume
            gc.collect() # Force garbage collection

        except Exception as e:
            print(f"Error processing volume pair: {img_path} and {label_path}: {e}")
            continue # Skip to the next volume


# --- Function to create TensorFlow Datasets from generators ---
def create_tf_dataset(filepaths_list, batch_size, shuffle_buffer_size=1000, is_training=True):
    """
    Creates a TensorFlow Dataset from the data generator.

    Args:
        filepaths_list (list): List of (input_nii_path, output_nii_path) tuples.
        batch_size (int): Batch size for the dataset.
        shuffle_buffer_size (int): Size of the buffer for shuffling elements.
        is_training (bool): If True, apply .repeat() and larger shuffle buffer.

    Returns:
        tf.data.Dataset: A TensorFlow dataset that yields batches of (image, label).
    """
    output_signature = (
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32), # Image
        tf.TensorSpec(shape=(256, 256), dtype=tf.uint8)      # Label
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(filepaths_list), # Wrap generator call in a lambda
        output_signature=output_signature
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat() # *** IMPORTANT: Add .repeat() for training dataset ***
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        # For validation/test, typically you don't repeat, and shuffle might not be needed
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def count_slices_in_filepaths(filepaths_list, slices_per_volume=None):
    """
    Counts the total number of 2D slices that would be yielded by the data_generator
    for a given list of file paths. This involves loading each 3D volume
    to get its shape.
    """
    # This tqdm is fine because it's run *before* the tf.data.Dataset is built.
    total_slices = 0
    for img_path, _ in tqdm(filepaths_list, desc="Counting Slices", ncols=75, leave=False):
        try:
            img_volume = nib.load(img_path).get_fdata()
            axes = [0, 1, 2] # Saggital, Coronal, Axial
            for axis in axes:
                slice_shape = img_volume.shape[axis]
                if slices_per_volume is not None and slices_per_volume > 0:
                    # If slices_per_volume is specified, use that count
                    total_slices += slices_per_volume
                else:
                    # Otherwise, count all slices along this axis
                    total_slices += slice_shape
            del img_volume # Free memory after getting shape
            gc.collect()
        except Exception as e:
            print(f"Warning: Could not count slices for {img_path}: {e}")
            continue
    return total_slices

# --- Main create_dataset function for external calls ---
# This is the function you will call from your Colab notebook
def create_dataset(path1, path2, n=40, s=0.05):
    """
    The main function to prepare data for training and testing.
    It returns file paths split for train/test.

    Args:
        path1 (str): Path to the folder containing 3D input MRI volumes (.nii.gz).
        path2 (str): Path to the folder containing 3D output segmentation masks (.nii.gz).
        n (int): Number of volumes to consider from the dataset.
        s (float): Split ratio for test set (e.g., 0.1 for 10% test, 90% train).
                     If s=0, all data is used for training (no test set returned).

    Returns:
        tuple: (train_filepaths, test_filepaths) where each is a list of
               (input_nii_path, output_nii_path) tuples.
               If s=0, returns (all_filepaths, None).
    """
    all_filepaths = prepare_filepaths(path1, path2, n)

    if s > 0:
        train_filepaths, test_filepaths = train_test_split(all_filepaths, test_size=s, random_state=38)
        print(f"Dataset prepared: {len(train_filepaths)} volumes for training, {len(test_filepaths)} for testing.")
        return train_filepaths, test_filepaths
    else:
        print(f"Dataset prepared: {len(all_filepaths)} volumes for training (no test split).")
        return all_filepaths, None # Return None for test_filepaths if s=0
