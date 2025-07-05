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
    Splies resizing, normalization, and adds channel dimension to a single 2D slice.
    """
    img_resized = resize(img_slice_2d, (256, 256), preserve_range=True, anti_aliasing=True)
    label_resized = resize(label_slice_2d, (256, 256), order=0, anti_aliasing=False, preserve_range=True)

    max_val = np.max(img_resized)
    if max_val > 0:
        img_normalized = img_resized.astype(np.float32) / max_val
    else:
        img_normalized = img_resized.astype(np.float32) # Keep as float even if all zeros

    img_final = np.expand_dims(img_normalized, axis=-1)
    label_final = label_resized.astype(np.uint8)

    return img_final, label_final

# --- This function now correctly filters and matches file paths ---
def prepare_filepaths(path1, path2, n=40):
    """
    Prepares lists of (input_nii_path, output_nii_path) tuples.
    It now explicitly filters for T2w images and their corresponding dseg labels.

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

    # Filter for T2w images in the input directory
    input_image_filenames = sorted([f for f in os.listdir(path1) if f.endswith('_T2w.nii.gz')])
    
    # Store output filenames with their base names for easy lookup
    # We'll prioritize _dseg.nii.gz as the target label for now.
    output_label_map = {}
    for f in os.listdir(path2):
        if f.endswith('_dseg.nii.gz'):
            # Extract base name like 'sub-040_rec-mial'
            base_name = f.replace('_dseg.nii.gz', '')
            output_label_map[base_name] = os.path.join(path2, f)
        # If you wanted to use _ddseg.nii.gz as the label, you'd add:
        # elif f.endswith('_ddseg.nii.gz'):
        #     base_name = f.replace('_ddseg.nii.gz', '')
        #     output_label_map[base_name] = os.path.join(path2, f)


    matched_filepaths = []
    # Limit to 'n' volumes from the filtered input images
    for i in range(min(n, len(input_image_filenames))):
        img_filename = input_image_filenames[i]
        
        # Extract the base name for matching (e.g., 'sub-040_rec-mial')
        base_name = img_filename.replace('_T2w.nii.gz', '')
        
        # Look up the corresponding label file in the output directory map
        label_path = output_label_map.get(base_name)

        if label_path and os.path.exists(label_path): # Check if the corresponding output label file exists
            matched_filepaths.append((os.path.join(path1, img_filename), label_path))
        else:
            print(f"Warning: Corresponding _dseg.nii.gz label not found in {path2} for input {img_filename}. Skipping.")

    if not matched_filepaths:
        raise ValueError("No matching T2w image and _dseg label NIfTI files found in the specified paths.")

    print(f"Prepared {len(matched_filepaths)} T2w image-label pairs for processing.")
    return matched_filepaths


# --- Data Generator Function ---
def data_generator(filepaths_list, slices_per_volume=None):
    """
    A Python generator that yields preprocessed 2D slices from 3D NIfTI volumes.
    """
    for img_path, label_path in filepaths_list:
        try:
            img_volume = nib.load(img_path).get_fdata()
            label_volume = nib.load(label_path).get_fdata()

            img_volume = img_volume.astype(np.float32)
            label_volume = label_volume.astype(np.uint8)

            axes = [0, 1, 2] # Saggital, Coronal, Axial

            for axis in axes:
                slice_shape = img_volume.shape[axis]
                if slices_per_volume is not None and slices_per_volume > 0:
                    selected_slice_indices = np.linspace(0, slice_shape - 1, slices_per_volume, dtype=int)
                else:
                    selected_slice_indices = range(slice_shape)

                for j in selected_slice_indices:
                    # Slicing logic
                    if axis == 0:
                        d1_slice, d2_slice = img_volume[j, :, :], label_volume[j, :, :]
                    elif axis == 1:
                        d1_slice, d2_slice = img_volume[:, j, :], label_volume[:, j, :]
                    else: # axis == 2
                        d1_slice, d2_slice = img_volume[:, :, j], label_volume[:, :, j]

                    preprocessed_img, preprocessed_label = preprocess_slice(d1_slice, d2_slice)

                    # Yield the preprocessed slices
                    yield preprocessed_img, preprocessed_label

            del img_volume, label_volume
            gc.collect()

        except Exception as e:
            print(f"Error processing volume pair: {img_path} and {label_path}: {e}")
            continue

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
        tf.TensorSpec(shape=(256, 256), dtype=tf.uint8)       # Label
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
