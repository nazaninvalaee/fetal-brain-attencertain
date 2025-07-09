import os
import gc
import time
import numpy as np
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize

from Dataset.preprocessing import reduce_2d, flip, blur  # Assuming compatible with 2D slices


# --- Preprocess a Single 2D Slice ---
def preprocess_slice(img_slice_2d, label_slice_2d):
    """
    Resizes, normalizes, and reshapes a 2D slice and its label.

    Args:
        img_slice_2d (np.ndarray): Image slice.
        label_slice_2d (np.ndarray): Corresponding label slice.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed image and label.
    """
    img_resized = resize(img_slice_2d, (256, 256), preserve_range=True, anti_aliasing=True)
    label_resized = resize(label_slice_2d, (256, 256), order=0, anti_aliasing=False, preserve_range=True)

    max_val = np.max(img_resized)
    img_normalized = img_resized.astype(np.float32) / max_val if max_val > 0 else img_resized.astype(np.float32)

    img_final = np.expand_dims(img_normalized, axis=-1)
    label_final = label_resized.astype(np.uint8)

    return img_final, label_final


# --- Prepare List of Input/Output Filepaths ---
def prepare_filepaths(path1, path2, n=40):
    """
    Filters and matches T2w input volumes with corresponding _dseg labels.

    Args:
        path1 (str): Input NIfTI directory (T2w).
        path2 (str): Output NIfTI directory (_dseg labels).
        n (int): Number of volumes to select.

    Returns:
        List[Tuple[str, str]]: Matched filepaths list.
    """
    path1 = path1.rstrip('/') + '/'
    path2 = path2.rstrip('/') + '/'

    input_filenames = sorted([f for f in os.listdir(path1) if f.endswith('_T2w.nii.gz')])
    output_label_map = {
        f.replace('_dseg.nii.gz', ''): os.path.join(path2, f)
        for f in os.listdir(path2) if f.endswith('_dseg.nii.gz')
    }

    matched_filepaths = []
    for i in range(min(n, len(input_filenames))):
        img_filename = input_filenames[i]
        base_name = img_filename.replace('_T2w.nii.gz', '')
        label_path = output_label_map.get(base_name)

        if label_path and os.path.exists(label_path):
            matched_filepaths.append((os.path.join(path1, img_filename), label_path))
        else:
            print(f"Warning: No _dseg label found for {img_filename}. Skipping.")

    if not matched_filepaths:
        raise ValueError("No matching image-label pairs found.")

    print(f"Prepared {len(matched_filepaths)} T2w image-label pairs for processing.")
    return matched_filepaths


# --- Slice-by-Slice Generator ---
def data_generator(filepaths_list, slices_per_volume=None):
    """
    Yields 2D preprocessed image-label pairs from 3D volumes.

    Args:
        filepaths_list (list): List of (img_path, label_path).
        slices_per_volume (int or None): Slices per volume per axis.

    Yields:
        Tuple[np.ndarray, np.ndarray]: Preprocessed image and label slices.
    """
    for img_path, label_path in filepaths_list:
        try:
            img_volume = nib.load(img_path).get_fdata().astype(np.float32)
            label_volume = nib.load(label_path).get_fdata().astype(np.uint8)

            for axis in [0, 1, 2]:
                slice_count = img_volume.shape[axis]
                indices = np.linspace(0, slice_count - 1, slices_per_volume, dtype=int) if slices_per_volume else range(slice_count)

                for j in indices:
                    if axis == 0:
                        d1_slice, d2_slice = img_volume[j, :, :], label_volume[j, :, :]
                    elif axis == 1:
                        d1_slice, d2_slice = img_volume[:, j, :], label_volume[:, j, :]
                    else:
                        d1_slice, d2_slice = img_volume[:, :, j], label_volume[:, :, j]

                    img, label = preprocess_slice(d1_slice, d2_slice)
                    yield img, label

            del img_volume, label_volume
            gc.collect()

        except Exception as e:
            print(f"Error loading volume pair {img_path}, {label_path}: {e}")
            continue


# --- Convert to tf.data.Dataset ---
def create_tf_dataset(filepaths_list, batch_size, shuffle_buffer_size=1000, is_training=True, slices_per_volume=None):
    """
    Wraps the data_generator in a TensorFlow Dataset pipeline.

    Args:
        filepaths_list (list): List of (img_path, label_path).
        batch_size (int): Batch size.
        shuffle_buffer_size (int): Shuffle buffer size.
        is_training (bool): If True, apply shuffle/repeat.
        slices_per_volume (int): Number of slices per volume per axis.

    Returns:
        tf.data.Dataset: Batched dataset.
    """
    output_signature = (
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256), dtype=tf.uint8)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(filepaths_list, slices_per_volume=slices_per_volume),
        output_signature=output_signature
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# --- Count 2D Slices in Given Filepaths ---
def count_slices_in_filepaths(filepaths_list, slices_per_volume=None):
    """
    Counts total number of 2D slices across all axes and volumes.

    Args:
        filepaths_list (list): List of (img_path, label_path).
        slices_per_volume (int or None): Fixed slices per axis or full.

    Returns:
        int: Total slice count.
    """
    total_slices = 0
    for img_path, _ in tqdm(filepaths_list, desc="Counting Slices", ncols=75, leave=False):
        try:
            volume = nib.load(img_path).get_fdata()
            if slices_per_volume and slices_per_volume > 0:
                total_slices += slices_per_volume * 3
            else:
                total_slices += sum(volume.shape)
            del volume
            gc.collect()
        except Exception as e:
            print(f"Warning: Could not count slices for {img_path}: {e}")
            continue
    return total_slices


# --- High-Level Dataset Preparer ---
def create_dataset(path1, path2, n=40, s=0.05):
    """
    Splits dataset into train/test filepaths after verifying pairs.

    Args:
        path1 (str): Path to input T2w volumes.
        path2 (str): Path to segmentation labels (_dseg).
        n (int): Number of volumes to include.
        s (float): Test split ratio (0 = no test split).

    Returns:
        Tuple[list, list or None]: (train_filepaths, test_filepaths)
    """
    all_filepaths = prepare_filepaths(path1, path2, n)

    if s > 0:
        train_fp, test_fp = train_test_split(all_filepaths, test_size=s, random_state=38)
        print(f"Dataset prepared: {len(train_fp)} for training, {len(test_fp)} for testing.")
        return train_fp, test_fp
    else:
        print(f"Dataset prepared: {len(all_filepaths)} for training (no test split).")
        return all_filepaths, None
