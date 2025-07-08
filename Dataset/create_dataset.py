import numpy as np
import os
import nibabel as nib
import time
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from Dataset.preprocessing import reduce_2d, flip, blur, \
                                  random_brightness_contrast, random_gamma_correction, \
                                  random_affine_transform, elastic_transform
from skimage.transform import resize # Ensure this is imported
import tensorflow as tf
import random
from skimage.transform import resize 

def preprocess_slice(img_slice_2d, label_slice_2d):
    img_resized = resize(img_slice_2d, (256, 256), preserve_range=True, anti_aliasing=True)
    label_resized = resize(label_slice_2d, (256, 256), order=0, anti_aliasing=False, preserve_range=True)

    max_val = np.max(img_resized)
    if max_val > 0:
        img_normalized = img_resized.astype(np.float32) / max_val
    else:
        img_normalized = img_resized.astype(np.float32)

    img_final = np.expand_dims(img_normalized, axis=-1)

    label_processed = np.squeeze(label_resized)

    label_final = np.expand_dims(label_processed, axis=-1)

    label_final = label_final.astype(np.uint8) 

    return img_final, label_final

# --- This function now correctly filters and matches file paths ---
def prepare_filepaths(path1, path2, n=40):
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
def data_generator(filepaths_list, slices_per_volume=None, apply_augmentation=False):
    """
    A Python generator that yields preprocessed 2D slices from 3D NIfTI volumes.
    Optionally applies data augmentation.
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

                    # Preprocess first (resize, normalize, expand_dims)
                    current_img, current_label = preprocess_slice(d1_slice, d2_slice)

                    # --- Apply Data Augmentation (if enabled and for training) ---
                    if apply_augmentation:
                        # Randomly apply augmentations with a certain probability
                        # Order: Intensity -> Geometric -> Blur (as per best practices)

                        # 1. Intensity Augmentations (apply only to image)
                        if random.random() < 0.5: # 50% chance for brightness/contrast
                            current_img = random_brightness_contrast(current_img)
                        if random.random() < 0.5: # 50% chance for gamma correction
                            current_img = random_gamma_correction(current_img)

                        # 2. Geometric Augmentations (apply to both image and label)
                        if random.random() < 0.5: # 50% chance for random affine transform (rotation + translation)
                            current_img, current_label = random_affine_transform(current_img, current_label)
                            # --- Ensure label has channel dim after transform if it loses it ---
                            if current_label.ndim == 2:
                                current_label = np.expand_dims(current_label, axis=-1)

                        if random.random() < 0.2: # 20% chance for elastic deformation (can be computationally heavier)
                            current_img, current_label = elastic_transform(current_img, current_label)
                            # --- Ensure label has channel dim after transform if it loses it ---
                            if current_label.ndim == 2:
                                current_label = np.expand_dims(current_label, axis=-1)

                        if random.random() < 0.5: # 50% chance for random flip
                            flip_code = random.choice([0, 1, -1]) # 0: vertical, 1: horizontal, -1: both
                            current_img, current_label = flip(current_img, current_label, flip_code)
                            # --- Ensure label has channel dim after flip if it loses it ---
                            if current_label.ndim == 2:
                                current_label = np.expand_dims(current_label, axis=-1)

                        # 3. Blur (apply only to image)
                        if random.random() < 0.5: # 50% chance for blur
                           current_img = blur(current_img, apply_blur=True) # blur function already has internal randomness for kernel size

                    # --- Convert to TensorFlow Tensors with explicit dtypes before yielding ---
                    yield tf.convert_to_tensor(current_img, dtype=tf.float32), tf.convert_to_tensor(current_label, dtype=tf.uint8)

            del img_volume, label_volume
            gc.collect()

        except Exception as e:
            print(f"Error processing volume pair: {img_path} and {label_path}: {e}")
            continue

# --- Function to create TensorFlow Datasets from generators ---
def create_tf_dataset(filepaths_list, batch_size, shuffle_buffer_size=1000,
                      is_training=True, slices_per_volume=None,
                      force_no_augmentation=False): # <--- ADD THIS PARAMETER
    output_signature = (
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32), # Image
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.uint8)    # Label
    )

    # Determine whether to apply augmentation: only if is_training and not forced off
    _apply_augmentation = is_training and not force_no_augmentation # <--- USE THIS LOGIC

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(filepaths_list, slices_per_volume=slices_per_volume,
                               apply_augmentation=_apply_augmentation), # <--- USE _apply_augmentation
        output_signature=output_signature
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def count_slices_in_filepaths(filepaths_list, slices_per_volume=None):
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
def create_dataset(path1, path2, n=40, s=0.05):
    all_filepaths = prepare_filepaths(path1, path2, n)

    if s > 0:
        train_filepaths, test_filepaths = train_test_split(all_filepaths, test_size=s, random_state=38)
        print(f"Dataset prepared: {len(train_filepaths)} volumes for training, {len(test_filepaths)} for testing.")
        return train_filepaths, test_filepaths
    else:
        print(f"Dataset prepared: {len(all_filepaths)} volumes for training (no test split).")
        return all_filepaths, None
