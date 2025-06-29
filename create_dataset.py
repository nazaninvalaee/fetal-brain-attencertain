import numpy as np
import os
import nibabel as nib
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from Dataset.preprocessing import reduce_2d, flip, blur
from skimage.transform import resize

# Function for splitting the dataset into train and test set and normalizing the values
def split_dataset(input_mri, output_mri, s):
    input_mri = np.array(input_mri, dtype=np.uint8)
    output_mri = np.array(output_mri, dtype=np.uint8)
    
    if s == 0:
        # Normalizing the input to the range 0 to 2
        input_mri = np.array(input_mri / 128, dtype=np.float16)
        return input_mri, output_mri
    else:  
        # Splitting up the dataset into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(input_mri, output_mri, test_size=s, random_state=38)
        
        del input_mri, output_mri
        
        # Normalizing the input to the range 0 to 2
        X_train = np.array(X_train / 128, dtype=np.float16)
        X_test = np.array(X_test / 128, dtype=np.float16)
        
        return X_train, X_test, y_train, y_test

'''  
     Function to create the dataset for both training and testing the model
     It will now include multi-scale images, so for each image slice, additional downscaled versions are added.
     It will also prepare the data for Edge-Aware Loss by highlighting boundaries.
'''

def create_dataset(path1, path2, n=40, s=0.05):
    if not path1.endswith('/'):
        path1 += '/'
    if not path2.endswith('/'):
        path2 += '/'

    l = os.listdir(path1)
    num = len(l)

    input_mri = []
    output_mri = []

    for i in tqdm(range(num), desc="Executing", ncols=75):
        f1 = l[i]
        f2 = f1[:-10] + 'dseg' + f1[-7:]

        data1 = nib.load(path1 + f1).get_fdata()
        data2 = nib.load(path2 + f2).get_fdata()

        data2 = np.array(data2, dtype=np.uint8)

        for axis in range(3):
            data_1_slice, data_2_slice = reduce_2d(data1, data2, axis)
            slice_shape = np.asarray(data_1_slice).shape[axis]
            slice_step = int(slice_shape / n)
            selected_slices = list(range(0, slice_shape, slice_step))

            for j in selected_slices:
                if axis == 0:
                    d1, d2 = data_1_slice[j, :, :], data_2_slice[j, :, :]
                elif axis == 1:
                    d1, d2 = data_1_slice[:, j, :], data_2_slice[:, j, :]
                else:
                    d1, d2 = data_1_slice[:, :, j], data_2_slice[:, :, j]

                # Ensure all images are the same shape (256x256)
                d1_resized = resize(d1, (256, 256), preserve_range=True, anti_aliasing=True)
                d2_resized = resize(d2, (256, 256), preserve_range=True, anti_aliasing=True)

                input_mri.append(d1_resized)
                output_mri.append(d2_resized)

    return split_dataset(input_mri, output_mri, s)
