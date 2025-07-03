import numpy as np
import os
import nibabel as nib
from ipywidgets import interact, interactive, fixed, AppLayout
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from IPython.display import display # Import display for AppLayout

# Function to visualize a 2D slice and apply optional colormap (EXISTING FUNCTION)
def visualize_2d(img, cmap='gray'):
    m = max(np.reshape(img, (-1)))
    # Ensure image data type is suitable for plotting (e.g., float or uint8)
    if m > 0:
        # Normalize to 0-1 for float, or 0-255 for uint8 if not already in that range
        if img.dtype != np.uint8: # Avoid unnecessary conversion if already uint8
            if not (m <= 1.0 and img.dtype == np.float32 or img.dtype == np.float64): # If not float [0,1]
                img = img.astype(np.float32) / m # Normalize to [0,1] for float plotting
            # If it's already [0,1] float or can be directly plotted, leave it.
    else: # Handle all-zero images gracefully
        img = np.zeros_like(img, dtype=np.uint8)

    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    # plt.show() # Remove plt.show() if you plan to use this in subplots

# Function to visualize the 3D MRI along each axis (EXISTING FUNCTION)
def explore_image(layer, n, data, cmap='gray'):
    if n == 0:
        visualize_2d(data[layer, :, :], cmap)  # Saggital plane
    elif n == 1:
        visualize_2d(data[:, layer, :], cmap)  # Coronal plane
    elif n == 2:
        visualize_2d(data[:, :, layer], cmap)  # Horizontal plane
    plt.show() # Add plt.show() back here if not used in subplots. For interactive use, it's fine.

# Creating a slider to select the layer to visualize in each axis (EXISTING FUNCTION)
def return_3d(select_file, c=0, x=0, cmap='gray'):
    if c == 1:  # Load 3D MRI from file
        data1 = nib.load(x + select_file).get_fdata() if x else nib.load(select_file).get_fdata()
    else:  # File already loaded
        data1 = select_file

    m = np.max(data1)
    if m > 0:
        data = np.array((data1 * 255 / m), dtype=np.uint8)  # Normalize to 0-255
    else:
        data = np.zeros_like(data1, dtype=np.uint8)

    # Creating 3 interactive sliders for each axis
    i1 = interactive(explore_image, layer=(0, data.shape[0] - 1), n=fixed(0), data=fixed(data), cmap=fixed(cmap))
    i2 = interactive(explore_image, layer=(0, data.shape[1] - 1), n=fixed(1), data=fixed(data), cmap=fixed(cmap))
    i3 = interactive(explore_image, layer=(0, data.shape[2] - 1), n=fixed(2), data=fixed(data), cmap=fixed(cmap))

    # Layout to visualize all axes side by side
    layout = AppLayout(header=None, left_sidebar=i1, center=i2, right_sidebar=i3, footer=None, pane_widths=[1, 1, 1])
    display(layout)

# Function to create an interface for visualizing 3D MRI with optional colormap (EXISTING FUNCTION)
def visualize_3d():
    x = input('Enter path containing image folder: ')  # Take folder path input
    if not x.endswith('/'):
        x = x + '/'
    l = os.listdir(x)  # List all files in the folder
    cmap = input('Enter colormap (default "gray"): ') or 'gray'
    interact(return_3d, select_file=l, c=fixed(1), x=fixed(x), cmap=fixed(cmap))  # Dropdown to select 3D image

# Overlay predictions on original MRI (EXISTING FUNCTION)
def overlay_predictions(mri_img, pred_img, alpha=0.4, cmap='jet'):
    plt.imshow(mri_img, cmap='gray')
    plt.imshow(pred_img, cmap=cmap, alpha=alpha)  # Overlay prediction with transparency
    plt.axis('off')
    plt.show()

# Visualize and focus on a particular brain part (segmentation) (EXISTING FUNCTION)
def brain_part_focus(data1, data2):
    print('Enter the segmented brain part to view:\n')
    print('1. Intracranial space and extra-axial CSF spaces')
    print('2. Gray matter')
    print('3. White matter')
    print('4. Ventricles')
    print('5. Cerebellum')
    print('6. Deep gray matter')
    print('7. Brainstem and spinal cord')
    print('8. Segmented brain without noise')

    while True:
        i = int(input('\nEnter your choice: '))
        if i < 1 or i > 8:
            print('Invalid choice. Retry!')
        else:
            break

    if i == 8:
        d1 = np.where(data2 > 0, data1, 0)  # Keep all brain parts
    else:
        d1 = np.where(data2 == i, data1, 0)  # Keep only the selected brain part

    visualize_2d(d1)
    plt.show() # Ensure plot is shown

# Visualize boundaries in segmentation (EXISTING FUNCTION)
def visualize_boundaries(segmentation):
    boundaries = find_boundaries(segmentation, mode='outer')
    plt.imshow(boundaries, cmap='hot')
    plt.axis('off')
    plt.show()

# Save a visualization to a file (EXISTING FUNCTION)
def save_visualization(img, filename, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0) # Added pad_inches=0
    print(f'Saved visualization as {filename}')

# --- NEW FUNCTIONS FOR UNCERTAINTY VISUALIZATION ---

def visualize_uncertainty_map(uncertainty_map_2d, title, cmap='magma_r', cbar_label='Uncertainty'):
    """
    Visualizes a 2D uncertainty map (e.g., entropy or variance).

    Args:
        uncertainty_map_2d: A 2D numpy array representing the uncertainty map.
        title: The title for the plot.
        cmap: Colormap for the uncertainty. 'magma_r' or 'viridis' work well for high values = high uncertainty.
        cbar_label: Label for the color bar.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(uncertainty_map_2d, cmap=cmap)
    plt.title(title)
    plt.colorbar(label=cbar_label)
    plt.axis('off')
    plt.show()

def plot_combined_segmentation_and_uncertainty(mri_img_2d, gt_seg_2d, pred_seg_2d, uncertainty_map_2d, uncertainty_type='Entropy'):
    """
    Plots the original MRI, ground truth, predicted segmentation, and an uncertainty map
    for a single 2D slice side-by-side.

    Args:
        mri_img_2d: 2D numpy array of the original MRI slice.
        gt_seg_2d: 2D numpy array of the ground truth segmentation slice.
        pred_seg_2d: 2D numpy array of the predicted segmentation slice.
        uncertainty_map_2d: 2D numpy array of the uncertainty map (e.g., entropy or variance).
        uncertainty_type: String, 'Entropy' or 'Variance' for plot titles.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original MRI
    axes[0].imshow(mri_img_2d, cmap='gray')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')

    # Ground Truth Segmentation
    axes[1].imshow(gt_seg_2d, cmap='jet') # Using 'jet' for segmentation labels
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Predicted Segmentation
    axes[2].imshow(pred_seg_2d, cmap='jet') # Using 'jet' for segmentation labels
    axes[2].set_title('Predicted Segmentation')
    axes[2].axis('off')

    # Uncertainty Map
    im = axes[3].imshow(uncertainty_map_2d, cmap='magma_r') # Colormap for uncertainty
    axes[3].set_title(f'{uncertainty_type} Map')
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3], label=f'{uncertainty_type} Value', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def plot_overlay_uncertainty_on_segmentation(mri_img_2d, pred_seg_2d, uncertainty_map_2d, uncertainty_type='Entropy', alpha_seg=0.4, alpha_unc=0.6, seg_cmap='jet', unc_cmap='Reds'):
    """
    Overlays the predicted segmentation and uncertainty map on the original MRI.
    Useful for seeing uncertainty directly on the anatomical context.

    Args:
        mri_img_2d: 2D numpy array of the original MRI slice.
        pred_seg_2d: 2D numpy array of the predicted segmentation slice.
        uncertainty_map_2d: 2D numpy array of the uncertainty map.
        uncertainty_type: String ('Entropy' or 'Variance') for title.
        alpha_seg: Transparency for segmentation overlay.
        alpha_unc: Transparency for uncertainty overlay.
        seg_cmap: Colormap for segmentation.
        unc_cmap: Colormap for uncertainty overlay (e.g., 'Reds', 'Greens' - should be sequential).
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(mri_img_2d, cmap='gray')
    plt.imshow(pred_seg_2d, cmap=seg_cmap, alpha=alpha_seg) # Overlay segmentation
    im = plt.imshow(uncertainty_map_2d, cmap=unc_cmap, alpha=alpha_unc) # Overlay uncertainty

    plt.title(f'MRI with Predicted Segmentation and {uncertainty_type} Overlay')
    plt.colorbar(im, label=f'{uncertainty_type} Value')
    plt.axis('off')
    plt.show()
