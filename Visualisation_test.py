import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os


def visualize_modalities(t2_path, dwi_path, adc_path, slice_idx):
    """
    Visualizes T2, DWI, and ADC images side-by-side for comparison.

    Args:
        t2_path (str): Path to the T2-weighted NIfTI file.
        dwi_path (str): Path to the DWI NIfTI file.
        adc_path (str): Path to the ADC NIfTI file.
        slice_idx (int): Index of the slice to visualize.
    """
    # Load the NIfTI files
    t2_img = nib.load(t2_path)
    dwi_img = nib.load(dwi_path)
    adc_img = nib.load(adc_path)

    t2_data = t2_img.get_fdata()
    dwi_data = dwi_img.get_fdata()
    adc_data = adc_img.get_fdata()

    # Normalize the images for better visualization
    t2_norm = Normalize(vmin=np.min(t2_data), vmax=np.max(t2_data))
    dwi_norm = Normalize(vmin=np.min(dwi_data), vmax=np.max(dwi_data))
    adc_norm = Normalize(vmin=np.min(adc_data), vmax=np.max(adc_data))

    # Plot the slices side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # T2-weighted image
    axes[0].imshow(t2_data[:, :, slice_idx], cmap='gray', norm=t2_norm)
    axes[0].set_title("T2-Weighted")
    axes[0].axis("off")

    # DWI image
    axes[1].imshow(dwi_data[:, :, slice_idx], cmap='gray', norm=dwi_norm)
    axes[1].set_title("DWI")
    axes[1].axis("off")

    # ADC map
    axes[2].imshow(adc_data[:, :, slice_idx], cmap='gray', norm=adc_norm)
    axes[2].set_title("ADC")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
t2_path = os.path.join(".", "image_with_masks", "P-10104751", "t2.nii.gz")
dwi_path = os.path.join(".", "image_with_masks", "P-10104751", "dwi.nii.gz")
adc_path = os.path.join(".", "image_with_masks", "P-10104751", "adc.nii.gz")

slice_idx = 2  # Select a slice index
visualize_modalities(t2_path, dwi_path, adc_path, slice_idx)
