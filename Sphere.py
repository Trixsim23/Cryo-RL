import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize
import torch 

#importing other classes
from MIP import MaximumIntensityProjection


def visualize_removal_with_overlay(original_mri, modified_mri, original_mask, modified_mask, slice_idx):
    """
    Visualizes the MRI with an overlay of the mask before and after removal of the spherical volume.

    Args:
        original_mri (numpy.ndarray): Original MRI data.
        modified_mri (numpy.ndarray): Modified MRI data.
        original_mask (numpy.ndarray): Original mask data.
        modified_mask (numpy.ndarray): Modified mask data.
        slice_idx (int): Index of the slice to visualize.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Define normalization for overlay consistency
    norm = Normalize(vmin=np.min(original_mri), vmax=np.max(original_mri))


    #removing the parts of the masks which do not have data


    # Original MRI 
    axes[0, 0].imshow(original_mri[:, :, slice_idx], cmap='gray', norm=norm)
    # axes[0, 0].imshow(original_mask[:, :, slice_idx], cmap='Reds', alpha=0.5)
    axes[0, 0].imshow(np.max(original_mask[:, :, :], axis =2), cmap='summer', alpha=0.2)
    axes[0, 0].set_title("Original MRI with Mask")
    axes[0, 0].axis("off")

    # Modified MRI with mask overlay
    axes[0, 1].imshow(modified_mri[:, :, slice_idx], cmap='gray', norm=norm)
    axes[0, 1].imshow(modified_mask[:, :, slice_idx], cmap='Reds', alpha=0.5)
    # axes[0, 0].imshow(np.max(modified_mask[:, :, :], axis =2), cmap='summer', alpha=0.2)
    axes[0, 1].set_title("Modified MRI with Mask")
    axes[0, 1].axis("off")

    # Original mask only
    axes[1, 0].imshow(original_mask[:, :, slice_idx], cmap='Reds')
    axes[1, 0].set_title("Original Mask")
    axes[1, 0].axis("off")

    # Modified mask only
    axes[1, 1].imshow(modified_mask[:, :, slice_idx], cmap='Reds')
    axes[1, 1].set_title("Modified Mask")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


def remove_sphere_from_prostate(mri_path, mask_path, sphere_radius, output_mri_path, output_mask_path):
    """
    Removes a defined spherical volume from the prostate MRI and mask volumes, with the sphere centered 
    on the prostate volume's center of mass, and visualizes the results with overlay.

    Args:
        mri_path (str): Path to the prostate MRI NIfTI file.
        mask_path (str): Path to the prostate mask NIfTI file.
        sphere_radius (float): Radius of the sphere to remove.
        output_mri_path (str): Path to save the modified MRI volume.
        output_mask_path (str): Path to save the modified mask volume.
    """
    # Load the MRI and mask volumes
    mri_img = nib.load(mri_path)
    mask_img = nib.load(mask_path)

    mri_data = mri_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # rotating the volume 90 degrees anti clockwise
    mri_data = np.rot90(mri_data, 1)
    mask_data = np.rot90(mask_data, 1)

    # Compute the center of mass of the prostate mask
    sphere_center = tuple(map(int, center_of_mass(mask_data)))
    print(f"Center of the prostate volume (sphere center): {sphere_center}")

    # Get the affine transform for saving modified data
    affine = mri_img.affine

    # Create a spherical mask
    x, y, z = np.ogrid[:mri_data.shape[0],
                       :mri_data.shape[1], :mri_data.shape[2]]
    distance = np.sqrt((x - sphere_center[0])**2 +
                       (y - sphere_center[1])**2 +
                       (z - sphere_center[2])**2)
    sphere_mask = distance <= sphere_radius

    # Remove the spherical volume from the MRI and mask
    mri_data_modified = mri_data.copy()
    mask_data_modified = mask_data.copy()

    mri_data_modified[sphere_mask] = 0
    mask_data_modified[sphere_mask] = 0

    # Visualize the removal with overlay
    # Visualize the slice through the sphere's center
    slice_idx = sphere_center[2]
    visualize_removal_with_overlay(
        mri_data, mri_data_modified, mask_data, mask_data_modified, slice_idx)

    # Save the modified volumes
    new_mri_img = nib.Nifti1Image(mri_data_modified, affine)
    new_mask_img = nib.Nifti1Image(mask_data_modified, affine)

    nib.save(new_mri_img, output_mri_path)
    nib.save(new_mask_img, output_mask_path)

    print(f"Modified MRI saved to: {output_mri_path}")
    print(f"Modified mask saved to: {output_mask_path}")


# Example usage
mri_file = os.path.join(".", "image_with_masks", "P-10104751", "t2.nii.gz")
mask_file = os.path.join(".", "image_with_masks", "P-10104751", "gland.nii.gz")
output_mri_file = "modified_prostate_mri.nii"
output_mask_file = "modified_prostate_mask.nii"

sphere_radius = 10  # Example radius in voxels

remove_sphere_from_prostate(
    mri_file, mask_file, sphere_radius, output_mri_file, output_mask_file)

# data on the t2 - xyz(dimensions) -512x512x23 spacing (0.3516x0.3516x3.3) voxel units (253x253x23)
#data on the gland mask - xyz(dimensions) -512x512x23 spacing (0.3516x0.3516x3.3) voxel units (253x253x23)
#data on the  - xyz(dimensions) -512x512x23 spacing (0.3516x0.3516x3.3) voxel units (253x253x23)
#initialising the MIP class 
# mip = MaximumIntensityProjection()