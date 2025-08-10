import os
import time 
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gym import spaces
from scipy.ndimage import center_of_mass
from matplotlib.colors import Normalize
import matplotlib.patches as patches
from main_env import SpherePlacementEnv

def resize_volume(volume, target_shape=(128, 128, 20)):
    """
    Resize a 3D volume to target shape using appropriate interpolation
    """
    from skimage.transform import resize
    
    # For binary masks, use order=0 (nearest neighbor)
    # For continuous data like MRI, use order=1 (linear)
    if volume.dtype == bool or np.array_equal(np.unique(volume), np.array([0, 1])):
        # Binary data - use nearest neighbor
        resized = resize(volume.astype(float), target_shape, order=0, preserve_range=True, anti_aliasing=False)
        return resized.astype(volume.dtype)
    else:
        # Continuous data - use linear interpolation
        resized = resize(volume, target_shape, order=1, preserve_range=True, anti_aliasing=True)
        return resized.astype(volume.dtype)


def load_and_preprocess_patient_data(patient_dir, target_shape=(128, 128, 20)):
    """
    Load and preprocess MRI data for a single patient to ensure consistent dimensions
    """
    mri_file = os.path.join(patient_dir, "t2.nii.gz")
    mask_file = os.path.join(patient_dir, "gland.nii.gz")
    lesion_file = os.path.join(patient_dir, "l_a1.nii.gz")

    # Load the images
    mri_img = nib.load(mri_file)
    mask_img = nib.load(mask_file)
    lesion_img = nib.load(lesion_file)

    # Get the data and rotate 90 degrees anti-clockwise
    mri_data = np.rot90(mri_img.get_fdata(), 1)
    mask_data = np.rot90(mask_img.get_fdata(), 1)
    lesion_data = np.rot90(lesion_img.get_fdata(), 1)

    print(f"Original shapes - MRI: {mri_data.shape}, Mask: {mask_data.shape}, Lesion: {lesion_data.shape}")

    # Resize all volumes to target shape
    mri_resized = resize_volume(mri_data, target_shape)
    mask_resized = resize_volume(mask_data, target_shape)
    lesion_resized = resize_volume(lesion_data, target_shape)

    print(f"Resized shapes - MRI: {mri_resized.shape}, Mask: {mask_resized.shape}, Lesion: {lesion_resized.shape}")

    return mri_resized, mask_resized, lesion_resized


def create_sphere_mask(sphere_positions, sphere_radius, volume_shape):
    """
    Create a binary mask representing all placed spheres
    """
    mask = np.zeros(volume_shape, dtype=bool)
    
    for center in sphere_positions:
        x, y, z = center
        
        # Define the ranges for the sphere
        x_range = np.arange(max(0, x - sphere_radius), min(volume_shape[0], x + sphere_radius))
        y_range = np.arange(max(0, y - sphere_radius), min(volume_shape[1], y + sphere_radius))
        z_range = np.arange(max(0, z - sphere_radius), min(volume_shape[2], z + sphere_radius))
        
        # Create a grid of points in the range
        x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # Calculate the distance of each point in the grid from the center
        distances = np.sqrt((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2)
        
        # Create a mask for points within the radius
        sphere_mask = distances <= sphere_radius
        
        # Update the relevant slices of the mask
        mask[x_range[:, None, None], y_range[None, :, None], z_range[None, None, :]] |= sphere_mask
    
    return mask


def calculate_dice_score(prediction_mask, target_mask):
    """
    Calculate Dice coefficient between two binary masks
    """
    # Ensure masks are binary
    pred_binary = prediction_mask > 0
    target_binary = target_mask > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_binary, target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()
    
    # Avoid division by zero
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2.0 * intersection / total
    return dice


# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================================================================

def enhanced_visualize_spheres_with_numbers(env, slice_idx=None, save_path=None, show=True, step_info=""):
    """
    Enhanced visualization with improved slice selection to show lesions and spheres optimally
    """
    # IMPROVED: Auto-select slice that shows both lesions and spheres optimally
    if slice_idx is None:
        # Find slice that best shows both lesion and spheres
        lesion_slices_with_content = []
        
        # Find all slices that contain lesion data
        for z in range(env.lesion_data.shape[2]):
            if np.any(env.lesion_data[:, :, z] > 0):
                lesion_volume_in_slice = np.sum(env.lesion_data[:, :, z] > 0)
                lesion_slices_with_content.append((z, lesion_volume_in_slice))
        
        if lesion_slices_with_content and env.sphere_positions:
            # Find slice that optimizes both lesion visibility and sphere proximity
            sphere_z_positions = [pos[2] for pos in env.sphere_positions]
            median_sphere_z = np.median(sphere_z_positions)
            
            # Score each lesion-containing slice based on lesion content and sphere proximity
            best_slice = None
            best_score = -1
            
            for z, lesion_vol in lesion_slices_with_content:
                # Score based on lesion volume and sphere proximity
                sphere_proximity_score = max(0, 5 - abs(z - median_sphere_z))  # Higher score for closer slices
                lesion_volume_score = lesion_vol / 100  # Normalize lesion volume
                
                total_score = lesion_volume_score + sphere_proximity_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_slice = z
            
            slice_idx = best_slice if best_slice is not None else int(median_sphere_z)
            print(f"Auto-selected slice {slice_idx} - optimized for lesion visibility and sphere proximity")
            
        elif lesion_slices_with_content:
            # If no spheres, choose slice with most lesion content
            slice_idx = max(lesion_slices_with_content, key=lambda x: x[1])[0]
            print(f"Auto-selected slice {slice_idx} - based on maximum lesion content")
            
        elif env.sphere_positions:
            # If no lesion visible, fall back to sphere-based selection
            z_positions = [pos[2] for pos in env.sphere_positions]
            slice_idx = int(np.median(z_positions))
            print(f"Auto-selected slice {slice_idx} - based on sphere positions (no lesion visible)")
            
        else:
            # Default to center slice
            slice_idx = env.mri_data.shape[2] // 2
            print(f"Auto-selected slice {slice_idx} - using center slice (no spheres or lesions)")
    
    # Ensure slice_idx is within bounds
    slice_idx = max(0, min(slice_idx, env.mri_data.shape[2] - 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Normalize MRI volumes to [0, 1]
    original_mri_norm = (env.mri_data - np.min(env.mri_data)) / (np.max(env.mri_data) - np.min(env.mri_data) + 1e-10)

    # Normalize masks to [0, 1]
    original_mask_norm = env.mask_data.astype(float)
    if np.max(env.mask_data) > 0:
        original_mask_norm = original_mask_norm / np.max(env.mask_data)
        
    modified_mask_norm = env.modified_mask.astype(float)
    if np.max(env.modified_mask) > 0:
        modified_mask_norm = modified_mask_norm / np.max(env.modified_mask)
        
    original_lesion_norm = env.lesion_data.astype(float)
    if np.max(env.lesion_data) > 0:
        original_lesion_norm = original_lesion_norm / np.max(env.lesion_data)
        
    modified_lesion_norm = env.modified_lesion.astype(float)
    if np.max(env.modified_lesion) > 0:
        modified_lesion_norm = modified_lesion_norm / np.max(env.modified_lesion)

    # Create masked arrays
    original_mask_masked = np.ma.masked_where(original_mask_norm[:, :, slice_idx] == 0, original_mask_norm[:, :, slice_idx])
    modified_mask_masked = np.ma.masked_where(modified_mask_norm[:, :, slice_idx] == 0, modified_mask_norm[:, :, slice_idx])
    original_lesion_masked = np.ma.masked_where(original_lesion_norm[:, :, slice_idx] == 0, original_lesion_norm[:, :, slice_idx])
    modified_lesion_masked = np.ma.masked_where(modified_lesion_norm[:, :, slice_idx] == 0, modified_lesion_norm[:, :, slice_idx])

    # Use more distinct colors and larger markers
    sphere_colors = ['red', 'cyan', 'yellow', 'magenta', 'lime']
    sphere_markers = ['o', 's', '^', 'D', 'P']

    # Original MRI with mask overlay
    axes[0, 0].imshow(original_mri_norm[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].imshow(original_mask_masked, cmap='Blues', alpha=0.3, vmin=0, vmax=1)
    axes[0, 0].imshow(original_lesion_masked, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    axes[0, 0].set_title("Original MRI with Masks")
    axes[0, 0].axis("off")

    # Modified MRI with enhanced sphere markers
    axes[0, 1].imshow(original_mri_norm[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[0, 1].imshow(modified_mask_masked, cmap='Blues', alpha=0.3, vmin=0, vmax=1)
    axes[0, 1].imshow(modified_lesion_masked, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    
    # Enhanced sphere visualization with debugging
    sphere_info = []
    spheres_visible = 0
    
    print(f"Visualizing {len(env.sphere_positions)} spheres on slice {slice_idx}")
    
    # Check lesion visibility on selected slice
    lesion_pixels_in_slice = np.sum(env.lesion_data[:, :, slice_idx] > 0)
    print(f"Lesion pixels in selected slice {slice_idx}: {lesion_pixels_in_slice}")
    
    for i, sphere_pos in enumerate(env.sphere_positions):
        x, y, z = sphere_pos
        z_distance = abs(z - slice_idx)
        
        print(f"  Sphere {i+1}: pos=({x}, {y}, {z}), z_dist={z_distance}")
        
        # Show spheres within larger range with transparency
        if z_distance <= 3:  # Increased from 2 to 3
            color = sphere_colors[i % len(sphere_colors)]
            marker = sphere_markers[i % len(sphere_markers)]
            
            # Calculate alpha based on distance
            alpha = max(0.4, 1.0 - (z_distance / 3.0))
            
            # Much larger, more visible markers with black outline
            scatter = axes[0, 1].scatter(y, x, s=800, c=color, marker=marker, 
                                       edgecolors='black', linewidth=4, alpha=alpha, zorder=10)
            
            # Larger text with better contrast
            text = axes[0, 1].text(y, x, str(i+1), ha='center', va='center', 
                                 fontsize=20, fontweight='bold', color='white', zorder=11,
                                 bbox=dict(boxstyle="circle,pad=0.15", facecolor='black', alpha=0.8))
            
            spheres_visible += 1
            sphere_info.append(f'Sphere {i+1}: ({x}, {y}, {z}) - Z-dist: {z_distance}')
    
    print(f"  {spheres_visible} spheres visible on this slice")
    
    axes[0, 1].set_title(f"Modified MRI with Numbered Spheres {step_info}\nSlice: {slice_idx} ({spheres_visible}/{len(env.sphere_positions)} visible)")
    axes[0, 1].axis("off")

    # Original masks
    axes[1, 0].imshow(original_mask_masked, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
    axes[1, 0].imshow(original_lesion_masked, cmap='Reds', alpha=0.8, vmin=0, vmax=1)
    axes[1, 0].set_title("Original Masks")
    axes[1, 0].axis("off")

    # Modified masks with enhanced sphere markers
    axes[1, 1].imshow(modified_mask_masked, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
    axes[1, 1].imshow(modified_lesion_masked, cmap='Reds', alpha=0.8, vmin=0, vmax=1)
    
    # Add enhanced sphere markers to mask view
    for i, sphere_pos in enumerate(env.sphere_positions):
        x, y, z = sphere_pos
        z_distance = abs(z - slice_idx)
        if z_distance <= 3:
            color = sphere_colors[i % len(sphere_colors)]
            marker = sphere_markers[i % len(sphere_markers)]
            alpha = max(0.4, 1.0 - (z_distance / 3.0))
            
            axes[1, 1].scatter(y, x, s=800, c=color, marker=marker, 
                             edgecolors='black', linewidth=4, alpha=alpha, zorder=10)
            axes[1, 1].text(y, x, str(i+1), ha='center', va='center', 
                           fontsize=20, fontweight='bold', color='white', zorder=11,
                           bbox=dict(boxstyle="circle,pad=0.15", facecolor='black', alpha=0.8))
    
    axes[1, 1].set_title(f"Modified Masks with Numbered Spheres {step_info}")
    axes[1, 1].axis("off")

    # Enhanced legend with detailed sphere information
    if env.sphere_positions:
        legend_elements = []
        for i in range(len(env.sphere_positions)):
            x, y, z = env.sphere_positions[i]
            color = sphere_colors[i % len(sphere_colors)]
            marker = sphere_markers[i % len(sphere_markers)]
            z_distance = abs(z - slice_idx)
            
            # Indicate if sphere is visible on current slice
            visibility = "visible" if z_distance <= 3 else "hidden"
            label = f'Sphere {i+1} (z={z}, {visibility})'
            
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor=color, markersize=15, 
                                            markeredgecolor='black', markeredgewidth=3,
                                            label=label))
        
        axes[0, 1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=9)

    plt.tight_layout()
    
    # Print detailed sphere information
    if sphere_info:
        print(f"\nDetailed sphere information for slice {slice_idx}:")
        for info in sphere_info:
            print(f"  {info}")
    
    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        folder = os.path.join(".", "results")
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(os.path.join('results', f'enhanced_visualize_{step_info}_{current_time}.png'), dpi=200)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_3d_volume_rendering(env, save_path=None, show=True, step_info="", opacity_threshold=0.3):
    """
    FIXED: Create 3D volume rendering with proper title spacing
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create 2x2 subplot layout for different 3D views
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    axes = [ax1, ax2, ax3, ax4]
    view_angles = [(30, 45), (-30, 45), (30, -45), (60, 30)]  # Different viewing angles
    view_names = ['Anterior-Superior', 'Posterior-Superior', 'Anterior-Inferior', 'Lateral-Superior']
    
    for idx, (ax, (elev, azim), view_name) in enumerate(zip(axes, view_angles, view_names)):
        ax.clear()
        
        # Plot prostate boundary (mask) as semi-transparent surface
        if np.any(env.mask_data > 0):
            mask_coords = np.where(env.mask_data > opacity_threshold)
            if len(mask_coords[0]) > 0:
                # Subsample for performance (every 2nd voxel)
                subsample = slice(None, None, 2)
                ax.scatter(mask_coords[1][subsample], mask_coords[0][subsample], mask_coords[2][subsample], 
                          c='blue', alpha=0.1, s=1, label='Prostate Boundary')
        
        # Plot lesion as more opaque surface
        if np.any(env.lesion_data > 0):
            lesion_coords = np.where(env.lesion_data > opacity_threshold)
            if len(lesion_coords[0]) > 0:
                ax.scatter(lesion_coords[1], lesion_coords[0], lesion_coords[2], 
                          c='red', alpha=0.4, s=4, label='Lesion Volume')
        
        # Show 3D spheres at placement locations
        sphere_colors = ['yellow', 'cyan', 'magenta', 'lime', 'orange']
        
        for i, sphere_pos in enumerate(env.sphere_positions):
            x_center, y_center, z_center = sphere_pos
            color = sphere_colors[i % len(sphere_colors)]
            
            # Create sphere surface using parametric equations
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            radius = env.sphere_radius
            
            x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + y_center  # Note: swapped for display
            y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + x_center
            z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center
            
            # Plot sphere surface
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=0.8)
            
            # Add sphere number label
            ax.text(y_center, x_center, z_center + radius + 2, f'{i+1}', 
                   fontsize=14, fontweight='bold', color='black',
                   bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', alpha=0.8))
        
        # Set equal aspect ratio and labels
        ax.set_xlim([0, env.mri_data.shape[1]])
        ax.set_ylim([0, env.mri_data.shape[0]])
        ax.set_zlim([0, env.mri_data.shape[2]])
        
        # Medical imaging coordinate labels
        ax.set_xlabel('Y (Anterior-Posterior)', fontsize=10)
        ax.set_ylabel('X (Left-Right)', fontsize=10)
        ax.set_zlabel('Z (Superior-Inferior)', fontsize=10)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # FIXED: Title for each view with reduced font size and padding
        ax.set_title(f'{view_name} View {step_info}', fontsize=11, fontweight='bold', pad=10)
        
        # Grid and background
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    
    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=8, alpha=0.6, label='Prostate Volume'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                   markersize=8, alpha=0.8, label='Lesion Volume')
    ]
    
    # Add sphere legend entries
    for i in range(len(env.sphere_positions)):
        color = sphere_colors[i % len(sphere_colors)]
        x, y, z = env.sphere_positions[i]
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      markersize=10, alpha=0.9, 
                      label=f'Sphere {i+1} ({x}, {y}, {z})')
        )
    
    # FIXED: Use tight_layout with proper spacing for title and legend
    plt.tight_layout(rect=[0, 0.08, 1, 0.90])  # [left, bottom, right, top] - leaves space for title and legend
    
    # FIXED: Place legend in the reserved bottom space
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
              ncol=min(4, len(legend_elements)), fontsize=10)
    
    # FIXED: Overall title positioned in the reserved top space
    fig.suptitle(f'3D Volume Rendering: Prostate Cryoablation Planning {step_info}', 
                fontsize=16, fontweight='bold', y=0.96)  # Moved higher from y=0.95 to y=0.96
    
    # Print sphere placement summary
    print(f"\n3D Volume Rendering Summary:")
    print(f"  Prostate volume: {np.sum(env.mask_data > 0)} voxels")
    print(f"  Lesion volume: {np.sum(env.lesion_data > 0)} voxels")
    print(f"  Spheres placed: {len(env.sphere_positions)}")
    for i, pos in enumerate(env.sphere_positions):
        print(f"    Sphere {i+1}: Position ({pos[0]}, {pos[1]}, {pos[2]}), Radius: {env.sphere_radius}")
    
    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D volume rendering saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# ============================================================================
# ENHANCED STEP VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_individual_step_placement(env, step_num, save_folder, patient_id, dice_score=None, create_3d=False):
    """
    FIXED: Individual step visualization with improved slice selection for lesion visibility
    Now uses optimal slice selection that prioritizes showing both lesions and spheres
    
    Args:
        env: Environment containing the sphere placement data
        step_num: Current step number
        save_folder: Folder to save visualizations
        patient_id: Patient identifier
        dice_score: Optional dice score for this step
        create_3d: Whether to also create 3D visualization (computationally expensive)
    
    Returns:
        tuple: Paths to created visualizations (enhanced_2d_path, volume_3d_path)
    """
    # IMPROVED: Better slice selection for optimal lesion and sphere visibility
    if env.sphere_positions:
        # Find slice that best shows both lesion and spheres
        lesion_slices_with_content = []
        
        # Find all slices that contain lesion data
        for z in range(env.lesion_data.shape[2]):
            if np.any(env.lesion_data[:, :, z] > 0):
                lesion_volume_in_slice = np.sum(env.lesion_data[:, :, z] > 0)
                lesion_slices_with_content.append((z, lesion_volume_in_slice))
        
        if lesion_slices_with_content:
            # Find slice that optimizes both lesion visibility and sphere proximity
            sphere_z_positions = [pos[2] for pos in env.sphere_positions]
            median_sphere_z = np.median(sphere_z_positions)
            
            # Score each lesion-containing slice
            best_slice = None
            best_score = -1
            
            for z, lesion_vol in lesion_slices_with_content:
                # Score based on lesion volume and sphere proximity
                sphere_proximity_score = max(0, 5 - abs(z - median_sphere_z))
                lesion_volume_score = lesion_vol / 100
                
                total_score = lesion_volume_score + sphere_proximity_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_slice = z
            
            slice_idx = best_slice if best_slice is not None else int(median_sphere_z)
        else:
            # No lesion visible, use sphere-based selection
            z_positions = [pos[2] for pos in env.sphere_positions]
            slice_idx = int(np.median(z_positions))
    else:
        # No spheres placed yet - find slice with most lesion content
        lesion_slices_with_content = []
        for z in range(env.lesion_data.shape[2]):
            if np.any(env.lesion_data[:, :, z] > 0):
                lesion_volume_in_slice = np.sum(env.lesion_data[:, :, z] > 0)
                lesion_slices_with_content.append((z, lesion_volume_in_slice))
        
        if lesion_slices_with_content:
            slice_idx = max(lesion_slices_with_content, key=lambda x: x[1])[0]
        else:
            slice_idx = env.mri_data.shape[2] // 2
    
    # Prepare step info
    step_info = f"Step {step_num}"
    if dice_score is not None:
        step_info += f" (Dice: {dice_score:.3f})"
    
    # CREATE ENHANCED 2D VISUALIZATION with improved slice selection
    enhanced_2d_path = os.path.join(save_folder, f'enhanced_step_patient_{patient_id}_step_{step_num}.png')
    enhanced_visualize_spheres_with_numbers(
        env=env,
        slice_idx=slice_idx,
        save_path=enhanced_2d_path,
        show=False,
        step_info=step_info
    )
    
    # OPTIONALLY CREATE 3D VISUALIZATION
    volume_3d_path = None
    if create_3d:
        volume_3d_path = os.path.join(save_folder, f'3d_step_patient_{patient_id}_step_{step_num}.png')
        try:
            visualize_3d_volume_rendering(
                env=env,
                save_path=volume_3d_path,
                show=False,
                step_info=step_info
            )
            print(f"    ✓ 3D visualization: {volume_3d_path}")
        except Exception as e:
            print(f"    ⚠ 3D visualization failed (optional): {e}")
            volume_3d_path = None
    
    print(f"    ✓ Enhanced 2D (optimized slice {slice_idx}): {enhanced_2d_path}")
    
    return enhanced_2d_path, volume_3d_path


def create_sphere_progression_summary(env, save_folder, patient_id):
    """
    Create a summary showing progression of sphere placements
    (Your original function, unchanged)
    """
    slice_idx = env.mri_data.shape[2] // 2
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Normalize MRI data
    mri_norm = (env.mri_data - np.min(env.mri_data)) / (np.max(env.mri_data) - np.min(env.mri_data) + 1e-10)
    
    # Define colors for spheres
    sphere_colors = ['red', 'blue', 'green']
    sphere_markers = ['o', 's', '^']
    
    # Show progression: 0, 1, 2, 3 spheres
    for step in range(4):
        axes[step].imshow(mri_norm[:, :, slice_idx], cmap='gray')
        
        # Show lesion overlay
        if np.any(env.lesion_data[:, :, slice_idx] > 0):
            axes[step].imshow(np.where(env.lesion_data[:, :, slice_idx] > 0, 
                                     env.lesion_data[:, :, slice_idx], np.nan), 
                            cmap='Reds', alpha=0.5)
        
        # Show spheres up to current step
        spheres_to_show = min(step, len(env.sphere_positions))
        for i in range(spheres_to_show):
            sphere_pos = env.sphere_positions[i]
            x, y, z = sphere_pos
            if abs(z - slice_idx) <= 2:
                color = sphere_colors[i % len(sphere_colors)]
                marker = sphere_markers[i % len(sphere_markers)]
                
                # Add numbered marker
                axes[step].scatter(y, x, s=300, c=color, marker=marker, 
                                 edgecolors='white', linewidth=3, alpha=0.9, zorder=10)
                axes[step].text(y, x, str(i+1), ha='center', va='center', 
                               fontsize=14, fontweight='bold', color='white', zorder=11)
        
        axes[step].set_title(f'Step {step}: {spheres_to_show} Sphere(s)')
        axes[step].axis('off')
    
    plt.suptitle(f'Sphere Placement Progression - Patient {patient_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, f'sphere_progression_patient_{patient_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def create_detailed_analysis_plot(env, save_path, patient_id, final_dice, coverage_percentage, sphere_volumes):
    """
    NEW: Create detailed quantitative analysis visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Lesion coverage heatmap
    if env.sphere_positions:
        # Create coverage map
        sphere_mask = create_sphere_mask(env.sphere_positions, env.sphere_radius, env.mri_data.shape)
        coverage_map = (sphere_mask.astype(float) + env.lesion_data.astype(float))
        
        # Show central slice
        central_slice = env.mri_data.shape[2] // 2
        im1 = axes[0, 0].imshow(coverage_map[:, :, central_slice], cmap='RdYlBu_r', alpha=0.8)
        axes[0, 0].set_title(f'Lesion Coverage Map\nCentral Slice (Z={central_slice})')
        plt.colorbar(im1, ax=axes[0, 0], label='Coverage Intensity')
    else:
        axes[0, 0].text(0.5, 0.5, 'No spheres placed', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Lesion Coverage Map')
    axes[0, 0].axis('off')
    
    # 2. Sphere volume distribution
    if sphere_volumes:
        bars = axes[0, 1].bar(range(1, len(sphere_volumes) + 1), sphere_volumes, 
                             color=['red', 'blue', 'green', 'orange', 'purple'][:len(sphere_volumes)])
        axes[0, 1].set_title('Sphere Volume Distribution\n(Voxels within Prostate)')
        axes[0, 1].set_xlabel('Sphere Number')
        axes[0, 1].set_ylabel('Volume (voxels)')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(sphere_volumes)*0.01,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'No spheres placed', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Sphere Volume Distribution')
    
    # 3. Distance analysis between spheres
    if len(env.sphere_positions) > 1:
        distances = []
        labels = []
        for i in range(len(env.sphere_positions)):
            for j in range(i + 1, len(env.sphere_positions)):
                pos1 = np.array(env.sphere_positions[i])
                pos2 = np.array(env.sphere_positions[j])
                dist = np.linalg.norm(pos1 - pos2)
                distances.append(dist)
                labels.append(f'S{i+1}-S{j+1}')
        
        bars = axes[1, 0].bar(range(len(distances)), distances, color='skyblue', edgecolor='navy')
        axes[1, 0].set_title('Inter-Sphere Distances')
        axes[1, 0].set_xlabel('Sphere Pairs')
        axes[1, 0].set_ylabel('Distance (voxels)')
        axes[1, 0].set_xticks(range(len(distances)))
        axes[1, 0].set_xticklabels(labels, rotation=45)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(distances)*0.01,
                           f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'Need >1 sphere for\ndistance analysis', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Inter-Sphere Distances')
    
    # 4. Performance metrics summary
    axes[1, 1].axis('off')
    
    # Create performance summary text
    performance_text = f"""
PERFORMANCE SUMMARY
Patient {patient_id}

Final Dice Score: {final_dice:.3f}
Lesion Coverage: {coverage_percentage:.1f}%

SPHERE PLACEMENTS:
Total Spheres: {len(env.sphere_positions)}
"""
    
    if env.sphere_positions:
        performance_text += "\nSphere Coordinates:\n"
        for i, pos in enumerate(env.sphere_positions):
            performance_text += f"  Sphere {i+1}: ({pos[0]}, {pos[1]}, {pos[2]})\n"
    
    performance_text += f"""
VOLUME ANALYSIS:
Total Lesion: {np.sum(env.lesion_data > 0)} voxels
Covered Lesion: {np.sum((create_sphere_mask(env.sphere_positions, env.sphere_radius, env.mri_data.shape) > 0) & (env.lesion_data > 0))} voxels

CLINICAL ASSESSMENT:
Coverage Quality: {'Excellent' if coverage_percentage > 80 else 'Good' if coverage_percentage > 60 else 'Adequate' if coverage_percentage > 40 else 'Poor'}
Dice Quality: {'Excellent' if final_dice > 0.8 else 'Good' if final_dice > 0.6 else 'Adequate' if final_dice > 0.4 else 'Poor'}
"""
    
    axes[1, 1].text(0.05, 0.95, performance_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle(f'Detailed Analysis: Patient {patient_id} Sphere Placement Results', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_final_comprehensive_evaluation_2d_only(env, save_folder, patient_id, final_dice_score, create_3d=False):
    """
    FIXED: Create comprehensive final evaluation with only improved 2D visualization (multiview removed)
    
    Args:
        env: Environment containing the sphere placement data
        save_folder: Folder to save all visualizations
        patient_id: Patient identifier
        final_dice_score: Final dice score for this patient
        create_3d: Whether to create 3D visualization (computationally expensive)
    
    Returns:
        dict: Dictionary with paths to all created visualizations and results summary
    """
    print(f"\nCreating comprehensive final evaluation for Patient {patient_id}...")
    
    # Calculate additional metrics
    sphere_mask = create_sphere_mask(env.sphere_positions, env.sphere_radius, env.mri_data.shape)
    final_dice = calculate_dice_score(sphere_mask, env.lesion_data)
    
    # Lesion coverage analysis
    lesion_volume = np.sum(env.lesion_data > 0)
    covered_volume = np.sum((sphere_mask > 0) & (env.lesion_data > 0))
    coverage_percentage = (covered_volume / lesion_volume * 100) if lesion_volume > 0 else 0
    
    # Sphere placement analysis
    sphere_volumes = []
    for i, pos in enumerate(env.sphere_positions):
        # Calculate volume of each sphere within prostate
        x, y, z = pos
        sphere_coords = []
        radius = env.sphere_radius
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx*dx + dy*dy + dz*dz <= radius*radius:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < env.mask_data.shape[0] and 
                            0 <= ny < env.mask_data.shape[1] and 
                            0 <= nz < env.mask_data.shape[2]):
                            if env.mask_data[nx, ny, nz] > 0:  # Within prostate
                                sphere_coords.append((nx, ny, nz))
        
        sphere_volumes.append(len(sphere_coords))
    
    step_info = f"Final (Dice: {final_dice:.3f}, Coverage: {coverage_percentage:.1f}%)"
    
    # Create enhanced 2D visualization with optimal slice selection
    enhanced_2d_path = os.path.join(save_folder, f'final_enhanced_2d_patient_{patient_id}.png')
    enhanced_visualize_spheres_with_numbers(
        env=env,
        slice_idx=None,  # Use automatic optimal slice selection
        save_path=enhanced_2d_path,
        show=False,
        step_info=step_info
    )
    
    # Optionally create 3D volume rendering
    volume_3d_path = None
    if create_3d:
        volume_3d_path = os.path.join(save_folder, f'final_volume_3d_patient_{patient_id}.png')
        try:
            visualize_3d_volume_rendering(
                env=env,
                save_path=volume_3d_path,
                show=False,
                step_info=step_info
            )
            print(f"  ✓ 3D volume rendering: {volume_3d_path}")
        except Exception as e:
            print(f"  ⚠ 3D visualization failed (optional): {e}")
            volume_3d_path = None
    
    # Create detailed analysis plot
    analysis_path = os.path.join(save_folder, f'final_analysis_patient_{patient_id}.png')
    create_detailed_analysis_plot(env, analysis_path, patient_id, final_dice, coverage_percentage, sphere_volumes)
    
    # Create sphere progression summary
    progression_path = os.path.join(save_folder, f'sphere_progression_patient_{patient_id}.png')
    create_sphere_progression_summary(env, save_folder, patient_id)
    
    print(f"  ✓ Enhanced 2D visualization: {enhanced_2d_path}")
    print(f"  ✓ Detailed analysis plot: {analysis_path}")
    print(f"  ✓ Progression summary: {progression_path}")
    
    # Save quantitative results
    results_summary = {
        'patient_id': patient_id,
        'final_dice_score': final_dice,
        'lesion_coverage_percentage': coverage_percentage,
        'total_lesion_volume': int(lesion_volume),
        'covered_lesion_volume': int(covered_volume),
        'sphere_positions': [list(map(int, pos)) for pos in env.sphere_positions],
        'sphere_volumes': sphere_volumes,
        'total_spheres': len(env.sphere_positions)
    }
    
    # Save as JSON for easy analysis
    json_path = os.path.join(save_folder, f'final_results_patient_{patient_id}.json')
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"  ✓ Quantitative results: {json_path}")
    
    return {
        'enhanced_2d_path': enhanced_2d_path,
        'volume_3d_path': volume_3d_path,
        'analysis_path': analysis_path,
        'progression_path': progression_path,
        'json_path': json_path,
        'results': results_summary
    }


# ============================================================================
# ENHANCED EVALUATION FUNCTIONS
# ============================================================================

def run_enhanced_evaluation_loop(model, eval_envs, results_folder, create_3d=False):
    """
    FIXED: Enhanced evaluation loop with improved 2D visualization (multiview removed)
    
    Args:
        model: Trained RL model
        eval_envs: List of evaluation environments
        results_folder: Folder to save results
        create_3d: Whether to create 3D visualizations (computationally expensive)
    
    Returns:
        tuple: (final_rewards, final_dice_scores, comprehensive_results)
    """
    print(f"\nRunning enhanced evaluation with improved 2D visualization...")
    print(f"3D visualization: {'Enabled' if create_3d else 'Disabled (for speed)'}")
    
    final_rewards = []
    final_dice_scores = []
    comprehensive_results = []
    
    for i, eval_env in enumerate(eval_envs):
        print(f"\nEvaluating Patient {i+1}/{len(eval_envs)} with improved 2D analysis...")
        
        # Create individual patient folder
        patient_folder = os.path.join(results_folder, f'patient_{i+1}_analysis')
        os.makedirs(patient_folder, exist_ok=True)
        
        # Run evaluation
        obs = eval_env.reset()
        done = False
        total_reward = 0
        step_rewards = []
        step_dice_scores = []
        step_num = 0
        
        # Initial state visualization
        print(f"  Creating initial visualizations...")
        enhanced_path, volume_3d_path = visualize_individual_step_placement(
            eval_env, step_num, patient_folder, i+1, dice_score=0.0, create_3d=create_3d
        )
        
        # Calculate initial dice score
        initial_sphere_mask = create_sphere_mask([], eval_env.sphere_radius, eval_env.mri_data.shape)
        initial_dice = calculate_dice_score(initial_sphere_mask, eval_env.lesion_data)
        step_dice_scores.append(initial_dice)
        
        # Step-by-step evaluation with improved 2D visualization
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
            step_rewards.append(reward)
            step_num += 1
            
            # Calculate dice score after this placement
            sphere_mask = create_sphere_mask(eval_env.sphere_positions, 
                                           eval_env.sphere_radius, 
                                           eval_env.mri_data.shape)
            dice_score = calculate_dice_score(sphere_mask, eval_env.lesion_data)
            step_dice_scores.append(dice_score)
            
            # Create improved 2D visualization for this step
            print(f"    Step {step_num}: Reward = {reward:.2f}, Dice = {dice_score:.3f}")
            enhanced_path, volume_3d_path = visualize_individual_step_placement(
                eval_env, step_num, patient_folder, i+1, dice_score, create_3d=create_3d
            )
        
        # Create comprehensive final evaluation (without multiview)
        final_results = create_final_comprehensive_evaluation_2d_only(
            eval_env, patient_folder, i+1, step_dice_scores[-1], create_3d=create_3d
        )
        
        # Store results
        final_rewards.append(total_reward)
        final_dice_scores.append(step_dice_scores[-1])
        comprehensive_results.append(final_results)
        
        # Create step-by-step progress plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot step rewards
        ax1.plot(range(1, len(step_rewards) + 1), step_rewards, 'ro-', linewidth=2, markersize=6)
        ax1.set_title(f'Step-by-Step Rewards - Patient {i+1}', fontsize=16)
        ax1.set_xlabel('Step', fontsize=14)
        ax1.set_ylabel('Reward', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot step dice scores
        ax2.plot(range(len(step_dice_scores)), step_dice_scores, 'go-', linewidth=2, markersize=6)
        ax2.set_title(f'Step-by-Step Dice Scores - Patient {i+1}', fontsize=16)
        ax2.set_xlabel('Step', fontsize=14)
        ax2.set_ylabel('Dice Score', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        step_progress_path = os.path.join(patient_folder, f'step_progress_patient_{i+1}.png')
        plt.savefig(step_progress_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Final Results - Reward: {total_reward:.2f}, Dice: {step_dice_scores[-1]:.3f}")
        print(f"    Analysis saved in: {patient_folder}")
    
    return final_rewards, final_dice_scores, comprehensive_results


# ============================================================================
# TRAINING CALLBACK (Your original, unchanged)
# ============================================================================

class TrainingCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.timesteps = []
        self.dice_scores = []
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation
            obs = self.eval_env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                total_reward += reward
            
            # Calculate Dice score
            sphere_mask = create_sphere_mask(self.eval_env.sphere_positions, 
                                           self.eval_env.sphere_radius, 
                                           self.eval_env.mri_data.shape)
            dice_score = calculate_dice_score(sphere_mask, self.eval_env.lesion_data)
            
            self.episode_rewards.append(total_reward)
            self.dice_scores.append(dice_score)
            self.timesteps.append(self.num_timesteps)
            
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Evaluation reward = {total_reward:.2f}, Dice score = {dice_score:.3f}")
        
        return True


# ============================================================================
# MAIN FUNCTION (Enhanced with new visualization capabilities)
# ============================================================================

def main():
    print("Starting Enhanced Reinforcement Learning Training with Improved 2D Visualization...")
    
    # Create results folder
    current_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = f"./enhanced_2d_results_{current_time}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Directory containing patient data
    directory = "./image_with_masks"
    
    # Get a list of patient folders
    patient_folders = [
        os.path.join(directory, folder)
        for folder in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, folder))
    ]
    
    # Filter out patients without the required files
    valid_patients = []
    for folder in patient_folders:
        if (os.path.exists(os.path.join(folder, "t2.nii.gz")) and
            os.path.exists(os.path.join(folder, "gland.nii.gz")) and
            os.path.exists(os.path.join(folder, "l_a1.nii.gz"))):
            valid_patients.append(folder)
    
    print(f"Found {len(valid_patients)} valid patients")
    
    if len(valid_patients) < 20:
        print(f"Warning: Only {len(valid_patients)} patients available, need at least 20 for 10 train + 10 eval")
        return
    
    # Split patients into training (first 10) and evaluation (next 10) sets
    train_patients = valid_patients[:10]
    eval_patients = valid_patients[10:20]
    
    print("Loading and preprocessing training data...")
    train_envs = []
    for i, patient_dir in enumerate(train_patients):
        patient_name = os.path.basename(patient_dir)
        print(f"Loading training patient {i+1}/10: {patient_name}")
        
        mri_data, mask_data, lesion_data = load_and_preprocess_patient_data(patient_dir)
        
        # Create environment
        try:
            env = SpherePlacementEnv(mri_data, mask_data, lesion_data)
            train_envs.append(env)
        except ValueError as e:
            print(f"Skipping patient {patient_name}: {e}")
    
    print("Loading and preprocessing evaluation data...")
    eval_envs = []
    for i, patient_dir in enumerate(eval_patients):
        patient_name = os.path.basename(patient_dir)
        print(f"Loading evaluation patient {i+1}/10: {patient_name}")
        
        mri_data, mask_data, lesion_data = load_and_preprocess_patient_data(patient_dir)
        
        # Create environment
        try:
            env = SpherePlacementEnv(mri_data, mask_data, lesion_data)
            eval_envs.append(env)
        except ValueError as e:
            print(f"Skipping patient {patient_name}: {e}")
    
    if not train_envs or not eval_envs:
        print("Error: No valid environments created!")
        return
    
    print(f"Created {len(train_envs)} training environments and {len(eval_envs)} evaluation environments")
    
    # Use first training environment for training
    train_env = train_envs[0]
    eval_env = eval_envs[0]  # Use first evaluation environment
    
    # Training parameters
    n_trials = 500
    steps_per_trial = 1000
    total_timesteps = n_trials * steps_per_trial
    
    print(f"Training setup: {n_trials} trials × {steps_per_trial} steps = {total_timesteps} total steps")
    
    # Initialize PPO model
    model = PPO("MlpPolicy", train_env, verbose=1, n_steps=512)
    
    # Create callback for evaluation
    callback = TrainingCallback(eval_env, eval_freq=steps_per_trial, verbose=1)
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save the trained model
    model_path = os.path.join(results_folder, "trained_model")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training progress
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    ax1.plot(callback.timesteps, callback.episode_rewards, 'b-o', linewidth=2, markersize=4)
    ax1.set_title('Enhanced Training Progress: Evaluation Rewards Over Time', fontsize=16)
    ax1.set_xlabel('Training Steps', fontsize=14)
    ax1.set_ylabel('Evaluation Reward', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot Dice scores
    ax2.plot(callback.timesteps, callback.dice_scores, 'r-o', linewidth=2, markersize=4)
    ax2.set_title('Enhanced Training Progress: Dice Scores Over Time', fontsize=16)
    ax2.set_xlabel('Training Steps', fontsize=14)
    ax2.set_ylabel('Dice Score', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    progress_path = os.path.join(results_folder, 'training_progress.png')
    plt.savefig(progress_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved: {progress_path}")
    
    # Enhanced evaluation with improved 2D visualization
    print("Running enhanced evaluation with improved 2D visualization...")
    
    # For faster execution, disable 3D by default (can be enabled by changing to True)
    create_3d_visualizations = False  # Set to True if you want 3D visualizations
    
    final_rewards, final_dice_scores, comprehensive_results = run_enhanced_evaluation_loop(
        model, eval_envs, results_folder, create_3d=create_3d_visualizations
    )
    
    # Summary statistics
    mean_reward = np.mean(final_rewards)
    std_reward = np.std(final_rewards)
    mean_dice = np.mean(final_dice_scores)
    std_dice = np.std(final_dice_scores)
    
    print(f"\nENHANCED EVALUATION RESULTS:")
    print(f"Mean reward across all environments: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Dice score across all environments: {mean_dice:.3f} ± {std_dice:.3f}")
    print(f"Individual rewards: {final_rewards}")
    print(f"Individual Dice scores: {[f'{score:.3f}' for score in final_dice_scores]}")
    
    # Plot final results summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot final rewards
    bars1 = ax1.bar(range(1, len(final_rewards) + 1), final_rewards, color='skyblue', edgecolor='navy', linewidth=1.5)
    ax1.axhline(y=mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
    ax1.set_title('Enhanced Final Evaluation Rewards Across All Environments', fontsize=14)
    ax1.set_xlabel('Environment Number', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(final_rewards),
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot final Dice scores
    bars2 = ax2.bar(range(1, len(final_dice_scores) + 1), final_dice_scores, color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
    ax2.axhline(y=mean_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dice:.3f}')
    ax2.set_title('Enhanced Final Dice Scores Across All Environments', fontsize=14)
    ax2.set_xlabel('Environment Number', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(final_dice_scores),
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    final_summary_path = os.path.join(results_folder, 'enhanced_2d_final_evaluation_summary.png')
    plt.savefig(final_summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Enhanced 2D final evaluation summary saved: {final_summary_path}")
    
    # Save comprehensive numerical results
    results_data = {
        'training_timesteps': callback.timesteps,
        'training_rewards': callback.episode_rewards,
        'training_dice_scores': callback.dice_scores,
        'final_rewards': final_rewards,
        'final_dice_scores': final_dice_scores,
        'mean_final_reward': mean_reward,
        'std_final_reward': std_reward,
        'mean_final_dice': mean_dice,
        'std_final_dice': std_dice,
        'training_time': training_time,
        'comprehensive_results': [result['results'] for result in comprehensive_results],
        'visualization_types_created': ['enhanced_2d', 'detailed_analysis', 'progression_summary'] + (['3d_volume'] if create_3d_visualizations else [])
    }
    
    np.savez(os.path.join(results_folder, 'enhanced_2d_results_data.npz'), **results_data)
    
    # Save summary report
    with open(os.path.join(results_folder, 'enhanced_2d_summary.txt'), 'w') as f:
        f.write("ENHANCED REINFORCEMENT LEARNING WITH IMPROVED 2D VISUALIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training completed: {current_time}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Total timesteps: {total_timesteps:,}\n")
        f.write(f"Visualization types created: Enhanced 2D, Detailed Analysis, Progression Summary")
        if create_3d_visualizations:
            f.write(", 3D Volume Rendering")
        f.write("\n\n")
        f.write(f"Final Evaluation Results:\n")
        f.write(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"Mean Dice score: {mean_dice:.3f} ± {std_dice:.3f}\n\n")
        f.write(f"Individual Patient Results:\n")
        for i, (reward, dice, comp_result) in enumerate(zip(final_rewards, final_dice_scores, comprehensive_results)):
            f.write(f"Patient {i+1}: Reward={reward:.2f}, Dice={dice:.3f}, Coverage={comp_result['results']['lesion_coverage_percentage']:.1f}%\n")
    
    print(f"\nEnhanced training and evaluation complete!")
    print(f"All results saved in folder: {results_folder}")
    print(f"Features created:")
    print(f"  ✓ Enhanced 2D visualizations (improved slice selection for lesion visibility)")
    print(f"  ✓ Detailed quantitative analysis plots")
    print(f"  ✓ Sphere placement progression summaries")
    print(f"  ✓ Comprehensive JSON results for each patient")
    if create_3d_visualizations:
        print(f"  ✓ 3D volume renderings")
    else:
        print(f"  ⚪ 3D volume renderings (disabled for speed - set create_3d_visualizations=True to enable)")


if __name__ == "__main__":
    main()