"""
Standalone Evaluation Script for Trained RL Models
================================================

This script loads a pre-trained RL model and evaluates it on test patients,
creating grids of projection visualizations (dots and ablation).

Usage:
    python standalone_evaluation.py

Results saved in: ./standalone_evaluation_results_TIMESTAMP/
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from matplotlib.gridspec import GridSpec
import tempfile
from mpl_toolkits.mplot3d import Axes3D

# Import necessary functions
from main_env import get_max_action_space_size, create_standardized_environments
from main_agent import (
    load_and_preprocess_patient_data,
    create_sphere_mask,
    calculate_dice_score,
    enhanced_visualize_spheres_with_numbers
)


def get_filtered_dataset(filtered_dir="./filtered_dataset"):
    """Get all patient folders from the filtered dataset."""
    if not os.path.exists(filtered_dir):
        # Fallback to original dataset
        filtered_dir = "./image_with_masks"
        if not os.path.exists(filtered_dir):
            raise FileNotFoundError(f"Dataset not found at {filtered_dir}")
    
    patient_folders = [
        os.path.join(filtered_dir, folder)
        for folder in os.listdir(filtered_dir)
        if os.path.isdir(os.path.join(filtered_dir, folder)) and not folder.startswith('.')
    ]
    
    # Filter out patients without the required files
    valid_patients = []
    for folder in patient_folders:
        if (os.path.exists(os.path.join(folder, "t2.nii.gz")) and
            os.path.exists(os.path.join(folder, "gland.nii.gz")) and
            os.path.exists(os.path.join(folder, "l_a1.nii.gz"))):
            valid_patients.append(folder)
    
    return valid_patients


def visualize_placement_projection_dots(env, save_path=None, show=True, step_info="", projection_axis='axial'):
    from mpl_toolkits.mplot3d import Axes3D
    from skimage import measure
    
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    prostate_mask = env.mask_data > 0
    lesion_mask = env.lesion_data > 0
    
    if np.any(prostate_mask):
        try:
            verts_p, faces_p, _, _ = measure.marching_cubes(prostate_mask.astype(float), level=0.5)
            ax.plot_trisurf(verts_p[:, 0], verts_p[:, 1], verts_p[:, 2], 
                           triangles=faces_p, color=[1, 0.4, 0.5], alpha=0.3, 
                           linewidth=0, shade=True)
        except:
            pass
    
    if np.any(lesion_mask):
        try:
            verts_l, faces_l, _, _ = measure.marching_cubes(lesion_mask.astype(float), level=0.5)
            ax.plot_trisurf(verts_l[:, 0], verts_l[:, 1], verts_l[:, 2], 
                           triangles=faces_l, color=[0.4, 1, 0.8], alpha=0.8, 
                           linewidth=0, shade=True)
        except:
            pass
    
    # Calculate bounds for zooming
    if np.any(prostate_mask):
        prostate_coords = np.where(prostate_mask)
        min_x, max_x = np.min(prostate_coords[0]), np.max(prostate_coords[0])
        min_y, max_y = np.min(prostate_coords[1]), np.max(prostate_coords[1])
        min_z, max_z = np.min(prostate_coords[2]), np.max(prostate_coords[2])
        
        # Add padding
        padding = 5
        min_x = max(0, min_x - padding)
        max_x = min(env.mri_data.shape[0], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(env.mri_data.shape[1], max_y + padding)
        min_z = max(0, min_z - padding)
        max_z = min(env.mri_data.shape[2], max_z + padding)
    else:
        min_x, max_x = 0, env.mri_data.shape[0]
        min_y, max_y = 0, env.mri_data.shape[1]
        min_z, max_z = 0, env.mri_data.shape[2]
    
    for i, sphere_pos in enumerate(env.sphere_positions):
        x, y, z = sphere_pos
        
        top_z = max_z + 2  
        ax.scatter(x, y, top_z, c='black', s=250, marker='o', edgecolors='white', 
                linewidth=5, zorder=200, alpha=1.0)
    
    
        ax.plot([x, x], [y, y], [z, top_z], 'k-', linewidth=8, alpha=0.4, zorder=150)
    
    # Zoom in on the data
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    
    ax.view_init(elev=70, azim=-0.3)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_facecolor('white')
    
   
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    ax.set_axis_off()
    
    try:
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Remove tick lines
        ax.xaxis.set_tick_params(which='both', length=0)
        ax.yaxis.set_tick_params(which='both', length=0) 
        ax.zaxis.set_tick_params(which='both', length=0)
        
        # Make axis lines transparent
        ax.xaxis.line.set_linewidth(0)
        ax.yaxis.line.set_linewidth(0)
        ax.zaxis.line.set_linewidth(0)
    except:
        pass  

    if env.sphere_positions:
        sphere_mask = create_sphere_mask(env.sphere_positions, env.sphere_radius, env.mri_data.shape)
        dice_score = calculate_dice_score(sphere_mask, env.lesion_data)
    else:
        dice_score = 0.0
    
    placement_count = len(env.sphere_positions)
    ax.set_title(f'Dice: {dice_score:.3f}', 
                fontsize=30, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_placement_projection_ablation(env, save_path=None, show=True, step_info="", projection_axis='axial'):
    from mpl_toolkits.mplot3d import Axes3D
    from skimage import measure
    
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    prostate_mask = env.mask_data > 0
    lesion_mask = env.lesion_data > 0
    
    if np.any(prostate_mask):
        try:
            verts_p, faces_p, _, _ = measure.marching_cubes(prostate_mask.astype(float), level=0.5)
            ax.plot_trisurf(verts_p[:, 0], verts_p[:, 1], verts_p[:, 2], 
                           triangles=faces_p, color=[1, 0.4, 0.5], alpha=0.3, 
                           linewidth=0, shade=True)
        except:
            pass
    
    if np.any(lesion_mask):
        try:
            verts_l, faces_l, _, _ = measure.marching_cubes(lesion_mask.astype(float), level=0.5)
            ax.plot_trisurf(verts_l[:, 0], verts_l[:, 1], verts_l[:, 2], 
                           triangles=faces_l, color=[0.4, 1, 0.8], alpha=0.8, 
                           linewidth=0, shade=True)
        except:
            pass
    
    if env.sphere_positions:
        sphere_mask = create_sphere_mask(env.sphere_positions, env.sphere_radius, env.mri_data.shape)
        ablation_mask = sphere_mask > 0
        
        if np.any(ablation_mask):
            try:
                verts_a, faces_a, _, _ = measure.marching_cubes(ablation_mask.astype(float), level=0.5)
                ax.plot_trisurf(verts_a[:, 0], verts_a[:, 1], verts_a[:, 2], 
                               triangles=faces_a, color=[0.3, 0.6, 0.9], alpha=0.5, 
                               linewidth=0, shade=True)
            except:
                pass
    
   
    if np.any(prostate_mask):
        prostate_coords = np.where(prostate_mask)
        min_x, max_x = np.min(prostate_coords[0]), np.max(prostate_coords[0])
        min_y, max_y = np.min(prostate_coords[1]), np.max(prostate_coords[1])
        min_z, max_z = np.min(prostate_coords[2]), np.max(prostate_coords[2])
        
        padding = 5
        min_x = max(0, min_x - padding)
        max_x = min(env.mri_data.shape[0], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(env.mri_data.shape[1], max_y + padding)
        min_z = max(0, min_z - padding)
        max_z = min(env.mri_data.shape[2], max_z + padding)
    else:
        min_x, max_x = 0, env.mri_data.shape[0]
        min_y, max_y = 0, env.mri_data.shape[1]
        min_z, max_z = 0, env.mri_data.shape[2]
    
    for i, sphere_pos in enumerate(env.sphere_positions):
        x, y, z = sphere_pos
        
        top_z = max_z + 2  
        ax.scatter(x, y, top_z, c='black', s=250, marker='o', edgecolors='white', 
                linewidth=5, zorder=200, alpha=1.0)
        ax.plot([x, x], [y, y], [z, top_z], 'k-', linewidth=8, alpha=0.4, zorder=150)
    

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    
    ax.view_init(elev=70, azim=-0.3)
    

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_facecolor('white')
   
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
 
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
   
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
  
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    
 
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    
    ax.set_axis_off()
    
    # Remove any remaining axis elements (with error handling)
    try:
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Remove tick lines
        ax.xaxis.set_tick_params(which='both', length=0)
        ax.yaxis.set_tick_params(which='both', length=0) 
        ax.zaxis.set_tick_params(which='both', length=0)
        
        # Make axis lines transparent
        ax.xaxis.line.set_linewidth(0)
        ax.yaxis.line.set_linewidth(0)
        ax.zaxis.line.set_linewidth(0)
    except:
        pass  # Some matplotlib versions may not have these attributes
    
    if env.sphere_positions:
        sphere_mask = create_sphere_mask(env.sphere_positions, env.sphere_radius, env.mri_data.shape)
        dice_score = calculate_dice_score(sphere_mask, env.lesion_data)
        lesion_volume = np.sum(env.lesion_data > 0)
        covered_volume = np.sum((sphere_mask > 0) & (env.lesion_data > 0))
        coverage_percentage = (covered_volume / lesion_volume * 100) if lesion_volume > 0 else 0
    else:
        dice_score = 0.0
        coverage_percentage = 0.0
    
    placement_count = len(env.sphere_positions)
    ax.set_title(f'Dice: {dice_score:.3f}', 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def evaluate_single_patient(model, env, patient_id):
    """Evaluate model on a single patient and return results."""
    obs = env.reset()
    done = False
    total_reward = 0
    step_num = 0
    
    print(f"  Evaluating Patient {patient_id}...")
    
    # Step-by-step evaluation
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_num += 1
        
        print(f"    Step {step_num}: Reward = {reward:.2f}")
    
    # Calculate final metrics
    sphere_mask = create_sphere_mask(env.sphere_positions, env.sphere_radius, env.mri_data.shape)
    dice_score = calculate_dice_score(sphere_mask, env.lesion_data)
    
    # Calculate coverage
    lesion_volume = np.sum(env.lesion_data > 0)
    covered_volume = np.sum((sphere_mask > 0) & (env.lesion_data > 0))
    coverage_percentage = (covered_volume / lesion_volume * 100) if lesion_volume > 0 else 0
    
    print(f"    Final Results - Reward: {total_reward:.2f}, Dice: {dice_score:.3f}, Coverage: {coverage_percentage:.1f}%")
    
    return {
        'env': env,
        'total_reward': total_reward,
        'dice_score': dice_score,
        'coverage_percentage': coverage_percentage,
        'sphere_positions': env.sphere_positions.copy()
    }


def create_projection_grid(results_list, save_folder, visualization_type="", title="Projection Grid"):
    """Create a grid of projection visualizations by first creating individual images."""
    import matplotlib.image as mpimg
    from PIL import Image
    import tempfile
    
    # Select 4 specific patients: 2, 3, 9, 12 (convert to 0-indexed: 1, 2, 8, 11)
    target_patients = [1, 2, 8, 11]  # 0-indexed positions for patients 2, 3, 9, 12
    selected_indices = []
    selected_results = []
    total_patients = len(results_list)
    
    # Add available target patients
    for patient_idx in target_patients:
        if patient_idx < total_patients:
            selected_indices.append(patient_idx)
            selected_results.append(results_list[patient_idx])
    
    # If we don't have enough target patients, fill with first available patients
    if len(selected_indices) < 4:
        print(f"Warning: Only {len(selected_indices)} of target patients available, filling with first patients")
        for i in range(min(4, total_patients)):
            if i not in selected_indices:
                selected_indices.append(i)
                selected_results.append(results_list[i])
                if len(selected_indices) >= 4:
                    break
    
    # Limit to 4 patients maximum
    selected_indices = selected_indices[:4]
    selected_results = selected_results[:4]
    
    print(f"Selected specific patients: {[i+1 for i in selected_indices]}")
    
    # Fixed grid dimensions for 4 patients - 2 rows x 4 columns (projection + masks)
    rows, cols = 2, 4
    
    # SIZE CONTROL: Larger sizes for better visibility
    individual_subplot_size = 10  
    mask_figure_size = 32        
    sphere_marker_size = 3000    
    
    # Create temporary directory for individual images
    temp_dir = tempfile.mkdtemp()
    temp_projection_images = []
    temp_mask_images = []
    
    print(f"Creating individual {visualization_type} projections and zoomed mask views for 4-patient grid...")
    
    # Generate individual projection images
    for i, result in enumerate(selected_results):
        env = result['env']
        dice_score = result['dice_score']
        coverage = result['coverage_percentage']
        
        step_info = f"(Dice: {dice_score:.3f}, Cov: {coverage:.1f}%)"
        
        # Create projection image
        temp_proj_path = os.path.join(temp_dir, f'{visualization_type}_{i}.png')
        
        try:
            if visualization_type == "dots":
                visualize_placement_projection_dots(
                    env=env,
                    save_path=temp_proj_path,
                    show=False,
                    step_info=step_info
                )
            elif visualization_type == "ablation":
                visualize_placement_projection_ablation(
                    env=env,
                    save_path=temp_proj_path,
                    show=False,
                    step_info=step_info
                )
            
            temp_projection_images.append(temp_proj_path)
            print(f"  ✓ Created {visualization_type} for Patient {selected_indices[i]+1}")
            
        except Exception as e:
            print(f"  ✗ Error creating {visualization_type} for Patient {selected_indices[i]+1}: {e}")
            temp_projection_images.append(None)
        
        
        temp_mask_path = os.path.join(temp_dir, f'masks_{i}.png')
        
        try:
            
            slice_idx = None  
            
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
                    print(f"  Patient {selected_indices[i]+1}: Auto-selected slice {slice_idx} - optimized for lesion visibility and sphere proximity")
                    
                elif lesion_slices_with_content:
                    
                    slice_idx = max(lesion_slices_with_content, key=lambda x: x[1])[0]
                    print(f"  Patient {selected_indices[i]+1}: Auto-selected slice {slice_idx} - based on maximum lesion content")
                    
                elif env.sphere_positions:
                    
                    z_positions = [pos[2] for pos in env.sphere_positions]
                    slice_idx = int(np.median(z_positions))
                    print(f"  Patient {selected_indices[i]+1}: Auto-selected slice {slice_idx} - based on sphere positions (no lesion visible)")
                    
                else:
                   
                    slice_idx = env.mri_data.shape[2] // 2
                    print(f"  Patient {selected_indices[i]+1}: Auto-selected slice {slice_idx} - using center slice (no spheres or lesions)")
            
           
            slice_idx = max(0, min(slice_idx, env.mri_data.shape[2] - 1))
            
            
            fig_mask, ax_mask = plt.subplots(1, 1, figsize=(mask_figure_size, mask_figure_size))
            
            
            mri_slice = env.mri_data[:, :, slice_idx]
            mask_slice = env.mask_data[:, :, slice_idx].astype(float)
            lesion_slice = env.lesion_data[:, :, slice_idx].astype(float)
            
          
            if np.any(mask_slice > 0):
                mask_coords = np.where(mask_slice > 0)
                min_row, max_row = np.min(mask_coords[0]), np.max(mask_coords[0])
                min_col, max_col = np.min(mask_coords[1]), np.max(mask_coords[1])
                
                
                padding = 15  
                min_row = max(0, min_row - padding)
                max_row = min(mri_slice.shape[0], max_row + padding)
                min_col = max(0, min_col - padding)
                max_col = min(mri_slice.shape[1], max_col + padding)
                
                
                mri_slice_cropped = mri_slice[min_row:max_row, min_col:max_col]
                mask_slice_cropped = mask_slice[min_row:max_row, min_col:max_col]
                lesion_slice_cropped = lesion_slice[min_row:max_row, min_col:max_col]
                
                print(f"  Patient {selected_indices[i]+1}: Cropped from {mri_slice.shape} to {mri_slice_cropped.shape}")
                
            else:
                
                mri_slice_cropped = mri_slice
                mask_slice_cropped = mask_slice
                lesion_slice_cropped = lesion_slice
                min_row, min_col = 0, 0
                print(f"  Patient {selected_indices[i]+1}: No mask found, using full slice")
            
            
            mask_unique = np.unique(mask_slice_cropped)
            lesion_unique = np.unique(lesion_slice_cropped)
            print(f"  Patient {selected_indices[i]+1}: Mask values: {mask_unique}, Lesion values: {lesion_unique}")
            
           
            if np.max(mri_slice_cropped) > np.min(mri_slice_cropped):
                mri_normalized = (mri_slice_cropped - np.min(mri_slice_cropped)) / (np.max(mri_slice_cropped) - np.min(mri_slice_cropped))
            else:
                mri_normalized = mri_slice_cropped
            
            if np.max(mask_slice_cropped) > 0:
                mask_slice_cropped = mask_slice_cropped / np.max(mask_slice_cropped)
            if np.max(lesion_slice_cropped) > 0:
                lesion_slice_cropped = lesion_slice_cropped / np.max(lesion_slice_cropped)
            
            # Display MRI as background in grayscale
            ax_mask.imshow(mri_normalized, cmap='gray', vmin=0, vmax=1)
            
            # Overlay masks on top of MRI
            # Create masked arrays - only mask where values are very small
            mask_masked = np.ma.masked_where(mask_slice_cropped <= 0.01, mask_slice_cropped)
            lesion_masked = np.ma.masked_where(lesion_slice_cropped <= 0.01, lesion_slice_cropped)
            
            # Show masks overlaid on MRI with enhanced visibility
            if np.max(mask_slice_cropped) > 0:
                ax_mask.imshow(mask_masked, cmap='Blues', alpha=0.4, vmin=0, vmax=1)  # Semi-transparent blue for prostate
            else:
                print(f"  Patient {selected_indices[i]+1}: No prostate mask content on slice {slice_idx}")
                
            if np.max(lesion_slice_cropped) > 0:
                ax_mask.imshow(lesion_masked, cmap='Reds', alpha=0.6, vmin=0, vmax=1)  # More visible red for lesions
            else:
                print(f"  Patient {selected_indices[i]+1}: No lesion mask content on slice {slice_idx}")
            
            # Add debug info to track what's being displayed
            mask_pixels = np.sum(mask_slice_cropped > 0.01)
            lesion_pixels = np.sum(lesion_slice_cropped > 0.01)
            sphere_count = len([pos for pos in env.sphere_positions 
                              if abs(pos[2] - slice_idx) <= 4])
            
            print(f"  Patient {selected_indices[i]+1}: Slice {slice_idx}, "
                  f"Prostate pixels: {mask_pixels}, Lesion pixels: {lesion_pixels}, "
                  f"Visible spheres: {sphere_count}")
            
            # Add sphere markers - MUCH larger dots with better positioning for cropped view
            spheres_added = 0
            for j, sphere_pos in enumerate(env.sphere_positions):
                x, y, z = sphere_pos
                z_distance = abs(z - slice_idx)
                if z_distance <= 3:
                    alpha = max(0.8, 1.0 - (z_distance / 3.0))  # Higher minimum alpha for better visibility
                    
                    # Adjust coordinates for cropped view
                    x_cropped = x - min_row
                    y_cropped = y - min_col
                    
                    # Validate coordinates are within cropped bounds
                    if (0 <= x_cropped < mri_slice_cropped.shape[0] and 
                        0 <= y_cropped < mri_slice_cropped.shape[1]):
                        # Much larger sphere markers for zoomed view with yellow color for high contrast
                        ax_mask.scatter(y_cropped, x_cropped, s=sphere_marker_size, c='yellow', marker='o', 
                                       edgecolors='black', linewidth=10, alpha=alpha, zorder=10)
                        spheres_added += 1
            
            print(f"  Patient {selected_indices[i]+1}: Added {spheres_added} sphere markers to zoomed MRI view")
            
            # Fallback: If no masks are visible, show debug information
            if mask_pixels == 0 and lesion_pixels == 0:
                print(f"  WARNING: No mask content visible for Patient {selected_indices[i]+1}")
                print(f"    Original mask range: {np.min(env.mask_data)} to {np.max(env.mask_data)}")
                print(f"    Original lesion range: {np.min(env.lesion_data)} to {np.max(env.lesion_data)}")
                print(f"    Mask shape: {env.mask_data.shape}, Lesion shape: {env.lesion_data.shape}")
                
                # Show text indicating the issue with larger font
                ax_mask.text(0.5, 0.5, f'Patient {selected_indices[i]+1}\nNo mask data visible\nSlice: {slice_idx}', 
                           ha='center', va='center', transform=ax_mask.transAxes, 
                           fontsize=32, fontweight='bold', color='red',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
            
            # Remove axis and set background
            ax_mask.set_xticks([])
            ax_mask.set_yticks([])
            ax_mask.axis('off')
            ax_mask.set_facecolor('black')  # Black background for medical imaging
            
            # Set black background for figure (medical imaging standard)
            fig_mask.patch.set_facecolor('black')
            
            # Save with higher DPI for better quality in zoomed view
            plt.savefig(temp_mask_path, dpi=450, bbox_inches='tight', facecolor='black', pad_inches=0.01)
            plt.close(fig_mask)
            
            temp_mask_images.append(temp_mask_path)
            print(f"  ✓ Created zoomed MRI+mask view for Patient {selected_indices[i]+1}")
            
        except Exception as e:
            print(f"  ✗ Error creating zoomed MRI+mask view for Patient {selected_indices[i]+1}: {e}")
            temp_mask_images.append(None)
    
    # Create the main grid with larger size to accommodate bigger mask views
    fig = plt.figure(figsize=(cols * individual_subplot_size, rows * individual_subplot_size), facecolor='white')
    
    # Create custom grid layout with more space for the mask row
    gs = fig.add_gridspec(rows, cols, 
                         left=0.02, bottom=0.02, right=0.98, top=0.98,
                         wspace=0.03, hspace=0.03,  # Reduced spacing for more image area
                         height_ratios=[1, 1.2])    # Make mask row slightly taller
    
    # First row: projection images (dots or ablation)
    for i in range(cols):
        ax = fig.add_subplot(gs[0, i])
        
        if i < len(temp_projection_images) and temp_projection_images[i] is not None:
            try:
                img = mpimg.imread(temp_projection_images[i])
                ax.imshow(img)
            except Exception as e:
                print(f"Error loading projection image for Patient {selected_indices[i]+1}: {e}")
                ax.text(0.5, 0.5, f'Error\nPatient {selected_indices[i]+1}', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=24, fontweight='bold')  # Increased font size
        else:
            ax.set_visible(False)
        
        # Remove axis elements and set white background
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.set_facecolor('white')
        
        # Remove any spines
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Second row: ZOOMED MRI+mask views (larger and more prominent)
    for i in range(cols):
        ax = fig.add_subplot(gs[1, i])
        
        if i < len(temp_mask_images) and temp_mask_images[i] is not None:
            try:
                img = mpimg.imread(temp_mask_images[i])
                ax.imshow(img)
            except Exception as e:
                print(f"Error loading MRI+mask image for Patient {selected_indices[i]+1}: {e}")
                ax.text(0.5, 0.5, f'Error\nPatient {selected_indices[i]+1}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=24, fontweight='bold')  # Increased font size
        else:
            ax.set_visible(False)
        
        # Remove axis elements and set black background for medical imaging
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.set_facecolor('black')
        
        # Remove any spines
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Save the grid with higher DPI for better quality
    save_path = os.path.join(save_folder, f'projection_{visualization_type}_grid_4patients_mri_overlay.png')
    fig.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    plt.close(fig)
    
    # Clean up temporary files
    for temp_path in temp_projection_images + temp_mask_images:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    os.rmdir(temp_dir)
    
    print(f"4-patient MRI+mask overlay grid saved: {save_path}")
    return save_path

def create_individual_projections(results_list, save_folder):
    """Create individual projection visualizations for each patient."""
    print("\nCreating individual projection visualizations...")
    
    for i, result in enumerate(results_list):
        env = result['env']
        dice_score = result['dice_score']
        coverage = result['coverage_percentage']
        
        step_info = f"Final (Dice: {dice_score:.3f}, Cov: {coverage:.1f}%)"
        
        # Create dots projection
        dots_path = os.path.join(save_folder, f'projection_dots_patient_{i+1}.png')
        try:
            visualize_placement_projection_dots(
                env=env,
                save_path=dots_path,
                show=False,
                step_info=step_info
            )
            print(f"  ✓ Dots projection: Patient {i+1}")
        except Exception as e:
            print(f"  ✗ Error creating dots for Patient {i+1}: {e}")
        
        # Create ablation projection
        ablation_path = os.path.join(save_folder, f'projection_ablation_patient_{i+1}.png')
        try:
            visualize_placement_projection_ablation(
                env=env,
                save_path=ablation_path,
                show=False,
                step_info=step_info
            )
            print(f"  ✓ Ablation projection: Patient {i+1}")
        except Exception as e:
            print(f"  ✗ Error creating ablation for Patient {i+1}: {e}")

        #creating final 2d projection 
        final_projection_path = os.path.join(save_folder, f'final_projection_patient_{i+1}.png')
        try:
            enhanced_visualize_spheres_with_numbers(
                env=env,
                save_path=final_projection_path,
                show=False,
                step_info=step_info)
            print(f"  ✓ Final projection: Patient {i+1}")
        except Exception as e:
            print(f"  ✗ Error creating final projection for Patient {i+1}: {e}")




def main():
    """Main evaluation function."""
    print("=" * 80)
    print("STANDALONE MODEL EVALUATION WITH PROJECTION GRIDS")
    print("=" * 80)
    
    # Get experiment folder name from user
    # experiment_folder = input("Enter the experiment folder name (e.g., 'experiment_1_20250101-123456'): ").strip()
    
    experiment_folder = "./SPARSE_P1_FINAL_RUN_20250727-143240"
    if not os.path.exists(experiment_folder):
        print(f"Error: Experiment folder '{experiment_folder}' not found!")
        return
    
    # Check if model exists
    model_path = os.path.join(experiment_folder, "trained_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(experiment_folder, "trained_model")
        if not os.path.exists(model_path + ".zip"):
            print(f"Error: No trained model found in '{experiment_folder}'!")
            print("Looking for 'trained_model.zip' or 'trained_model'")
            return
    
    print(f"Found experiment folder: {experiment_folder}")
    print(f"Loading model from: {model_path}")
    
    # Create results folder
    current_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = f"./{experiment_folder}_REVAL_{current_time}"
    os.makedirs(results_folder, exist_ok=True)
    
    try:
        # Load dataset
        print("\nLoading dataset...")
        patient_folders = get_filtered_dataset()
        print(f"Found {len(patient_folders)} valid patients")
        
        # Create test split (using same logic as experiments)
        import random
        random.seed(42)  # Fixed seed for reproducibility
        
        shuffled_patients = patient_folders.copy()
        random.shuffle(shuffled_patients)
        
        # Use test patients (similar to experiment splits)
        num_patients = len(shuffled_patients)
        if num_patients >= 35:
            test_patients = shuffled_patients[15:30]  # Use 15 patients max
        elif num_patients >= 15:
            test_patients = shuffled_patients[10:15]
        else:
            test_patients = shuffled_patients[len(shuffled_patients)//2:]  # Use second half
        
        # Limit to reasonable number for visualization
        test_patients = test_patients[:15]  # Maximum 15 patients for grid
        
        print(f"Using {len(test_patients)} test patients for evaluation")
        
        # Create test environments
        print("\nCreating test environments...")
        max_action_space = get_max_action_space_size(test_patients)
        test_envs, _ = create_standardized_environments(test_patients, max_action_space)
        
        if not test_envs:
            print("Error: No valid test environments created!")
            return
        
        print(f"Successfully created {len(test_envs)} test environments")
        
        # Load the trained model
        print(f"\nLoading trained model...")
        try:
            model = PPO.load(model_path)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Run evaluation on all test patients
        print(f"\nRunning evaluation on {len(test_envs)} patients...")
        results = []
        
        for i, test_env in enumerate(test_envs):
            try:
                result = evaluate_single_patient(model, test_env, i+1)
                results.append(result)
            except Exception as e:
                print(f"  Error evaluating Patient {i+1}: {e}")
                continue
        
        if not results:
            print("Error: No successful evaluations!")
            return
        
        print(f"\nSuccessfully evaluated {len(results)} patients")
        
        # Calculate summary statistics
        dice_scores = [r['dice_score'] for r in results]
        coverages = [r['coverage_percentage'] for r in results]
        rewards = [r['total_reward'] for r in results]
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"  Mean Dice Score: {np.mean(dice_scores):.3f} ± {np.std(dice_scores):.3f}")
        print(f"  Mean Coverage: {np.mean(coverages):.1f}% ± {np.std(coverages):.1f}%")
        print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        
        # Create individual projection visualizations
        create_individual_projections(results, results_folder)
        
        # Create projection grids
        print(f"\nCreating projection grids...")
        
        # Create dots grid
        dots_grid_path = create_projection_grid(
            results, 
            results_folder, 
            visualization_type="dots",
            title=f"Projection Dots Grid - {os.path.basename(experiment_folder)}"
        )
        
        # Create ablation grid
        ablation_grid_path = create_projection_grid(
            results, 
            results_folder, 
            visualization_type="ablation",
            title=f"Projection Ablation Grid - {os.path.basename(experiment_folder)}"
        )
        
        # Save summary results
        summary_path = os.path.join(results_folder, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("STANDALONE MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Experiment folder: {experiment_folder}\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Evaluation time: {current_time}\n")
            f.write(f"Patients evaluated: {len(results)}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"  Mean Dice Score: {np.mean(dice_scores):.3f} ± {np.std(dice_scores):.3f}\n")
            f.write(f"  Mean Coverage: {np.mean(coverages):.1f}% ± {np.std(coverages):.1f}%\n")
            f.write(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n\n")
            
            f.write("INDIVIDUAL PATIENT RESULTS:\n")
            for i, result in enumerate(results):
                f.write(f"  Patient {i+1}: Dice={result['dice_score']:.3f}, "
                       f"Coverage={result['coverage_percentage']:.1f}%, "
                       f"Reward={result['total_reward']:.2f}\n")
        
        print(f"\n" + "=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results saved in: {results_folder}")
        print(f"Files created:")
        print(f"  ✓ {dots_grid_path}")
        print(f"  ✓ {ablation_grid_path}")
        print(f"  ✓ {summary_path}")
        print(f"  ✓ Individual projection images for each patient")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()