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
from matplotlib.patches import Circle

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
                           triangles=faces_p, color=[0.4, 1, 0.4], alpha=0.15, 
                           linewidth=0, shade=True)
        except:
            pass
    
    if np.any(lesion_mask):
        try:
            verts_l, faces_l, _, _ = measure.marching_cubes(lesion_mask.astype(float), level=0.5)
            ax.plot_trisurf(verts_l[:, 0], verts_l[:, 1], verts_l[:, 2], 
                           triangles=faces_l, color=[1, 0.4, 0.4], alpha=0.3, 
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
        ax.scatter(x, y, z, c='blue', s=250, marker='o', edgecolors='white', 
                linewidth=6, zorder=200, alpha=1.0, facecolors='blue')
    
    # Zoom in on the data
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    
    ax.view_init(elev=35, azim=45)
    
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
    
    # if np.any(prostate_mask):
    #     try:
    #         verts_p, faces_p, _, _ = measure.marching_cubes(prostate_mask.astype(float), level=0.5)
    #         ax.plot_trisurf(verts_p[:, 0], verts_p[:, 1], verts_p[:, 2], 
    #                        triangles=faces_p, color=[0.4, 1, 0.4], alpha=0.05, 
    #                        linewidth=0, shade=True)
    #     except:
    #         pass
    
    if np.any(lesion_mask):
        try:
            verts_l, faces_l, _, _ = measure.marching_cubes(lesion_mask.astype(float), level=0.5)
            ax.plot_trisurf(verts_l[:, 0], verts_l[:, 1], verts_l[:, 2], 
                           triangles=faces_l, color=[1, 0.4, 0.4], alpha=0.15, 
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
                               triangles=faces_a, color=[0.3, 0.6, 0.9], alpha=0.25, 
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
        ax.scatter(x, y, z, c='blue', s=250, marker='o', edgecolors='white', 
                linewidth=6, zorder=200, alpha=1.0, facecolors='blue')
    

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    
    ax.view_init(elev=35, azim=45)
    

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
        
        ax.xaxis.set_tick_params(which='both', length=0)
        ax.yaxis.set_tick_params(which='both', length=0) 
        ax.zaxis.set_tick_params(which='both', length=0)
        
        ax.xaxis.line.set_linewidth(0)
        ax.yaxis.line.set_linewidth(0)
        ax.zaxis.line.set_linewidth(0)
    except:
        pass
    
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
                fontsize=30, fontweight='bold')
    
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
    individual_subplot_size = 8  
    mask_figure_size = 20       
    sphere_marker_size = 1500    
    
    # Create temporary directory for individual images
    temp_dir = tempfile.mkdtemp()
    temp_projection_images = []
    temp_mask_images = []
    
    print(f"Creating individual {visualization_type} projections and MRI views for 4-patient grid...")
    
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
            # Find optimal slice that shows spheres
            slice_idx = None
            
            if env.sphere_positions:
                # Find slice that shows the most spheres
                sphere_z_positions = [pos[2] for pos in env.sphere_positions]
                
                # Try slices around sphere positions
                best_slice = None
                max_visible_spheres = 0
                
                # Check slices around each sphere position
                candidate_slices = set()
                for sphere_z in sphere_z_positions:
                    for offset in range(-3, 4):  # Check 7 slices around each sphere
                        candidate_z = sphere_z + offset
                        if 0 <= candidate_z < env.mri_data.shape[2]:
                            candidate_slices.add(candidate_z)
                
                for z in candidate_slices:
                    # Count visible spheres on this slice
                    visible_spheres = sum(1 for pos in env.sphere_positions if abs(pos[2] - z) <= 2)
                    
                    # Also check for lesion content (use original lesion data)
                    lesion_content = np.sum(env.lesion_data[:, :, z] > 0)
                    
                    # Score based on visible spheres and lesion content
                    score = visible_spheres * 100 + lesion_content
                    
                    if visible_spheres > 0 and score > max_visible_spheres:
                        max_visible_spheres = score
                        best_slice = z
                
                slice_idx = best_slice if best_slice is not None else int(np.median(sphere_z_positions))
                print(f"  Patient {selected_indices[i]+1}: Selected slice {slice_idx} with {sum(1 for pos in env.sphere_positions if abs(pos[2] - slice_idx) <= 2)} visible spheres")
            else:
                # No spheres, find slice with most lesion content
                lesion_slices = []
                for z in range(env.lesion_data.shape[2]):
                    lesion_content = np.sum(env.lesion_data[:, :, z] > 0)
                    if lesion_content > 0:
                        lesion_slices.append((z, lesion_content))
                
                if lesion_slices:
                    slice_idx = max(lesion_slices, key=lambda x: x[1])[0]
                else:
                    slice_idx = env.mri_data.shape[2] // 2
                
                print(f"  Patient {selected_indices[i]+1}: Selected slice {slice_idx} based on lesion content")
            
            # Ensure slice_idx is within bounds
            slice_idx = max(0, min(slice_idx, env.mri_data.shape[2] - 1))
            
            # Create standardized MRI view
            fig_mask, ax_mask = plt.subplots(1, 1, figsize=(mask_figure_size, mask_figure_size))
            
            # Get the data for this slice - use ORIGINAL masks
            mri_slice = env.mri_data[:, :, slice_idx]
            mask_slice = env.mask_data[:, :, slice_idx].astype(float)
            lesion_slice = env.lesion_data[:, :, slice_idx].astype(float)
            
            # Standardized cropping with less aggressive cropping
            target_size = 100  # Increased from 80 to reduce pixelation
            
            # Find region of interest based on original mask for consistent cropping
            if np.any(mask_slice > 0):
                mask_coords = np.where(mask_slice > 0)
                center_row = int(np.mean(mask_coords[0]))
                center_col = int(np.mean(mask_coords[1]))
                
                # Calculate bounds for fixed size crop
                half_size = target_size // 2
                min_row = max(0, center_row - half_size)
                max_row = min(mri_slice.shape[0], center_row + half_size)
                min_col = max(0, center_col - half_size)
                max_col = min(mri_slice.shape[1], center_col + half_size)
                
                # Adjust if we hit boundaries
                if max_row - min_row < target_size:
                    if min_row == 0:
                        max_row = min(mri_slice.shape[0], target_size)
                    elif max_row == mri_slice.shape[0]:
                        min_row = max(0, mri_slice.shape[0] - target_size)
                
                if max_col - min_col < target_size:
                    if min_col == 0:
                        max_col = min(mri_slice.shape[1], target_size)
                    elif max_col == mri_slice.shape[1]:
                        min_col = max(0, mri_slice.shape[1] - target_size)
                
                # Crop to standardized size
                mri_slice_cropped = mri_slice[min_row:max_row, min_col:max_col]
                mask_slice_cropped = mask_slice[min_row:max_row, min_col:max_col]
                lesion_slice_cropped = lesion_slice[min_row:max_row, min_col:max_col]
                
                print(f"  Patient {selected_indices[i]+1}: Standardized crop to {mri_slice_cropped.shape}")
                
            else:
                # Use center crop if no mask
                center_row, center_col = mri_slice.shape[0] // 2, mri_slice.shape[1] // 2
                half_size = target_size // 2
                min_row = max(0, center_row - half_size)
                max_row = min(mri_slice.shape[0], center_row + half_size)
                min_col = max(0, center_col - half_size)
                max_col = min(mri_slice.shape[1], center_col + half_size)
                
                mri_slice_cropped = mri_slice[min_row:max_row, min_col:max_col]
                mask_slice_cropped = mask_slice[min_row:max_row, min_col:max_col]
                lesion_slice_cropped = lesion_slice[min_row:max_row, min_col:max_col]
                min_row, min_col = 0, 0
                print(f"  Patient {selected_indices[i]+1}: Center crop to {mri_slice_cropped.shape}")
            
            # Normalize MRI data
            if np.max(mri_slice_cropped) > np.min(mri_slice_cropped):
                mri_normalized = (mri_slice_cropped - np.min(mri_slice_cropped)) / (np.max(mri_slice_cropped) - np.min(mri_slice_cropped))
            else:
                mri_normalized = mri_slice_cropped
            
            # Normalize masks
            mask_slice_cropped = np.clip(mask_slice_cropped, 0, 1)
            lesion_slice_cropped = np.clip(lesion_slice_cropped, 0, 1)
            
            # Display MRI as background
            ax_mask.imshow(mri_normalized, cmap='gray', vmin=0, vmax=1)
            
            # Overlay masks with decreased opacity (green for prostate, red for lesion)
            mask_masked = np.ma.masked_where(mask_slice_cropped < 0.001, mask_slice_cropped)
            lesion_masked = np.ma.masked_where(lesion_slice_cropped < 0.001, lesion_slice_cropped)
            
            # Reduced opacity for prostate (0.2) and higher for lesion (0.4)
            if np.any(mask_slice_cropped > 0):
                ax_mask.imshow(mask_masked, cmap='Greens', alpha=0.2, vmin=0, vmax=1)
                
            if np.any(lesion_slice_cropped > 0):
                ax_mask.imshow(lesion_masked, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
            
            # Add spheres with variable sizes based on distance from slice
            spheres_added = 0
            for j, sphere_pos in enumerate(env.sphere_positions):
                x, y, z = sphere_pos
                z_distance = abs(z - slice_idx)
                
                # Only show spheres within reasonable distance
                if z_distance <= env.sphere_radius:
                    # Calculate circle radius based on 3D geometry
                    # For a sphere with radius R at distance d from slice, 
                    # the circle radius is sqrt(R^2 - d^2)
                    if z_distance <= env.sphere_radius:
                        circle_radius = np.sqrt(env.sphere_radius**2 - z_distance**2)
                    else:
                        continue  # Skip if too far
                    
                    alpha = max(0.6, 1.0 - (z_distance / env.sphere_radius))
                    
                    # Adjust coordinates for cropped view
                    x_cropped = x - min_row
                    y_cropped = y - min_col
                    
                    # Check if within cropped bounds
                    if (0 <= x_cropped < mri_slice_cropped.shape[0] and 
                        0 <= y_cropped < mri_slice_cropped.shape[1]):
                        
                        # Add circle with size based on actual 3D intersection
                        circle = Circle((y_cropped, x_cropped), circle_radius, 
                                      fill=False, color='blue', linewidth=3, alpha=alpha)
                        ax_mask.add_patch(circle)
                        
                        # Add center dot at actual sphere position
                        ax_mask.scatter(y_cropped, x_cropped, s=50, c='blue', marker='o', 
                                       edgecolors='white', linewidth=2, alpha=alpha, zorder=10)
                        spheres_added += 1
            
            print(f"  Patient {selected_indices[i]+1}: Added {spheres_added} sphere visualizations")
            
            # Set background and remove axes
            ax_mask.set_xticks([])
            ax_mask.set_yticks([])
            ax_mask.axis('off')
            ax_mask.set_facecolor('black')
            fig_mask.patch.set_facecolor('black')
            
            # Save with consistent settings
            plt.savefig(temp_mask_path, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0.01)
            plt.close(fig_mask)
            
            temp_mask_images.append(temp_mask_path)
            print(f"  ✓ Created standardized MRI view for Patient {selected_indices[i]+1}")
            
        except Exception as e:
            print(f"  ✗ Error creating MRI view for Patient {selected_indices[i]+1}: {e}")
            temp_mask_images.append(None)
    
    # Create the main grid with reduced spacing
    fig = plt.figure(figsize=(cols * individual_subplot_size, rows * individual_subplot_size), facecolor='white')
    
    # Reduced spacing for tighter layout
    gs = fig.add_gridspec(rows, cols, 
                         left=0.01, bottom=0.01, right=0.99, top=0.99,
                         wspace=0.01, hspace=0.01,  # Much tighter spacing
                         height_ratios=[1, 1])
    
    # First row: projection images
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
                       fontsize=24, fontweight='bold')
        else:
            ax.set_visible(False)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.set_facecolor('white')
        
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Second row: MRI views
    for i in range(cols):
        ax = fig.add_subplot(gs[1, i])
        
        if i < len(temp_mask_images) and temp_mask_images[i] is not None:
            try:
                img = mpimg.imread(temp_mask_images[i])
                ax.imshow(img)
            except Exception as e:
                print(f"Error loading MRI image for Patient {selected_indices[i]+1}: {e}")
                ax.text(0.5, 0.5, f'Error\nPatient {selected_indices[i]+1}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=24, fontweight='bold')
        else:
            ax.set_visible(False)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.set_facecolor('black')
        
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Save with higher DPI
    save_path = os.path.join(save_folder, f'projection_{visualization_type}_grid_4patients_improved.png')
    fig.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white', pad_inches=0.01)
    plt.close(fig)
    
    # Clean up temporary files
    for temp_path in temp_projection_images + temp_mask_images:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    os.rmdir(temp_dir)
    
    print(f"Improved 4-patient grid saved: {save_path}")
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