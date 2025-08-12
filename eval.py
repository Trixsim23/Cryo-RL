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
    
    fig = plt.figure(figsize=(10, 8))
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
        # Use same coordinate system as the ablation zones - don't swap x and y
        top_z = max_z + 2  # Place dots above everything else
        ax.scatter(x, y, top_z, c='black', s=250, marker='o', edgecolors='white', 
                linewidth=5, zorder=200, alpha=1.0)
    
        # Optional: add a thin line showing the actual depth position
        ax.plot([x, x], [y, y], [z, top_z], 'k-', linewidth=8, alpha=0.4, zorder=150)
    
    # Zoom in on the data
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    
    ax.view_init(elev=70, azim=-0.3)
    # Remove all axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_facecolor('white')

    if env.sphere_positions:
        sphere_mask = create_sphere_mask(env.sphere_positions, env.sphere_radius, env.mri_data.shape)
        dice_score = calculate_dice_score(sphere_mask, env.lesion_data)
        # lesion_volume = np.sum(env.lesion_data > 0)
        # covered_volume = np.sum((sphere_mask > 0) & (env.lesion_data > 0))
        # coverage_percentage = (covered_volume / lesion_volume * 100) if lesion_volume > 0 else 0
    else:
        dice_score = 0.0
    
    
    placement_count = len(env.sphere_positions)
    ax.set_title(f'Dice: {dice_score:.3f}', 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def visualize_placement_projection_ablation(env, save_path=None, show=True, step_info="", projection_axis='axial'):
    from mpl_toolkits.mplot3d import Axes3D
    from skimage import measure
    
    fig = plt.figure(figsize=(10, 8))
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
        # Use same coordinate system as the ablation zones - don't swap x and y
        top_z = max_z + 2  # Place dots above everything else
        ax.scatter(x, y, top_z, c='black', s=250, marker='o', edgecolors='white', 
                linewidth=5, zorder=200, alpha=1.0)
        # ax.text(x, y, top_z+2, str(i+1), fontsize=30, fontweight='bold', 
        #     color='black', zorder=201)
        # Optional: add a thin line showing the actual depth position
        ax.plot([x, x], [y, y], [z, top_z], 'k-', linewidth=8, alpha=0.4, zorder=150)
    
    # Zoom in on the data
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    
    ax.view_init(elev=70, azim=-0.3)
    # Remove all axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_facecolor('white')
    
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
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
    
    # Select 8 patients using every other patient logic
    total_patients = len(results_list)
    selected_indices = []
    
    # First, try every other patient starting from 0 (patient 1, 3, 5, 7...)
    for i in range(0, total_patients, 2):
        selected_indices.append(i)
        if len(selected_indices) >= 8:
            break
    
    # If we don't have 8 patients, work backwards to fill remaining slots
    if len(selected_indices) < 8:
        # Add remaining patients working backwards from even indices
        for i in range(1, total_patients, 2):  # 1, 3, 5, 7... (patient 2, 4, 6, 8...)
            if i not in selected_indices:
                selected_indices.append(i)
                if len(selected_indices) >= 8:
                    break
    
    # Limit to 8 patients maximum
    selected_indices = selected_indices[:8]
    selected_results = [results_list[i] for i in selected_indices]
    
    print(f"Selected patients: {[i+1 for i in selected_indices]}")
    
    # Fixed grid dimensions for 8 patients
    rows, cols = 2, 4
    
    # Create temporary directory for individual images
    temp_dir = tempfile.mkdtemp()
    temp_images = []
    
    print(f"Creating individual {visualization_type} projections for grid...")
    
    # Generate individual projection images
    for i, result in enumerate(selected_results):
        env = result['env']
        dice_score = result['dice_score']
        coverage = result['coverage_percentage']
        
        step_info = f"(Dice: {dice_score:.3f}, Cov: {coverage:.1f}%)"
        temp_path = os.path.join(temp_dir, f'{visualization_type}_{i}.png')
        
        try:
            if visualization_type == "dots":
                visualize_placement_projection_dots(
                    env=env,
                    save_path=temp_path,
                    show=False,
                    step_info=step_info
                )
            elif visualization_type == "ablation":
                visualize_placement_projection_ablation(
                    env=env,
                    save_path=temp_path,
                    show=False,
                    step_info=step_info
                )
            
            temp_images.append(temp_path)
            print(f"  ✓ Created {visualization_type} for Patient {selected_indices[i]+1}")
            
        except Exception as e:
            print(f"  ✗ Error creating {visualization_type} for Patient {selected_indices[i]+1}: {e}")
            temp_images.append(None)
    
    # Create the grid with minimal spacing
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    # Display images in grid
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        if i < len(temp_images) and temp_images[i] is not None:
            try:
                # Load and display the image
                img = mpimg.imread(temp_images[i])
                ax.imshow(img)
                # Remove subplot title as requested
            except Exception as e:
                print(f"Error loading image for Patient {selected_indices[i]+1}: {e}")
                ax.text(0.5, 0.5, f'Error\nPatient {selected_indices[i]+1}', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.set_visible(False)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    
    # Remove main title and minimize spacing
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.05, hspace=0.05)
    
    # Save the grid
    save_path = os.path.join(save_folder, f'projection_{visualization_type}_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()
    
    # Clean up temporary files
    for temp_path in temp_images:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    os.rmdir(temp_dir)
    
    print(f"Grid saved: {save_path}")
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
    experiment_folder = input("Enter the experiment folder name (e.g., 'experiment_1_20250101-123456'): ").strip()
    
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