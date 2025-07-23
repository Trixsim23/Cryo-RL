import os
import time 
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

# QUICK FIX: Replace your enhanced_visualize_spheres_with_numbers function with this:

def enhanced_visualize_spheres_with_numbers(env, slice_idx, save_path=None, show=True, step_info=""):
    """
    FIXED: Enhanced visualization with consistent display across all panels
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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

    # FIXED: Use same slice for all panels
    original_mask_masked = np.ma.masked_where(original_mask_norm[:, :, slice_idx] == 0, original_mask_norm[:, :, slice_idx])
    modified_mask_masked = np.ma.masked_where(modified_mask_norm[:, :, slice_idx] == 0, modified_mask_norm[:, :, slice_idx])
    original_lesion_masked = np.ma.masked_where(original_lesion_norm[:, :, slice_idx] == 0, original_lesion_norm[:, :, slice_idx])
    modified_lesion_masked = np.ma.masked_where(modified_lesion_norm[:, :, slice_idx] == 0, modified_lesion_norm[:, :, slice_idx])

    # Define normalization for overlay consistency
    norm = Normalize(vmin=0, vmax=1)

    # Define colors for each sphere (up to 3 spheres)
    sphere_colors = ['red', 'blue', 'green']
    sphere_markers = ['o', 's', '^']  # circle, square, triangle

    # FIXED: Original MRI with mask overlay - NOW CONSISTENT
    axes[0, 0].imshow(original_mri_norm[:, :, slice_idx], cmap='gray', norm=norm)
    axes[0, 0].imshow(original_mask_masked, cmap='Blues', alpha=0.4,vmin=0, vmax=1)     # FIXED: Same colormap
    axes[0, 0].imshow(original_lesion_masked, cmap='Reds', alpha=0.6,vmin=0, vmax=1)    # FIXED: Same colormap
    axes[0, 0].set_title("Original MRI with Masks")
    axes[0, 0].axis("off")

    # Modified MRI with mask overlay + Individual Sphere Markers
    axes[0, 1].imshow(original_mri_norm[:, :, slice_idx], cmap='gray', norm=norm)  # Use original MRI
    axes[0, 1].imshow(modified_mask_masked, cmap='Blues', alpha=0.4,vmin=0, vmax=1)
    axes[0, 1].imshow(modified_lesion_masked, cmap='Reds', alpha=0.6,vmin=0, vmax=1)
    
    # Add individual sphere markers
    for i, sphere_pos in enumerate(env.sphere_positions):
        x, y, z = sphere_pos
        if abs(z - slice_idx) <= 2:  # Show spheres close to current slice
            color = sphere_colors[i % len(sphere_colors)]
            marker = sphere_markers[i % len(sphere_markers)]
            
            # Add numbered marker
            axes[0, 1].scatter(y, x, s=200, c=color, marker=marker, 
                             edgecolors='white', linewidth=2, alpha=0.9, zorder=10)
            axes[0, 1].text(y, x, str(i+1), ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='white', zorder=11)
    
    axes[0, 1].set_title(f"Modified MRI with Numbered Spheres {step_info}")
    axes[0, 1].axis("off")

    # Original masks
    axes[1, 0].imshow(original_mask_masked, cmap='Blues', alpha=0.7,vmin=0, vmax=1)
    axes[1, 0].imshow(original_lesion_masked, cmap='Reds', alpha=0.8,vmin=0, vmax=1)
    axes[1, 0].set_title("Original Masks")
    axes[1, 0].axis("off")

    # Modified masks with Individual Sphere Markers
    axes[1, 1].imshow(modified_mask_masked, cmap='Blues', alpha=0.7,vmin=0, vmax=1)
    axes[1, 1].imshow(modified_lesion_masked, cmap='Reds', alpha=0.8,vmin=0, vmax=1)
    
    # Add individual sphere markers to mask view
    for i, sphere_pos in enumerate(env.sphere_positions):
        x, y, z = sphere_pos
        if abs(z - slice_idx) <= 2:  # Show spheres close to current slice
            color = sphere_colors[i % len(sphere_colors)]
            marker = sphere_markers[i % len(sphere_markers)]
            
            # Add numbered marker
            axes[1, 1].scatter(y, x, s=200, c=color, marker=marker, 
                             edgecolors='white', linewidth=2, alpha=0.9, zorder=10)
            axes[1, 1].text(y, x, str(i+1), ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='white', zorder=11)
    
    axes[1, 1].set_title(f"Modified Masks with Numbered Spheres {step_info}")
    axes[1, 1].axis("off")

    # Add legend for sphere markers
    if env.sphere_positions:
        legend_elements = []
        for i in range(len(env.sphere_positions)):
            color = sphere_colors[i % len(sphere_colors)]
            marker = sphere_markers[i % len(sphere_markers)]
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor=color, markersize=10, 
                                            markeredgecolor='white', markeredgewidth=2,
                                            label=f'Sphere {i+1}'))
        
        axes[0, 1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        # Use default save path if none provided
        current_time = time.strftime("%Y%m%d-%H%M%S")
        folder = os.path.join(".", "results")
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(os.path.join('results', f'enhanced_visualize_{step_info}_{current_time}.png'))
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def visualize_individual_step_placement(env, step_num, save_folder, patient_id, dice_score=None):
    """
    Visualize individual step with enhanced sphere visualization
    """
    slice_idx = env.mri_data.shape[2] // 2
    
    # Prepare step info
    step_info = f"Step {step_num}"
    if dice_score is not None:
        step_info += f" (Dice: {dice_score:.3f})"
    
    # Create enhanced visualization
    save_path = os.path.join(save_folder, f'enhanced_step_patient_{patient_id}_step_{step_num}.png')
    
    fig = enhanced_visualize_spheres_with_numbers(
        env=env,
        slice_idx=slice_idx,
        save_path=save_path,
        show=False,
        step_info=step_info
    )
    
    return save_path

def create_sphere_progression_summary(env, save_folder, patient_id):
    """
    Create a summary showing progression of sphere placements
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


def main():
    print("Starting Enhanced Reinforcement Learning Training...")
    
    # Create results folder
    current_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = f"./enhanced_results_{current_time}"
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
    
    # training params
    #the faster version of the code is 50 * 100
    #change this to 500 and 1000
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
    ax1.set_title('Training Progress: Evaluation Rewards Over Time', fontsize=16)
    ax1.set_xlabel('Training Steps', fontsize=14)
    ax1.set_ylabel('Evaluation Reward', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot Dice scores
    ax2.plot(callback.timesteps, callback.dice_scores, 'r-o', linewidth=2, markersize=4)
    ax2.set_title('Training Progress: Dice Scores Over Time', fontsize=16)
    ax2.set_xlabel('Training Steps', fontsize=14)
    ax2.set_ylabel('Dice Score', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    progress_path = os.path.join(results_folder, 'training_progress.png')
    plt.savefig(progress_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved: {progress_path}")
    
    # Final evaluation on all evaluation environments with enhanced visualization
    print("Running final evaluation with enhanced visualization...")
    final_rewards = []
    final_dice_scores = []
    
    for i, eval_env in enumerate(eval_envs):
        print(f"Evaluating on environment {i+1}/{len(eval_envs)}...")
        
        # Create individual placement folder for this patient
        patient_folder = os.path.join(results_folder, f'patient_{i+1}_enhanced_placements')
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)
        
        obs = eval_env.reset()
        done = False
        total_reward = 0
        step_rewards = []
        step_dice_scores = []
        step_num = 0
        
        # Visualize initial state (no spheres)
        visualize_individual_step_placement(eval_env, step_num, patient_folder, i+1, dice_score=0.0)
        
        # Calculate initial dice score (should be 0)
        initial_sphere_mask = create_sphere_mask([], eval_env.sphere_radius, eval_env.mri_data.shape)
        initial_dice = calculate_dice_score(initial_sphere_mask, eval_env.lesion_data)
        step_dice_scores.append(initial_dice)
        
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
            
            # Enhanced visualization for this placement
            visualize_individual_step_placement(eval_env, step_num, patient_folder, i+1, dice_score=dice_score)
            
            print(f"  Step {step_num}: Reward = {reward:.2f}, Dice = {dice_score:.3f}")
        
        # Create sphere progression summary
        create_sphere_progression_summary(eval_env, patient_folder, i+1)
        
        # Create final enhanced visualization
        slice_idx = eval_env.mri_data.shape[2] // 2
        final_viz_path = os.path.join(patient_folder, f'final_enhanced_result_patient_{i+1}.png')
        enhanced_visualize_spheres_with_numbers(
            env=eval_env,
            slice_idx=slice_idx,
            save_path=final_viz_path,
            show=False,
            step_info=f"Final (Dice: {step_dice_scores[-1]:.3f})"
        )
        
        final_rewards.append(total_reward)
        final_dice_scores.append(step_dice_scores[-1])  # Final dice score
        
        # Plot step-by-step progress for this patient
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
    
    # Summary statistics
    mean_reward = np.mean(final_rewards)
    std_reward = np.std(final_rewards)
    mean_dice = np.mean(final_dice_scores)
    std_dice = np.std(final_dice_scores)
    
    print(f"\nFinal Evaluation Results:")
    print(f"Mean reward across all environments: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Dice score across all environments: {mean_dice:.3f} ± {std_dice:.3f}")
    print(f"Individual rewards: {final_rewards}")
    print(f"Individual Dice scores: {[f'{score:.3f}' for score in final_dice_scores]}")
    
    # Plot final results summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot final rewards
    bars1 = ax1.bar(range(1, len(final_rewards) + 1), final_rewards, color='skyblue', edgecolor='navy', linewidth=1.5)
    ax1.axhline(y=mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
    ax1.set_title('Final Evaluation Rewards Across All Environments', fontsize=14)
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
    ax2.set_title('Final Dice Scores Across All Environments', fontsize=14)
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
    
    final_summary_path = os.path.join(results_folder, 'final_evaluation_summary.png')
    plt.savefig(final_summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Final evaluation summary saved: {final_summary_path}")
    
    # Save numerical results
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
        'training_time': training_time
    }
    
    np.savez(os.path.join(results_folder, 'results_data.npz'), **results_data)
    
    print(f"\nAll results saved in folder: {results_folder}")
    print("Enhanced training and evaluation complete!")


if __name__ == "__main__":
    main()