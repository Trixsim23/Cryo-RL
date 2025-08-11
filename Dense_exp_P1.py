"""
Experiment 3: Dense Reward RL - ENHANCED WITH COMPREHENSIVE COVERAGE TRACKING
============================================================================

This experiment modifies the reward function to provide dense feedback at each step
with comprehensive coverage tracking similar to Experiment 1.

Usage:
    python exp3.py

Results saved in: ./experiment_3_enhanced_coverage_TIMESTAMP/
"""

import os
import time 
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gym import spaces
from scipy.ndimage import center_of_mass

# FIXED: Import from fixed environment
from main_env import SpherePlacementEnv, get_max_action_space_size, create_standardized_environments
from typing import List, Dict, Tuple

# Import all necessary functions from main_agent.py
from main_agent import (
    load_and_preprocess_patient_data,
    create_sphere_mask,
    run_enhanced_evaluation_loop,                    
    TrainingCallback
)


# ============================================================================
# DATASET SETUP WITH FILTERED PATIENTS
# ============================================================================

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


# ============================================================================
# DENSE REWARD ENVIRONMENT - FULLY FIXED
# ============================================================================

class DenseRewardSpherePlacementEnv(SpherePlacementEnv):
    """
    FULLY FIXED: Modified SpherePlacementEnv with balanced dense reward function.
    Provides immediate positive feedback at each step based on coverage improvement.
    """
    
    def __init__(self, mri_data, mask_data, lesion_data, sphere_radius=7, max_action_space=None):
        # FIXED: Pass max_action_space to parent constructor
        super().__init__(mri_data, mask_data, lesion_data, sphere_radius, max_action_space)
        
        # Track coverage history for dense rewards
        self.previous_coverage = 0.0
        self.total_lesion_volume = np.sum(self.lesion_data > 0)
        
        print(f"Dense Reward Environment initialized with {self.total_lesion_volume} lesion voxels")
    
    def _calculate_coverage_reward(self, new_sphere_position):
        """ENHANCED: Calculate reward based on UNIQUE coverage improvement + placement bonus"""
        # Create mask for ALL spheres (including new one)
        all_sphere_mask = create_sphere_mask(self.sphere_positions, self.sphere_radius, self.mri_data.shape)
        
        # Calculate UNIQUE lesion coverage (prevents benefit from overlapping spheres)
        current_unique_coverage = np.sum(np.logical_and(all_sphere_mask, self.lesion_data > 0))
        
        # Coverage improvement reward based on unique coverage only
        coverage_improvement = current_unique_coverage - self.previous_coverage
        coverage_reward = coverage_improvement / self.total_lesion_volume  # Normalize to [0,1]
        
        # BONUS: Give extra reward for any lesion contact
        new_sphere_mask = self._create_single_sphere_mask(new_sphere_position)
        lesion_contact = np.sum(np.logical_and(new_sphere_mask, self.lesion_data > 0))
        if lesion_contact > 0:
            contact_bonus = 0.1  # Small bonus for touching lesion
            coverage_reward += contact_bonus
        
        # BONUS: Progressive reward for increasing total coverage
        total_coverage_fraction = current_unique_coverage / self.total_lesion_volume
        if total_coverage_fraction > 0.1:  # 10% coverage
            coverage_reward += 0.05
        if total_coverage_fraction > 0.3:  # 30% coverage  
            coverage_reward += 0.1
        if total_coverage_fraction > 0.5:  # 50% coverage
            coverage_reward += 0.15
        
        # Update previous coverage
        self.previous_coverage = current_unique_coverage
        
        # Ensure minimum positive coverage reward if any improvement
        if coverage_improvement > 0:
            coverage_reward = max(coverage_reward, 0.05)  # Minimum 0.05 for any improvement
        
        return coverage_reward
    
    def _calculate_spacing_reward(self, new_sphere_position):
        """Calculate reward based on spacing from existing spheres - BALANCED"""
        if len(self.sphere_positions) <= 1:
            return 0.0  # No spacing penalty for first sphere
        
        # Calculate distances to existing spheres
        distances = []
        for existing_pos in self.sphere_positions[:-1]:  # Exclude the current sphere
            distance = np.linalg.norm(np.array(new_sphere_position) - np.array(existing_pos))
            distances.append(distance)
        
        min_distance = min(distances)
        
        # Reward good spacing (not too close, not too far) - LESS HARSH
        optimal_distance = self.sphere_radius * 2.5  # Optimal spacing
        
        if min_distance < self.sphere_radius * 1.5:
            # Too close - REDUCED penalty
            spacing_reward = -0.2  # REDUCED from -0.5
        elif min_distance > self.sphere_radius * 4.0:
            # Too far - small penalty
            spacing_reward = -0.05  # REDUCED from -0.1
        else:
            # Good spacing - INCREASED reward
            spacing_reward = 0.3  # INCREASED from 0.2
        
        return spacing_reward
    
    def _calculate_lesion_center_reward(self, new_sphere_position):
        """Calculate reward based on proximity to lesion center - ENHANCED"""
        lesion_center = np.array(center_of_mass(self.lesion_data))
        distance_to_center = np.linalg.norm(np.array(new_sphere_position) - lesion_center)
        
        # Normalize by lesion size
        lesion_span = np.max(np.argwhere(self.lesion_data > 0), axis=0) - np.min(np.argwhere(self.lesion_data > 0), axis=0)
        max_distance = np.linalg.norm(lesion_span)
        
        # ENHANCED: More generous center proximity reward
        center_reward = 0.2 * (1.0 - distance_to_center / max_distance)  # INCREASED from 0.1
        
        return center_reward
    
    def _calculate_boundary_penalty(self, new_sphere_position):
        """Penalty for placing spheres outside prostate mask - BALANCED"""
        x, y, z = new_sphere_position
        
        # Check if sphere center is within mask
        if not self.mask_data[x, y, z]:
            return -0.5  # REDUCED penalty from -1.0
        
        # Check if sphere extends outside mask
        sphere_mask = self._create_single_sphere_mask(new_sphere_position)
        overlap_with_mask = np.logical_and(sphere_mask, self.mask_data > 0)
        
        # Penalty based on how much of sphere is outside mask
        sphere_volume = np.sum(sphere_mask)
        valid_volume = np.sum(overlap_with_mask)
        
        if sphere_volume > 0:
            boundary_penalty = -0.15 * (1.0 - valid_volume / sphere_volume)  # REDUCED from -0.3
        else:
            boundary_penalty = 0.0
        
        return boundary_penalty
    
    def _calculate_overlap_penalty(self, new_sphere_position):
        """Calculate penalty for overlapping or too-close spheres - NEW METHOD"""
        if len(self.sphere_positions) <= 1:
            return 0.0  # No penalty for first sphere
        
        overlap_penalty = 0.0
        new_sphere_mask = self._create_single_sphere_mask(new_sphere_position)
        
        # Check overlap with each existing sphere
        for existing_pos in self.sphere_positions[:-1]:  # Exclude the current (new) sphere
            existing_sphere_mask = self._create_single_sphere_mask(existing_pos)
            
            # Calculate actual volume overlap
            overlap_volume = np.sum(np.logical_and(new_sphere_mask, existing_sphere_mask))
            
            if overlap_volume > 0:
                # Moderate penalty for any overlap
                overlap_penalty += 0.5  # Reasonable penalty for overlap
            else:
                # Check distance
                distance = np.linalg.norm(np.array(new_sphere_position) - np.array(existing_pos))
                min_safe_distance = self.sphere_radius * 1.8  # Slightly more than diameter
                
                if distance < min_safe_distance:
                    # Small penalty based on how close they are
                    proximity_penalty = (min_safe_distance - distance) / min_safe_distance
                    overlap_penalty += proximity_penalty * 0.3  # Moderate proximity penalty
        
        return -overlap_penalty  # Return negative value as penalty
    
    def _create_single_sphere_mask(self, center):
        """Create mask for a single sphere"""
        mask = np.zeros(self.mri_data.shape, dtype=bool)
        x, y, z = center
        
        x_range = np.arange(max(0, x - self.sphere_radius), min(self.mri_data.shape[0], x + self.sphere_radius))
        y_range = np.arange(max(0, y - self.sphere_radius), min(self.mri_data.shape[1], y + self.sphere_radius))
        z_range = np.arange(max(0, z - self.sphere_radius), min(self.mri_data.shape[2], z + self.sphere_radius))
        
        x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        distances = np.sqrt((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2)
        sphere_mask = distances <= self.sphere_radius
        
        mask[x_range[:, None, None], y_range[None, :, None], z_range[None, None, :]] = sphere_mask
        
        return mask
    
    def step(self, action):
        """Modified step function with BALANCED dense reward for positive outcomes"""
        # FIXED: Use the same action mapping as parent class
        if self.lesion_coords.shape[0] == 0:
            raise ValueError("No valid lesion coordinates available")
        
        coord_index = int(action) % self.lesion_coords.shape[0]
        coord = self.lesion_coords[coord_index]
        
        # Add random noise to the placement
        noise = np.random.randint(-5, 5, size=3)
        coord = np.clip(coord + noise, [0, 0, 0], np.array(self.mri_data.shape) - 1)
        
        # Add sphere position
        self.sphere_positions.append(coord)
        self.sphere_count += 1
        
        # Calculate all reward components
        coverage_reward = self._calculate_coverage_reward(coord)
        spacing_reward = self._calculate_spacing_reward(coord)
        center_reward = self._calculate_lesion_center_reward(coord)
        boundary_penalty = self._calculate_boundary_penalty(coord)
        overlap_penalty = self._calculate_overlap_penalty(coord)  # Use the new method
        
        # BASE POSITIVE REWARD: Give points just for valid placement
        base_reward = 1.0  # Base positive reward for each placement
        
        # BALANCED WEIGHTS: Much less punitive, more rewarding
        total_reward = (
            base_reward +                   # NEW: Always positive starting point
            coverage_reward * 10.0 +        # INCREASED: Main reward for coverage
            spacing_reward * 2.0 +          # REDUCED: Less harsh spacing penalty
            center_reward * 3.0 +           # INCREASED: More reward for good placement
            boundary_penalty * 2.0 +        # REDUCED: Less harsh boundary penalty
            overlap_penalty * 3.0           # NEW: Moderate overlap penalty
        )
        
        # SAFETY NET: Ensure reward isn't too negative
        total_reward = max(total_reward, -2.0)  # Cap negative rewards
        
        # Update environment state
        self._remove_sphere_from_data(coord)
        
        # Check if done
        done = self.sphere_count >= self.max_spheres
        
        # Create observation
        obs = np.concatenate([
            np.expand_dims(self.mri_data, -1), 
            np.expand_dims(self.modified_mask, -1),
            np.expand_dims(self.modified_lesion, -1)
        ], axis=-1)
        
        # ENHANCED INFO: Track all components for debugging
        info = {
            "sphere_positions": self.sphere_positions,
            "base_reward": base_reward,
            "coverage_reward": coverage_reward,
            "spacing_reward": spacing_reward,
            "center_reward": center_reward,
            "boundary_penalty": boundary_penalty,
            "overlap_penalty": overlap_penalty,
            "total_reward": total_reward,
            "reward_breakdown": f"Base:{base_reward:.2f} + Cov:{coverage_reward*10:.2f} + Spc:{spacing_reward*2:.2f} + Ctr:{center_reward*3:.2f} + Bnd:{boundary_penalty*2:.2f} + Ovr:{overlap_penalty*3:.2f} = {total_reward:.2f}"
        }
        
        return obs, total_reward, done, info
    
    def reset(self):
        """Reset environment and coverage tracking"""
        self.previous_coverage = 0.0
        return super().reset()


def create_dense_reward_environments(patient_dirs, max_action_space=None, target_shape=(128, 128, 20)):
    """
    Create dense reward environments with standardized action spaces
    """
    # If max_action_space not provided, calculate it
    if max_action_space is None:
        max_action_space = get_max_action_space_size(patient_dirs, target_shape)
    
    environments = []
    
    print(f"\nCreating dense reward environments with action space size: {max_action_space}")
    
    for i, patient_dir in enumerate(patient_dirs):
        try:
            patient_name = os.path.basename(patient_dir)
            print(f"Loading patient {i+1}/{len(patient_dirs)}: {patient_name}")
            
            mri_data, mask_data, lesion_data = load_and_preprocess_patient_data(patient_dir, target_shape)
            
            # Create environment with standardized action space
            env = DenseRewardSpherePlacementEnv(mri_data, mask_data, lesion_data, 
                                              sphere_radius=7, max_action_space=max_action_space)
            environments.append(env)
            
            print(f"  Successfully created dense reward environment for {patient_name}")
            
        except Exception as e:
            print(f"  Error creating environment for {patient_name}: {e}")
            continue
    
    return environments, max_action_space


# ============================================================================
# EXPERIMENT 3: DENSE REWARD RL - ENHANCED WITH COMPREHENSIVE COVERAGE TRACKING
# ============================================================================
def run_experiment_3_enhanced_coverage():
    """Run Experiment 3: """
    
    print("=" * 80)
    print("EXPERIMENT 3: DENSE REWARD RL ")
    print("=" * 80)

    
    # Setup results folder
    current_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = f"./experiment_3_alt{current_time}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Load dataset
    try:
        patient_folders = get_filtered_dataset()
        print(f"Found {len(patient_folders)} valid patients")
        
        if len(patient_folders) < 22:
            print(f"Warning: Only {len(patient_folders)} patients available")
            if len(patient_folders) < 5:
                raise ValueError(f"Need at least 5 patients for training, found {len(patient_folders)}")
        
        # ============================================================================
        # DATASET SPLIT WITH RANDOM SEED
        # ============================================================================
        
        import random
        random.seed(42)  # Fixed seed for reproducibility
        
        # Shuffle all patients for random selection
        shuffled_patients = patient_folders.copy()
        random.shuffle(shuffled_patients)
        
        # Adaptive split based on available patients
        num_patients = len(shuffled_patients)
        if num_patients >= 35:
            # Ideal case: 1/15/15 split (1 for training like exp1.py)
            train_patients = shuffled_patients[:1]
            val_patients = shuffled_patients[1:15]
            test_patients = shuffled_patients[15:30]
        elif num_patients >= 15:
            # Medium case: 5/5/5 split
            train_patients = shuffled_patients[:5]
            val_patients = shuffled_patients[5:10]
            test_patients = shuffled_patients[10:15]
        else:
            # Minimum case: use what we have
            train_patients = shuffled_patients[:min(5, num_patients)]
            val_patients = shuffled_patients[len(train_patients):min(len(train_patients) + 3, num_patients)]
            test_patients = shuffled_patients[len(train_patients) + len(val_patients):]
        
        print(f"Using {len(train_patients)}/{len(val_patients)}/{len(test_patients)} split:")
        print(f"  Training: {len(train_patients)} patients")
        print(f"  Validation: {len(val_patients)} patients") 
        print(f"  Testing: {len(test_patients)} patients")
        
        if len(train_patients) == 0:
            raise ValueError("No training patients available!")
        
        # Log selected patients
        print(f"\nSelected training patients:")
        for i, patient_dir in enumerate(train_patients):
            patient_name = os.path.basename(patient_dir)
            print(f"  {i+1}. {patient_name}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # ============================================================================
    # STANDARDIZED ENVIRONMENT CREATION - KEY FIX
    # ============================================================================
    
    print("\n" + "="*50)
    print("CREATING STANDARDIZED DENSE REWARD ENVIRONMENTS")
    print("="*50)
    
    # Get maximum action space size across all patients that will be used
    all_patients_for_analysis = train_patients + val_patients + test_patients
    max_action_space = get_max_action_space_size(all_patients_for_analysis)
    
    # Create standardized dense reward training environments
    print("\nCreating training environments...")
    train_envs, _ = create_dense_reward_environments(train_patients, max_action_space)
    
    # Create validation environments
    print("\nCreating validation environments...")
    if val_patients:
        # Use a subset of validation patients if too many
        val_subset = val_patients[:min(5, len(val_patients))]
        val_envs, _ = create_dense_reward_environments(val_subset, max_action_space)
    else:
        val_envs = [train_envs[0]]  # Use first training env for validation if no val patients
    
    # Create testing environments
    print("\nCreating testing environments...")
    if test_patients:
        # Use a subset of test patients if too many
        test_subset = test_patients[:min(10, len(test_patients))]
        test_envs, _ = create_dense_reward_environments(test_subset, max_action_space)
    else:
        test_envs = train_envs[:3] if len(train_envs) >= 3 else train_envs  # Use training envs for testing
    
    if not train_envs:
        print("Error: No valid training environments created!")
        return
    
    print(f"\nSuccessfully created:")
    print(f"  {len(train_envs)} training environments") 
    print(f"  {len(val_envs)} validation environments")
    print(f"  {len(test_envs)} testing environments")
    print(f"  Standardized action space size: {max_action_space}")
    
    # Verify all environments have same action space
    for i, env in enumerate(train_envs + val_envs + test_envs):
        assert env.action_space.n == max_action_space, f"Environment {i} has wrong action space size: {env.action_space.n} != {max_action_space}"
    print("✓ Verified: All environments have identical action spaces")
    
    # ============================================================================
    # TRAINING PARAMETERS
    # ============================================================================
    
    timesteps_per_patient = 500000000  # Increased for better training
    total_timesteps = len(train_envs) * timesteps_per_patient
    eval_freq = 2000
    
    print(f"\nTraining setup:")
    print(f"- Algorithm: PPO")
    print(f"- Policy: MlpPolicy") 
    print(f"- Training patients: {len(train_envs)}")
    print(f"- Timesteps per patient: {timesteps_per_patient:,}")
    print(f"- Total timesteps: {total_timesteps:,}")
    print(f"- Evaluation frequency: {eval_freq:,} steps")
    print(f"- Environment: DenseRewardSpherePlacementEnv (BALANCED)")
    print(f"- Standardized action space size: {max_action_space}")
    
    # Use first validation environment for callback evaluation
    val_env = val_envs[0]
    
    # ============================================================================
    # TRAINING LOOP - Train sequentially with standardized action spaces
    # ============================================================================
    
    # Initialize model with first training environment
    print(f"\nInitializing PPO model with action space size: {train_envs[0].action_space.n}")
    model = PPO("MlpPolicy", train_envs[0], verbose=1, n_steps=512)
    
    # Training callback for monitoring
    callback = TrainingCallback(val_env, eval_freq=eval_freq, verbose=1)
    
    print("\nStarting sequential training on patients...")
    start_time = time.time()
    
    # Train on each patient sequentially with the same model
    for patient_idx, train_env in enumerate(train_envs):
        patient_name = os.path.basename(train_patients[patient_idx])
        print(f"\n{'='*50}")
        print(f"Training on Patient {patient_idx + 1}/{len(train_envs)}: {patient_name}")
        print(f"Timesteps: {timesteps_per_patient:,}")
        print(f"Environment action space: {train_env.action_space.n}")
        print(f"Model action space: {model.action_space.n}")
        print(f"{'='*50}")
        
        # Now this should work since all environments have the same action space
        model.set_env(train_env)
        
        # Train on this patient
        model.learn(total_timesteps=timesteps_per_patient, callback=callback, reset_num_timesteps=False)
        
        print(f"Completed training on {patient_name}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Total timesteps trained: {total_timesteps:,}")
    
    # Save model
    model_path = os.path.join(results_folder, "trained_model")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # ============================================================================
    # ENHANCED TRAINING PROGRESS VISUALIZATION
    # ============================================================================
    
    # Training progress plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    if callback.episode_rewards and callback.dice_scores:
        # Calculate means for horizontal reference lines
        mean_reward = np.mean(callback.episode_rewards)
        mean_dice = np.mean(callback.dice_scores)
        
        ax1.plot(callback.timesteps, callback.episode_rewards, 'b-o', linewidth=2, markersize=4)
        ax1.axhline(y=mean_reward, color='green', linestyle='-', linewidth=2, alpha=0.7, 
                   label=f'Mean Reward: {mean_reward:.2f}')
        ax1.set_title('Training Progress -  Dense Rewards', fontsize=16)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Validation Reward')
        ax1.grid(True, alpha=0.3)
        
        # Add vertical lines to show patient transitions
        for i in range(1, len(train_envs)):
            ax1.axvline(x=i * timesteps_per_patient, color='red', linestyle='--', alpha=0.5, 
                       label=f'Patient {i+1}' if i == 1 else "")
        ax1.legend()
        
        ax2.plot(callback.timesteps, callback.dice_scores, 'r-o', linewidth=2, markersize=4)
        ax2.axhline(y=mean_dice, color='green', linestyle='-', linewidth=2, alpha=0.7, 
                   label=f'Mean Dice: {mean_dice:.3f}')
        ax2.set_title('Dense Rewards Training Progress - Dice Scores', fontsize=16)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Dice Score')
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines to show patient transitions
        for i in range(1, len(train_envs)):
            ax2.axvline(x=i * timesteps_per_patient, color='red', linestyle='--', alpha=0.5)
        
        ax2.legend()
    else:
        ax1.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    training_progress_path = os.path.join(results_folder, 'enhanced_training_progress.png')
    plt.savefig(training_progress_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # ENHANCED FINAL EVALUATION WITH COMPREHENSIVE COVERAGE TRACKING
    # ============================================================================
    
 
    print("evals")

    # Option to enable 3D visualization (computationally expensive)
    create_projections=True  # Set to True if you want 3D visualizations
    if create_projections:
        print("3D volume rendering (ENABLED)")
    else:
        print("no 3d viz)")
    
    print("="*60)
    
    # Use the enhanced evaluation loop from Experiment 1 for comprehensive coverage tracking
    final_rewards, final_dice_scores, comprehensive_results = run_enhanced_evaluation_loop(
        model, test_envs, results_folder, create_projections=True
    )
    
    # ============================================================================
    # ENHANCED RESULTS SUMMARY WITH COMPREHENSIVE COVERAGE TRACKING
    # ============================================================================
    
    # Summary statistics
    mean_reward = np.mean(final_rewards) if final_rewards else 0
    std_reward = np.std(final_rewards) if final_rewards else 0
    mean_dice = np.mean(final_dice_scores) if final_dice_scores else 0
    std_dice = np.std(final_dice_scores) if final_dice_scores else 0
    
    # Extract coverage metrics from comprehensive results
    if comprehensive_results:
        coverage_percentages = [result['results']['lesion_coverage_percentage'] for result in comprehensive_results]
        mean_coverage = np.mean(coverage_percentages)
        std_coverage = np.std(coverage_percentages)
        
        total_spheres_placed = [result['results']['total_spheres'] for result in comprehensive_results]
        mean_spheres = np.mean(total_spheres_placed)
        
        # Additional detailed metrics (with error handling)
        lesion_volumes = []
        covered_voxels = []
        
        # Check what keys are available in the first result
        if comprehensive_results:
            print(f"Available result keys: {list(comprehensive_results[0]['results'].keys())}")
            
            # Try to extract additional metrics safely
            for result in comprehensive_results:
                result_data = result['results']
                
                # Try different possible key names for lesion volume
                lesion_vol = (result_data.get('total_lesion_voxels') or 
                             result_data.get('lesion_volume') or 
                             result_data.get('total_lesion_volume') or 0)
                lesion_volumes.append(lesion_vol)
                
                # Try different possible key names for covered voxels
                covered_vol = (result_data.get('covered_lesion_voxels') or 
                              result_data.get('covered_voxels') or 
                              result_data.get('spheres_covering_lesion') or 0)
                covered_voxels.append(covered_vol)
        
        print(f"Evaluation Results:")
        print(f"  Patients evaluated: {len(test_subset) if 'test_subset' in locals() else len(test_envs)}")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Mean Dice score: {mean_dice:.3f} ± {std_dice:.3f}")
        print(f"  Mean lesion coverage: {mean_coverage:.1f}% ± {std_coverage:.1f}%")
        print(f"  Mean spheres per patient: {mean_spheres:.1f}")
        
        if lesion_volumes and any(v > 0 for v in lesion_volumes):
            print(f"  Mean lesion volume: {np.mean(lesion_volumes):.0f} voxels")
        if covered_voxels and any(v > 0 for v in covered_voxels):
            print(f"  Mean covered voxels: {np.mean(covered_voxels):.0f}")
        
        print(f"\nIndividual Patient Results:")
        for i, (reward, dice) in enumerate(zip(final_rewards, final_dice_scores)):
            coverage = coverage_percentages[i]
            spheres = total_spheres_placed[i]
            patient_name = f"Patient {i+1}"
            print(f"  {patient_name}: Reward={reward:.2f}, Dice={dice:.3f}, Coverage={coverage:.1f}%, Spheres={spheres}")
    
    else:
        coverage_percentages = []
        mean_coverage = std_coverage = mean_spheres = 0
        lesion_volumes = []
        covered_voxels = []
        total_spheres_placed = []
        print(f"Basic Results:")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Mean Dice score: {mean_dice:.3f} ± {std_dice:.3f}")
    
    print(f"\n" + "="*60)
    print("EXPERIMENT 3 RESULTS")
    print("="*60)
    print(f"Training Configuration:")
    print(f"  Algorithm: Dense Reward RL (PPO)")
    print(f"  Environment: DenseRewardSpherePlacementEnv (BALANCED)")
    print(f"  Patients trained: {len(train_envs)}")
    print(f"  Timesteps per patient: {timesteps_per_patient:,}")
    print(f"  Total training timesteps: {total_timesteps:,}")
    print(f"  Standardized action space size: {max_action_space}")
    print(f"  Training time: {training_time:.2f} seconds")
    
    # ============================================================================
    # ENHANCED SUMMARY VISUALIZATIONS WITH COVERAGE
    # ============================================================================
    
    # Create comprehensive summary plot (like Experiment 1)
    if final_rewards and final_dice_scores:
        if coverage_percentages:
            # Three-metric summary plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        else:
            # Two-metric summary plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot final rewards
        bars1 = ax1.bar(range(1, len(final_rewards) + 1), final_rewards, 
                       color='purple', edgecolor='darkviolet', linewidth=1.5)
        ax1.axhline(y=mean_reward, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_reward:.2f}')
        ax1.set_title('Experiment 3  Dense Reward RL - Final Test Rewards per Patient', fontsize=14)
        ax1.set_xlabel('Test Patient', fontsize=12)
        ax1.set_ylabel('Total Reward', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height != 0:  # Only add label if non-zero
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(final_rewards),
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot final Dice scores
        bars2 = ax2.bar(range(1, len(final_dice_scores) + 1), final_dice_scores, 
                       color='mediumpurple', edgecolor='darkviolet', linewidth=1.5)
        ax2.axhline(y=mean_dice, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_dice:.3f}')
        ax2.set_title('Experiment 3  Dense Reward RL - Final Test Dice Scores per Patient', fontsize=14)
        ax2.set_xlabel('Test Patient', fontsize=12)
        ax2.set_ylabel('Dice Score', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            if height != 0:  # Only add label if non-zero
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(final_dice_scores),
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot lesion coverage if available
        if coverage_percentages:
            bars3 = ax3.bar(range(1, len(coverage_percentages) + 1), coverage_percentages, 
                           color='plum', edgecolor='darkviolet', linewidth=1.5)
            ax3.axhline(y=mean_coverage, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_coverage:.1f}%')
            ax3.set_title('Experiment 3  Dense Reward RL - Lesion Coverage per Patient', fontsize=14)
            ax3.set_xlabel('Test Patient', fontsize=12)
            ax3.set_ylabel('Coverage Percentage (%)', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars3:
                height = bar.get_height()
                if height != 0:  # Only add label if non-zero
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(coverage_percentages),
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        enhanced_summary_path = os.path.join(results_folder, 'final_results_summary.png')
        plt.savefig(enhanced_summary_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ============================================================================
    # SAVE COMPREHENSIVE RESULTS DATA WITH COVERAGE TRACKING
    # ============================================================================
    
    # Save enhanced results data (like Experiment 1 but for Dense Reward RL)
    results_data = {
        'experiment_name': 'Experiment 3 dense rl ',
        'method': 'Dense Reward RL Enhanced - Comprehensive Coverage Tracking',
        'algorithm': 'PPO',
        'policy': 'MlpPolicy',
        'environment': 'DenseRewardSpherePlacementEnv (BALANCED)',
        'max_action_space_size': max_action_space,
        'training_patients': [os.path.basename(p) for p in train_patients],
        'timesteps_per_patient': timesteps_per_patient,
        'total_training_timesteps': total_timesteps,
        'training_timesteps': callback.timesteps if hasattr(callback, 'timesteps') else [],
        'training_rewards': callback.episode_rewards if hasattr(callback, 'episode_rewards') else [],
        'training_dice_scores': callback.dice_scores if hasattr(callback, 'dice_scores') else [],
        'final_rewards': final_rewards,
        'final_dice_scores': final_dice_scores,
        'lesion_coverage_percentages': coverage_percentages,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_dice': mean_dice,
        'std_dice': std_dice,
        'mean_coverage': mean_coverage,
        'std_coverage': std_coverage,
        'mean_spheres_per_patient': mean_spheres,
        'training_time': training_time,
        'comprehensive_patient_results': [result['results'] for result in comprehensive_results] if comprehensive_results else [],
        'reward_balance_changes': {
            'base_reward': 1.0,
            'coverage_weight': 10.0,
            'spacing_weight': 2.0,
            'center_weight': 3.0,
            'boundary_weight': 2.0,
            'overlap_weight': 3.0,
            'safety_net_minimum': -2.0
        },
        'visualization_types_created': [
            'enhanced_2d', 'multiview', 'detailed_analysis', 'progression_summary'
        ] + (['3d_volume'] if create_projections else []),
        'dataset_split': {
            'train_patients': len(train_patients),
            'val_patients': len(val_patients),
            'test_patients': len(test_patients),
            'total_patients': len(patient_folders)
        }
    }
    
    results_file = os.path.join(results_folder, 'experiment_evaluation_results.npz')
    np.savez(results_file, **results_data)
    
    # Save comprehensive summary report
    summary_file = os.path.join(results_folder, 'experiment_evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("EXPERIMENT 3 ENHANCED: DENSE REWARD RL WITH COMPREHENSIVE COVERAGE TRACKING\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Experiment completed: {current_time}\n")
        f.write(f"Method: Dense Reward RL Enhanced with Comprehensive Coverage Tracking\n")
        f.write(f"Algorithm: PPO\n")
        f.write(f"Policy: MlpPolicy\n")
        f.write(f"Environment: DenseRewardSpherePlacementEnv (BALANCED)\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Timesteps per patient: {timesteps_per_patient:,}\n")
        f.write(f"Total timesteps: {total_timesteps:,}\n")
        f.write(f"Max action space size: {max_action_space}\n")
        f.write(f"Evaluation frequency: {eval_freq:,} steps\n\n")
        

        if create_projections:
            f.write(f"visualisations enabled\n")
        else:
            f.write(f" visualisations disabled for speed)\n")
        f.write("\n")
        
        
        f.write(f"Training Patients:\n")
        for i, patient_dir in enumerate(train_patients):
            patient_name = os.path.basename(patient_dir)
            f.write(f"  {i+1}. {patient_name}\n")
        f.write("\n")
        
        f.write(f"Dataset Split:\n")
        f.write(f"  Total patients available: {len(patient_folders)}\n")
        f.write(f"  Training patients: {len(train_patients)}\n")
        f.write(f"  Validation patients: {len(val_patients)}\n")
        f.write(f"  Testing patients: {len(test_patients)}\n\n")
        
        f.write(f"Test Set Results:\n")
        f.write(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"  Mean Dice score: {mean_dice:.3f} ± {std_dice:.3f}\n")
        if coverage_percentages:
            f.write(f"  Mean lesion coverage: {mean_coverage:.1f}% ± {std_coverage:.1f}%\n")
            f.write(f"  Mean spheres per patient: {mean_spheres:.1f}\n")
        f.write("\n")
        
        if final_rewards:
            f.write(f"Individual Test Results:\n")
            for i, (reward, dice) in enumerate(zip(final_rewards, final_dice_scores)):
                coverage = coverage_percentages[i] if i < len(coverage_percentages) else 0
                spheres = total_spheres_placed[i] if i < len(total_spheres_placed) else 0
                f.write(f"  Patient {i+1}: Reward={reward:.2f}, Dice={dice:.3f}, Coverage={coverage:.1f}%, Spheres={spheres}\n")
        

        if create_projections:
            f.write(f"  -visual renderings (step-by-step + final)\n")
    
    print(f"\n" + "="*80)
    print("EXPERIMENT 3 Dense rewards")
    print("="*80)
    print(f"Results saved in: {results_folder}")
    print(f"Summary report: {summary_file}")
    print(f"\nKey files created:")
    print(f"✓ {enhanced_summary_path} - Comprehensive 3-panel summary")
    print(f"✓ {training_progress_path} - Training overview")
    print(f"✓ {results_file} - Complete data")
    print(f"✓ {summary_file} - Detailed report")
    
    if create_projections:
        print(f"")
    else:
        print(f"volume renderings (disabled for speed)")
    print(f"\nEvaluation Summary:")
    print(f"  Patients evaluated: {len(test_subset) if 'test_subset' in locals() else len(test_envs)}")
    if coverage_percentages:
        print(f"  Mean coverage: {mean_coverage:.1f}% ± {std_coverage:.1f}%")
    print(f"  Mean Dice: {mean_dice:.3f} ± {std_dice:.3f}")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("="*80)


if __name__ == "__main__":
    run_experiment_3_enhanced_coverage()