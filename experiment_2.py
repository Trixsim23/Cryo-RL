"""
Experiment 2: Random Policy Baseline - FIXED
===========================================

This experiment uses random action selection instead of a learned policy.
Same environment and setup as Experiment 1, but actions are chosen randomly.
This provides a lower bound for RL performance.

Usage:
    python experiment_2.py

Results saved in: ./experiment_2_results_TIMESTAMP/
"""

import os
import time 
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


from main_env import SpherePlacementEnv, get_max_action_space_size, create_standardized_environments
from typing import List, Dict, Tuple

# Import all necessary functions from main_agent.py
from main_agent_2 import (
    load_and_preprocess_patient_data,
    create_sphere_mask,
    calculate_dice_score,
    enhanced_visualize_spheres_with_numbers,
    visualize_individual_step_placement,
    create_sphere_progression_summary
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
# RANDOM POLICY IMPLEMENTATION
# ============================================================================

class RandomPolicy:
    """Random policy that selects actions randomly from the action space"""
    
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        if seed is not None:
            np.random.seed(seed)
    
    def predict(self, obs, deterministic=False):
        """Predict action (randomly)"""
        action = self.action_space.sample()
        return action, None
    
    def learn(self, *args, **kwargs):
        """No learning for random policy"""
        pass


def run_random_evaluation(env, policy, num_episodes=5):
    """Run multiple episodes with random policy to get statistics"""
    rewards = []
    dice_scores = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        # Calculate dice score
        sphere_mask = create_sphere_mask(env.sphere_positions, env.sphere_radius, env.mri_data.shape)
        dice_score = calculate_dice_score(sphere_mask, env.lesion_data)
        
        rewards.append(total_reward)
        dice_scores.append(dice_score)
    
    return rewards, dice_scores


# ============================================================================
# EXPERIMENT 2: RANDOM POLICY - FIXED
# ============================================================================
def run_experiment_2():
    """Run Experiment 2: Random Policy Baseline - FIXED with standardized action spaces"""
    
    print("=" * 60)
    print("EXPERIMENT 2: RANDOM POLICY BASELINE - FIXED")
    print("=" * 60)
    print("This experiment uses random action selection with fixes:")
    print("- Standardized action spaces across environments")
    print("- Fixed visualization functions")
    print("- Random action selection (no learning)")
    print("- Testing on 5 randomly selected patients")
    print("- Multiple runs per patient for statistics")
    print("- Provides lower bound for RL performance")
    print("=" * 60)
    
    # Setup results folder
    current_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = f"./experiment_2_fixed_results_{current_time}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Load dataset
    try:
        patient_folders = get_filtered_dataset()
        print(f"Found {len(patient_folders)} valid patients")
        
        if len(patient_folders) < 22:
            print(f"Warning: Only {len(patient_folders)} patients available")
            if len(patient_folders) < 5:
                raise ValueError(f"Need at least 5 patients for testing, found {len(patient_folders)}")
        
        # ============================================================================
        # DATASET SPLIT: Adaptive split based on available patients
        # ============================================================================
        import random
        random.seed(42)  # Fixed seed for reproducibility
        
        # Shuffle all patients for random selection
        shuffled_patients = patient_folders.copy()
        random.shuffle(shuffled_patients)
        
        # Adaptive split based on available patients
        num_patients = len(shuffled_patients)
        if num_patients >= 35:
            # Ideal case: 5/15/15 split
            train_patients = shuffled_patients[:5]
            val_patients = shuffled_patients[5:20]
            test_patients = shuffled_patients[20:35]
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
    print("CREATING STANDARDIZED ENVIRONMENTS")
    print("="*50)
    
    # Get maximum action space size across selected test patients
    max_action_space = get_max_action_space_size(selected_test_patients)
    
    # Create standardized testing environments
    print("\nCreating testing environments...")
    test_envs, _ = create_standardized_environments(selected_test_patients, max_action_space)
    
    if not test_envs:
        print("Error: No valid test environments created!")
        return
    
    print(f"\nSuccessfully created {len(test_envs)} testing environments")
    print(f"Standardized action space size: {max_action_space}")
    
    # Verify all environments have same action space
    for i, env in enumerate(test_envs):
        assert env.action_space.n == max_action_space, f"Environment {i} has wrong action space size: {env.action_space.n} != {max_action_space}"
    print("✓ Verified: All environments have identical action spaces")
    
    # Create random policy
    sample_env = test_envs[0]
    random_policy = RandomPolicy(sample_env.action_space, seed=42)
    
    # ============================================================================
    # EVALUATION PARAMETERS
    # ============================================================================
    
    episodes_per_patient = 10  # Multiple runs per patient for statistics
    
    print(f"\nEvaluation setup:")
    print(f"- Policy: Random")
    print(f"- Episodes per patient: {episodes_per_patient}")
    print(f"- Selected test patients: {len(test_envs)}")
    print(f"- Environment: SpherePlacementEnv (standardized)")
    print(f"- Action space: Discrete({max_action_space})")
    
    # Run evaluation
    print("\nRunning random policy evaluation on selected test patients...")
    all_patient_rewards = []
    all_patient_dice_scores = []
    final_rewards = []
    final_dice_scores = []
    
    start_time = time.time()
    
    for i, test_env in enumerate(test_envs):
        patient_name = os.path.basename(selected_test_patients[i])
        print(f"\nTesting on patient {i+1}/{len(test_envs)}: {patient_name}...")
        
        # Create patient results folder
        patient_folder = os.path.join(results_folder, f'test_patient_{i+1}')
        os.makedirs(patient_folder, exist_ok=True)
        
        # Run multiple episodes for this patient
        patient_rewards, patient_dice_scores = run_random_evaluation(test_env, random_policy, num_episodes=episodes_per_patient)
        
        all_patient_rewards.extend(patient_rewards)
        all_patient_dice_scores.extend(patient_dice_scores)
        
        # Use the best run for visualization
        best_episode_idx = np.argmax(patient_dice_scores)
        best_reward = patient_rewards[best_episode_idx]
        best_dice = patient_dice_scores[best_episode_idx]
        
        final_rewards.append(best_reward)
        final_dice_scores.append(best_dice)
        
        # Run the best episode again for visualization
        obs = test_env.reset()
        done = False
        total_reward = 0
        step_rewards = []
        step_dice_scores = []
        step_num = 0
        
        # Set random seed to reproduce the best episode
        np.random.seed(42 + best_episode_idx)
        random_policy = RandomPolicy(test_env.action_space, seed=42 + best_episode_idx)
        
        # Visualize initial state (no spheres)
        visualize_individual_step_placement(test_env, step_num, patient_folder, i+1, dice_score=0.0)
        
        # Calculate initial dice score (should be 0)
        initial_sphere_mask = create_sphere_mask([], test_env.sphere_radius, test_env.mri_data.shape)
        initial_dice = calculate_dice_score(initial_sphere_mask, test_env.lesion_data)
        step_dice_scores.append(initial_dice)
        
        while not done:
            action, _ = random_policy.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
            step_rewards.append(reward)
            step_num += 1
            
            # Calculate dice score after this placement
            sphere_mask = create_sphere_mask(test_env.sphere_positions, 
                                           test_env.sphere_radius, 
                                           test_env.mri_data.shape)
            dice_score = calculate_dice_score(sphere_mask, test_env.lesion_data)
            step_dice_scores.append(dice_score)
            
            # Enhanced visualization for this placement
            visualize_individual_step_placement(test_env, step_num, patient_folder, i+1, dice_score=dice_score)
            
            print(f"  Step {step_num}: Reward = {reward:.2f}, Dice = {dice_score:.3f}")
        
        # Create sphere progression summary
        create_sphere_progression_summary(test_env, patient_folder, i+1)
        
        # Create final enhanced visualization
        slice_idx = test_env.mri_data.shape[2] // 2
        final_viz_path = os.path.join(patient_folder, f'final_enhanced_result_patient_{i+1}.png')
        enhanced_visualize_spheres_with_numbers(
            env=test_env,
            slice_idx=slice_idx,
            save_path=final_viz_path,
            show=False,
            step_info=f"Final (Dice: {step_dice_scores[-1]:.3f})"
        )
        
        print(f"  Best reward: {best_reward:.2f}, Best dice: {best_dice:.3f}")
        print(f"  Mean reward: {np.mean(patient_rewards):.2f} ± {np.std(patient_rewards):.2f}")
        print(f"  Mean dice: {np.mean(patient_dice_scores):.3f} ± {np.std(patient_dice_scores):.3f}")
        
        # Plot step-by-step progress for this patient with mean lines
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Calculate means for reference lines
        mean_step_reward = np.mean(step_rewards)
        mean_step_dice = np.mean(step_dice_scores)
        
        # Plot step rewards
        ax1.plot(range(1, len(step_rewards) + 1), step_rewards, 'ro-', linewidth=2, markersize=6)
        ax1.axhline(y=mean_step_reward, color='green', linestyle='-', linewidth=2, alpha=0.7, 
                   label=f'Mean Reward: {mean_step_reward:.2f}')
        ax1.set_title(f'Step-by-Step Rewards - Test Patient {i+1}', fontsize=16)
        ax1.set_xlabel('Step', fontsize=14)
        ax1.set_ylabel('Reward', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot step dice scores
        ax2.plot(range(len(step_dice_scores)), step_dice_scores, 'go-', linewidth=2, markersize=6)
        ax2.axhline(y=mean_step_dice, color='green', linestyle='-', linewidth=2, alpha=0.7, 
                   label=f'Mean Dice: {mean_step_dice:.3f}')
        ax2.set_title(f'Step-by-Step Dice Scores - Test Patient {i+1}', fontsize=16)
        ax2.set_xlabel('Step', fontsize=14)
        ax2.set_ylabel('Dice Score', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        step_progress_path = os.path.join(patient_folder, f'step_progress_patient_{i+1}.png')
        plt.savefig(step_progress_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save patient-specific results
        patient_results = {
            'patient_id': i+1,
            'patient_name': patient_name,
            'episode_rewards': patient_rewards,
            'episode_dice_scores': patient_dice_scores,
            'best_reward': best_reward,
            'best_dice': best_dice,
            'mean_reward': np.mean(patient_rewards),
            'std_reward': np.std(patient_rewards),
            'mean_dice': np.mean(patient_dice_scores),
            'std_dice': np.std(patient_dice_scores)
        }
        
        np.savez(os.path.join(patient_folder, f'patient_{i+1}_results.npz'), **patient_results)
    
    evaluation_time = time.time() - start_time
    print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
    
    # Summary statistics
    mean_reward = np.mean(final_rewards)
    std_reward = np.std(final_rewards)
    mean_dice = np.mean(final_dice_scores)
    std_dice = np.std(final_dice_scores)
    
    # Overall statistics across all episodes
    overall_mean_reward = np.mean(all_patient_rewards)
    overall_std_reward = np.std(all_patient_rewards)
    overall_mean_dice = np.mean(all_patient_dice_scores)
    overall_std_dice = np.std(all_patient_dice_scores)
    
    print(f"\nEXPERIMENT 2 FIXED RESULTS:")
    print(f"Tested on {len(test_envs)} randomly selected patients with {episodes_per_patient} episodes each")
    print(f"Standardized action space size: {max_action_space}")
    print(f"Test set performance:")
    print(f"Best performance per patient:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Mean Dice score: {mean_dice:.3f} ± {std_dice:.3f}")
    print(f"Overall performance (all episodes):")
    print(f"  Mean reward: {overall_mean_reward:.2f} ± {overall_std_reward:.2f}")
    print(f"  Mean Dice score: {overall_mean_dice:.3f} ± {overall_std_dice:.3f}")
    
    # Summary plot with mean lines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    bars1 = ax1.bar(range(1, len(final_rewards) + 1), final_rewards, color='orange', edgecolor='red')
    ax1.axhline(y=mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
    ax1.set_title(f'Experiment 2 Fixed: Best Random Policy Test Rewards per Patient ({len(test_envs)} Selected)')
    ax1.set_xlabel('Test Patient')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    bars2 = ax2.bar(range(1, len(final_dice_scores) + 1), final_dice_scores, color='lightcoral', edgecolor='darkred')
    ax2.axhline(y=mean_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dice:.3f}')
    ax2.set_title(f'Experiment 2 Fixed: Best Random Policy Test Dice Scores per Patient ({len(test_envs)} Selected)')
    ax2.set_xlabel('Test Patient')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'final_results_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Distribution plots with mean lines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.hist(all_patient_rewards, bins=10, color='orange', alpha=0.7, edgecolor='red')
    ax1.axvline(overall_mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_mean_reward:.2f}')
    ax1.set_title('Experiment 2 Fixed: Distribution of All Random Policy Test Rewards')
    ax1.set_xlabel('Total Reward')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(all_patient_dice_scores, bins=10, color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.axvline(overall_mean_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_mean_dice:.3f}')
    ax2.set_title('Experiment 2 Fixed: Distribution of All Random Policy Test Dice Scores')
    ax2.set_xlabel('Dice Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'distribution_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results data
    results_data = {
        'method': f'Random Policy Fixed - {len(test_envs)} Selected Patients',
        'algorithm': 'Random',
        'policy': 'Random',
        'max_action_space_size': max_action_space,
        'selected_test_patients': [os.path.basename(p) for p in selected_test_patients],
        'episodes_per_patient': episodes_per_patient,
        'all_rewards': all_patient_rewards,
        'all_dice_scores': all_patient_dice_scores,
        'final_rewards': final_rewards,
        'final_dice_scores': final_dice_scores,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_dice': mean_dice,
        'std_dice': std_dice,
        'overall_mean_reward': overall_mean_reward,
        'overall_std_reward': overall_std_reward,
        'overall_mean_dice': overall_mean_dice,
        'overall_std_dice': overall_std_dice,
        'evaluation_time': evaluation_time,
        'dataset_split': {
            'train_patients_pool': len(train_patients),
            'val_patients': len(val_patients),
            'test_patients_pool': len(test_patients),
            'selected_test_patients': len(selected_test_patients),
            'total_patients': len(patient_folders)
        }
    }
    
    np.savez(os.path.join(results_folder, 'experiment_2_fixed_results.npz'), **results_data)
    
    # Save summary report
    with open(os.path.join(results_folder, 'experiment_2_summary.txt'), 'w') as f:
        f.write("EXPERIMENT 2 FIXED: RANDOM POLICY BASELINE RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Method: Random Policy Fixed ({len(test_envs)} Selected Patients)\n")
        f.write(f"Algorithm: Random Action Selection\n")
        f.write(f"Episodes per patient: {episodes_per_patient}\n")
        f.write(f"Max action space size: {max_action_space}\n")
        f.write(f"Evaluation time: {evaluation_time:.2f} seconds\n\n")
        f.write(f"Selected Test Patients:\n")
        for i, patient_dir in enumerate(selected_test_patients):
            patient_name = os.path.basename(patient_dir)
            f.write(f"  {i+1}. {patient_name}\n")
        f.write(f"\nDataset Split:\n")
        f.write(f"  Training patients pool: {len(train_patients)} (not used)\n")
        f.write(f"  Validation patients: {len(val_patients)} (not used)\n")
        f.write(f"  Testing patients pool: {len(test_patients)}\n")
        f.write(f"  Selected testing patients: {len(selected_test_patients)}\n")
        f.write(f"  Total available patients: {len(patient_folders)}\n\n")
        f.write(f"Test Set Results:\n")
        f.write(f"Best Performance Per Patient:\n")
        f.write(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"Mean Dice score: {mean_dice:.3f} ± {std_dice:.3f}\n\n")
        f.write(f"Overall Performance (All Episodes):\n")
        f.write(f"Mean reward: {overall_mean_reward:.2f} ± {overall_std_reward:.2f}\n")
        f.write(f"Mean Dice score: {overall_mean_dice:.3f} ± {overall_std_dice:.3f}\n\n")
        f.write(f"Individual Best Test Results:\n")
        for i, (reward, dice) in enumerate(zip(final_rewards, final_dice_scores)):
            patient_name = os.path.basename(selected_test_patients[i])
            f.write(f"Test Patient {i+1} ({patient_name}): Reward={reward:.2f}, Dice={dice:.3f}\n")
    
    print(f"\nExperiment 2 Fixed completed successfully!")
    print(f"Results saved in: {results_folder}")
    print(f"Summary saved in: {os.path.join(results_folder, 'experiment_2_fixed_summary.txt')}")
    print(f"Tested on {len(test_envs)} randomly selected patients: {[os.path.basename(p) for p in selected_test_patients]}")
    print(f"Dataset used: {len(patient_folders)} total patients, {len(test_patients)} test pool, {len(selected_test_patients)} tested")
    print(f"Key fix: Standardized action space size to {max_action_space} across all environments")
    
if __name__ == "__main__":
    run_experiment_2()