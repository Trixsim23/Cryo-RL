"""
Fixed Experiment 1: Current Method (Baseline) with Standardized Action Spaces
============================================================================

This fixes the action space mismatch error by standardizing action spaces across all environments.

Usage:
    python experiment_1.py

Results saved in: ./experiment_1_results_TIMESTAMP/
"""

import os
import time 
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import the fixed environment functions
from main_env import SpherePlacementEnv, get_max_action_space_size, create_standardized_environments

# Import all enhanced functions from the new main_agent.py
from agent_vis_2 import (
    resize_volume,
    load_and_preprocess_patient_data,
    create_sphere_mask,
    calculate_dice_score,
    enhanced_visualize_spheres_with_numbers,
    visualize_multi_view_spheres,                    
    visualize_3d_volume_rendering,                   
    visualize_individual_step_placement,             
    create_sphere_progression_summary,
    create_final_comprehensive_evaluation,           
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
# EXPERIMENT 1: CURRENT METHOD WITH FIXED ACTION SPACES
# ============================================================================

def run_experiment_1():
    """Run Experiment 1: Current Method (Baseline) with action spaces"""
   
    print("EXPERIMENT 1: CURRENT METHOD (BASELINE) - FIXED")
    
    # Setup results folder
    current_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = f"./experiment_1_results_{current_time}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Load dataset
    try:
        patient_folders = get_filtered_dataset()
        print(f"Found {len(patient_folders)} valid patients")
        
        if len(patient_folders) < 35:
            print(f"Warning: Only {len(patient_folders)} patients available")
            if len(patient_folders) < 5:
                raise ValueError(f"Need at least 5 patients for training, found {len(patient_folders)}")
        
        # ============================================================================
        # DATASET SPLIT: Use available patients with minimum 5 for training
        # ============================================================================
        
        import random
        random.seed(42)  # Fixed seed for reproducibility
        
        # Shuffle all patients for random selection
        shuffled_patients = patient_folders.copy()
        random.shuffle(shuffled_patients)
        
        # Adaptive split based on available patients
        num_patients = len(shuffled_patients)
        if num_patients >= 35:
            # Ideal case: 10/15/15 split
            train_patients = shuffled_patients[:10]
            val_patients = shuffled_patients[10:25]
            test_patients = shuffled_patients[25:40]
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
    
    # Get maximum action space size across all patients that will be used
    all_patients_for_analysis = train_patients + val_patients + test_patients
    max_action_space = get_max_action_space_size(all_patients_for_analysis)
    
    # Create standardized training environments
    print("\nCreating training environments...")
    train_envs, _ = create_standardized_environments(train_patients, max_action_space)
    
    # Create validation environments  
    print("\nCreating validation environments...")
    if val_patients:
        val_envs, _ = create_standardized_environments(val_patients, max_action_space)
    else:
        val_envs = [train_envs[0]]  # Use first training env for validation if no val patients
    
    # Create testing environments
    print("\nCreating testing environments...")
    if test_patients:
        test_envs, _ = create_standardized_environments(test_patients, max_action_space)
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
    #supposed to be 100000
    timesteps_per_patient = 50000000  # increased for more thorough training
    total_timesteps = len(train_envs) * timesteps_per_patient
    eval_freq = 2000
    
    print(f"\nTraining setup:")
    print(f"- Algorithm: PPO")
    print(f"- Policy: MlpPolicy") 
    print(f"- Training patients: {len(train_envs)}")
    print(f"- Timesteps per patient: {timesteps_per_patient:,}")
    print(f"- Total timesteps: {total_timesteps:,}")
    print(f"- Evaluation frequency: {eval_freq:,} steps")
    print(f"- Standardized action space size: {max_action_space}")
    
    # Use first validation environment for callback evaluation
    val_env = val_envs[0]
    
    # ============================================================================
    # TRAINING APPROACH: Train sequentially but with model continuity
    # ============================================================================
    
    # Initialize model with first training environment
    print(f"\nInitializing PPO model with action space size: {train_envs[0].action_space.n}")
    model = PPO("MlpPolicy", train_envs[0], verbose=1, n_steps=512)
    
    # Training callback for monitoring
    callback = TrainingCallback(val_env, eval_freq=eval_freq, verbose=1)
    
    print("\nStarting sequential training on patients...")
    start_time = time.time()
    
    # Option 1: Train on all patients in sequence using the same model
    # This works now because all environments have the same action space
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
    # TRAINING PROGRESS VIZ
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
        ax1.set_title('Experiment 1: Training Progress - Rewards', fontsize=16)
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
        ax2.set_title('Experiment 1: Training Progress - Dice Scores', fontsize=16)
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
    plt.savefig(os.path.join(results_folder, 'enhanced_training_progress.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # FINAL EVAL
    # ============================================================================
    
    print ("\n Running final evaluation on the test set")
    
    # Option to enable 3D visualization (computationally expensive)
    create_3d_visualizations = True  # Set to True if you want 3D visualizations
    if create_3d_visualizations:
        print("3D RENDER ENABLED")
    else:
        print("NO 3D RENDER")
    
    
    # Use the new enhanced evaluation loop
    final_rewards, final_dice_scores, comprehensive_results = run_enhanced_evaluation_loop(
        model, test_envs, results_folder, create_3d=create_3d_visualizations
    )
    
    # ============================================================================
    #  RESULTS SUMMARY AND ANALYSIS
    # ============================================================================
    
    # Summary statistics
    mean_reward = np.mean(final_rewards) if final_rewards else 0
    std_reward = np.std(final_rewards) if final_rewards else 0
    mean_dice = np.mean(final_dice_scores) if final_dice_scores else 0
    std_dice = np.std(final_dice_scores) if final_dice_scores else 0
    
    # Calculate additional comprehensive metrics
    if comprehensive_results:
        coverage_percentages = [result['results']['lesion_coverage_percentage'] for result in comprehensive_results]
        mean_coverage = np.mean(coverage_percentages)
        std_coverage = np.std(coverage_percentages)
        
        total_spheres_placed = [result['results']['total_spheres'] for result in comprehensive_results]
        mean_spheres = np.mean(total_spheres_placed)
    else:
        coverage_percentages = []
        mean_coverage = std_coverage = mean_spheres = 0
    
    print(f"\n" + "="*60)
    print("EXPERIMENT 1  RESULTS")
    print("="*60)
    print(f"Training Configuration:")
    print(f"  Patients trained: {len(train_envs)}")
    print(f"  Timesteps per patient: {timesteps_per_patient:,}")
    print(f"  Total training timesteps: {total_timesteps:,}")
    print(f"  Standardized action space size: {max_action_space}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"\nTest Set Performance:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Mean Dice score: {mean_dice:.3f} ± {std_dice:.3f}")
    if coverage_percentages:
        print(f"  Mean lesion coverage: {mean_coverage:.1f}% ± {std_coverage:.1f}%")
        print(f"  Mean spheres per patient: {mean_spheres:.1f}")
    
    if final_rewards:
        print(f"\nIndividual Patient Results:")
        for i, (reward, dice) in enumerate(zip(final_rewards, final_dice_scores)):
            coverage = coverage_percentages[i] if i < len(coverage_percentages) else 0
            print(f"  Patient {i+1}: Reward={reward:.2f}, Dice={dice:.3f}, Coverage={coverage:.1f}%")
    
    # ============================================================================
    # ENHANCED SUMMARY VISUALIZATIONS
    # ============================================================================
    
    # Create comprehensive summary plot
    if final_rewards and final_dice_scores:
        if coverage_percentages:
            # Three-metric summary plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        else:
            # Two-metric summary plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot final rewards
        bars1 = ax1.bar(range(1, len(final_rewards) + 1), final_rewards, 
                       color='skyblue', edgecolor='navy', linewidth=1.5)
        ax1.axhline(y=mean_reward, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_reward:.2f}')
        ax1.set_title('Enhanced Experiment 1: Final Test Rewards per Patient', fontsize=14)
        ax1.set_xlabel('Test Patient', fontsize=12)
        ax1.set_ylabel('Total Reward', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(final_rewards),
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot final Dice scores
        bars2 = ax2.bar(range(1, len(final_dice_scores) + 1), final_dice_scores, 
                       color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
        ax2.axhline(y=mean_dice, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_dice:.3f}')
        ax2.set_title('Enhanced Experiment 1: Final Test Dice Scores per Patient', fontsize=14)
        ax2.set_xlabel('Test Patient', fontsize=12)
        ax2.set_ylabel('Dice Score', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(final_dice_scores),
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot lesion coverage if available
        if coverage_percentages:
            bars3 = ax3.bar(range(1, len(coverage_percentages) + 1), coverage_percentages, 
                           color='lightcoral', edgecolor='darkred', linewidth=1.5)
            ax3.axhline(y=mean_coverage, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_coverage:.1f}%')
            ax3.set_title('Enhanced Experiment 1: Lesion Coverage per Patient', fontsize=14)
            ax3.set_xlabel('Test Patient', fontsize=12)
            ax3.set_ylabel('Coverage Percentage (%)', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(coverage_percentages),
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, 'enhanced_final_results_summary.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # ============================================================================
    # SAVE COMPREHENSIVE RESULTS DATA
    # ============================================================================
    
    # Save enhanced results data
    results_data = {
        'experiment_name': 'SPARSE REWARDS Visualization',
        'method': 'SPARSE REWARDS',
        'algorithm': 'PPO',
        'policy': 'MlpPolicy',
        'training_patients': [os.path.basename(p) for p in train_patients],
        'timesteps_per_patient': timesteps_per_patient,
        'total_training_timesteps': total_timesteps,
        'max_action_space_size': max_action_space,
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
        'mean_coverage': mean_coverage if coverage_percentages else 0,
        'std_coverage': std_coverage if coverage_percentages else 0,
        'mean_spheres_per_patient': mean_spheres if coverage_percentages else 0,
        'training_time': training_time,
        'comprehensive_patient_results': [result['results'] for result in comprehensive_results],
        'visualization_types_created': [
            'enhanced_2d', 'multiview', 'detailed_analysis', 'progression_summary'
        ] + (['3d_volume'] if create_3d_visualizations else []),
        'dataset_split': {
            'train_patients': len(train_patients),
            'val_patients': len(val_patients),
            'test_patients': len(test_patients),
            'total_patients': len(patient_folders)
        }
    }
    
    np.savez(os.path.join(results_folder, 'experiment_1_results.npz'), **results_data)
    
    # Save comprehensive summary report
    with open(os.path.join(results_folder, 'experiment_1_summary.txt'), 'w') as f:
        f.write("EXPERIMENT 1\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Experiment completed: {current_time}\n")
        f.write(f"Method: SPARSE REWARDS\n")
        f.write(f"Algorithm: PPO\n")
        f.write(f"Policy: MlpPolicy\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Timesteps per patient: {timesteps_per_patient:,}\n")
        f.write(f"Total timesteps: {total_timesteps:,}\n")
        f.write(f"Max action space size: {max_action_space}\n")
        f.write(f"Evaluation frequency: {eval_freq:,} steps\n\n")
        
        if create_3d_visualizations:
            f.write(f" 3D volume rendering\n")
        else:
            f.write(f"NO 3D\n")
        f.write("\n")
        
        f.write(f"Training Patients:\n")
        for i, patient_dir in enumerate(train_patients):
            patient_name = os.path.basename(patient_dir)
            f.write(f"  {i+1}. {patient_name}\n")
        
        f.write(f"\nDataset Split:\n")
        f.write(f"  Training patients: {len(train_patients)}\n")
        f.write(f"  Validation patients: {len(val_patients)}\n")
        f.write(f"  Testing patients: {len(test_patients)}\n")
        f.write(f"  Total available patients: {len(patient_folders)}\n\n")
        
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
                f.write(f"  Patient {i+1}: Reward={reward:.2f}, Dice={dice:.3f}, Coverage={coverage:.1f}%\n")
        
        f.write(f"\nVisualization Output Structure:\n")
        f.write(f"  Each patient folder contains:\n")
        f.write(f"  - Enhanced 2D visualizations (step-by-step + final)\n")
        f.write(f"  - Multi-view medical imaging (step-by-step + final)\n")
        f.write(f"  - Detailed quantitative analysis plot\n")
        f.write(f"  - Sphere placement progression summary\n")
        f.write(f"  - Comprehensive JSON results\n")
        f.write(f"  - Step-by-step progress plots\n")
        if create_3d_visualizations:
            f.write(f"  - 3D volume renderings (step-by-step + final)\n")
    
    print(f"\n" + "="*60)
    print("ENHANCED EXPERIMENT 1 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved in: {results_folder}")
    print(f"Summary report: {os.path.join(results_folder, 'experiment_1_summary.txt')}")

    
    if create_3d_visualizations:
        print(f"  3D volume renderings")
    else:
        print(f"   3D volume renderings (disabled for speed)")
    
    

if __name__ == "__main__":
    run_experiment_1()