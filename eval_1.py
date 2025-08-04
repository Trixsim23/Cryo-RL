"""
Model Re-evaluation with Comprehensive Coverage Tracking
=======================================================

This script loads a pre-trained model from a zip file and re-evaluates it with 
comprehensive coverage tracking, using the EXACT same evaluation patients that 
were used during original training.

Features:
✓ Load any saved PPO model (Dense Reward, Current Method, etc.)
✓ Reproduce exact same patient split using same random seed
✓ Create same test environments with standardized action spaces
✓ Run comprehensive evaluation with coverage tracking
✓ Generate all visualization types (2D, multi-view, 3D optional)
✓ Calculate detailed coverage statistics
✓ Save comprehensive results and comparisons

Usage:
    python reevaluate_model.py --model_path ./path/to/trained_model.zip 
                               --results_folder ./reevaluation_results
                               --experiment_name "Dense_Reward_RL_Reevaluation"

Or modify the script parameters directly and run:
    python reevaluate_model.py
"""

import os
import time 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import the fixed environment functions
from main_env import SpherePlacementEnv, get_max_action_space_size, create_standardized_environments

# Import all enhanced functions for comprehensive evaluation
from agent_vis_2 import (
    resize_volume,
    load_and_preprocess_patient_data,
    create_sphere_mask,
    calculate_dice_score,
    enhanced_visualize_spheres_with_numbers,
    visualize_3d_volume_rendering,
    visualize_individual_step_placement,
    create_sphere_progression_summary,
    create_final_comprehensive_evaluation_2d_only,
    run_enhanced_evaluation_loop,
    TrainingCallback
)

# Also support Dense Reward environments if needed
try:
    from paste import DenseRewardSpherePlacementEnv, create_dense_reward_environments
    DENSE_REWARD_AVAILABLE = True
except ImportError:
    print("Dense Reward environment not available - will use standard environments")
    DENSE_REWARD_AVAILABLE = False


# ============================================================================
# DATASET SETUP WITH FILTERED PATIENTS (SAME AS TRAINING)
# ============================================================================

def get_filtered_dataset(filtered_dir="./filtered_dataset"):
    """Get all patient folders from the filtered dataset - SAME AS TRAINING."""
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


def reproduce_patient_split(patient_folders, random_seed=42):
    """
    Reproduce the EXACT same patient split as used during training.
    This ensures we evaluate on the same test patients.
    """
    import random
    random.seed(random_seed)  # SAME seed as training
    
    # Shuffle all patients for random selection (SAME as training)
    shuffled_patients = patient_folders.copy()
    random.shuffle(shuffled_patients)
    
    # Adaptive split based on available patients (SAME logic as training)
    num_patients = len(shuffled_patients)
    if num_patients >= 35:
        # Ideal case: 5/15/15 split
        train_patients = shuffled_patients[:1]  # Matches training script
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
    
    return train_patients, val_patients, test_patients


# ============================================================================
# MODEL RE-EVALUATION WITH COMPREHENSIVE COVERAGE TRACKING
# ============================================================================

def reevaluate_model_with_coverage(model_path, results_folder, experiment_name, 
                                 environment_type="standard", create_3d=True, 
                                 random_seed=42):
    """
    Re-evaluate a pre-trained model with comprehensive coverage tracking.
    
    Args:
        model_path: Path to the saved model zip file
        results_folder: Folder to save re-evaluation results
        experiment_name: Name for this re-evaluation
        environment_type: "standard" or "dense_reward"
        create_3d: Whether to create 3D visualizations
        random_seed: Random seed to reproduce patient split (default: 42)
    """
    
    print("=" * 80)
    print(f"MODEL RE-EVALUATION WITH COMPREHENSIVE COVERAGE TRACKING")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Experiment: {experiment_name}")
    print(f"Environment type: {environment_type}")
    print(f"Results folder: {results_folder}")
    print(f"Random seed: {random_seed} (for reproducible patient split)")
    print(f"3D visualizations: {'ENABLED' if create_3d else 'DISABLED'}")
    print("=" * 80)
    
    # Create results folder
    current_time = time.strftime("%Y%m%d-%H%M%S")
    full_results_folder = f"{results_folder}_{current_time}"
    os.makedirs(full_results_folder, exist_ok=True)
    
    # ============================================================================
    # LOAD PRE-TRAINED MODEL
    # ============================================================================
    
    print(f"\n{'='*50}")
    print("LOADING PRE-TRAINED MODEL")
    print(f"{'='*50}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    try:
        # Load the model
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path)
        print(f"✓ Model loaded successfully")
        print(f"✓ Model policy: {type(model.policy).__name__}")
        print(f"✓ Model action space: {model.action_space}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # ============================================================================
    # REPRODUCE EXACT SAME DATASET SPLIT
    # ============================================================================
    
    print(f"\n{'='*50}")
    print("REPRODUCING DATASET SPLIT")
    print(f"{'='*50}")
    
    try:
        patient_folders = get_filtered_dataset()
        print(f"Found {len(patient_folders)} valid patients")
        
        # Reproduce the exact same split
        train_patients, val_patients, test_patients = reproduce_patient_split(
            patient_folders, random_seed=random_seed
        )
        
        print(f"Reproduced split: {len(train_patients)}/{len(val_patients)}/{len(test_patients)}")
        print(f"  Training: {len(train_patients)} patients")
        print(f"  Validation: {len(val_patients)} patients") 
        print(f"  Testing: {len(test_patients)} patients")
        
        # Log the test patients we'll evaluate on
        print(f"\nTest patients for evaluation:")
        for i, patient_dir in enumerate(test_patients):
            patient_name = os.path.basename(patient_dir)
            print(f"  {i+1}. {patient_name}")
        
        if len(test_patients) == 0:
            raise ValueError("No test patients available for evaluation!")
            
    except Exception as e:
        print(f"Error reproducing dataset split: {e}")
        return None
    
    # ============================================================================
    # CREATE STANDARDIZED TEST ENVIRONMENTS
    # ============================================================================
    
    print(f"\n{'='*50}")
    print("CREATING TEST ENVIRONMENTS")
    print(f"{'='*50}")
    
    try:
        # Get maximum action space size (same as training)
        all_patients_for_analysis = train_patients + val_patients + test_patients
        max_action_space = get_max_action_space_size(all_patients_for_analysis)
        print(f"Calculated max action space size: {max_action_space}")
        
        # Verify model compatibility
        if hasattr(model.action_space, 'n'):
            model_action_space = model.action_space.n
            if model_action_space != max_action_space:
                print(f"WARNING: Model action space ({model_action_space}) != calculated ({max_action_space})")
                print(f"Using model's action space size: {model_action_space}")
                max_action_space = model_action_space
        
        # Create test environments
        if environment_type == "dense_reward" and DENSE_REWARD_AVAILABLE:
            print("Creating dense reward test environments...")
            test_envs, _ = create_dense_reward_environments(test_patients, max_action_space)
        else:
            print("Creating standard test environments...")
            test_envs, _ = create_standardized_environments(test_patients, max_action_space)
        
        if not test_envs:
            raise ValueError("No valid test environments created!")
        
        print(f"✓ Successfully created {len(test_envs)} test environments")
        print(f"✓ Standardized action space size: {max_action_space}")
        
        # Verify environment compatibility
        for i, env in enumerate(test_envs):
            if env.action_space.n != max_action_space:
                print(f"WARNING: Environment {i} has action space {env.action_space.n} != {max_action_space}")
        
    except Exception as e:
        print(f"Error creating test environments: {e}")
        return None
    
    # ============================================================================
    # RUN COMPREHENSIVE EVALUATION WITH COVERAGE TRACKING
    # ============================================================================
    
    print(f"\n{'='*60}")
    print("RUNNING COMPREHENSIVE EVALUATION WITH COVERAGE TRACKING")
    print(f"{'='*60}")
    print("This will create for each patient:")
    print("✓ Enhanced 2D visualization")
    print("✓ Multi-view medical imaging (sagittal, coronal, axial)")
    print("✓ Detailed quantitative analysis plots")
    print("✓ Sphere placement progression summaries")
    print("✓ COMPREHENSIVE COVERAGE TRACKING AND STATISTICS")
    print("✓ Coverage percentage tracking")
    print("✓ Comprehensive JSON results")
    print("✓ Step-by-step progress tracking")
    if create_3d:
        print("✓ 3D volume rendering (ENABLED)")
    else:
        print("⚪ 3D volume rendering (DISABLED)")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Use the enhanced evaluation loop for comprehensive coverage tracking
        final_rewards, final_dice_scores, comprehensive_results = run_enhanced_evaluation_loop(
            model, test_envs, full_results_folder, create_3d=create_3d
        )
        
        evaluation_time = time.time() - start_time
        print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None
    
    # ============================================================================
    # COMPREHENSIVE RESULTS ANALYSIS
    # ============================================================================
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate comprehensive statistics
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
        print(f"  Patients evaluated: {len(test_patients)}")
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
            patient_name = os.path.basename(test_patients[i])
            print(f"  {patient_name}: Reward={reward:.2f}, Dice={dice:.3f}, Coverage={coverage:.1f}%, Spheres={spheres}")
    
    else:
        coverage_percentages = []
        mean_coverage = std_coverage = mean_spheres = 0
        print(f"Basic Results:")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Mean Dice score: {mean_dice:.3f} ± {std_dice:.3f}")
    
    # ============================================================================
    # CREATE COMPREHENSIVE SUMMARY VISUALIZATIONS (MATCHING EXPERIMENT 1)
    # ============================================================================
    
    print(f"\nCreating comprehensive summary visualizations...")
    
    # 1. ENHANCED FINAL RESULTS SUMMARY (matches experiment 1 format)
    if final_rewards and final_dice_scores:
        if coverage_percentages:
            # Three-metric summary plot (same as experiment 1)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        else:
            # Two-metric summary plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot final rewards (matching experiment 1 style)
        bars1 = ax1.bar(range(1, len(final_rewards) + 1), final_rewards, 
                       color='skyblue', edgecolor='navy', linewidth=1.5)
        ax1.axhline(y=mean_reward, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_reward:.2f}')
        ax1.set_title(f'Re-evaluation: {experiment_name} - Final Test Rewards per Patient', fontsize=14)
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
        
        # Plot final Dice scores (matching experiment 1 style)
        bars2 = ax2.bar(range(1, len(final_dice_scores) + 1), final_dice_scores, 
                       color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
        ax2.axhline(y=mean_dice, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_dice:.3f}')
        ax2.set_title(f'Re-evaluation: {experiment_name} - Final Test Dice Scores per Patient', fontsize=14)
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
        
        # Plot lesion coverage if available (matching experiment 1 style)
        if coverage_percentages:
            bars3 = ax3.bar(range(1, len(coverage_percentages) + 1), coverage_percentages, 
                           color='lightcoral', edgecolor='darkred', linewidth=1.5)
            ax3.axhline(y=mean_coverage, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_coverage:.1f}%')
            ax3.set_title(f'Re-evaluation: {experiment_name} - Lesion Coverage per Patient', fontsize=14)
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
        # Use experiment 1 naming convention
        enhanced_summary_path = os.path.join(full_results_folder, 'final_results_summary.png')
        plt.savefig(enhanced_summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Enhanced final results summary saved: {enhanced_summary_path}")
    
    # 2. CREATE TRAINING PROGRESS VISUALIZATION (even for re-evaluation)
    # This shows that no additional training was done during re-evaluation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Since this is re-evaluation, show final results as single points
    ax1.scatter([1], [mean_reward], color='blue', s=100, zorder=5)
    ax1.axhline(y=mean_reward, color='green', linestyle='-', linewidth=2, alpha=0.7, 
               label=f'Re-evaluation Mean Reward: {mean_reward:.2f}')
    ax1.set_title(f'Re-evaluation: {experiment_name} - Final Results Overview', fontsize=16)
    ax1.set_xlabel('Re-evaluation Run')
    ax1.set_ylabel('Mean Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0.5, 1.5)
    
    ax2.scatter([1], [mean_dice], color='red', s=100, zorder=5)
    ax2.axhline(y=mean_dice, color='green', linestyle='-', linewidth=2, alpha=0.7, 
               label=f'Re-evaluation Mean Dice: {mean_dice:.3f}')
    ax2.set_title(f'Re-evaluation: {experiment_name} - Dice Score Overview', fontsize=16)
    ax2.set_xlabel('Re-evaluation Run')
    ax2.set_ylabel('Mean Dice Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0.5, 1.5)
    
    plt.tight_layout()
    # Use experiment 1 naming convention for training progress
    training_progress_path = os.path.join(full_results_folder, 'training_progress.png')
    plt.savefig(training_progress_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Enhanced training progress overview saved: {training_progress_path}")
    
    # ============================================================================
    # SAVE COMPREHENSIVE RE-EVALUATION RESULTS (MATCHING EXPERIMENT 1 FORMAT)
    # ============================================================================
    
    print(f"\nSaving comprehensive re-evaluation results...")
    
    # Save comprehensive results data (matching experiment 1 structure)
    results_data = {
        'experiment_name': f'Re-evaluation: {experiment_name} with Comprehensive Coverage Tracking',
        'method': f'{experiment_name} Re-evaluation - Comprehensive Coverage Tracking',
        'algorithm': 'PPO',
        'policy': type(model.policy).__name__ if model else 'Unknown',
        'environment_type': environment_type,
        'reevaluation_info': {
            'original_model_path': model_path,
            'reevaluation_timestamp': current_time,
            'random_seed_used': random_seed,
            'evaluation_time_seconds': evaluation_time
        },
        'model_info': {
            'max_action_space_size': max_action_space,
            'policy_type': type(model.policy).__name__ if model else 'Unknown'
        },
        'training_patients': [os.path.basename(p) for p in train_patients],  # For reference
        'timesteps_per_patient': 'N/A - Re-evaluation only',
        'total_training_timesteps': 'N/A - Re-evaluation only',
        'training_timesteps': [],  # Empty for re-evaluation
        'training_rewards': [],    # Empty for re-evaluation
        'training_dice_scores': [], # Empty for re-evaluation
        'final_rewards': final_rewards,
        'final_dice_scores': final_dice_scores,
        'lesion_coverage_percentages': coverage_percentages,  # Key addition matching experiment 1
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_dice': mean_dice,
        'std_dice': std_dice,
        'mean_coverage': mean_coverage,  # Key addition matching experiment 1
        'std_coverage': std_coverage,    # Key addition matching experiment 1
        'mean_spheres_per_patient': mean_spheres,  # Key addition matching experiment 1
        'training_time': 0,  # No training time for re-evaluation
        'comprehensive_patient_results': [result['results'] for result in comprehensive_results] if comprehensive_results else [],  # Key addition
        'visualization_types_created': [
            'enhanced_2d', 'multiview', 'detailed_analysis', 'progression_summary'
        ] + (['3d_volume'] if create_3d else []),
        'dataset_split': {
            'train_patients': len(train_patients),
            'val_patients': len(val_patients),
            'test_patients': len(test_patients),
            'total_patients': len(patient_folders)
        }
    }
    
    # Save as npz (matching experiment 1 naming convention)
    results_file = os.path.join(full_results_folder, 'enhanced_experiment_evaluation_results.npz')
    np.savez(results_file, **results_data)
    print(f"✓ Enhanced results data saved: {results_file}")
    
    # Save comprehensive summary report (matching experiment 1 format and detail level)
    summary_file = os.path.join(full_results_folder, 'enhanced_experiment_evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("MODEL RE-EVALUATION WITH COMPREHENSIVE COVERAGE TRACKING\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Re-evaluation completed: {current_time}\n")
        f.write(f"Experiment name: {experiment_name}\n")
        f.write(f"Method: {experiment_name} Re-evaluation - Comprehensive Coverage Tracking\n")
        f.write(f"Algorithm: PPO\n")
        f.write(f"Policy: {type(model.policy).__name__ if model else 'Unknown'}\n")
        f.write(f"Environment type: {environment_type}\n")
        f.write(f"Original model: {model_path}\n")
        f.write(f"Random seed used: {random_seed}\n")
        f.write(f"Evaluation time: {evaluation_time:.2f} seconds\n")
        f.write(f"Max action space size: {max_action_space}\n\n")
        
        f.write(f"Coverage Tracking Features (like Experiment 1):\n")
        f.write(f"✓ Enhanced 2D visualization (original style)\n")
        f.write(f"✓ Multi-view medical imaging (sagittal, coronal, axial)\n")
        f.write(f"✓ Detailed quantitative analysis plots\n")
        f.write(f"✓ Sphere placement progression summaries\n")
        f.write(f"✓ COMPREHENSIVE coverage percentage tracking\n")
        f.write(f"✓ Coverage statistics calculation and reporting\n")
        f.write(f"✓ Comprehensive JSON results for each patient\n")
        if create_3d:
            f.write(f"✓ 3D volume rendering\n")
        else:
            f.write(f"⚪ 3D volume rendering (disabled for speed)\n")
        f.write("\n")
        
        f.write(f"Training Patients (Used in Original Training):\n")
        for i, patient_dir in enumerate(train_patients):
            patient_name = os.path.basename(patient_dir)
            f.write(f"  {i+1}. {patient_name}\n")
        f.write("\n")
        
        f.write(f"Dataset Split (Reproduced from Original Training):\n")
        f.write(f"  Total patients available: {len(patient_folders)}\n")
        f.write(f"  Training patients: {len(train_patients)}\n")
        f.write(f"  Validation patients: {len(val_patients)}\n") 
        f.write(f"  Testing patients: {len(test_patients)} (re-evaluated)\n\n")
        
        f.write(f"Test Patients Re-evaluated:\n")
        for i, patient_dir in enumerate(test_patients):
            patient_name = os.path.basename(patient_dir)
            f.write(f"  {i+1}. {patient_name}\n")
        f.write("\n")
        
        f.write(f"Re-evaluation Results:\n")
        f.write(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"  Mean Dice score: {mean_dice:.3f} ± {std_dice:.3f}\n")
        if coverage_percentages:
            f.write(f"  Mean lesion coverage: {mean_coverage:.1f}% ± {std_coverage:.1f}%\n")
            f.write(f"  Mean spheres per patient: {mean_spheres:.1f}\n")
        f.write("\n")
        
        if final_rewards:
            f.write(f"Individual Patient Results:\n")
            for i, (reward, dice) in enumerate(zip(final_rewards, final_dice_scores)):
                coverage = coverage_percentages[i] if i < len(coverage_percentages) else 0
                patient_name = os.path.basename(test_patients[i])
                f.write(f"  {patient_name}: Reward={reward:.2f}, Dice={dice:.3f}, Coverage={coverage:.1f}%\n")
        
        f.write(f"\nVisualization Output Structure:\n")
        f.write(f"  Each patient folder contains:\n")
        f.write(f"  - Enhanced 2D visualizations (step-by-step + final)\n")
        f.write(f"  - Multi-view medical imaging (step-by-step + final)\n")
        f.write(f"  - Detailed quantitative analysis plot\n")
        f.write(f"  - Sphere placement progression summary\n")
        f.write(f"  - Comprehensive JSON results with coverage data\n")
        f.write(f"  - Step-by-step progress plots\n")
        if create_3d:
            f.write(f"  - 3D volume renderings (step-by-step + final)\n")
        
        f.write(f"\nRe-evaluation Notes:\n")
        f.write(f"  - No additional training was performed\n")
        f.write(f"  - Same test patients as original training evaluation\n")
        f.write(f"  - Comprehensive coverage tracking added\n")
        f.write(f"  - All visualization types from Experiment 1 included\n")
        f.write(f"  - Results directly comparable to original training\n")
        
    print(f"✓ Enhanced summary report saved: {summary_file}")
    
    print(f"\n{'='*80}")
    print("ENHANCED MODEL RE-EVALUATION WITH COMPREHENSIVE COVERAGE TRACKING COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved in: {full_results_folder}")
    print(f"Summary report: {summary_file}")
    print(f"\nKey files created (matching Experiment 1 format):")
    print(f"✓ enhanced_final_results_summary.png - Comprehensive 3-panel summary")
    print(f"✓ enhanced_training_progress.png - Re-evaluation overview")
    print(f"✓ enhanced_experiment_reevaluation_results.npz - Complete data")
    print(f"✓ enhanced_experiment_reevaluation_summary.txt - Detailed report")
    print(f"\nPatient-level outputs (in individual folders):")
    print(f"✓ Enhanced 2D visualizations (step-by-step + final)")
    print(f"✓ Multi-view medical imaging (sagittal, coronal, axial)")
    print(f"✓ Detailed quantitative analysis plots")
    print(f"✓ Sphere placement progression summaries")
    print(f"✓ Comprehensive JSON results with coverage data")
    print(f"✓ Step-by-step progress plots")
    if create_3d:
        print(f"✓ 3D volume renderings (step-by-step + final)")
    else:
        print(f"⚪ 3D volume renderings (disabled for speed)")
    print(f"\nEvaluation Summary:")
    print(f"  Patients evaluated: {len(test_patients)}")
    if coverage_percentages:
        print(f"  Mean coverage: {mean_coverage:.1f}% ± {std_coverage:.1f}%")
    print(f"  Mean Dice: {mean_dice:.3f} ± {std_dice:.3f}")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"{'='*80}")
    
    return {
        'results_folder': full_results_folder,
        'mean_reward': mean_reward,
        'mean_dice': mean_dice,
        'mean_coverage': mean_coverage if coverage_percentages else None,
        'patients_evaluated': len(test_patients)
    }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Re-evaluate a trained model with comprehensive coverage tracking')
    parser.add_argument('--model_path', type=str, 
                       default='./experiment_3_alt_20250130-143422/trained_model.zip',
                       help='Path to the saved model zip file')
    parser.add_argument('--results_folder', type=str, 
                       default='./reevaluation_results',
                       help='Base folder name for saving results')
    parser.add_argument('--experiment_name', type=str, 
                       default='Model_Reevaluation',
                       help='Name for this re-evaluation experiment')
    parser.add_argument('--environment_type', type=str, 
                       choices=['standard', 'dense_reward'], 
                       default='dense_reward',
                       help='Type of environment to use')
    parser.add_argument('--create_3d', type=bool, 
                       default=True,
                       help='Whether to create 3D visualizations')
    parser.add_argument('--random_seed', type=int, 
                       default=42,
                       help='Random seed for reproducible patient split')
    
    args = parser.parse_args()
    
    # Run the re-evaluation
    results = reevaluate_model_with_coverage(
        model_path=args.model_path,
        results_folder=args.results_folder,
        experiment_name=args.experiment_name,
        environment_type=args.environment_type,
        create_3d=args.create_3d,
        random_seed=args.random_seed
    )
    
    if results:
        print(f"\nRe-evaluation successful!")
        print(f"Results: {results}")
    else:
        print(f"\nRe-evaluation failed!")


if __name__ == "__main__":
    # OPTION 1: Direct parameters (modify these as needed)
    MODEL_PATH = "./experiment_1_results_20250727-143438/trained_model.zip"  # UPDATE THIS PATH
    RESULTS_FOLDER = "./dense_reward_exp1_1_patients"
    EXPERIMENT_NAME = "Sparse_Reward_RL_Reevaluation"
    ENVIRONMENT_TYPE = "sparse_reward"  # or "standard"
    CREATE_3D = True  # Set to False to speed up evaluation
    RANDOM_SEED = 42  # Keep as 42 to match original training
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script to point to your saved model.")
        print("Example paths:")
        print("  ./experiment_1_enhanced_TIMESTAMP/trained_model.zip")
        print("  ./experiment_3_enhanced_coverage_TIMESTAMP/trained_model.zip")
        exit(1)
    
    print(f"Using model: {MODEL_PATH}")
    print(f"Environment type: {ENVIRONMENT_TYPE}")
    print("Starting re-evaluation...")
    
    # Run re-evaluation with direct parameters
    results = reevaluate_model_with_coverage(
        model_path=MODEL_PATH,
        results_folder=RESULTS_FOLDER,
        experiment_name=EXPERIMENT_NAME,
        environment_type=ENVIRONMENT_TYPE,
        create_3d=CREATE_3D,
        random_seed=RANDOM_SEED
    )
    
    if results:
        print(f"\nRe-evaluation successful!")
    else:
        print(f"\nRe-evaluation failed!")
    
    # OPTION 2: Use command line arguments (uncomment next line instead of above)
    # main()