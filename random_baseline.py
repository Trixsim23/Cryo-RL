# random_baseline_standalone.py - Standalone random placement baseline

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import center_of_mass
from typing import List, Dict

# Import your existing functions
from main_agent import  calculate_dice_score,load_and_preprocess_patient_data

def create_sphere_mask(sphere_positions, sphere_radius, volume_shape):
    mask = np.zeros(volume_shape, dtype=bool)

    for center in sphere_positions:
        x, y, z = center

        x_range = np.arange(max(0, x - sphere_radius), min(volume_shape[0], x + sphere_radius + 1))
        y_range = np.arange(max(0, y - sphere_radius), min(volume_shape[1], y + sphere_radius + 1))
        z_range = np.arange(max(0, z - sphere_radius), min(volume_shape[2], z + sphere_radius + 1))

        x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        distances = np.sqrt((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2)

        sphere_mask = distances <= sphere_radius
        mask[x_range[:, None, None], y_range[None, :, None], z_range[None, None, :]] |= sphere_mask

    return mask

class RandomBaseline:
    """Simple random placement baseline for sphere placement"""
    
    def __init__(self, mri_data, mask_data, lesion_data, sphere_radius=7, max_spheres=3):
        self.mri_data = mri_data
        self.mask_data = mask_data
        self.lesion_data = lesion_data
        self.sphere_radius = sphere_radius
        self.max_spheres = max_spheres
        
        # Get all valid coordinates within prostate mask
        self.prostate_coords = np.argwhere(self.mask_data > 0)
        if len(self.prostate_coords) == 0:
            raise ValueError("No valid coordinates found within the prostate mask.")
        
        print(f"Random baseline initialized with {len(self.prostate_coords)} valid prostate coordinates")
    
    def evaluate_placement(self, center):
        """Evaluate placement using same criteria as RL agent"""
        lesion_center = np.array(center_of_mass(self.lesion_data))
        distance_to_center = np.linalg.norm(center - lesion_center)

        # Overlap with lesion
        x, y, z = center
        overlap_score = np.sum(self.lesion_data[
            max(0, x - self.sphere_radius):min(self.mri_data.shape[0], x + self.sphere_radius),
            max(0, y - self.sphere_radius):min(self.mri_data.shape[1], y + self.sphere_radius),
            max(0, z - self.sphere_radius):min(self.mri_data.shape[2], z + self.sphere_radius)
        ])

        # Avoid overlap with existing spheres
        overlap_penalty = 0
        for existing_center in self.sphere_positions:
            if np.linalg.norm(center - existing_center) <= 2 * self.sphere_radius:
                overlap_penalty += 100

        score = overlap_score - distance_to_center - overlap_penalty
        return score
    
    def run_episode(self, seed=None):
        """Run one random placement episode"""
        if seed is not None:
            np.random.seed(seed)
        
        self.sphere_positions = []
        episode_rewards = []
        episode_dice_scores = []
        
        # Initial dice score
        initial_mask = create_sphere_mask([], self.sphere_radius, self.mri_data.shape)
        initial_dice = calculate_dice_score(initial_mask, self.lesion_data)
        episode_dice_scores.append(initial_dice)
        
        for step in range(self.max_spheres):
            # Random placement within prostate
            random_idx = np.random.randint(0, len(self.prostate_coords))
            coord = self.prostate_coords[random_idx].copy()
            
            # Add small noise
            noise = np.random.randint(-5, 5, size=3)
            coord = np.clip(coord + noise, [0, 0, 0], np.array(self.mri_data.shape) - 1)
            
            # Ensure still within prostate
            if not self.mask_data[coord[0], coord[1], coord[2]]:
                coord = self.prostate_coords[random_idx].copy()
            
            # Evaluate and place
            reward = self.evaluate_placement(coord)
            self.sphere_positions.append(coord)
            episode_rewards.append(reward)
            
            # Calculate dice after placement
            sphere_mask = create_sphere_mask(self.sphere_positions, self.sphere_radius, self.mri_data.shape)
            dice_score = calculate_dice_score(sphere_mask, self.lesion_data)
            episode_dice_scores.append(dice_score)
        
        # Final metrics
        final_mask = create_sphere_mask(self.sphere_positions, self.sphere_radius, self.mri_data.shape)
        final_dice = calculate_dice_score(final_mask, self.lesion_data)
        
        lesion_volume = np.sum(self.lesion_data > 0)
        covered_volume = np.sum((final_mask > 0) & (self.lesion_data > 0))
        coverage_percentage = (covered_volume / lesion_volume * 100) if lesion_volume > 0 else 0
        
        return {
            'sphere_positions': self.sphere_positions.copy(),
            'rewards': episode_rewards,
            'dice_scores': episode_dice_scores,
            'total_reward': sum(episode_rewards),
            'final_dice': final_dice,
            'coverage': coverage_percentage,
        }


def visualize_episode(baseline, episode_result, save_path, patient_name, episode_num):
    """Create visualization for one episode"""
    
    sphere_mask = create_sphere_mask(
        episode_result['sphere_positions'], 
        baseline.sphere_radius, 
        baseline.mri_data.shape
    )
    
    slice_idx = baseline.mri_data.shape[2] // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original MRI
    axes[0, 0].imshow(baseline.mri_data[:, :, slice_idx], cmap='gray')
    axes[0, 0].set_title('Original MRI')
    axes[0, 0].axis('off')
    
    # MRI with masks
    axes[0, 1].imshow(baseline.mri_data[:, :, slice_idx], cmap='gray')
    axes[0, 1].contour(baseline.mask_data[:, :, slice_idx], colors='blue', levels=[0.5], linewidths=2)
    axes[0, 1].contour(baseline.lesion_data[:, :, slice_idx], colors='red', levels=[0.5], linewidths=2)
    axes[0, 1].contour(sphere_mask[:, :, slice_idx], colors='yellow', levels=[0.5], linewidths=3)
    
    # Mark sphere centers
    for j, pos in enumerate(episode_result['sphere_positions']):
        if pos[2] == slice_idx:
            axes[0, 1].plot(pos[1], pos[0], 'yo', markersize=8, markeredgecolor='black')
            axes[0, 1].text(pos[1]+2, pos[0]+2, f'S{j+1}', color='yellow', fontweight='bold')
    
    axes[0, 1].set_title(f'Random Placement (Dice: {episode_result["final_dice"]:.3f})')
    axes[0, 1].axis('off')
    
    # Dice progression
    axes[1, 0].plot(range(len(episode_result['dice_scores'])), episode_result['dice_scores'], 'g-o')
    axes[1, 0].set_title('Dice Score Progression')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rewards
    if episode_result['rewards']:
        axes[1, 1].bar(range(1, len(episode_result['rewards'])+1), episode_result['rewards'], color='purple', alpha=0.7)
        axes[1, 1].set_title('Step Rewards')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Random Baseline - {patient_name} - Episode {episode_num}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_random_baseline():
    """Main function to run random baseline experiment"""
    
    print("=" * 60)
    print("RANDOM PLACEMENT BASELINE")
    print("=" * 60)
    
    # Setup
    current_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = f"./random_baseline_{current_time}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Get dataset
    def get_patients():
        dataset_dir = "./filtered_dataset"
        if not os.path.exists(dataset_dir):
            dataset_dir = "./image_with_masks"
            if not os.path.exists(dataset_dir):
                raise FileNotFoundError(f"Dataset not found")
        
        patient_folders = []
        for folder in os.listdir(dataset_dir):
            folder_path = os.path.join(dataset_dir, folder)
            if os.path.isdir(folder_path) and not folder.startswith('.'):
                if (os.path.exists(os.path.join(folder_path, "t2.nii.gz")) and
                    os.path.exists(os.path.join(folder_path, "gland.nii.gz")) and
                    os.path.exists(os.path.join(folder_path, "l_a1.nii.gz"))):
                    patient_folders.append(folder_path)
        
        return patient_folders
    
    try:
        patient_dirs = get_patients()
        print(f"Found {len(patient_dirs)} valid patients")
        
        # Use first 10 patients (or all if less than 10)
        test_patients = patient_dirs[:10] if len(patient_dirs) >= 10 else patient_dirs
        
        all_results = []
        patient_stats = []
        
        num_episodes = 5  # Episodes per patient
        
        for i, patient_dir in enumerate(test_patients):
            patient_name = os.path.basename(patient_dir)
            print(f"\nProcessing Patient {i+1}: {patient_name}")
            
            # Create patient folder
            patient_folder = os.path.join(results_folder, f'patient_{i+1}_{patient_name}')
            os.makedirs(patient_folder, exist_ok=True)
            
            try:
                # Load data
                mri_data, mask_data, lesion_data = load_and_preprocess_patient_data(
                    patient_dir, target_shape=(128, 128, 20)
                )
                
                # Create baseline
                baseline = RandomBaseline(mri_data, mask_data, lesion_data)
                
                # Run episodes
                patient_episodes = []
                for episode in range(num_episodes):
                    print(f"  Episode {episode+1}/{num_episodes}")
                    
                    result = baseline.run_episode(seed=42+episode)
                    patient_episodes.append(result)
                    all_results.append(result)
                    
                    # Create visualization
                    viz_path = os.path.join(patient_folder, f'episode_{episode+1}.png')
                    visualize_episode(baseline, result, viz_path, patient_name, episode+1)
                    
                    print(f"    Dice: {result['final_dice']:.3f}, Coverage: {result['coverage']:.1f}%, Reward: {result['total_reward']:.1f}")
                
                # Patient statistics
                dice_scores = [ep['final_dice'] for ep in patient_episodes]
                coverages = [ep['coverage'] for ep in patient_episodes]
                rewards = [ep['total_reward'] for ep in patient_episodes]
                
                patient_stat = {
                    'name': patient_name,
                    'episodes': num_episodes,
                    'mean_dice': np.mean(dice_scores),
                    'std_dice': np.std(dice_scores),
                    'mean_coverage': np.mean(coverages),
                    'std_coverage': np.std(coverages),
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                }
                patient_stats.append(patient_stat)
                
                print(f"  Patient Summary: Dice={patient_stat['mean_dice']:.3f}±{patient_stat['std_dice']:.3f}, Coverage={patient_stat['mean_coverage']:.1f}%±{patient_stat['std_coverage']:.1f}%")
                
            except Exception as e:
                print(f"  Error processing {patient_name}: {e}")
                continue
        
        # Overall statistics
        if all_results:
            all_dice = [r['final_dice'] for r in all_results]
            all_coverage = [r['coverage'] for r in all_results]
            all_rewards = [r['total_reward'] for r in all_results]
            
            overall_stats = {
                'total_episodes': len(all_results),
                'total_patients': len(patient_stats),
                'episodes_per_patient': num_episodes,
                'mean_dice': np.mean(all_dice),
                'std_dice': np.std(all_dice),
                'mean_coverage': np.mean(all_coverage),
                'std_coverage': np.std(all_coverage),
                'mean_reward': np.mean(all_rewards),
                'std_reward': np.std(all_rewards),
            }
            
            # Create summary visualization
            create_summary_plot(patient_stats, overall_stats, results_folder)
            
            # Save results
            np.savez(os.path.join(results_folder, 'results.npz'), 
                    overall_stats=overall_stats, 
                    patient_stats=patient_stats, 
                    all_results=all_results)
            
            # Save text summary
            with open(os.path.join(results_folder, 'summary.txt'), 'w') as f:
                f.write("RANDOM BASELINE RESULTS\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Total Episodes: {overall_stats['total_episodes']}\n")
                f.write(f"Total Patients: {overall_stats['total_patients']}\n")
                f.write(f"Episodes per Patient: {overall_stats['episodes_per_patient']}\n\n")
                f.write(f"OVERALL PERFORMANCE:\n")
                f.write(f"Mean Dice Score: {overall_stats['mean_dice']:.3f} ± {overall_stats['std_dice']:.3f}\n")
                f.write(f"Mean Coverage: {overall_stats['mean_coverage']:.1f}% ± {overall_stats['std_coverage']:.1f}%\n")
                f.write(f"Mean Reward: {overall_stats['mean_reward']:.2f} ± {overall_stats['std_reward']:.2f}\n\n")
                f.write(f"PER-PATIENT RESULTS:\n")
                for i, stat in enumerate(patient_stats):
                    f.write(f"Patient {i+1} ({stat['name']}):\n")
                    f.write(f"  Dice: {stat['mean_dice']:.3f} ± {stat['std_dice']:.3f}\n")
                    f.write(f"  Coverage: {stat['mean_coverage']:.1f}% ± {stat['std_coverage']:.1f}%\n")
                    f.write(f"  Reward: {stat['mean_reward']:.2f} ± {stat['std_reward']:.2f}\n\n")
            
            print("\n" + "=" * 60)
            print("RANDOM BASELINE SUMMARY")
            print("=" * 60)
            print(f"Total Episodes: {overall_stats['total_episodes']}")
            print(f"Total Patients: {overall_stats['total_patients']}")
            print(f"Mean Dice Score: {overall_stats['mean_dice']:.3f} ± {overall_stats['std_dice']:.3f}")
            print(f"Mean Coverage: {overall_stats['mean_coverage']:.1f}% ± {overall_stats['std_coverage']:.1f}%")
            print(f"Mean Reward: {overall_stats['mean_reward']:.2f} ± {overall_stats['std_reward']:.2f}")
            print(f"Results saved in: {results_folder}")
            print("=" * 60)
            
            return overall_stats
        
        else:
            print("No successful episodes completed.")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None


def create_summary_plot(patient_stats, overall_stats, save_folder):
    """Create summary visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Per-patient dice scores
    patient_names = [p['name'][:8] for p in patient_stats]  # Truncate names
    dice_means = [p['mean_dice'] for p in patient_stats]
    dice_stds = [p['std_dice'] for p in patient_stats]
    
    axes[0, 0].bar(range(len(dice_means)), dice_means, yerr=dice_stds, 
                   color='green', alpha=0.7, capsize=5)
    axes[0, 0].axhline(overall_stats['mean_dice'], color='red', linestyle='--', 
                      label=f"Overall Mean: {overall_stats['mean_dice']:.3f}")
    axes[0, 0].set_title('Dice Score by Patient')
    axes[0, 0].set_xlabel('Patient')
    axes[0, 0].set_ylabel('Dice Score')
    axes[0, 0].set_xticks(range(len(patient_names)))
    axes[0, 0].set_xticklabels(patient_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Per-patient coverage
    coverage_means = [p['mean_coverage'] for p in patient_stats]
    coverage_stds = [p['std_coverage'] for p in patient_stats]
    
    axes[0, 1].bar(range(len(coverage_means)), coverage_means, yerr=coverage_stds,
                   color='orange', alpha=0.7, capsize=5)
    axes[0, 1].axhline(overall_stats['mean_coverage'], color='red', linestyle='--',
                      label=f"Overall Mean: {overall_stats['mean_coverage']:.1f}%")
    axes[0, 1].set_title('Coverage by Patient')
    axes[0, 1].set_xlabel('Patient')
    axes[0, 1].set_ylabel('Coverage (%)')
    axes[0, 1].set_xticks(range(len(patient_names)))
    axes[0, 1].set_xticklabels(patient_names, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Distribution histogram
    all_dice = []
    for stat in patient_stats:
        # Approximate individual scores from mean/std (for visualization)
        episode_scores = np.random.normal(stat['mean_dice'], stat['std_dice'], stat['episodes'])
        all_dice.extend(episode_scores)
    
    axes[1, 0].hist(all_dice, bins=15, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(overall_stats['mean_dice'], color='red', linestyle='--', linewidth=2,
                      label=f"Mean: {overall_stats['mean_dice']:.3f}")
    axes[1, 0].set_title('Dice Score Distribution')
    axes[1, 0].set_xlabel('Dice Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
RANDOM BASELINE SUMMARY

Total Episodes: {overall_stats['total_episodes']}
Total Patients: {overall_stats['total_patients']}
Episodes per Patient: {overall_stats['episodes_per_patient']}

PERFORMANCE METRICS:
Mean Dice Score: {overall_stats['mean_dice']:.3f} ± {overall_stats['std_dice']:.3f}
Mean Coverage: {overall_stats['mean_coverage']:.1f}% ± {overall_stats['std_coverage']:.1f}%
Mean Reward: {overall_stats['mean_reward']:.2f} ± {overall_stats['std_reward']:.2f}

QUALITY ASSESSMENT:
Dice Quality: {'Excellent' if overall_stats['mean_dice'] > 0.6 else 'Good' if overall_stats['mean_dice'] > 0.4 else 'Adequate' if overall_stats['mean_dice'] > 0.2 else 'Poor'}
Coverage Quality: {'Excellent' if overall_stats['mean_coverage'] > 80 else 'Good' if overall_stats['mean_coverage'] > 60 else 'Adequate' if overall_stats['mean_coverage'] > 40 else 'Poor'}

VARIABILITY:
Dice CV: {(overall_stats['std_dice']/overall_stats['mean_dice']*100):.1f}%
Coverage CV: {(overall_stats['std_coverage']/overall_stats['mean_coverage']*100):.1f}%
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Random Baseline Performance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Running Random Placement Baseline...")
    results = run_random_baseline()
    if results:
        print("Random baseline completed successfully!")
    else:
        print("Random baseline failed.")