#this file is the one with out the downsampling 

import gymnasium as gym
from gym import spaces
import numpy as np
import nibabel as nib
import time 
from scipy.ndimage import center_of_mass
import os
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

class SpherePlacementEnv(gym.Env):
    def __init__(self, mri_data, mask_data, lesion_data, sphere_radius=7):
        super(SpherePlacementEnv, self).__init__()

        self.mri_data = mri_data
        self.mask_data = mask_data
        self.lesion_data = lesion_data

        #radius is hard coded
        self.sphere_radius = sphere_radius

        # Get all possible locations within the lesion grid
        self.lesion_coords = self._get_lesion_grid()
        if len(self.lesion_coords) == 0:
            raise ValueError("No valid lesion coordinates found within the prostate mask.")
        
        # Define action space: single action to place a sphere
        self.action_space = spaces.Discrete(self.lesion_coords.shape[0])
        self.observation_space = spaces.Box(
            low=0, high= 1, shape=(mri_data.shape[0],mri_data.shape[1],mri_data.shape[2],3), dtype=np.int32
        )
        print(f"Environment initialized with {self.lesion_coords.shape[0]} valid lesion coordinates")
        
        self.max_spheres = 3  # Max spheres per episode
        self.sphere_count = 0
        self.sphere_positions = []  # Track sphere placements

        self.modified_mask = mask_data.copy()
        self.modified_lesion = lesion_data.copy()
        
        self.reset()

    def _get_lesion_grid(self):
        # Get the coordinates of the lesion within the prostate mask
        lesion_coords = np.argwhere((self.lesion_data > 0) & (self.mask_data > 0))
        # print(f"Found {lesion_coords.shape[0]} valid lesion coordinates")
        return lesion_coords

    def _evaluate_location(self, center):
        """
        Evaluate a potential sphere placement location based on:
        1. Proximity to the center of the lesion.
        2. Overlap with the lesion.
        3. Avoidance of overlap with existing spheres.
        """
        # Proximity to the center of the lesion
        lesion_center = np.array(center_of_mass(self.lesion_data))
        distance_to_center = np.linalg.norm(center - lesion_center)

        # Overlap with the lesion
        x, y, z = center
        overlap_score = np.sum(self.lesion_data[
            max(0, x - self.sphere_radius):min(self.mri_data.shape[0], x + self.sphere_radius),
            max(0, y - self.sphere_radius):min(self.mri_data.shape[1], y + self.sphere_radius),
            max(0, z - self.sphere_radius):min(self.mri_data.shape[2], z + self.sphere_radius)
        ])

        # Avoidance of overlap with existing spheres
        overlap_with_existing = 0
        for existing_center in self.sphere_positions:
            if np.linalg.norm(center - existing_center) <= 2 * self.sphere_radius:
                overlap_with_existing += 1

        # Combine scores (higher is better)
        score = overlap_score - distance_to_center - 100 * overlap_with_existing
        return score

    def step(self, action):
        # Ensure action is within valid range
        action = int(action) % self.lesion_coords.shape[0]  # Use modulo to handle out-of-range actions
        
        # Evaluate all possible locations and select the best one
        coord = self.lesion_coords[action]

        # Add random noise to the placement (limited to around 5 voxels in x, y, z)
        noise = np.random.randint(-5, 5, size=3)
        coord = np.clip(coord + noise, [0, 0, 0], np.array(self.mri_data.shape) - 1)

        score = self._evaluate_location(coord)

        reward = score  # Reward for successful placement
        self.sphere_positions.append(coord)
        self.sphere_count += 1
        self._remove_sphere_from_data(coord)
        done = self.sphere_count >= self.max_spheres
        obs = np.concatenate([np.expand_dims(self.mri_data,-1), np.expand_dims(self.modified_mask,-1),np.expand_dims(self.modified_lesion,-1)], axis=-1)
        
        return obs, reward, done, {"sphere_positions": self.sphere_positions}

    def _remove_sphere_from_data(self, center):
        x, y, z = center
        radius = self.sphere_radius

        # Define the ranges for the sphere
        x_range = np.arange(max(0, x - radius), min(self.mri_data.shape[0], x + radius))
        y_range = np.arange(max(0, y - radius), min(self.mri_data.shape[1], y + radius))
        z_range = np.arange(max(0, z - radius), min(self.mri_data.shape[2], z + radius))

        # Create a grid of points in the range
        x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')

        # Calculate the distance of each point in the grid from the center
        distances = np.sqrt((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2)

        # Create a mask for points within the radius
        sphere_mask = distances <= radius

        # Update the relevant slices of the modified arrays
        self.modified_mask[x_range[:, None, None], y_range[None, :, None], z_range[None, None, :]] *= ~sphere_mask
        self.modified_lesion[x_range[:, None, None], y_range[None, :, None], z_range[None, None, :]] *= ~sphere_mask

    def reset(self):
        #should be added for reset 
        self.mask_data = self.mask_data.copy()
        self.lesion_data = self.lesion_data.copy()
        self.sphere_count = 0
        self.sphere_positions = []
        self.modified_mask = self.mask_data.copy()
        self.modified_lesion = self.lesion_data.copy()
        #new additions 
        self.lesion_coords= self._get_lesion_grid()
        obs = np.concatenate([np.expand_dims(self.mri_data,-1), np.expand_dims(self.modified_mask,-1),np.expand_dims(self.modified_lesion,-1)], axis=-1)
        return obs
    
    def render(self, mode='human'):
        print(f"Spheres placed: {self.sphere_count}/{self.max_spheres}")
        print(f"Sphere positions: {self.sphere_positions}")


#used for testing the environment itself 


# # Add this updated method to your SpherePlacementEnv class in main_env.py

#     def visualize_spheres(self, slice_idx, show=True, save_path=None):
#         """
#         Enhanced visualization showing individual sphere placements with numbers and colors
#         """
#         from main_agent import enhanced_visualize_spheres_with_numbers
        
#         fig = enhanced_visualize_spheres_with_numbers(
#             env=self,
#             slice_idx=slice_idx,
#             save_path=save_path,
#             show=show,
#             step_info=""
#         )
#         return fig

# # Alternative: Update the existing visualize_removal_with_overlay function in main_env.py

def visualize_removal_with_overlay(original_mri, modified_mri, original_mask, modified_mask, 
                                          original_lesion, modified_lesion, slice_idx, sphere_positions=None, 
                                          sphere_radius=7, show=True, save_path=None, step_info=""):
    """
    Enhanced visualization with individual sphere markers
    """
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Normalize MRI volumes to [0, 1]
    original_mri_norm = (original_mri - np.min(original_mri)) / (np.max(original_mri) - np.min(original_mri) + 1e-10)
    modified_mri_norm = (modified_mri - np.min(modified_mri)) / (np.max(modified_mri) - np.min(modified_mri) + 1e-10)

    # Normalize masks to [0, 1]
    original_mask_norm = original_mask.astype(float)
    if np.max(original_mask) > 0:
        original_mask_norm = original_mask_norm / np.max(original_mask)
        
    modified_mask_norm = modified_mask.astype(float)
    if np.max(modified_mask) > 0:
        modified_mask_norm = modified_mask_norm / np.max(modified_mask)
        
    original_lesion_norm = original_lesion.astype(float)
    if np.max(original_lesion) > 0:
        original_lesion_norm = original_lesion_norm / np.max(original_lesion)
        
    modified_lesion_norm = modified_lesion.astype(float)
    if np.max(modified_lesion) > 0:
        modified_lesion_norm = modified_lesion_norm / np.max(modified_lesion)

    # Mask out values in the prostate mask that are 0
    original_mask_masked = np.ma.masked_where(original_mask_norm == 0, original_mask_norm)
    modified_mask_masked = np.ma.masked_where(modified_mask_norm == 0, modified_mask_norm)

    # Mask out values in the lesion mask that are 0
    original_lesion_masked = np.ma.masked_where(original_lesion_norm == 0, original_lesion_norm)
    modified_lesion_masked = np.ma.masked_where(modified_lesion_norm == 0, modified_lesion_norm)

    # Define normalization for overlay consistency
    norm = Normalize(vmin=0, vmax=1)

    # Define colors and markers for each sphere
    sphere_colors = ['red', 'blue', 'green']
    sphere_markers = ['o', 's', '^']  # circle, square, triangle

    # Original MRI with mask overlay
    axes[0, 0].imshow(original_mri_norm[:, :, slice_idx], cmap='gray', norm=norm)
    axes[0, 0].imshow(np.max(original_mask_masked[:, :, :], axis=2), cmap='summer', alpha=0.5)
    axes[0, 0].imshow(np.max(original_lesion_masked[:, :, :], axis=2), cmap='jet', alpha=0.4)
    axes[0, 0].set_title("Original MRI with Masks")
    axes[0, 0].axis("off")

    # Modified MRI with mask overlay + Individual Sphere Markers
    axes[0, 1].imshow(modified_mri_norm[:, :, slice_idx], cmap='gray', norm=norm)
    axes[0, 1].imshow(modified_mask_masked[:, :, slice_idx], cmap='Reds', alpha=0.5)
    axes[0, 1].imshow(modified_lesion_masked[:, :, slice_idx], cmap='Reds', alpha=0.4)
    
    # Add individual sphere markers
    if sphere_positions:
        for i, sphere_pos in enumerate(sphere_positions):
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

    # Original mask with lesion overlay 
    axes[1, 0].imshow(original_mask_masked[:, :, slice_idx], cmap='Reds', norm=norm)
    axes[1, 0].imshow(original_lesion_masked[:, :, slice_idx], cmap='Reds', alpha=0.4)
    axes[1, 0].set_title("Original Masks")
    axes[1, 0].axis("off")

    # Modified mask with lesion overlay + Individual Sphere Markers
    axes[1, 1].imshow(modified_mask_masked[:, :, slice_idx], cmap='Reds', norm=norm)
    axes[1, 1].imshow(modified_lesion_masked[:, :, slice_idx], cmap='Reds', alpha=0.4)
    
    # Add individual sphere markers to mask view
    if sphere_positions:
        for i, sphere_pos in enumerate(sphere_positions):
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
    if sphere_positions:
        legend_elements = []
        for i in range(len(sphere_positions)):
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

# Example usage
if __name__ == "__main__":
    mri_file = os.path.join(".", "image_with_masks", "P-10104751", "t2.nii.gz")
    mask_file = os.path.join(".", "image_with_masks", "P-10104751", "gland.nii.gz")
    lesion_mask = os.path.join(".", "image_with_masks", "P-10104751", "l_a1.nii.gz")

    mri_img = nib.load(mri_file)
    mask_img = nib.load(mask_file)
    lesion_img = nib.load(lesion_mask)
    

    mri_data = mri_img.get_fdata()
    mask_data = mask_img.get_fdata()
    lesion_data = lesion_img.get_fdata()

    # Rotate the volume 90 degrees anti-clockwise
    mri_data = np.rot90(mri_data, 1)
    mask_data = np.rot90(mask_data, 1)
    lesion_data = np.rot90(lesion_data, 1)

    env = SpherePlacementEnv(mri_data, mask_data, lesion_data)
    # print(env.mri_data.shape)
    obs = env.reset()
    done = False
    rewards = []
    #change the range here 
    for i in range(20):
        if done:
            obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        rewards.append(reward) 
        # env.render()
    # print (rewards)
    #PRINT AN AVERAGE REWARD 
    print (np.mean(rewards))
    plt.figure()
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o', linestyle='-', color='b')
    plt.title('Rewards Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True)
    plt.show()

    # Visualize the effect of the spheres
    slice_idx = mri_data.shape[2] // 2
    env.visualize_spheres(slice_idx)