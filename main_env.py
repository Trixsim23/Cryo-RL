# main_env_fixed.py - Fixed version of SpherePlacementEnv

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
    def __init__(self, mri_data, mask_data, lesion_data, sphere_radius=7, max_action_space=None):
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
        
        # FIXED: Use standardized action space
        if max_action_space is None:
            # If no max specified, use the current environment's size (for single environment training)
            self.max_action_space = self.lesion_coords.shape[0]
        else:
            # Use the provided max action space size (for multi-environment training)
            self.max_action_space = max_action_space
        
        # Define action space: standardized size across all environments
        self.action_space = spaces.Discrete(self.max_action_space)
        self.observation_space = spaces.Box(
            low=0, high= 1, shape=(mri_data.shape[0],mri_data.shape[1],mri_data.shape[2],3), dtype=np.int32
        )
        print(f"Environment initialized with {self.lesion_coords.shape[0]} valid lesion coordinates, action space size: {self.max_action_space}")
        
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
        # FIXED: Map action to valid coordinate index using modulo
        if self.lesion_coords.shape[0] == 0:
            raise ValueError("No valid lesion coordinates available")
        
        coord_index = int(action) % self.lesion_coords.shape[0]
        coord = self.lesion_coords[coord_index]

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


def get_max_action_space_size(patient_dirs, target_shape=(128, 128, 20)):
    """
    Determine the maximum action space size across all patients
    """
    from main_agent import load_and_preprocess_patient_data
    
    max_lesion_coords = 0
    lesion_coord_counts = []
    
    print("Determining maximum action space size across all patients...")
    
    for i, patient_dir in enumerate(patient_dirs):
        try:
            patient_name = os.path.basename(patient_dir)
            print(f"Checking patient {i+1}/{len(patient_dirs)}: {patient_name}")
            
            mri_data, mask_data, lesion_data = load_and_preprocess_patient_data(patient_dir, target_shape)
            
            # Count lesion coordinates
            lesion_coords = np.argwhere((lesion_data > 0) & (mask_data > 0))
            num_coords = lesion_coords.shape[0]
            lesion_coord_counts.append(num_coords)
            
            if num_coords > max_lesion_coords:
                max_lesion_coords = num_coords
                
            print(f"  {patient_name}: {num_coords} lesion coordinates")
            
        except Exception as e:
            print(f"  Error processing {patient_name}: {e}")
            continue
    
    print(f"\nAction space analysis:")
    print(f"  Minimum lesion coordinates: {min(lesion_coord_counts)}")
    print(f"  Maximum lesion coordinates: {max_lesion_coords}")
    print(f"  Mean lesion coordinates: {np.mean(lesion_coord_counts):.1f}")
    print(f"  Standard deviation: {np.std(lesion_coord_counts):.1f}")
    print(f"  Using maximum size: {max_lesion_coords}")
    
    return max_lesion_coords


def create_standardized_environments(patient_dirs, max_action_space=None, target_shape=(128, 128, 20)):
    """
    Create environments with standardized action spaces
    """
    from main_agent import load_and_preprocess_patient_data
    
    # If max_action_space not provided, calculate it
    if max_action_space is None:
        max_action_space = get_max_action_space_size(patient_dirs, target_shape)
    
    environments = []
    
    print(f"\nCreating standardized environments with action space size: {max_action_space}")
    
    for i, patient_dir in enumerate(patient_dirs):
        try:
            patient_name = os.path.basename(patient_dir)
            print(f"Loading patient {i+1}/{len(patient_dirs)}: {patient_name}")
            
            mri_data, mask_data, lesion_data = load_and_preprocess_patient_data(patient_dir, target_shape)
            
            # Create environment with standardized action space
            env = SpherePlacementEnv(mri_data, mask_data, lesion_data, 
                                   sphere_radius=7, max_action_space=max_action_space)
            environments.append(env)
            
            print(f"  Successfully created environment for {patient_name}")
            
        except Exception as e:
            print(f"  Error creating environment for {patient_name}: {e}")
            continue
    
    return environments, max_action_space