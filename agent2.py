import os
import time 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from env2 import SpherePlacementEnv, visualize_removal_with_overlay

# Function to preprocess and load MRI data for a single patient
def load_patient_data(patient_dir):
    mri_file = os.path.join(patient_dir, "t2.nii.gz")
    mask_file = os.path.join(patient_dir, "gland.nii.gz")
    lesion_file = os.path.join(patient_dir, "l_a1.nii.gz")

    mri_img = nib.load(mri_file)
    mask_img = nib.load(mask_file)
    lesion_img = nib.load(lesion_file)

    mri_data = np.rot90(mri_img.get_fdata(), 1)
    mask_data = np.rot90(mask_img.get_fdata(), 1)
    lesion_data = np.rot90(lesion_img.get_fdata(), 1)

    # Get original dimensions
    orig_x, orig_y, orig_z = mri_data.shape
    
    # Define target shape
    target_x, target_y, target_z = 128, 128, 20
    
    # Handle Z dimension (depth)
    if orig_z > target_z:
        # Select evenly spaced slices using array indexing
        z_indices = np.linspace(0, orig_z - 1, target_z, dtype=int)
        mri_data = mri_data[:, :, z_indices]
        mask_data = mask_data[:, :, z_indices]
        lesion_data = lesion_data[:, :, z_indices]
    else:
        # Pad with zeros if fewer than target_z slices
        pad_size = target_z - orig_z
        mri_data = np.pad(mri_data, ((0, 0), (0, 0), (0, pad_size)))
        mask_data = np.pad(mask_data, ((0, 0), (0, 0), (0, pad_size)))
        lesion_data = np.pad(lesion_data, ((0, 0), (0, 0), (0, pad_size)))
    
    # Handle X and Y dimensions using vectorized operations
    # Create index arrays for the original data positions
    x_indices = np.clip((np.arange(target_x) * orig_x / target_x).astype(int), 0, orig_x - 1)
    y_indices = np.clip((np.arange(target_y) * orig_y / target_y).astype(int), 0, orig_y - 1)
    
    # For MRI data, use a block averaging approach
    # We'll use a simplified approach with meshgrid for indexing
    mri_downsampled = np.zeros((target_x, target_y, target_z))
    X, Y = np.meshgrid(x_indices, y_indices, indexing='ij')
    
    # Simple point sampling for all data types
    mri_downsampled = mri_data[X, Y, :]
    mask_downsampled = mask_data[X, Y, :]
    lesion_downsampled = lesion_data[X, Y, :]
    
    # For binary masks, ensure they remain binary after resampling
    if mask_data.dtype == bool or np.array_equal(np.unique(mask_data), np.array([0, 1])):
        mask_downsampled = mask_downsampled.round().astype(mask_data.dtype)
    
    if lesion_data.dtype == bool or np.array_equal(np.unique(lesion_data), np.array([0, 1])):
        lesion_downsampled = lesion_downsampled.round().astype(lesion_data.dtype)
    
    # Verify final shape
    assert mri_downsampled.shape == (target_x, target_y, target_z), f"Expected shape ({target_x}, {target_y}, {target_z}), got {mri_downsampled.shape}"
    
    return mri_downsampled, mask_downsampled, lesion_downsampled

# Custom callback for evaluation at checkpoints
class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.all_rewards = []
        self.timesteps = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation
            obs = self.eval_env.reset()
            done = False
            total_reward = 0
            episode_rewards = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # No need to clip actions - environment now handles invalid actions
                obs, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                episode_rewards.append(reward)
            
            # Store the current timestep and mean reward
            self.timesteps.append(self.num_timesteps)
            self.all_rewards.append(total_reward)
            
            if self.verbose > 0:
                print(f"Timestep: {self.num_timesteps}, Evaluation reward: {total_reward}")
            
            # Save visualization at this checkpoint
            current_time = time.strftime("%Y%m%d-%H%M%S")
            slice_idx = self.eval_env.mri_data.shape[2] // 2
            
            # Create results folder if it doesn't exist
            results_folder = os.path.join(".", "results")
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
                
            # Save the current spheres visualization
            fig = self.eval_env.visualize_spheres(slice_idx, show=False)
            if fig is not None:  # Add check in case visualization method doesn't return a figure
                fig.savefig(os.path.join(results_folder, f'spheres_placement_step_{self.num_timesteps}.png'))
                plt.close(fig)
            
            # Update and plot the rewards graph
            self.plot_rewards(results_folder)
            
            if total_reward > self.best_mean_reward:
                self.best_mean_reward = total_reward
                # Optionally save the best model
                if self.verbose > 0:
                    print(f"New best mean reward: {self.best_mean_reward}")
                
        return True
    
    def plot_rewards(self, save_folder):
        # Plot rewards over all evaluation points
        plt.figure(figsize=(10, 6))
        plt.plot(self.timesteps, self.all_rewards, marker='o', linestyle='-', color='b')
        plt.title('Evaluation Rewards Over Training', fontsize=16)
        plt.xlabel('Training Timesteps', fontsize=14)
        plt.ylabel('Total Evaluation Reward', fontsize=14)
        plt.grid(True)
        
        # Save the figure
        plt.savefig(os.path.join(save_folder, f'evaluation_rewards_progress.png'))
        plt.close()
        
        # Also save the raw data
        np.save(os.path.join(save_folder, 'evaluation_timesteps.npy'), self.timesteps)
        np.save(os.path.join(save_folder, 'evaluation_rewards.npy'), self.all_rewards)


if __name__ == "__main__":
    # Directory containing patient data
    directory = "./image_with_masks"

    # Get a list of patient folders
    patient_folders = [
        os.path.join(directory, folder)
        for folder in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, folder))
    ]

    # Filter out patients without the required files
    valid_patients = [
        folder
        for folder in patient_folders
        if os.path.exists(os.path.join(folder, "t2.nii.gz"))
        and os.path.exists(os.path.join(folder, "gland.nii.gz"))
        and os.path.exists(os.path.join(folder, "l_a1.nii.gz"))
    ]

    # Split the patients into training and evaluation sets
    train_patients = valid_patients[:10]  # First 10 patients
    eval_patient = valid_patients[10]     # 11th patient for evaluation

    # Create the training environment
    train_envs = []
    for patient_dir in train_patients:
        mri_data, mask_data, lesion_data = load_patient_data(patient_dir)
        train_envs.append(SpherePlacementEnv(mri_data, mask_data, lesion_data))

    # Use the first training environment (can be parallelized later)
    train_env = train_envs[0]

    # Create the evaluation environment
    eval_mri, eval_mask, eval_lesion = load_patient_data(eval_patient)
    eval_env = SpherePlacementEnv(eval_mri, eval_mask, eval_lesion, sphere_radius=20)
    
    # Print information about action spaces
    print(f"Training env action space size: {train_env.action_space.n}")
    print(f"Evaluation env action space size: {eval_env.action_space.n}")

    # Initialize PPO agent
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./ppo_sphere_logs", n_steps=512)

    # Create folder for results if it doesn't exist
    results_folder = os.path.join(".", "results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Add callbacks for evaluation, checkpointing, and custom evaluation
    # Regular eval callback for logging to tensorboard
    eval_callback = EvalCallback(
        train_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )
    
    # Custom callback for more detailed evaluation on the test patient
    custom_eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        eval_freq=200,  # Evaluate more frequently to get more data points
        verbose=1
    )
    
    # Checkpoint callback to save models
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, 
        save_path="./logs/checkpoints/",
        name_prefix="ppo_sphere"
    )

    # Train the model with all callbacks
    total_timesteps = 10000  # Adjust based on your needs
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[eval_callback, checkpoint_callback, custom_eval_callback]
    )

    # Save the final trained model
    model.save("ppo_sphere_placement_final")

    # Final evaluation on the test patient
    obs = eval_env.reset()
    done = False
    total_reward = 0
    sphere_positions = []
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # No need to clip - environment handles invalid actions
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
        sphere_positions.append(info["sphere_positions"])
        rewards.append(reward)

    print(f"Final evaluation complete. Total reward: {total_reward}")
    
    # Plot final rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o', linestyle='-', color='b')
    plt.title('Final Evaluation Rewards per Episode Step', fontsize=16)
    plt.xlabel('Episode Step', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.grid(True)

    # Save the figure to the results folder
    current_time = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(results_folder, f'final_rewards_{current_time}.png'))
    plt.close()

    # Visualize the final result
    slice_idx = eval_mri.shape[2] // 2
    fig = eval_env.visualize_spheres(slice_idx, show=True)
    if fig is not None:
        fig.savefig(os.path.join(results_folder, f'final_sphere_placement_{current_time}.png'))
    
    # Save sphere positions for analysis
    np.save(os.path.join(results_folder, "sphere_positions_eval.npy"), sphere_positions)
    
    # Create a summary plot showing all evaluation rewards
    custom_eval_callback.plot_rewards(results_folder)
    
    print(f"All results saved to {results_folder}")