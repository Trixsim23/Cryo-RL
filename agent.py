import os
import time 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from env import SpherePlacementEnv, visualize_removal_with_overlay

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

    # Split the patients into training and evaluation sets using first 10 
    train_patients = valid_patients[:10]  # All except the last patient
    eval_patient = valid_patients[10]  # Last patient for evaluation

    # Create the training environment
    train_envs = []
    for patient_dir in train_patients:
        mri_data, mask_data, lesion_data = load_patient_data(patient_dir)
        train_envs.append(SpherePlacementEnv(mri_data, mask_data, lesion_data))

    # Use the first training environment (can be parallelized later)
    train_env = train_envs[0]

    # Initialize PPO agent
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./ppo_sphere_logs", n_steps=512)

    # Add callbacks for evaluation and checkpointing
    eval_callback = EvalCallback(
        train_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./logs/checkpoints/")

    # Train the model
    #maybe change the time steps to less......
    model.learn(total_timesteps=1000, callback=[eval_callback, checkpoint_callback])

    # Save the trained model
    model.save("ppo_sphere_placement")

    # Evaluation on the last patient
    eval_mri, eval_mask, eval_lesion = load_patient_data(eval_patient)
    eval_env = SpherePlacementEnv(eval_mri, eval_mask, eval_lesion, sphere_radius=20)

    obs = eval_env.reset()
    done = False
    total_reward = 0
    sphere_positions = []
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
        sphere_positions.append(info["sphere_positions"])
        rewards.append(reward)

    print(f"Evaluation complete. Total reward: {total_reward}")
    print (rewards)

    # Plot rewards
    plt.figure()
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o', linestyle='-', color='b')
    plt.title('Rewards Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True)
    

    #the current date and time is 
    current_time = time.strftime("%Y%m%d-%H%M%S")
    # Save the figure to the results folder
    folder = os.path.join(".", "results")
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join('results', f'rewards{current_time}.png'))
    plt.close()

    # Visualize the result
    slice_idx = eval_mri.shape[2] // 2
    eval_env.visualize_spheres(slice_idx)

    # Save results
    np.save("sphere_positions_eval.npy", sphere_positions)
