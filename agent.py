import os
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

    #downsample the data so that it is (128, 128, 20, 3)
    mri_data = mri_data[:, :, :20]
    mri_data = mri_data[::4, ::4, :]
    mask_data = mask_data[::4, ::4, :]
    lesion_data = lesion_data[::4, ::4, :]

    return mri_data, mask_data, lesion_data

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
    model.learn(total_timesteps=50000, callback=[eval_callback, checkpoint_callback])

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

    # Plot rewards
    plt.figure()
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o', linestyle='-', color='b')
    plt.title('Rewards Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True)
    
    # Save the figure to the results folder
    plt.savefig(os.path.join('results', 'rewards.png'))
    plt.close()

    # Visualize the result
    slice_idx = eval_mri.shape[2] // 2
    eval_env.visualize_spheres(slice_idx)

    # Save results
    np.save("sphere_positions_eval.npy", sphere_positions)
