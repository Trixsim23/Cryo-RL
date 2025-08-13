import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import json

# Import the existing functions from your modules
from main_agent import load_and_preprocess_patient_data, calculate_dice_score, create_sphere_mask

# Import enhanced_visualize_spheres_with_numbers with filename fix
def enhanced_visualize_spheres_with_numbers_fixed(env, slice_idx=None, save_path=None, show=True, step_info=""):
    """
    Wrapper for enhanced_visualize_spheres_with_numbers that fixes filename issues
    """
    # Import the original function
    from main_agent import enhanced_visualize_spheres_with_numbers as original_viz
    
    # Sanitize step_info to avoid filename issues
    step_info_sanitized = sanitize_filename(step_info) if step_info else ""
    
    # Call the original function with sanitized step_info
    return original_viz(env, slice_idx, save_path, show, step_info_sanitized)

# Import additional visualization functions from agent_vis_2.py
try:
    from main_agent  import (
        visualize_multi_view_spheres,
        visualize_3d_volume_rendering,
        visualize_individual_step_placement,
        create_final_comprehensive_evaluation
    )
    ADVANCED_VIS_AVAILABLE = True
    print("✓ Advanced visualization functions imported successfully")
except ImportError as e:
    ADVANCED_VIS_AVAILABLE = False
    print(f"⚠ Advanced visualization functions not available: {e}")
    print("  Only basic visualization will be available")


def sanitize_filename(filename):
    """
    Sanitize filename by removing or replacing invalid characters for Windows/Unix
    """
    import re
    # Replace invalid characters with underscores
    invalid_chars = r'[<>:"/\\|?*%]'
    sanitized = re.sub(invalid_chars, '_', filename)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


# Test the sanitization function
def test_filename_sanitization():
    """Test filename sanitization with problematic characters"""
    test_cases = [
        "Clinical Placement (Dice: 0.398, Coverage: 97.0%)",
        "Test<file>name",
        "File:with/invalid\\chars",
        "Normal_filename",
        "Multiple___underscores",
    ]
    
    print("Testing filename sanitization:")
    for test in test_cases:
        sanitized = sanitize_filename(test)
        print(f"  '{test}' -> '{sanitized}'")

# Run test if this module is imported
# if __name__ == "__main__":
#     # Uncomment the line below to test filename sanitization
#     # test_filename_sanitization()
#     # main()


def calculate_comprehensive_coverage_metrics(sphere_positions, sphere_radius, mri_shape, lesion_data):
    """
    Calculate comprehensive coverage metrics similar to the experiments.
    
    Returns:
        dict: Dictionary containing all coverage metrics
    """
    if not sphere_positions:
        return {
            'lesion_coverage_percentage': 0.0,
            'total_lesion_voxels': np.sum(lesion_data > 0),
            'covered_lesion_voxels': 0,
            'sphere_coverage_efficiency': 0.0,
            'total_sphere_volume': 0,
            'effective_sphere_volume': 0,
            'overlap_percentage': 0.0
        }
    
    # Create sphere mask for all spheres
    sphere_mask = create_sphere_mask(sphere_positions, sphere_radius, mri_shape)
    
    # Calculate lesion coverage metrics
    total_lesion_voxels = np.sum(lesion_data > 0)
    covered_lesion_voxels = np.sum(np.logical_and(sphere_mask, lesion_data > 0))
    lesion_coverage_percentage = (covered_lesion_voxels / total_lesion_voxels * 100) if total_lesion_voxels > 0 else 0.0
    
    # Calculate sphere efficiency metrics
    total_sphere_volume = np.sum(sphere_mask)
    effective_sphere_volume = covered_lesion_voxels  # Volume that actually covers lesion
    sphere_coverage_efficiency = (effective_sphere_volume / total_sphere_volume * 100) if total_sphere_volume > 0 else 0.0
    
    # Calculate overlap between spheres
    if len(sphere_positions) > 1:
        individual_sphere_volumes = []
        for pos in sphere_positions:
            single_sphere_mask = np.zeros(mri_shape, dtype=bool)
            x, y, z = pos
            x_range = np.arange(max(0, x - sphere_radius), min(mri_shape[0], x + sphere_radius))
            y_range = np.arange(max(0, y - sphere_radius), min(mri_shape[1], y + sphere_radius))
            z_range = np.arange(max(0, z - sphere_radius), min(mri_shape[2], z + sphere_radius))
            
            x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            distances = np.sqrt((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2)
            sphere_coords = distances <= sphere_radius
            
            single_sphere_mask[x_range[:, None, None], y_range[None, :, None], z_range[None, None, :]] = sphere_coords
            individual_sphere_volumes.append(np.sum(single_sphere_mask))
        
        expected_total_volume = sum(individual_sphere_volumes)
        actual_total_volume = total_sphere_volume
        overlap_volume = expected_total_volume - actual_total_volume
        overlap_percentage = (overlap_volume / expected_total_volume * 100) if expected_total_volume > 0 else 0.0
    else:
        overlap_percentage = 0.0
    
    return {
        'lesion_coverage_percentage': lesion_coverage_percentage,
        'total_lesion_voxels': int(total_lesion_voxels),
        'covered_lesion_voxels': int(covered_lesion_voxels),
        'sphere_coverage_efficiency': sphere_coverage_efficiency,
        'total_sphere_volume': int(total_sphere_volume),
        'effective_sphere_volume': int(effective_sphere_volume),
        'overlap_percentage': overlap_percentage
    }


class ClinicalPlacementCollector:
    def __init__(self, data_directory="./collection_tool/Clinical_set", sphere_radius=7):
        # Use static data directory path
        self.data_directory = data_directory
        self.sphere_radius = sphere_radius
        self.current_patient = None
        self.current_patient_data = None
        self.sphere_placements = []
        self.max_spheres = 3
        
        # Get available patients
        print(f"🔍 Initializing ClinicalPlacementCollector...")
        self.available_patients = self._get_available_patients()
        print(f"🔍 After scanning: available_patients = {len(self.available_patients)} patients")
        
        # CSV file path
        self.csv_file = "clinical_placements.csv"
        self.load_existing_placements()
    
    def _find_data_directory(self):
        """Auto-detect the correct data directory path"""
        # Possible data directory names and locations
        possible_dirs = [
            "./Clinical_set",           # Current directory
            "../Clinical_set",          # Parent directory
            "./image_with_masks",       # Alternative name
            "../image_with_masks",      # Alternative in parent
            "../../Clinical_set",       # Two levels up
            "../../image_with_masks",   # Two levels up alternative
            "./filtered_dataset",       # Filtered dataset
            "../filtered_dataset",      # Filtered in parent
        ]
        
        print("🔍 Auto-detecting data directory...")
        
        for data_dir in possible_dirs:
            if os.path.exists(data_dir) and os.path.isdir(data_dir):
                # Check if it contains valid patient folders
                patient_count = 0
                try:
                    for folder in os.listdir(data_dir):
                        folder_path = os.path.join(data_dir, folder)
                        if os.path.isdir(folder_path):
                            required_files = ["t2.nii.gz", "gland.nii.gz", "l_a1.nii.gz"]
                            if all(os.path.exists(os.path.join(folder_path, f)) for f in required_files):
                                patient_count += 1
                except:
                    continue
                
                if patient_count > 0:
                    print(f"✅ Found data directory: {data_dir} (with {patient_count} valid patients)")
                    return data_dir
                else:
                    print(f"⚠️  Found directory {data_dir} but no valid patients inside")
        
        # If no valid directory found, ask user to specify
        print("❌ Could not auto-detect data directory!")
        print("\nPlease check that your data directory contains patient folders with:")
        print("  - t2.nii.gz")
        print("  - gland.nii.gz") 
        print("  - l_a1.nii.gz")
        print("\nTried looking in:")
        for data_dir in possible_dirs:
            exists = "✅" if os.path.exists(data_dir) else "❌"
            print(f"  {exists} {data_dir}")
        
        return "./Clinical_set"  # Default fallback
        
    def _get_available_patients(self):
        """Get list of available patients with required files"""
        patients = []
        
        if not os.path.exists(self.data_directory):
            print(f"❌ Data directory not found: {self.data_directory}")
            return patients
        
        print(f"📁 Scanning data directory: {self.data_directory}")
        
        try:
            folders = os.listdir(self.data_directory)
            total_folders = len([f for f in folders if os.path.isdir(os.path.join(self.data_directory, f))])
            print(f"   Found {total_folders} folders to check...")
            
            for folder in folders:
                folder_path = os.path.join(self.data_directory, folder)
                if os.path.isdir(folder_path):
                    # Check if required files exist
                    required_files = ["t2.nii.gz", "gland.nii.gz", "l_a1.nii.gz"]
                    missing_files = []
                    
                    for file in required_files:
                        if not os.path.exists(os.path.join(folder_path, file)):
                            missing_files.append(file)
                    
                    if not missing_files:
                        patients.append(folder)
                        print(f"   ✅ {folder} - Valid patient")
                    else:
                        print(f"   ❌ {folder} - Missing: {', '.join(missing_files)}")
            
            print(f"\n🎯 Found {len(patients)} valid patients out of {total_folders} folders")
            if patients:
                print("Valid patients:")
                for i, patient in enumerate(patients, 1):
                    print(f"   {i}. {patient}")
            
        except Exception as e:
            print(f"❌ Error scanning directory: {e}")

        return patients 
        
    def set_data_directory(self, new_directory):
        """Manually set a new data directory and refresh patient list"""
        print(f"🔄 Changing data directory to: {new_directory}")
        self.data_directory = new_directory
        self.available_patients = self._get_available_patients()
        return len(self.available_patients) > 0
    
    def load_existing_placements(self):
        """Load existing placements from CSV file"""
        if os.path.exists(self.csv_file):
            try:
                self.placements_df = pd.read_csv(self.csv_file)
                print(f"Loaded {len(self.placements_df)} existing placements from {self.csv_file}")
            except Exception as e:
                print(f"Error loading existing placements: {e}")
                self.placements_df = pd.DataFrame()
        else:
            self.placements_df = pd.DataFrame()
    
    def save_placement_to_csv(self, patient_id, sphere_positions, user_id="default_user", notes=""):
        """Enhanced save sphere placements to CSV file with comprehensive coverage metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate comprehensive coverage metrics
        if self.current_patient_data is not None:
            mri_data, mask_data, lesion_data = self.current_patient_data
            
            # Calculate Dice score (existing)
            sphere_mask = create_sphere_mask(sphere_positions, self.sphere_radius, mri_data.shape)
            dice_score = calculate_dice_score(sphere_mask, lesion_data)
            
            # Calculate comprehensive coverage metrics (NEW)
            coverage_metrics = calculate_comprehensive_coverage_metrics(
                sphere_positions, self.sphere_radius, mri_data.shape, lesion_data
            )
            
        else:
            dice_score = 0.0
            coverage_metrics = {
                'lesion_coverage_percentage': 0.0,
                'total_lesion_voxels': 0,
                'covered_lesion_voxels': 0,
                'sphere_coverage_efficiency': 0.0,
                'total_sphere_volume': 0,
                'effective_sphere_volume': 0,
                'overlap_percentage': 0.0
            }
        
        # Calculate sphere spacing metrics
        if len(sphere_positions) > 1:
            distances = []
            for i in range(len(sphere_positions)):
                for j in range(i + 1, len(sphere_positions)):
                    dist = np.linalg.norm(np.array(sphere_positions[i]) - np.array(sphere_positions[j]))
                    distances.append(dist)
            
            min_sphere_distance = min(distances)
            max_sphere_distance = max(distances)
            avg_sphere_distance = np.mean(distances)
        else:
            min_sphere_distance = max_sphere_distance = avg_sphere_distance = 0.0
        
        # Create new records for each sphere with enhanced metrics
        new_records = []
        for i, (x, y, z) in enumerate(sphere_positions):
            record = {
                'timestamp': timestamp,
                'patient_id': patient_id,
                'user_id': user_id,
                'sphere_number': i + 1,
                'x_coordinate': x,
                'y_coordinate': y,
                'z_coordinate': z,
                'sphere_radius': self.sphere_radius,
                'total_spheres': len(sphere_positions),
                
                # Enhanced coverage metrics
                'dice_score': dice_score,
                'lesion_coverage_percentage': coverage_metrics['lesion_coverage_percentage'],
                'total_lesion_voxels': coverage_metrics['total_lesion_voxels'],
                'covered_lesion_voxels': coverage_metrics['covered_lesion_voxels'],
                'sphere_coverage_efficiency': coverage_metrics['sphere_coverage_efficiency'],
                'total_sphere_volume': coverage_metrics['total_sphere_volume'],
                'effective_sphere_volume': coverage_metrics['effective_sphere_volume'],
                'overlap_percentage': coverage_metrics['overlap_percentage'],
                
                # Spacing metrics
                'min_sphere_distance': min_sphere_distance,
                'max_sphere_distance': max_sphere_distance,
                'avg_sphere_distance': avg_sphere_distance,
                
                'notes': notes
            }
            new_records.append(record)
        
        # Add to existing dataframe
        new_df = pd.DataFrame(new_records)
        self.placements_df = pd.concat([self.placements_df, new_df], ignore_index=True)
        
        # Save to CSV
        self.placements_df.to_csv(self.csv_file, index=False)
        
        print(f"Saved {len(sphere_positions)} sphere placements for patient {patient_id}")
        print(f"Dice score: {dice_score:.3f}")
        print(f"Lesion coverage: {coverage_metrics['lesion_coverage_percentage']:.1f}%")
        print(f"Sphere efficiency: {coverage_metrics['sphere_coverage_efficiency']:.1f}%")
        if coverage_metrics['overlap_percentage'] > 0:
            print(f"Sphere overlap: {coverage_metrics['overlap_percentage']:.1f}%")
    
    def load_patient_placements(self, patient_id):
        """Load existing placements for a specific patient"""
        if self.placements_df.empty:
            return []
        
        patient_placements = self.placements_df[self.placements_df['patient_id'] == patient_id]
        if patient_placements.empty:
            return []
        
        # Group by timestamp to get different placement sessions
        sessions = []
        for timestamp in patient_placements['timestamp'].unique():
            session_data = patient_placements[patient_placements['timestamp'] == timestamp]
            spheres = []
            for _, row in session_data.iterrows():
                spheres.append((row['x_coordinate'], row['y_coordinate'], row['z_coordinate']))
            sessions.append({
                'timestamp': timestamp,
                'spheres': spheres,
                'dice_score': session_data.iloc[0]['dice_score'],
                'user_id': session_data.iloc[0]['user_id'],
                'notes': session_data.iloc[0]['notes'],
                'lesion_coverage_percentage': session_data.iloc[0].get('lesion_coverage_percentage', 0.0),
                'sphere_coverage_efficiency': session_data.iloc[0].get('sphere_coverage_efficiency', 0.0),
                'overlap_percentage': session_data.iloc[0].get('overlap_percentage', 0.0)
            })
        return sessions


class InteractivePlacementGUI:
    def __init__(self, collector):
        self.collector = collector
        self.root = tk.Tk()
        self.root.title("Clinical Cryoablation Placement Tool")
        self.root.geometry("800x600")
        
        self.current_patient = None
        self.current_slice = 0
        self.sphere_placements = []
        self.fig = None
        self.ax = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the main GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Patient selection
        ttk.Label(main_frame, text="Select Patient:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.patient_var = tk.StringVar()
        self.patient_combo = ttk.Combobox(main_frame, textvariable=self.patient_var, 
                                         values=self.collector.available_patients, 
                                         state="readonly", width=30)
        self.patient_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.patient_combo.bind('<<ComboboxSelected>>', self.on_patient_selected)
        
        # User ID
        ttk.Label(main_frame, text="User ID:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.user_var = tk.StringVar(value="clinician_1")
        user_entry = ttk.Entry(main_frame, textvariable=self.user_var, width=30)
        user_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Change Data Dir", 
                  command=self.change_data_directory).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Load Patient", 
                  command=self.load_patient).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Start Placement", 
                  command=self.start_placement).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Reset Spheres", 
                  command=self.reset_spheres).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="Save Placement", 
                  command=self.save_placement).grid(row=0, column=4, padx=5)
        
        # Placement info
        info_frame = ttk.LabelFrame(main_frame, text="Placement Information", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.info_text = tk.Text(info_frame, height=8, width=70)
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Notes
        ttk.Label(main_frame, text="Notes:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.notes_var = tk.StringVar()
        notes_entry = ttk.Entry(main_frame, textvariable=self.notes_var, width=50)
        notes_entry.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Analysis buttons
        analysis_frame = ttk.Frame(main_frame)
        analysis_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        ttk.Button(analysis_frame, text="View Existing Placements", 
                  command=self.view_existing_placements).grid(row=0, column=0, padx=5)
        ttk.Button(analysis_frame, text="Export Results", 
                  command=self.export_results).grid(row=0, column=1, padx=5)
        ttk.Button(analysis_frame, text="Create Comprehensive Analysis", 
                  command=self.create_comprehensive_analysis).grid(row=0, column=2, padx=5)
        ttk.Button(analysis_frame, text="Compare with RL Agent", 
                  command=self.compare_with_rl).grid(row=0, column=3, padx=5)
        
        # Show current data directory and available patients
        status_message = f"Data Directory: {self.collector.data_directory}\n"
        if self.collector.available_patients:
            status_message += f"✅ Found {len(self.collector.available_patients)} valid patients. Select one to begin!"
        else:
            status_message += "❌ No valid patients found.\n"
            status_message += "Expected folder structure:\n"
            status_message += f"{self.collector.data_directory}/\n"
            status_message += "├── Patient_001/ (with t2.nii.gz, gland.nii.gz, l_a1.nii.gz)\n"
            status_message += "├── Patient_002/ (with required files)\n"
            status_message += "└── ...\n"
            status_message += "Use 'Change Data Dir' to select a different location."
        
        self.update_info(status_message)
    
    def change_data_directory(self):
        """Allow user to change the data directory"""
        new_dir = filedialog.askdirectory(title="Select Data Directory (containing patient folders)")
        if new_dir:
            self.update_info(f"Changing data directory to: {new_dir}")
            success = self.collector.set_data_directory(new_dir)
            if success:
                # Update the patient dropdown
                self.patient_combo['values'] = self.collector.available_patients
                self.patient_var.set('')  # Clear selection
                self.update_info(f"✅ Found {len(self.collector.available_patients)} valid patients")
            else:
                self.update_info("❌ No valid patients found in selected directory")
                messagebox.showwarning("Warning", "No valid patients found in the selected directory!")
    
    def update_info(self, message):
        """Update the information display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.info_text.see(tk.END)
        self.root.update()
    
    def on_patient_selected(self, event=None):
        """Handle patient selection"""
        selected = self.patient_var.get()
        if selected:
            self.update_info(f"Patient {selected} selected. Click 'Load Patient' to begin.")
    
    def load_patient(self):
        """Load patient data"""
        patient_id = self.patient_var.get()
        if not patient_id:
            messagebox.showwarning("Warning", "Please select a patient first.")
            return
        
        try:
            self.update_info(f"Loading patient {patient_id}...")
            patient_dir = os.path.join(self.collector.data_directory, patient_id)
            
            # Load and preprocess data
            mri_data, mask_data, lesion_data = load_and_preprocess_patient_data(patient_dir)
            
            self.collector.current_patient = patient_id
            self.collector.current_patient_data = (mri_data, mask_data, lesion_data)
            self.current_patient = patient_id
            self.sphere_placements = []
            
            self.update_info(f"Patient {patient_id} loaded successfully!")
            self.update_info(f"MRI shape: {mri_data.shape}")
            self.update_info(f"Lesion volume: {np.sum(lesion_data)} voxels")
            
            # Show existing placements for this patient
            existing = self.collector.load_patient_placements(patient_id)
            if existing:
                self.update_info(f"Found {len(existing)} existing placement sessions for this patient.")
                for i, session in enumerate(existing):
                    coverage_info = f", Coverage: {session.get('lesion_coverage_percentage', 0):.1f}%" if 'lesion_coverage_percentage' in session else ""
                    self.update_info(f"  Session {i+1}: {len(session['spheres'])} spheres, "
                                   f"Dice: {session['dice_score']:.3f}{coverage_info}, User: {session['user_id']}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load patient: {str(e)}")
            self.update_info(f"Error loading patient: {str(e)}")
    
    def start_placement(self):
        """Start interactive sphere placement"""
        if self.collector.current_patient_data is None:
            messagebox.showwarning("Warning", "Please load a patient first.")
            return
        
        self.update_info("Starting interactive placement mode...")
        self.update_info("Instructions:")
        self.update_info("- Use arrow keys or scroll to navigate slices")
        self.update_info("- Click on the image to place spheres")
        self.update_info("- Maximum 3 spheres per placement")
        self.update_info("- Close the plot window when done")
        
        # Start interactive matplotlib session
        self._start_interactive_placement()
    
    def _start_interactive_placement(self):
        """Start interactive matplotlib placement interface"""
        mri_data, mask_data, lesion_data = self.collector.current_patient_data
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 10))
        self.current_slice = mri_data.shape[2] // 2
        
        # Setup the display
        self._update_slice_display()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        plt.title(f"Patient {self.current_patient} - Interactive Placement\n"
                 f"Slice {self.current_slice}/{mri_data.shape[2]-1} | "
                 f"Spheres: {len(self.sphere_placements)}/{self.collector.max_spheres}")
        
        plt.show()
    
    def _update_slice_display(self):
        """Update the slice display"""
        mri_data, mask_data, lesion_data = self.collector.current_patient_data
        
        self.ax.clear()
        
        # Normalize and display MRI
        mri_norm = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data) + 1e-10)
        self.ax.imshow(mri_norm[:, :, self.current_slice], cmap='gray')
        
        # Overlay masks with proper normalization
        if np.any(mask_data[:, :, self.current_slice] > 0):
            # Normalize mask data
            mask_slice = mask_data[:, :, self.current_slice]
            mask_normalized = mask_slice / (np.max(mask_slice) + 1e-10)
            mask_overlay = np.ma.masked_where(mask_normalized == 0, mask_normalized)
            self.ax.imshow(mask_overlay, cmap='Blues', alpha=0.4, vmin=0, vmax=1)

        if np.any(lesion_data[:, :, self.current_slice] > 0):
            # Normalize lesion data
            lesion_slice = lesion_data[:, :, self.current_slice]
            lesion_normalized = lesion_slice / (np.max(lesion_slice) + 1e-10)
            lesion_overlay = np.ma.masked_where(lesion_normalized == 0, lesion_normalized)
            self.ax.imshow(lesion_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        
        # Display existing spheres
        sphere_colors = ['red', 'blue', 'green']
        sphere_markers = ['o', 's', '^']
        
        for i, (x, y, z) in enumerate(self.sphere_placements):
            if abs(z - self.current_slice) <= 2:  # Show spheres close to current slice
                color = sphere_colors[i % len(sphere_colors)]
                marker = sphere_markers[i % len(sphere_markers)]
                
                self.ax.scatter(y, x, s=300, c=color, marker=marker, 
                            edgecolors='white', linewidth=3, alpha=0.9, zorder=10)
                self.ax.text(y, x, str(i+1), ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='white', zorder=11)
        
        self.ax.set_title(f"Patient {self.current_patient} - Slice {self.current_slice}/{mri_data.shape[2]-1}\n"
                        f"Spheres: {len(self.sphere_placements)}/{self.collector.max_spheres} | "
                        f"Click to place, Arrow keys to navigate")
        self.ax.axis('off')
        
        self.fig.canvas.draw()
    
    def _on_click(self, event):
        """Handle mouse click for sphere placement"""
        if event.inaxes != self.ax:
            return
        
        if len(self.sphere_placements) >= self.collector.max_spheres:
            self.update_info(f"Maximum {self.collector.max_spheres} spheres reached!")
            return
        
        # Get click coordinates
        x, y = int(event.ydata), int(event.xdata)
        z = self.current_slice
        
        # Validate placement (within bounds and in lesion area)
        mri_data, mask_data, lesion_data = self.collector.current_patient_data
        
        if (0 <= x < mri_data.shape[0] and 0 <= y < mri_data.shape[1] and 
            mask_data[x, y, z] > 0):  # Within prostate mask
            
            # Add sphere
            self.sphere_placements.append((x, y, z))
            self.update_info(f"Sphere {len(self.sphere_placements)} placed at ({x}, {y}, {z})")
            
            # Update display
            self._update_slice_display()
        else:
            self.update_info(f"Invalid placement at ({x}, {y}, {z}) - outside prostate mask")
    
    def _on_key_press(self, event):
        """Handle keyboard navigation"""
        mri_data, _, _ = self.collector.current_patient_data
        
        if event.key == 'right' or event.key == 'up':
            self.current_slice = min(self.current_slice + 1, mri_data.shape[2] - 1)
            self._update_slice_display()
        elif event.key == 'left' or event.key == 'down':
            self.current_slice = max(self.current_slice - 1, 0)
            self._update_slice_display()
    
    def reset_spheres(self):
        """Reset all sphere placements"""
        self.sphere_placements = []
        self.update_info("All sphere placements reset.")
        
        if hasattr(self, 'fig') and self.fig is not None:
            self._update_slice_display()
    
    def save_placement(self):
        """Enhanced save current sphere placement with comprehensive metrics display"""
        if not self.sphere_placements:
            messagebox.showwarning("Warning", "No spheres to save!")
            return
        
        if self.collector.current_patient is None:
            messagebox.showwarning("Warning", "No patient loaded!")
            return
        
        try:
            user_id = self.user_var.get() or "default_user"
            notes = self.notes_var.get() or ""
            
            # Calculate comprehensive metrics before saving
            mri_data, mask_data, lesion_data = self.collector.current_patient_data
            coverage_metrics = calculate_comprehensive_coverage_metrics(
                self.sphere_placements, self.collector.sphere_radius, mri_data.shape, lesion_data
            )
            
            # Save placement (this will call the enhanced save method)
            self.collector.save_placement_to_csv(
                self.collector.current_patient,
                self.sphere_placements,
                user_id,
                notes
            )
            
            # Enhanced information display
            self.update_info(f"Placement saved! {len(self.sphere_placements)} spheres for patient {self.collector.current_patient}")
            
            # Calculate and display comprehensive metrics
            sphere_mask = create_sphere_mask(self.sphere_placements, self.collector.sphere_radius, mri_data.shape)
            dice_score = calculate_dice_score(sphere_mask, lesion_data)
            
            self.update_info(f"COMPREHENSIVE COVERAGE ANALYSIS:")
            self.update_info(f"  Dice score: {dice_score:.3f}")
            self.update_info(f"  Lesion coverage: {coverage_metrics['lesion_coverage_percentage']:.1f}%")
            self.update_info(f"  Total lesion voxels: {coverage_metrics['total_lesion_voxels']:,}")
            self.update_info(f"  Covered lesion voxels: {coverage_metrics['covered_lesion_voxels']:,}")
            self.update_info(f"  Sphere efficiency: {coverage_metrics['sphere_coverage_efficiency']:.1f}%")
            self.update_info(f"  Total sphere volume: {coverage_metrics['total_sphere_volume']:,} voxels")
            self.update_info(f"  Effective sphere volume: {coverage_metrics['effective_sphere_volume']:,} voxels")
            
            if coverage_metrics['overlap_percentage'] > 0:
                self.update_info(f"  Sphere overlap: {coverage_metrics['overlap_percentage']:.1f}%")
            
            # Enhanced success message
            success_message = (
                f"Placement saved successfully!\n\n"
                f"Coverage Analysis:\n"
                f"• Dice score: {dice_score:.3f}\n"
                f"• Lesion coverage: {coverage_metrics['lesion_coverage_percentage']:.1f}%\n"
                f"• Sphere efficiency: {coverage_metrics['sphere_coverage_efficiency']:.1f}%\n"
                f"• Total spheres: {len(self.sphere_placements)}"
            )
            
            if coverage_metrics['overlap_percentage'] > 5.0:  # Warning for significant overlap
                success_message += f"\n\n⚠️ Note: {coverage_metrics['overlap_percentage']:.1f}% sphere overlap detected"
            
            messagebox.showinfo("Success", success_message)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save placement: {str(e)}")
            self.update_info(f"Error saving placement: {str(e)}")
    
    def view_existing_placements(self):
        """View existing placements for current patient"""
        if not self.collector.current_patient:
            messagebox.showwarning("Warning", "Please load a patient first.")
            return
        
        existing = self.collector.load_patient_placements(self.collector.current_patient)
        if not existing:
            messagebox.showinfo("Info", f"No existing placements found for patient {self.collector.current_patient}")
            return
        
        # Create new window to display existing placements
        self._show_existing_placements_window(existing)
    
    def _show_existing_placements_window(self, placements):
        """Show window with existing placements"""
        window = tk.Toplevel(self.root)
        window.title(f"Existing Placements - {self.collector.current_patient}")
        window.geometry("800x400")
        
        # Create treeview with enhanced columns
        columns = ('Session', 'Timestamp', 'User', 'Spheres', 'Dice Score', 'Coverage %', 'Efficiency %', 'Overlap %', 'Notes')
        tree = ttk.Treeview(window, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            if col in ['Coverage %', 'Efficiency %', 'Overlap %']:
                tree.column(col, width=80)
            elif col == 'Timestamp':
                tree.column(col, width=150)
            else:
                tree.column(col, width=100)
        
        # Add data
        for i, session in enumerate(placements):
            tree.insert('', 'end', values=(
                i+1,
                session['timestamp'],
                session['user_id'],
                len(session['spheres']),
                f"{session['dice_score']:.3f}",
                f"{session.get('lesion_coverage_percentage', 0):.1f}",
                f"{session.get('sphere_coverage_efficiency', 0):.1f}",
                f"{session.get('overlap_percentage', 0):.1f}",
                session['notes']
            ))
        
        tree.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add buttons
        btn_frame = ttk.Frame(window)
        btn_frame.pack(pady=10)
        
        def visualize_selected():
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                session_idx = int(item['values'][0]) - 1
                self._visualize_placement_session(placements[session_idx])
        
        ttk.Button(btn_frame, text="Visualize Selected", command=visualize_selected).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Close", command=window.destroy).pack(side='left', padx=5)
    
    def _visualize_placement_session(self, session):
        """Enhanced visualize a specific placement session with multiple visualization options"""
        if self.collector.current_patient_data is None:
            messagebox.showwarning("Warning", "Patient data not loaded!")
            return
        
        mri_data, mask_data, lesion_data = self.collector.current_patient_data
        
        # Create a temporary environment-like object for visualization
        class TempEnv:
            def __init__(self, mri_data, mask_data, lesion_data, sphere_positions, sphere_radius):
                self.mri_data = mri_data
                self.mask_data = mask_data
                self.lesion_data = lesion_data
                self.modified_mask = mask_data.copy()
                self.modified_lesion = lesion_data.copy()
                self.sphere_positions = sphere_positions
                self.sphere_radius = sphere_radius
        
        temp_env = TempEnv(mri_data, mask_data, lesion_data, session['spheres'], self.collector.sphere_radius)
        
        slice_idx = mri_data.shape[2] // 2
        coverage = session.get('lesion_coverage_percentage', 0)
        
        # FIXED: Sanitize the step_info to avoid filename issues
        step_info_raw = f"Clinical Placement (Dice: {session['dice_score']:.3f}, Coverage: {coverage:.1f}%)"
        step_info = sanitize_filename(step_info_raw)
        
        # Ask user which visualization type they want
        if ADVANCED_VIS_AVAILABLE:
            choice = messagebox.askyesnocancel(
                "Visualization Options", 
                "Choose visualization type:\n\n"
                "• Yes: Enhanced 2D Visualization (Original)\n" 
                "• No: Multi-View Medical Imaging\n"
                "• Cancel: 3D Volume Rendering"
            )
            
            try:
                if choice is True:
                    # Enhanced 2D visualization (original)
                    enhanced_visualize_spheres_with_numbers_fixed(
                        env=temp_env,
                        slice_idx=slice_idx,
                        save_path=None,
                        show=True,
                        step_info=step_info_raw  # Use original, wrapper will sanitize
                    )
                elif choice is False:
                    # Multi-view visualization
                    visualize_multi_view_spheres(
                        env=temp_env,
                        save_path=None,
                        show=True,
                        step_info=step_info
                    )
                elif choice is None:
                    # 3D visualization
                    self.update_info("Creating 3D visualization - this may take a moment...")
                    visualize_3d_volume_rendering(
                        env=temp_env,
                        save_path=None,
                        show=True,
                        step_info=step_info
                    )
            except Exception as e:
                messagebox.showerror("Visualization Error", f"Failed to create visualization: {str(e)}")
                self.update_info(f"Visualization error: {str(e)}")
        else:
            # Fallback to basic visualization only
            enhanced_visualize_spheres_with_numbers_fixed(
                env=temp_env,
                slice_idx=slice_idx,
                save_path=None,
                show=True,
                step_info=step_info_raw  # Use original, wrapper will sanitize
            )
    
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis with all visualization types for current placement"""
        if not self.sphere_placements:
            messagebox.showwarning("Warning", "No spheres placed!")
            return
        
        if self.collector.current_patient is None:
            messagebox.showwarning("Warning", "No patient loaded!")
            return
            
        if not ADVANCED_VIS_AVAILABLE:
            messagebox.showinfo("Info", "Advanced visualization functions not available.\nOnly basic visualization can be created.")
            return
        
        self.update_info("Creating comprehensive analysis - this may take a moment...")
        
        try:
            mri_data, mask_data, lesion_data = self.collector.current_patient_data
            
            # Calculate comprehensive metrics
            coverage_metrics = calculate_comprehensive_coverage_metrics(
                self.sphere_placements, self.collector.sphere_radius, mri_data.shape, lesion_data
            )
            
            sphere_mask = create_sphere_mask(self.sphere_placements, self.collector.sphere_radius, mri_data.shape)
            dice_score = calculate_dice_score(sphere_mask, lesion_data)
            
            # Create temp environment for advanced visualizations
            class TempEnv:
                def __init__(self, mri_data, mask_data, lesion_data, sphere_positions, sphere_radius):
                    self.mri_data = mri_data
                    self.mask_data = mask_data
                    self.lesion_data = lesion_data
                    self.modified_mask = mask_data.copy()
                    self.modified_lesion = lesion_data.copy()
                    self.sphere_positions = sphere_positions
                    self.sphere_radius = sphere_radius
            
            temp_env = TempEnv(mri_data, mask_data, lesion_data, self.sphere_placements, self.collector.sphere_radius)
            
            # Create results folder
            import time
            current_time = time.strftime("%Y%m%d-%H%M%S")
            results_folder = f"./comprehensive_analysis_{self.collector.current_patient}_{current_time}"
            os.makedirs(results_folder, exist_ok=True)
            
            # Sanitized step info
            step_info_raw = f"Clinical Analysis (Dice: {dice_score:.3f}, Coverage: {coverage_metrics['lesion_coverage_percentage']:.1f}%)"
            step_info = sanitize_filename(step_info_raw)
            
            self.update_info("Creating visualizations...")
            
            # Create all visualization types
            viz_paths = {}
            
            # 1. Enhanced 2D
            enhanced_path = os.path.join(results_folder, f'enhanced_2d_{self.collector.current_patient}.png')
            slice_idx = mri_data.shape[2] // 2
            enhanced_visualize_spheres_with_numbers_fixed(
                env=temp_env,
                slice_idx=slice_idx,
                save_path=enhanced_path,
                show=False,
                step_info=step_info_raw  # Use original, wrapper will sanitize
            )
            viz_paths['Enhanced 2D'] = enhanced_path
            
            # 2. Multi-view
            multiview_path = os.path.join(results_folder, f'multiview_{self.collector.current_patient}.png')
            visualize_multi_view_spheres(
                env=temp_env,
                save_path=multiview_path,
                show=False,
                step_info=step_info
            )
            viz_paths['Multi-view'] = multiview_path
            
            # 3. 3D Volume (optional - computationally expensive)
            create_3d = messagebox.askyesno("3D Visualization", 
                                          "Create 3D volume rendering?\n\n"
                                          "This may take some time but provides detailed 3D views.")
            
            if create_3d:
                volume_3d_path = os.path.join(results_folder, f'3d_volume_{self.collector.current_patient}.png')
                try:
                    visualize_3d_volume_rendering(
                        env=temp_env,
                        save_path=volume_3d_path,
                        show=False,
                        step_info=step_info
                    )
                    viz_paths['3D Volume'] = volume_3d_path
                    self.update_info("✓ 3D volume rendering created")
                except Exception as e:
                    self.update_info(f"⚠ 3D visualization failed: {str(e)}")
            
            # 4. Create comprehensive final evaluation
            try:
                final_results = create_final_comprehensive_evaluation(
                    temp_env, results_folder, self.collector.current_patient, dice_score, create_3d=create_3d
                )
                viz_paths.update(final_results)
                self.update_info("✓ Comprehensive final evaluation created")
            except Exception as e:
                self.update_info(f"⚠ Comprehensive evaluation failed: {str(e)}")
            
            # Display success message with all created files
            success_msg = f"Comprehensive analysis complete!\n\n"
            success_msg += f"Results saved in: {results_folder}\n\n"
            success_msg += f"Created visualizations:\n"
            for viz_type, path in viz_paths.items():
                if path and os.path.exists(path):
                    success_msg += f"✓ {viz_type}\n"
            
            success_msg += f"\nMetrics:\n"
            success_msg += f"• Dice Score: {dice_score:.3f}\n"
            success_msg += f"• Lesion Coverage: {coverage_metrics['lesion_coverage_percentage']:.1f}%\n"
            success_msg += f"• Sphere Efficiency: {coverage_metrics['sphere_coverage_efficiency']:.1f}%\n"
            success_msg += f"• Total Spheres: {len(self.sphere_placements)}"
            
            messagebox.showinfo("Comprehensive Analysis Complete", success_msg)
            self.update_info(f"Comprehensive analysis saved in: {results_folder}")
            
            # Ask if user wants to open the results folder
            if messagebox.askyesno("Open Results", "Would you like to open the results folder?"):
                import subprocess
                import platform
                try:
                    if platform.system() == "Windows":
                        subprocess.Popen(f'explorer "{results_folder}"')
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.Popen(["open", results_folder])
                    else:  # Linux
                        subprocess.Popen(["xdg-open", results_folder])
                except:
                    self.update_info("Could not open results folder automatically")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create comprehensive analysis: {str(e)}")
            self.update_info(f"Comprehensive analysis error: {str(e)}")
    
    def export_results(self):
        """Export all results to various formats"""
        if self.collector.placements_df.empty:
            messagebox.showinfo("Info", "No placements to export!")
            return
        
        # Ask user for export location
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.xlsx'):
                    self.collector.placements_df.to_excel(filename, index=False)
                else:
                    self.collector.placements_df.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
                self.update_info(f"Results exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def compare_with_rl(self):
        """Compare clinical placements with RL agent results"""
        messagebox.showinfo("Info", "RL comparison feature would be implemented here.\n"
                                  "This would load RL agent results and create comparative visualizations.")
        self.update_info("RL comparison requested - feature to be implemented")
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def analyze_placement_results(csv_file="clinical_placements.csv"):
    """Enhanced analyze and visualize placement results from CSV with comprehensive coverage metrics"""
    if not os.path.exists(csv_file):
        print(f"No results file found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    print(f"\n=== ENHANCED CLINICAL PLACEMENT ANALYSIS ===")
    print(f"Total placements: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Summary statistics by placement session
    agg_dict = {
        'sphere_number': 'count',
        'dice_score': 'first',
        'user_id': 'first'
    }
    
    # Check if new columns exist before aggregating
    if 'lesion_coverage_percentage' in df.columns:
        agg_dict['lesion_coverage_percentage'] = 'first'
    if 'sphere_coverage_efficiency' in df.columns:
        agg_dict['sphere_coverage_efficiency'] = 'first'
    if 'overlap_percentage' in df.columns:
        agg_dict['overlap_percentage'] = 'first'
    
    placement_summary = df.groupby(['patient_id', 'timestamp']).agg(agg_dict).rename(columns={'sphere_number': 'num_spheres'})
    
    print(f"\n=== COMPREHENSIVE COVERAGE STATISTICS ===")
    print(f"Mean Dice score: {placement_summary['dice_score'].mean():.3f}")
    print(f"Std Dice score: {placement_summary['dice_score'].std():.3f}")
    print(f"Min Dice score: {placement_summary['dice_score'].min():.3f}")
    print(f"Max Dice score: {placement_summary['dice_score'].max():.3f}")
    
    if 'lesion_coverage_percentage' in placement_summary.columns:
        print(f"\nMean lesion coverage: {placement_summary['lesion_coverage_percentage'].mean():.1f}%")
        print(f"Std lesion coverage: {placement_summary['lesion_coverage_percentage'].std():.1f}%")
        print(f"Min lesion coverage: {placement_summary['lesion_coverage_percentage'].min():.1f}%")
        print(f"Max lesion coverage: {placement_summary['lesion_coverage_percentage'].max():.1f}%")
        
        print(f"\nMean sphere efficiency: {placement_summary['sphere_coverage_efficiency'].mean():.1f}%")
        print(f"Std sphere efficiency: {placement_summary['sphere_coverage_efficiency'].std():.1f}%")
        
        print(f"\nMean sphere overlap: {placement_summary['overlap_percentage'].mean():.1f}%")
        print(f"Max sphere overlap: {placement_summary['overlap_percentage'].max():.1f}%")
    
    # Enhanced plotting with coverage metrics
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Dice score distribution
    axes[0, 0].hist(placement_summary['dice_score'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Dice Score Distribution')
    axes[0, 0].set_xlabel('Dice Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Lesion coverage distribution
    if 'lesion_coverage_percentage' in placement_summary.columns:
        axes[0, 1].hist(placement_summary['lesion_coverage_percentage'], bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Lesion Coverage Distribution')
        axes[0, 1].set_xlabel('Coverage Percentage (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Coverage data\nnot available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Lesion Coverage Distribution')
    
    # Sphere efficiency distribution
    if 'sphere_coverage_efficiency' in placement_summary.columns:
        axes[0, 2].hist(placement_summary['sphere_coverage_efficiency'], bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 2].set_title('Sphere Efficiency Distribution')
        axes[0, 2].set_xlabel('Efficiency Percentage (%)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'Efficiency data\nnot available', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Sphere Efficiency Distribution')
    
    # Coverage vs Dice score correlation
    if 'lesion_coverage_percentage' in placement_summary.columns:
        axes[1, 0].scatter(placement_summary['lesion_coverage_percentage'], placement_summary['dice_score'], alpha=0.6, color='purple')
        axes[1, 0].set_title('Coverage vs Dice Score')
        axes[1, 0].set_xlabel('Lesion Coverage (%)')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = placement_summary['lesion_coverage_percentage'].corr(placement_summary['dice_score'])
        axes[1, 0].text(0.05, 0.95, f'r = {correlation:.3f}', transform=axes[1, 0].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        axes[1, 0].text(0.5, 0.5, 'Coverage data\nnot available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Coverage vs Dice Score')
    
    # Number of spheres vs coverage
    if 'lesion_coverage_percentage' in placement_summary.columns:
        axes[1, 1].scatter(placement_summary['num_spheres'], placement_summary['lesion_coverage_percentage'], alpha=0.6, color='red')
        axes[1, 1].set_title('Spheres vs Coverage')
        axes[1, 1].set_xlabel('Number of Spheres')
        axes[1, 1].set_ylabel('Lesion Coverage (%)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Coverage data\nnot available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Spheres vs Coverage')
    
    # Overlap percentage distribution
    if 'overlap_percentage' in placement_summary.columns:
        axes[1, 2].hist(placement_summary['overlap_percentage'], bins=20, edgecolor='black', alpha=0.7, color='salmon')
        axes[1, 2].set_title('Sphere Overlap Distribution')
        axes[1, 2].set_xlabel('Overlap Percentage (%)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Overlap data\nnot available', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Sphere Overlap Distribution')
    
    plt.tight_layout()
    plt.savefig('enhanced_clinical_placement_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Generate detailed coverage report
    print(f"\n=== DETAILED PLACEMENT ANALYSIS ===")
    
    if 'lesion_coverage_percentage' in placement_summary.columns:
        best_coverage = placement_summary.loc[placement_summary['lesion_coverage_percentage'].idxmax()]
        print(f"Best coverage placement:")
        print(f"  Patient: {best_coverage.name[0]}")
        print(f"  Coverage: {best_coverage['lesion_coverage_percentage']:.1f}%")
        print(f"  Dice: {best_coverage['dice_score']:.3f}")
        print(f"  Spheres: {best_coverage['num_spheres']}")
        print(f"  User: {best_coverage['user_id']}")
    
    best_dice = placement_summary.loc[placement_summary['dice_score'].idxmax()]
    print(f"\nBest Dice score placement:")
    print(f"  Patient: {best_dice.name[0]}")
    print(f"  Dice: {best_dice['dice_score']:.3f}")
    if 'lesion_coverage_percentage' in placement_summary.columns:
        print(f"  Coverage: {best_dice['lesion_coverage_percentage']:.1f}%")
    print(f"  Spheres: {best_dice['num_spheres']}")
    print(f"  User: {best_dice['user_id']}")
    
    return placement_summary


def main():
    """Main function to run the clinical placement collection tool"""
    print("=== ENHANCED Clinical Cryoablation Placement Collection Tool ===")
    print("🚀 Initializing...")
    print(f"📁 Using data directory: ./collection_tool/Clinical_set")
    
    # Check for advanced visualization capabilities
    if not ADVANCED_VIS_AVAILABLE:
        print("\n" + "⚠"*50)
        print("NOTICE: Advanced visualization functions not available!")
        print("To enable 3D visualizations and multi-view imaging:")
        print("1. Make sure 'agent_vis_2.py' is in the same directory")
        print("2. Ensure all required dependencies are installed")
        print("3. Only basic enhanced 2D visualization will be available")
        print("⚠"*50 + "\n")
    
    # Initialize collector with static directory
    collector = ClinicalPlacementCollector()
    
    # Debug: Check what patients were found
    print(f"🔍 Debug: collector.available_patients has {len(collector.available_patients)} patients")
    if collector.available_patients:
        print(f"🔍 Debug: First few patients: {collector.available_patients[:3]}")
    
    if not collector.available_patients:
        print("\n" + "="*60)
        print("🔧 NO VALID PATIENTS FOUND!")
        print("="*60)
        print(f"Expected data directory: {collector.data_directory}")
        print("\n📁 Required folder structure:")
        print("./collection_tool/Clinical_set/")
        print("├── Patient_001/")
        print("│   ├── t2.nii.gz")
        print("│   ├── gland.nii.gz")
        print("│   └── l_a1.nii.gz")
        print("├── Patient_002/")
        print("│   ├── t2.nii.gz")
        print("│   ├── gland.nii.gz")
        print("│   └── l_a1.nii.gz")
        print("└── ...")
        
        print(f"\n💡 SOLUTIONS:")
        print(f"1. Create the folder: {collector.data_directory}")
        print(f"2. Add patient folders with the required files")
        print(f"3. Each patient folder needs: t2.nii.gz, gland.nii.gz, l_a1.nii.gz")
        print(f"4. Use 'Change Data Dir' button in the GUI to select a different location")
        
        # Still create GUI for manual directory selection
        print(f"\n🖥️  Starting GUI for manual directory selection...")
    
    # Create GUI
    gui = InteractivePlacementGUI(collector)
    
    if collector.available_patients:
        print(f"\n🎯 Successfully initialized with {len(collector.available_patients)} patients!")
        print("✨ Enhanced Features:")
        print("✓ Comprehensive coverage analysis (lesion coverage %, sphere efficiency)")
        print("✓ Overlap detection and reporting")
        print("✓ Enhanced spacing metrics")
        print("✓ Real-time coverage feedback")
        print("✓ Detailed success messages with coverage breakdown")
        print("✓ Enhanced analysis with 6-panel visualization")
        if ADVANCED_VIS_AVAILABLE:
            print("✓ Multi-view medical imaging (sagittal, coronal, axial)")
            print("✓ 3D volume rendering capabilities")
            print("✓ Comprehensive analysis generation")
        else:
            print("⚠ Advanced visualizations not available (check agent_vis_2.py import)")
        print("\n📋 Instructions:")
        print("1. Select a patient from the dropdown")
        print("2. Enter your user ID")
        print("3. Click 'Load Patient' to load MRI data")
        print("4. Click 'Start Placement' for interactive sphere placement")
        print("5. Use mouse clicks to place spheres, arrow keys to navigate")
        print("6. Click 'Save Placement' when done - you'll see comprehensive coverage analysis")
        if ADVANCED_VIS_AVAILABLE:
            print("7. Use 'Create Comprehensive Analysis' for advanced multi-type visualizations")
            print("8. Use 'View Existing Placements' to see past placements with visualization options")
    
    # Run the application
    gui.run()
    
    # Analyze results if any exist
    if os.path.exists(collector.csv_file):
        print("\n=== Analyzing Enhanced Results ===")
        analyze_placement_results(collector.csv_file)


if __name__ == "__main__":
    main()