import os
import glob
import h5py
import pandas as pd

def load_mat_eeg(eeg_file, participant, task, dataset):
    """Load EEG data from a MATLAB v7.3 .mat file."""
    #mat_file = os.path.join(filepath, dataset, task, "Raw Data", participant, f"{participant}_EEG.mat")
    
    if not os.path.exists(eeg_file):
        print(f"File not found: {eeg_file}")
        return None
    
    with h5py.File(eeg_file, 'r') as f:
        eeg_data = f['EEG/data'][:].T  # Shape: (Channels, Timepoints)
        chanlocs_labels = f['EEG/chanlocs/labels'][:]
        
        # Extract channel names
        channel_names = []
        for ref in chanlocs_labels.flatten():
            label_data = f[ref][:]  # Dereference
            channel_name = ''.join(chr(i[0]) for i in label_data)  # Convert to string
            channel_names.append(channel_name)
        
        # Convert to DataFrame
        eeg_df = pd.DataFrame(eeg_data.T, columns=channel_names)  # (Timepoints, Channels)
        
        # Add metadata
        eeg_df['Participant'] = participant
        eeg_df['Task'] = task
        eeg_df['Dataset'] = dataset
        
    return eeg_df

def load_all_eeg(base_dir):
    """Load all EEG data from ZuCo1 and ZuCo2 datasets."""
    all_data = []
    datasets = ["ZuCo1", "ZuCo2"]
    tasks = ["task1", "task2"]
    
    for dataset in datasets:
        for task in tasks:
            task_dir = os.path.join(base_dir, dataset, task, "Preprocessed")
            print(task_dir)
            if not os.path.exists(task_dir):
                print(f"Could not find processed directory for: {dataset}, {task}")
                continue
            
            for participant in os.listdir(task_dir):
                participant_dir = os.path.join(task_dir, participant)
                if not os.path.isdir(participant_dir):
                    print(f"Processed directory not found for: {participant}")
                    continue
                
                # Load all _EEG.mat files
                eeg_files = glob.glob(os.path.join(participant_dir, "*_EEG.mat"))
                for eeg_file in eeg_files:
                    print(f"Processing file: {eeg_file}")
                    eeg_df = load_mat_eeg(eeg_file, participant, task, dataset)
                    if eeg_df is not None:
                        all_data.append(eeg_df)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Define dataset directories
#BASE_DIR = os.path.join("C:/", "Users", "meca8121", "Emotive Computing Dropbox", "Megan Caruso", 
#                        "MC Data Folders", "EyeMindLink_MC", "Projects", "eeg_gaze_deeplearning", "datasets")

BASE_DIR = "C:/Users/meca8121/Emotive Computing Dropbox/Megan Caruso/MC Data Folders/EyeMindLink_MC/Projects/eeg_gaze_deeplearning/datasets/"

participant = "ZPH"
task = "task1"
dataset = "ZuCo1"
base_dir= "C:/Users/meca8121/Emotive Computing Dropbox/Megan Caruso/MC Data Folders/EyeMindLink_MC/Projects/eeg_gaze_deeplearning/datasets/"
filepath = base_dir

# Load and display the final DataFrame
eeg_data_df = load_all_eeg(BASE_DIR)
if not eeg_data_df.empty:
    import ace_tools as tools
    tools.display_dataframe_to_user(name="EEG Data", dataframe=eeg_data_df)


file_path =  "C:/Users/meca8121/Emotive Computing Dropbox/Megan Caruso/MC Data Folders/EyeMindLink_MC/Projects/eeg_gaze_deeplearning/datasets/ZuCo1/task1/Preprocessed/ZMG/oip_ZMG_SR1_EEG.mat"
file_path = "C:/Users/meca8121/Emotive Computing Dropbox/Megan Caruso/MC Data Folders/EyeMindLink_MC/Projects/eeg_gaze_deeplearning/datasets/ZuCo1/task1/Preprocessed/ZMG/oip_ZMG_SNR7_EEG.mat"
try:
    with h5py.File(file_path, 'r') as f:
        print("File successfully opened as HDF5.")
except OSError:
    print("File is not in HDF5 format.")