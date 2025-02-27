import os
import glob
import h5py
import pandas as pd
from pathlib import Path
#load preprocessed data from Matlab files in Zuco Datasets


# DataFrame to track skipped files
skipped_files = pd.DataFrame(columns=["File", "Reason"])

def load_mat_eeg(eeg_file, participant, task, dataset):
    #grabs a single EEG file (for given participant)
    """Load EEG data from a MATLAB .mat file, skipping older formats."""
    if not os.path.exists(eeg_file):
        print(f"File not found: {eeg_file}")
        skipped_files.loc[len(skipped_files)] = [eeg_file, "File not found"]
        return None
    
    try:
        with h5py.File(eeg_file, 'r') as f:
            eeg_data = f['EEG/data'][:].T  # Shape: (Channels, Timepoints)
            chanlocs_labels = f['EEG/chanlocs/labels'][:]
            time_data = f['EEG/times'][:]
            
            # Extract channel names
            channel_names = ["".join(chr(i[0]) for i in f[ref][:]) for ref in chanlocs_labels.flatten()]
            
            print(f"EEG Data Shape: {eeg_data.shape}")
            print(f"Extracted Channels: {len(channel_names)}")
            
            # Convert to DataFrame
            eeg_df = pd.DataFrame(eeg_data.T, columns=channel_names)
            eeg_df['Time'] = time_data  # Add time column
            eeg_df['Participant'] = participant
            eeg_df['Task'] = task
            eeg_df['Dataset'] = dataset
            eeg_df['SNR'] = eeg_file.split('\\')[-1].split('_')[-2]
            eeg_df['FileName'] = eeg_file.split('\\')[-1]
            return eeg_df
    except OSError:
        print(f"Skipping file {eeg_file}: Not in HDF5 format")
        skipped_files.loc[len(skipped_files)] = [eeg_file, "Not in HDF5 format"]
        return None

def load_all_eeg():
    """Iterate through all Zuco EEG files"""
    #all_data = []
    #datasets = ["ZuCo1", "ZuCo2"]
    datasets = ["ZuCo1"]
    tasks = ["task1", "task2"]
    
    for dataset in datasets:
        for task in tasks:
            task_dir = "C:\\Users\\meca8121\\Emotive Computing Dropbox\\Megan Caruso\\MC Data Folders\\EyeMindLink_MC\\Projects\\eeg_gaze_deeplearning\\RawDatasets\\{}\\{}\\Preprocessed\\".format(dataset, task)
            print(task_dir)
            if not os.path.exists(task_dir):
                print(f"Could not find processed directory for: {dataset}, {task}")
                continue
            
            for participant in os.listdir(task_dir):
                participant_dir = os.path.join(task_dir, participant)
                
                # Iterate through all EEG files for that participant
                eeg_files = glob.glob(os.path.join(participant_dir, "*_EEG.mat"))
                for eeg_file in eeg_files:
                    print(f"Processing file: {eeg_file}")
                    eeg_df = load_mat_eeg(eeg_file, participant, task, dataset)
                    #if eeg_df is not None:
                    #    all_data.append(eeg_df)
    
    #return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# Load and display the final DataFrame
eeg_data_df = load_all_eeg()


# test
eeg_file = "C:\\Users\\meca8121\\Emotive Computing Dropbox\\Megan Caruso\\MC Data Folders\\EyeMindLink_MC\\Projects\\eeg_gaze_deeplearning\\RawDatasets\\ZuCo1\\task1\\Preprocessed\\ZAB\\gip_ZAB_SNR6_EEG.mat"
e = load_mat_eeg(eeg_file, 'ZAB', 'task1', 'Zuco1')
e.to_csv('gip_ZAB_SNR6_EEG_test.csv')
