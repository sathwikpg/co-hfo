"""
High-Frequency Oscillation (HFO) Session Processing Script
Created on: Mon Apr 24, 2023
Author: Sathwik
Purpose: Process post-processed HFO detections from sEEG data into task-specific trial segments.
"""

import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from argparse import ArgumentParser

# Utility Functions
def save_to_file(filename, data):
    """Save data to a file using pickle."""
    with open(filename, 'wb') as filehandle:
        pickle.dump(data, filehandle)  # Save as binary data stream


def load_from_file(filename):
    """Load data from a pickle file."""
    with open(filename, 'rb') as filehandle:
        return pickle.load(filehandle)


def create_folders_from_pkl(pkl_file, output_dir):
    """
    Create folders based on filenames listed in a pickle file.
    Args:
        pkl_file (str): Path to the pickle file containing filenames.
        output_dir (str): Output directory where folders will be created.
    Returns:
        list: List of created folder paths.
    """
    filenames = load_from_file(pkl_file)
    folders = []

    for filename in filenames:
        folder_name = os.path.splitext(filename)[0]
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        folders.append(folder_name)

    return folders


def process_session(session_idx, hfo_file, event_file, output_base_path, folders):
    """
    Process a single session by splitting HFO detections into task-specific trials.
    Args:
        session_idx (int): Index of the session to process.
        hfo_file (str): Path to the HFO pickle file.
        event_file (str): Path to the event pickle file.
        output_base_path (str): Base directory to save processed trials.
        folders (list): List of folder names corresponding to sessions.
    """
    # Load HFO and event data
    hfo_df = pd.read_pickle(hfo_file)
    event_df = pd.read_pickle(event_file)

    # Adjust time units to seconds
    event_df['uutc_time'] = event_df['uutc_time'] / 1_000_000
    hfo_df['event_start'] = hfo_df['event_start'] / 1_000_000
    event_df['event_start'] = event_df['uutc_time'] - 1.5
    event_df['event_stop'] = event_df['uutc_time'] + 1.5

    # Create output directory for the session
    session_output_path = os.path.join(output_base_path, folders[session_idx])
    os.makedirs(session_output_path, exist_ok=True)

    # Process each trial in the session
    for _, event_row in event_df.iterrows():
        # Filter HFO events overlapping with the current trial window
        event_data = hfo_df[
            hfo_df['event_start'].between(event_row['event_start'], event_row['event_stop']) |
            hfo_df['event_stop'].between(event_row['event_start'], event_row['event_stop'])
        ]

        # Save the trial-specific HFO data
        trial_folder = os.path.join(session_output_path, str(event_row['trial_type']))
        os.makedirs(trial_folder, exist_ok=True)

        trial_filename = f"trial_{event_row['trial_type']}_word_{event_row['text']}_start_{event_row['event_start']}_stop_{event_row['event_stop']}.pkl"
        trial_path = os.path.join(trial_folder, trial_filename)
        save_to_file(trial_path, event_data)
        print(f"Saved: {trial_path}")


def main():
    # Argument Parser
    parser = ArgumentParser(description="Process HFO session data into task-specific trials.")
    parser.add_argument("--hfo_dir", required=True, help="Directory containing HFO pickle files.")
    parser.add_argument("--event_dir", required=True, help="Directory containing event pickle files.")
    parser.add_argument("--output_dir", required=True, help="Base directory to save processed trials.")
    parser.add_argument("--folders_pkl", required=True, help="Pickle file containing folder names.")
    parser.add_argument("--num_processes", type=int, default=multiprocessing.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    # Load folder names
    folders = load_from_file(args.folders_pkl)

    # Process sessions in parallel
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        pool.starmap(
            process_session,
            [
                (
                    session_idx,
                    os.path.join(args.hfo_dir, folders[session_idx] + ".pkl"),
                    os.path.join(args.event_dir, folders[session_idx] + "_event.pkl"),
                    args.output_dir,
                    folders
                )
                for session_idx in range(len(folders))
            ]
        )
    print("HFO Session Processing is complete!")


if __name__ == "__main__":
    main()
