import os
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from multiprocessing import Pool, Manager
from scipy.stats import f_oneway

# Suppress warnings
warnings.filterwarnings("ignore")

# Default Parameters
DEFAULT_FREQ_RANGE = (60, 500)
DEFAULT_TIME_BIN_SIZE = 10  # in milliseconds
DEFAULT_TIME_WINDOW = 5  # in bins
DEFAULT_TRIALS = ['ENCODE', 'RECALL', 'DISTRACTOR', 'COUNTDOWN']


def plot_brain_map(coords, node_values, trial):
    """Plot brain map with node coordinates and values."""
    plt.figure(figsize=(30, 20), dpi=300)
    plotting.plot_glass_brain(None, display_mode='lyrz', colorbar=True, cmap='viridis')
    plotting.plot_markers(node_coords=coords, node_values=node_values, node_cmap='Reds', node_size=10)
    plt.title(f"Brain Map for {trial} Trial")
    plt.show()


def load_matching_files(encode_dir, recall_dir):
    """Return matching ENCODE and RECALL files based on common parts in filenames."""
    encode_files = os.listdir(encode_dir)
    recall_files = os.listdir(recall_dir)

    recall_parts = [filename.split('_word_')[1].split('_start')[0].lower() for filename in recall_files]
    matching_encode, matching_recall = [], []

    for encode_file in encode_files:
        common_part = encode_file.split('_word_')[1].split('_start')[0].lower()
        if common_part in recall_parts:
            matching_encode.append(os.path.join(encode_dir, encode_file))
            matching_recall.append(os.path.join(recall_dir, recall_files[recall_parts.index(common_part)]))

    return matching_encode, matching_recall


def find_coincidences(data, time_window):
    """Detect temporal coincidences in event data."""
    num_channels, num_timebins = data.shape
    coincidences = np.zeros((num_channels, num_timebins), dtype=int)

    for channel in range(num_channels):
        for timebin in range(num_timebins):
            if data[channel, timebin] == 1:
                coincidences[channel, max(0, timebin - time_window):timebin + time_window + 1] += 1

    return coincidences


def normalize_row(row):
    """Normalize a row by its sum."""
    total = sum(row)
    return [x / total if total > 0 else 0 for x in row]


def process_folder(args):
    """Process a single folder for the given trial type."""
    folder, trial, time_bin_size, time_window, freq_range = args
    folder_path = os.path.join(folder, trial)
    significant_means = []

    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return None

    try:
        files = os.listdir(folder_path)
        for event_file in files:
            with open(os.path.join(folder_path, event_file), 'rb') as f:
                data = pickle.load(f)

            # Filter frequency range
            data = data[(data['freq_at_max'] > freq_range[0]) & (data['freq_at_max'] < freq_range[1])]
            data['event_start'] = data['event_start'] * 1000  # Convert to milliseconds

            # Compute raster matrix
            min_time = int(data['event_start'].min())
            max_time = int(data['event_start'].max())
            num_bins = int((max_time - min_time) / time_bin_size) + 1
            channels = sorted(data['channel'].unique())
            channel_to_row = {channel: i for i, channel in enumerate(channels)}
            raster_matrix = np.zeros((len(channels), num_bins))

            for _, row in data.iterrows():
                channel = row['channel']
                time = int(row['event_start'])
                bin_index = int((time - min_time) / time_bin_size)
                raster_matrix[channel_to_row[channel], bin_index] += 1

            # Detect coincidences
            binary_raster = np.where(raster_matrix > 0, 1, 0)
            coincidence_matrix = find_coincidences(binary_raster, time_window)
            binary_coincidence = np.where(coincidence_matrix > 0, 1, 0)

            # Calculate mean coincidence rates
            mean_values = binary_coincidence.mean(axis=1)
            significant_means.append(mean_values)

    except Exception as e:
        print(f"Error processing folder {folder}: {str(e)}")
        return None

    return significant_means


def analyze_trials(trials, dataset_path, time_bin_size, time_window, freq_range):
    """Analyze all trials and return results."""
    results = {}

    for trial in trials:
        print(f"Processing trial: {trial}")
        trial_path = os.path.join(dataset_path, trial)
        if not os.path.exists(trial_path):
            print(f"Trial path does not exist: {trial_path}")
            continue

        with Pool() as pool:
            args = [(trial_path, trial, time_bin_size, time_window, freq_range)]
            trial_results = pool.map(process_folder, args)
            results[trial] = trial_results

    return results


def plot_results(results):
    """Plot mean and error bars for significant results across trials."""
    plt.figure(figsize=(30, 20), dpi=300)

    for trial, data in results.items():
        if data is None:
            continue
        means = np.mean(data, axis=0)
        std_error = np.std(data, axis=0) / np.sqrt(len(data))

        plt.plot(means, label=f"{trial} (Mean)", linewidth=2)
        plt.fill_between(range(len(means)), means - std_error, means + std_error, alpha=0.2)
    
    plt.xlabel("Time Bins")
    plt.ylabel("Normalized Mean")
    plt.title("Mean and Error Bars Across Trials")
    plt.legend()
    plt.show()


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze High-Frequency Oscillation (HFO) Synchronization")
    parser.add_argument("--dataset", required=True, help="Path to the dataset directory")
    parser.add_argument("--trials", nargs="+", default=DEFAULT_TRIALS, help="List of trials to analyze")
    parser.add_argument("--freq_range", type=int, nargs=2, default=DEFAULT_FREQ_RANGE, help="Frequency range (min max)")
    parser.add_argument("--time_bin_size", type=int, default=DEFAULT_TIME_BIN_SIZE, help="Time bin size in milliseconds")
    parser.add_argument("--time_window", type=int, default=DEFAULT_TIME_WINDOW, help="Time window in bins")
    args = parser.parse_args()

    # Run the analysis
    results = analyze_trials(args.trials, args.dataset, args.time_bin_size, args.time_window, tuple(args.freq_range))

    # Plot the results
    plot_results(results)


if __name__ == "__main__":
    main()
