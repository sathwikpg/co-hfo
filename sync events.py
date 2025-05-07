#%%
import os
import re
import pandas as pd
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from multiprocessing import Pool, Manager
import multiprocessing
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

f1 = 60
f2 = 500
# Path to local dataset
path_to_local = '/home/sathwik/brno/post/derivatives/bipolar/' # brno

def plot_brain_map(coords, node_values,trial):
    # Plot the brain
    plt.figure(figsize=(30, 20),dpi=300)

    display = plotting.plot_glass_brain(None, display_mode='lyrz', colorbar=True, cmap='viridis')
    
    # Plot the coordinates with the corresponding node values
    plotting.plot_markers(node_coords=coords, node_values=node_values, node_cmap='Reds', node_size=10)
    plt.title(trial)
    plt.show()


def load_matching_files(encode_directory, recall_directory):
    encode_files = os.listdir(encode_directory)
    recall_files = os.listdir(recall_directory)

    recall_common_parts = [filename.split('_word_')[1].split('_start')[0].lower() for filename in recall_files]

    matching_encode_files = []
    matching_recall_files = []

    for filename in encode_files:
        common_part = filename.split('_word_')[1].split('_start')[0].lower()
        if common_part in recall_common_parts:
            matching_encode_files.append(os.path.join(encode_directory, filename))
            matching_recall_files.append(os.path.join(
                recall_directory, recall_files[recall_common_parts.index(common_part)]))

    return matching_encode_files, matching_recall_files


def find_coincidences(data_array, time_window):
    num_channels, num_timebins = data_array.shape
    coincidences = np.zeros((num_channels, num_timebins), dtype=int)

    for channel in range(num_channels):
        for timebin in range(num_timebins):
            if data_array[channel, timebin] == 1:
                coincidences[channel, max(0, timebin - time_window):timebin + time_window + 1] += 1

    return coincidences


def normalize_row(row):
    total = sum(row)
    if total == 0:
        return [0] * len(row)
    return [x / total for x in row]

freq_ranges = [[60, 150], [150, 250], [250, 500]]


def process_folder(args):
    folder, trial, time_bin_size, time_window = args
    xyz = folder
    folder = path_to_local + "data/" + folder
    print(folder)
    if trial == 'ENCODE':
        encode_directory = folder + '/ENCODE/'
        recall_directory = folder + '/RECALL/'
        significant_counts = {}

        if os.path.exists(recall_directory):
            matching_encode_files, matching_recall_files = load_matching_files(encode_directory, recall_directory)
            significant_list = []
            non_significant_list = []
            lowest_mean_list = []
            node_coords = []
            node_values = []
            
            for event_file in matching_encode_files:
                try:
                    with open(event_file, 'rb') as f:
                        data = pickle.load(f)
                    data = data[(data['freq_at_max'] > f1) & (data['freq_at_max'] < f2)]
                    data['event_start'] = data['event_start'] * 1000
                    channel_to_coords = {row['channel']: (row['x'], row['y'], row['z']) for _, row in data.iterrows()}

                    min_time = int(data['event_start'].min())
                    max_time = int(data['event_start'].max())
                    num_bins = int((max_time - min_time) / time_bin_size) + 1

                    for y_axis_column in ['channel']:
                        if y_axis_column in data.columns:
                            channels = sorted(data[y_axis_column].unique())
                            channel_to_row = {channel: i for i, channel in enumerate(channels)}
                            raster_matrix = np.zeros((len(channels), num_bins))

                            for _, row in data.iterrows():
                                channel = row[y_axis_column]
                                time = int(row['event_start'])
                                bin_index = int((time - min_time) / time_bin_size)
                                raster_matrix[channel_to_row[channel], bin_index] += 1
                            binary_raster_matrix = np.where(raster_matrix > 0, 1, 0)
                            raster_matrix = find_coincidences(binary_raster_matrix, time_window)
                            binary_raster_matrix = np.where(raster_matrix > 0, 1, 0)

                            
                            
                            raster_df = pd.DataFrame(raster_matrix, index=channels)
                            corr_df = pd.DataFrame(raster_matrix, index=channels)
                            df_transposed = corr_df.T
                            corr_matrix = df_transposed.corr()

                            mean_per_row = corr_matrix.apply(lambda row: sorted(row)[-2], axis=1)
                            df_sorted = df_transposed[mean_per_row.sort_values(ascending=False).index].T
                        
                            significant_rows = df_sorted[mean_per_row > 0.5]
                            non_significant_rows = df_sorted[mean_per_row <= 0.5]
                            
                            lowest_per_row = corr_matrix.apply(min, axis=1)
                            df_sorted_lowest = df_transposed[lowest_per_row.sort_values(ascending=True).index].T
                            lowest_rows = df_sorted_lowest[lowest_per_row < -0.3]
                            
                            significant_mean = significant_rows.mean()
                            non_significant_mean = non_significant_rows.mean()
                            lowest_mean = lowest_rows.mean()
                            
                            significant_list.append(significant_mean)
                            non_significant_list.append(non_significant_mean)
                            lowest_mean_list.append(lowest_mean)
                            
                            significant_channels = significant_rows.index
                            non_significant_channels = non_significant_rows.index
                            
                            for channel in significant_channels:
                                if channel in channel_to_coords:
                                    if channel not in significant_counts:
                                        significant_counts[channel] = 0
                                    significant_counts[channel] += 1

                except Exception as e:
                    print(f"Error processing {event_file}: {str(e)}")
                    continue

            significant_df = pd.concat(significant_list, axis=1)
            non_significant_df = pd.concat(non_significant_list, axis=1)
            lowest_mean_df = pd.concat(lowest_mean_list, axis=1)

            # Normalize significant counts by the total number of files
            total_event_files = len(matching_encode_files)
            normalized_significant_counts = {channel: count / total_event_files for channel, count in significant_counts.items()}

            # Set node_coords and node_values based on normalized significant counts
            node_coords = []
            node_values = []
            for channel, count in normalized_significant_counts.items():
                if channel in channel_to_coords:
                    node_coords.append(channel_to_coords[channel])
                    node_values.append(count)

            # Plot the brain map using normalized significant counts as node values
            plot_brain_map(node_coords, node_values,trial)

            max_len = max(len(significant_df.columns), len(non_significant_df.columns))
            significant_df = significant_df.reindex(range(max_len), axis=1)
            non_significant_df = non_significant_df.reindex(range(max_len), axis=1)
            lowest_mean_df = lowest_mean_df.reindex(range(max_len), axis=1)
            
            significant_list_mean = significant_df.mean(axis=1)
            significant_list_std = significant_df.std(axis=1) / np.sqrt(len(significant_df.columns))
            non_significant_list_mean = non_significant_df.mean(axis=1)
            non_significant_list_std = non_significant_df.std(axis=1) / np.sqrt(len(non_significant_df.columns))
            lowest_mean_list_mean = lowest_mean_df.mean(axis=1)
            lowest_mean_list_std = lowest_mean_df.std(axis=1)

            plt.figure(figsize=(10, 6))
            plt.plot(significant_list_mean.index, significant_list_mean.values, label='Significant', color='blue')
            plt.fill_between(significant_list_mean.index, significant_list_mean - significant_list_std, significant_list_mean + significant_list_std, alpha=0.2, color='blue')
            plt.plot(non_significant_list_mean.index, non_significant_list_mean.values, label='Non-Significant', color='red')
            plt.fill_between(non_significant_list_mean.index, non_significant_list_mean - non_significant_list_std, non_significant_list_mean + non_significant_list_std, alpha=0.2, color='red')
            
            plt.title(xyz + ' trial ' + trial)
            plt.xlabel(f'Time Bin = {time_bin_size} ms, win = +/- {time_window} bins')
            plt.ylabel('Normalized Probability')
            plt.legend()

            plt.figure(figsize=(10, 6))
            plt.bar(normalized_significant_counts.keys(), normalized_significant_counts.values(), color='blue')
            plt.xlabel('Channels')
            plt.ylabel('Normalized Significant Count')
            plt.title('Normalized Significant Count per Channel')
            plt.close()
        else:
            print(f"ENCODE  do not exist in {folder}")
            return  # Skip further processing if directories don't exist
        
    if trial != 'ENCODE':
        file_directory = folder + f'/{trial}/'
        if os.path.exists(file_directory):
            files_list = os.listdir(file_directory)
            node_coords = []
            node_values = []
            significant_list = []
            non_significant_list = []
            lowest_mean_list = []
            significant_counts = {}  # Track significant counts per channel
            
            for event_file in files_list:
                try:
                    with open(file_directory + event_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Filter frequency range
                    data = data[(data['freq_at_max'] > f1) & (data['freq_at_max'] < f2)]
                    data['event_start'] = data['event_start'] * 1000
                    channel_to_coords = {row['channel']: (row['x'], row['y'], row['z']) for _, row in data.iterrows()}

                    # Determine the time range of the data
                    min_time = int(data['event_start'].min())
                    max_time = int(data['event_start'].max())
                    
                    # Calculate the number of time bins
                    num_bins = int((max_time - min_time) / time_bin_size) + 1

                    for y_axis_column in ['channel']:
                        if y_axis_column in data.columns:
                            channels = sorted(data[y_axis_column].unique())
                            channel_to_row = {channel: i for i, channel in enumerate(channels)}
                            raster_matrix = np.zeros((len(channels), num_bins))

                            # Fill raster matrix based on events
                            for _, row in data.iterrows():
                                channel = row[y_axis_column]
                                time = int(row['event_start'])
                                bin_index = int((time - min_time) / time_bin_size)
                                raster_matrix[channel_to_row[channel], bin_index] += 1
                            
                            # Apply coincidence detection
                            binary_raster_matrix = np.where(raster_matrix > 0, 1, 0)
                            raster_matrix = find_coincidences(binary_raster_matrix, time_window)
                            binary_raster_matrix = np.where(raster_matrix > 0, 1, 0)

                            
                            # Create correlation matrix
                            df_transposed = pd.DataFrame(raster_matrix, index=channels).T
                            corr_matrix = df_transposed.corr()

                            # Find mean per row of correlation matrix
                            mean_per_row = corr_matrix.apply(lambda row: sorted(row)[-2], axis=1)
                            df_sorted = df_transposed[mean_per_row.sort_values(ascending=False).index].T

                            # Filter significant and non-significant rows
                            significant_rows = df_sorted[mean_per_row > 0.5]
                            non_significant_rows = df_sorted[mean_per_row <= 0.5]

                            # Calculate lowest correlations
                            lowest_per_row = corr_matrix.apply(min, axis=1)
                            df_sorted_lowest = df_transposed[lowest_per_row.sort_values(ascending=True).index].T
                            lowest_rows = df_sorted_lowest[lowest_per_row < -0.3]

                            # Calculate means
                            significant_mean = significant_rows.mean()
                            non_significant_mean = non_significant_rows.mean()
                            lowest_mean = lowest_rows.mean()

                            # Append to lists
                            significant_list.append(significant_mean)
                            non_significant_list.append(non_significant_mean)
                            lowest_mean_list.append(lowest_mean)

                            # Extract coordinates and mark channels as significant or non-significant
                            significant_channels = significant_rows.index
                            non_significant_channels = non_significant_rows.index
                            
                            for channel in significant_channels:
                                if channel in channel_to_coords:
                                    if channel not in significant_counts:
                                        significant_counts[channel] = 0
                                    significant_counts[channel] += 1  # Increment count for significant channels
                            
                except Exception as e:
                    print(f"Error processing {event_file}: {str(e)}")
                    continue                    
            
            # Normalize significant counts by total files
            total_event_files = len(files_list)
            normalized_significant_counts = {channel: count / total_event_files for channel, count in significant_counts.items()}

            # Extract coordinates and values for brain plot
            for channel, count in normalized_significant_counts.items():
                if channel in channel_to_coords:
                    node_coords.append(channel_to_coords[channel])
                    node_values.append(count)  # Use normalized count

            # Plot brain map with node coordinates and normalized counts
            #plot_brain_map(node_coords, node_values,trial)

            # Combine the lists into DataFrames
            significant_df = pd.concat(significant_list, axis=1)
            non_significant_df = pd.concat(non_significant_list, axis=1)
            lowest_mean_df = pd.concat(lowest_mean_list, axis=1)

            # Align the DataFrames and fill missing values with NaN
            max_len = max(len(significant_df.columns), len(non_significant_df.columns))
            significant_df = significant_df.reindex(range(max_len), axis=1)
            non_significant_df = non_significant_df.reindex(range(max_len), axis=1)
            lowest_mean_df = lowest_mean_df.reindex(range(max_len), axis=1)
            
            # Calculate the mean and standard deviation
            significant_list_mean = significant_df.mean(axis=1)
            significant_list_std = significant_df.std(axis=1) / np.sqrt(len(significant_df.columns))
            non_significant_list_mean = non_significant_df.mean(axis=1)
            non_significant_list_std = non_significant_df.std(axis=1) / np.sqrt(len(non_significant_df.columns))
            lowest_mean_list_mean = lowest_mean_df.mean(axis=1)
            lowest_mean_list_std = lowest_mean_df.std(axis=1)

            # Plot smoothed means with error areas
            plt.figure(figsize=(10, 6))
            plt.plot(significant_list_mean.index, significant_list_mean.values, label='Significant', color='blue')
            plt.fill_between(significant_list_mean.index, significant_list_mean - significant_list_std, significant_list_mean + significant_list_std, alpha=0.2, color='blue')

            plt.plot(non_significant_list_mean.index, non_significant_list_mean.values, label='Non-Significant', color='red')
            plt.fill_between(non_significant_list_mean.index, non_significant_list_mean - non_significant_list_std, non_significant_list_mean + non_significant_list_std, alpha=0.2, color='red')
            
            # Plot titles and labels
            plt.title(f'{xyz} trial {trial}')
            plt.xlabel(f'Time Bin = {time_bin_size} ms, win = +/- {time_window} bins')
            plt.ylabel('Normalized Probability')
            plt.legend()
            #plt.show()
            plt.close()

        else:
            print(f"{trial} does not exist in {folder}")
            return
                    
    return significant_list_mean,non_significant_list_mean,lowest_mean_list_mean  # Placeholder for the actual implementation

def generate_sig_trial(trial, folder_path):
    folders_list = pd.read_pickle(folder_path)
    folders_list = [x for x in folders_list if 'run-01' in x and 'sub-018' not in x and 'sub-017'not in x]

    # Create a manager list to store the results from different processes
    manager = Manager()
    all_significant_means = manager.list()
    all_non_significant_means = manager.list()
    all_lowest_means = manager.list()
    all_node_coords = manager.list()
    all_node_values = manager.list()
    # Define the time bin size in milliseconds
    time_bin_size = 10
    time_window = 5

    # Create a pool of worker processes
    pool = multiprocessing.Pool()

    # Apply the process_folder function to each folder using the worker processes
    results = pool.map(process_folder, [(folder, trial, time_bin_size, time_window) for folder in folders_list])
    all_significant_means.extend([res[0] for res in results])
    all_non_significant_means.extend([res[1] for res in results])
    all_lowest_means.extend([res[2] for res in results])
    
    # Close the pool and wait for the processes to finish
    pool.close()
    pool.close()

    pool.join()

    # Normalize all rows in both DataFrames
    all_significant_means = [normalize_row(row) for row in all_significant_means]
    all_non_significant_means = [normalize_row(row) for row in all_non_significant_means]

    # Convert the lists to DataFrames
    all_significant_df = pd.DataFrame(all_significant_means)
    all_non_significant_df = pd.DataFrame(all_non_significant_means)

    # Calculate the mean and standard deviation for all_significant_df and all_non_significant_df
    significant_means = all_significant_df.mean()
    significant_std = all_significant_df.std() / np.sqrt(len(all_significant_df))
    non_significant_means = all_non_significant_df.mean()
    non_significant_std = all_non_significant_df.std() / np.sqrt(len(all_non_significant_df))

   # Set the figure size
    plt.figure(figsize=(30,20))

    # Crop the x-axis values and data points
    crop_start = 20
    crop_end = -20
    x_values = significant_means.index[crop_start:crop_end]
    significant_data = significant_means.values[crop_start:crop_end]
    non_significant_data = non_significant_means.values[crop_start:crop_end]
    significant_std_data = significant_std[crop_start:crop_end]
    non_significant_std_data = non_significant_std[crop_start:crop_end]

    # Plot the mean lines and error areas for significant rows
    plt.plot(x_values, significant_data, label='Significant', color='orange')
    plt.fill_between(x_values, significant_data - significant_std_data, significant_data + significant_std_data,
                    alpha=0.2, color='orange')

    # Plot the mean lines and error areas for non-significant rows
    plt.plot(x_values, non_significant_data, label='Non-Significant', color='black')
    plt.fill_between(x_values, non_significant_data - non_significant_std_data,
                    non_significant_data + non_significant_std_data, alpha=0.2, color='black')
    #add y lim from 0 to 0.005
    plt.ylim(0.0025,0.004)
    plt.axhline(y=0.005, color='r', linestyle='--')
    # Customize the plot
    plt.xlabel('time in seconds')
    plt.ylabel('Normalized Mean')
    plt.xticks([50, 150, 250], ['-1', '0', '1'])


    plt.title(trial)
    plt.legend()
    plt.grid()

    # Save and show the plot
    #plt.savefig(f'/home/sathwik/brno/post/scripts/plots/brno_sync_rates_Sig>0.5_bin={time_bin_size}_win={time_window}_svns_global_{trial}_S_vs_NS.png')
    plt.show()

    # Return the significant DataFrame
    return all_significant_df

trials = ['ENCODE', 'RECALL', 'DISTRACTOR', 'COUNTDOWN']
trial_dfs = {}

for trial in trials:
    #folder_path = '/home/sathwik/brno/post/mayo_dataset/post/folder_names.pkl' #mayo
    folder_path = '/home/sathwik/brno/post/derivatives/bipolar/folder_names.pkl' #brno
    #folder_path = '/home/sathwik/brno/wroclaw_memory_dataset/derivatives/post/folder_names.pkl'
    trial_df = generate_sig_trial(trial, folder_path)
    trial_dfs[f"sig_{trial}"] = trial_df

# Access the DataFrames by their respective trial names:
sig_recall = trial_dfs["sig_RECALL"]
sig_encode = trial_dfs["sig_ENCODE"]
sig_distractor = trial_dfs["sig_DISTRACTOR"]
sig_countdown = trial_dfs["sig_COUNTDOWN"]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Assuming you have the DataFrames sig_recall, sig_countdown, sig_distractor, and sig_encode.

# Calculate the standard error for each DataFrame
recall_means = sig_recall.mean()
countdown_means = sig_countdown.mean()
distractor_means = sig_distractor.mean()
encode_means = sig_encode.mean()

recall_se = sig_recall.std() / np.sqrt(len(sig_recall))
countdown_se = sig_countdown.std() / np.sqrt(len(sig_countdown))
distractor_se = sig_distractor.std() / np.sqrt(len(sig_distractor))
encode_se = sig_encode.std() / np.sqrt(len(sig_encode))

# Set the figure size
plt.figure(figsize=(30, 20),dpi=300)

# Plot the mean lines and error areas for each DataFrame using standard error
plt.plot(recall_means.index[20:-20], recall_means.values[20:-20], label='recall', color='blue')
plt.fill_between(recall_means.index[20:-20], recall_means[20:-20] - recall_se[20:-20], recall_means[20:-20] + recall_se[20:-20], alpha=0.2, color='blue')

plt.plot(countdown_means.index[20:-20], countdown_means.values[20:-20], label='countdown', color='green')
plt.fill_between(countdown_means.index[20:-20], countdown_means[20:-20] - countdown_se[20:-20], countdown_means[20:-20] + countdown_se[20:-20], alpha=0.2, color='green')

plt.plot(distractor_means.index[20:-20], distractor_means.values[20:-20], label='distractor', color='orange')
plt.fill_between(distractor_means.index[20:-20], distractor_means[20:-20] - distractor_se[20:-20], distractor_means[20:-20] + distractor_se[20:-20], alpha=0.2, color='orange')

plt.plot(encode_means.index[20:-20], encode_means.values[20:-20], label='encode', color='purple')
plt.fill_between(encode_means.index[20:-20], encode_means[20:-20] - encode_se[20:-20], encode_means[20:-20] + encode_se[20:-20], alpha=0.2, color='purple')

# Customize the plot
plt.xlabel('Columns')
plt.ylabel('Normalized Mean bin = 10ms, window = 5 bins')
plt.title('Mayo Mean and Error Area for Significant Rows (>0.5) '+f'freq={f1}-{f2}Hz')
plt.show()
#plt.savefig('/home/sathwik/brno/post/scripts/sync_time/brno_sync_rates_Sig>0.5_bin=10_win=5_'+f'freq={f1}-{f2}Hz.png')
# Perform ANOVA test and calculate p-values
window_size = 1
p_values = []

for i in range(len(sig_recall.columns)):
    smoothed_data = []
    
    for df in [sig_recall, sig_countdown, sig_distractor, sig_encode]:
        smoothed_column = df.iloc[:, i].rolling(window=window_size, min_periods=1).mean()
        smoothed_data.append(smoothed_column)
    
    _, p_value = f_oneway(*smoothed_data)
    p_values.append(p_value)

# Create a separate plot for the p-values
plt.figure(figsize=(10, 6))
plt.plot(recall_means.index, p_values, label='p-values', color='red')
plt.ylabel('p-values')
plt.title('ANOVA p-values vs Time Bins')
plt.grid()
plt.close()
plt.figure(figsize=(30, 20),dpi=300)
rolling_window = 5

# Use pandas' rolling function to calculate rolling mean for each data series
smooth_recall_means = recall_means.rolling(window=rolling_window).mean()
smooth_countdown_means = countdown_means.rolling(window=rolling_window).mean()
smooth_distractor_means = distractor_means.rolling(window=rolling_window).mean()
smooth_encode_means = encode_means.rolling(window=rolling_window).mean()

# Plot the mean lines and error areas for each DataFrame using standard error
plt.plot(recall_means.index[20:-20], smooth_recall_means[20:-20], label='recall', color='blue')
plt.fill_between(recall_means.index[20:-20], smooth_recall_means[20:-20] - recall_se[20:-20], smooth_recall_means[20:-20] + recall_se[20:-20], alpha=0.2, color='blue')

plt.plot(countdown_means.index[20:-20], smooth_countdown_means[20:-20], label='countdown', color='green')
plt.fill_between(countdown_means.index[20:-20], smooth_countdown_means[20:-20] - countdown_se[20:-20], smooth_countdown_means[20:-20] + countdown_se[20:-20], alpha=0.2, color='green')

plt.plot(distractor_means.index[20:-20], smooth_distractor_means[20:-20], label='distractor', color='orange')
plt.fill_between(distractor_means.index[20:-20], smooth_distractor_means[20:-20] - distractor_se[20:-20], smooth_distractor_means[20:-20] + distractor_se[20:-20], alpha=0.2, color='orange')

plt.plot(encode_means.index[20:-20], smooth_encode_means[20:-20], label='encode', color='purple')
plt.fill_between(encode_means.index[20:-20], smooth_encode_means[20:-20] - encode_se[20:-20], smooth_encode_means[20:-20] + encode_se[20:-20], alpha=0.2, color='purple')

# Customize the plot
plt.xlabel('Time in seconds')
plt.ylabel('Normalized Mean bin = 10ms, window = 5 bins')
plt.title('Mean and Error Area for Significant Rows (>0.5) '+f'freq={f1}-{f2}Hz') #freq=60-150Hz
plt.xticks([50, 150, 250], ['-1', '0', '1'])

#plt.grid()
plt.show()
# %%

