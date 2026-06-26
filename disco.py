#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:55:56 2024

@author: wi11iamk
"""
#%%

###
# Import libraries
###

import os, csv
import numpy as np, pandas as pd
import umap, umap.plot
import matplotlib.pyplot as plt, seaborn as sns
import spyder_kernels.utils.iofuncs as io
from fastdtw import fastdtw
from hubdt import data_loading, wavelets, t_sne, hdb_clustering, b_utils
from mycolours import custom_colour_list
from parameters import D1, D2
from scipy.integrate import simpson
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import jensenshannon
from scipy.stats import sem, t, norm, linregress, ttest_rel

#%%

###
# Import all DLC features and tracking data; apply Savitzky-Golay filter to
# each 'y' channel and normalise each first frame to 0
###

pt = '067' # Choose the participant number
day = 'D1'

if day == 'D1':
    params = D1[pt]
elif day == 'D2':
    params = D2[pt]
else:
    raise ValueError("Invalid day specified. Please use 'D1' or 'D2'.")
    
# Load DLC h5 file, extract features and tracking data
h5 = data_loading.load_dlc_hdf(f'./data/{day}/{pt}_{day}_NR.h5')
h5 = data_loading.dlc_remove_scorer(h5)
features = list(h5.T.index.get_level_values(0).unique())

del features[4:7] # Delete extra features should they exist

tracking = data_loading.load_tracking (pt=pt, day=day, dlc=True, feats=features)

# Apply Savitzky-Golay filter and normalise first frames to 0
tracking_filtered = tracking.copy() # Create copy of tracking to filter
for col in range(1, 8, 2):  # Iterate over y-value columns (1, 3, 5, 7)
    # Apply Savitzky-Golay filter
    tracking_filtered[:, col] = savgol_filter(tracking_filtered[:, col], 7, 1)
    
    # Normalise the filtered y-values
    tracking_filtered[:, col] -= tracking_filtered[0, col]
    
#%%

###
# Initialise dictionaries to store frame values for each trial over one or two
# days of testing as well as micro on- and offline subdivisions of trials
###
    
# Initialise a dictionary to store frames for each trial

trial = {}

if day == 'D1':
    total_trials = 36
elif day == 'D2':
    total_trials = 72
else:
    raise ValueError("Invalid day value. It should be 'D1' or 'D2'.")

# Populate the dictionary with a slice of frames per trial
for i in range(total_trials):
    key = f"{i+1}"  # Key name (e.g., '1' for trial 1)
    start_index = i * 1200
    end_index = (i + 1) * 1200 - 1
    trial[key] = (start_index, end_index)

# Initialise a dictionary to store micro-segments
micro = {}
# Number of frames to consider for each micro-segment
frames_per_micro = 120

# Subdivide each trial's frame range into first and last 120 frames (1 s)
for key, (start, end) in trial.items():
    # Calculate the frame range for the first 120 frames (m1, m3, m5, ...)
    micro[f"{key}_m1"] = (start, start + frames_per_micro - 1)
    # Calculate the frame range for the last 120 frames (m2, m4, m6, ...)
    micro[f"{key}_m2"] = (end - frames_per_micro + 1, end)

#%%

###
# Generate scales and frequencies for wavelet transform of the tracking data;
# store the wavelet projection into a variable and then transpose the output
###

scales, frequencies = wavelets.calculate_scales (0.75, 2.75, 120, 5)

proj = wavelets.wavelet_transform_np (tracking_filtered, scales, frequencies, 120)

proj = np.transpose(proj)

#%%

###
# Fit wavelet projection into two dimensional embedded space (UMAP); plot;
# calculate the KDE and dimensions of the entire data set
###

mapper = umap.UMAP(n_neighbors=35, n_components=2, min_dist=0).fit(proj)

embed = mapper.embedding_

plt.scatter(embed[:, 0], embed[:, 1], s=0.25, c='blue', alpha=0.25)

umap.plot.connectivity(mapper)

umap.plot.diagnostic(mapper, diagnostic_type='pca')

# Calculate a density grid for the entire dataset
entire_data_density, x_grid, y_grid = t_sne.calc_density(embed)
grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])


#%%

###
# Perform HDBSCAN clustering to obtain labels and probabilities from embedded
# data; plot cluster labels atop embedded data
###

clusterobj = hdb_clustering.hdb_scan(embed, params['min_cluster_size'], 20, selection='leaf', cluster_selection_epsilon=params['cluster_selection_epsilon'])

labels = clusterobj.labels_

probabilities = clusterobj.probabilities_

color_palette = custom_colour_list()

fig, cluster_colors = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, color_palette)

#%%

###
# Calculate the mean length in frames of each label; store the first frame of
# each occurrence of each label
###

a_labels = np.reshape(labels, (-1, 1))

# Function to tally continuous label occurrences, allowing for a threshold
def calculate_label_data(arr):
    continuous_counts = {}
    current_val = None
    count = 0
    start_frame = 0

    arr = arr.flatten()  # Ensure the array is flattened for iteration
    # Generate original indices array
    original_indices = np.arange(len(arr))

    # Filter out -1 values
    noiseless_mask = arr != -1
    a_labels_noiseless = arr[noiseless_mask]
    original_indices_noiseless = original_indices[noiseless_mask]

    # Combine the original indices and labels into a single array
    a_labels_combined = np.column_stack((original_indices_noiseless, a_labels_noiseless))

    for idx, (original_idx, val) in enumerate(a_labels_combined):
        if val == current_val:
            count += 1
        else:
            if current_val is not None:
                finalise_count(continuous_counts, current_val, count, start_frame, a_labels_combined)
            current_val = val
            count = 1
            start_frame = idx  # Start index is adjusted due to filtered array

    # Finalise the last sequence if any valid count exists
    if current_val is not None and count > 0:
        finalise_count(continuous_counts, current_val, count, start_frame, a_labels_combined)

    return process_counts(continuous_counts), a_labels_combined

# Function to update or initialise label data in 'continuous_counts'
def finalise_count(continuous_counts, val, count, start_frame, combined_array):
    if val not in continuous_counts:
        continuous_counts[val] = {'counts': [], 'start_frames': []}
    continuous_counts[val]['counts'].append(count)
    continuous_counts[val]['start_frames'].append(combined_array[start_frame, 0])

# Function to compute statistics for each label based on its counts
def process_counts(continuous_counts):
    results = []
    for label, data in continuous_counts.items():
        mean_count = np.mean(data['counts'])
        total_occurrences = len(data['counts'])
        firsts = data['start_frames'][:5]  # The first five start frames
        lasts = data['start_frames'][-5:]  # The last five start frames, if applicable
        results.append((label, mean_count, total_occurrences, firsts, lasts))
    return results

# Usage
a_labels_data, a_labels_combined = calculate_label_data(a_labels)

#%%

###
# Calculate the distribution of labels within each trial and plot with and
# without noise
###

normalised_trial_counts = []
normalised_trial_counts_including_noise = []

for trial_name, slice_range in trial.items():
    slice_labels = a_labels[slice_range[0]:slice_range[1]+1]
    # For non-negative labels
    non_negative_labels = slice_labels[slice_labels >= 0]
    unique_non_neg, counts_non_neg = np.unique(non_negative_labels, return_counts=True)
    normalised_counts_non_neg = counts_non_neg / counts_non_neg.sum()
    trial_count_non_neg = dict(zip(unique_non_neg, normalised_counts_non_neg))
    normalised_trial_counts.append(trial_count_non_neg)
    
    # For including noise labels
    unique, counts = np.unique(slice_labels, return_counts=True)  # This time, keep -1 labels
    normalised_counts = counts / counts.sum()
    trial_count = dict(zip(unique, normalised_counts))
    normalised_trial_counts_including_noise.append(trial_count)

# Define the maximum label value with and without noise label
max_label_non_neg = max(max(trial.keys()) for trial in normalised_trial_counts)
max_label = max(max(trial.keys()) for trial in normalised_trial_counts_including_noise)

# Prepare plot data
plot_matrix_non_neg = np.zeros((len(normalised_trial_counts), max_label_non_neg + 1))
plot_matrix = np.zeros((len(normalised_trial_counts_including_noise), max_label + 2))  # +2 for indexing and noise

# Populate plot matrices
for i, trial_counts in enumerate(normalised_trial_counts):
    for label, normalised_count in trial_counts.items():
        plot_matrix_non_neg[i, label] = normalised_count
        
for i, trial_counts in enumerate(normalised_trial_counts_including_noise):
    for label, normalised_count in trial_counts.items():
        plot_matrix[i, label+1] = normalised_count  # +1 to offset for noise at index 0

# Plot excluding noise
fig, ax = plt.subplots(figsize=(20, 8))
trial_indices = np.arange(1, len(normalised_trial_counts) + 1)
for label in range(0, max_label_non_neg + 1):
    bottom = np.sum(plot_matrix_non_neg[:, :label], axis=1)
    color = color_palette[label] if label < len(color_palette) else (1, 1, 1)  # Fallback to white
    ax.bar(trial_indices, plot_matrix_non_neg[:, label], bottom=bottom, color=color, label=f'Label {label}')
ax.set_xticks(trial_indices)
ax.set_title('Normalised Distribution Excluding Noise')
ax.legend(title="Labels", loc="best", bbox_to_anchor=(1.0, 1.0))

# Plot including noise
fig, ax = plt.subplots(figsize=(20, 8))
trial_indices = np.arange(1, len(normalised_trial_counts_including_noise) + 1)
for label in range(-1, max_label + 1):
    if label == -1:
        # Directly use noise_color for label -1
        color = (0.5,0.5,0.5)
    else:
        # Use label's index directly for other labels
        color = color_palette[label] if label < len(color_palette) - 1 else (1, 1, 1)  # Exclude the last noise color
    # Calculate the correct bottom for stacking
    if label == -1:
        bottom = np.zeros(len(trial_indices))
    else:
        bottom = np.sum(plot_matrix[:, :label+1], axis=1)  # Include noise counts in bottom calculation
    # Plot bars
    ax.bar(trial_indices, plot_matrix[:, label+1], bottom=bottom, color=color, label=f'Label {label}')

ax.set_xticks(trial_indices)
ax.set_title('Normalisd Distribution Including Noise')
ax.legend(title="Labels", loc="best", bbox_to_anchor=(1.0, 1.0))

plt.tight_layout()
plt.show()

#%%

###
# Generate a CSV file to store label specific data
###

labels_info = {}
if day == 'D1':
    total_trials = 36
elif day == 'D2':
    total_trials = 72
else:
    raise ValueError("Invalid day specified. Please use 'D1' or 'D2'.")

# Initialise label info structure
for i, trial_counts in enumerate(normalised_trial_counts_including_noise):
    for label in trial_counts:
        if label >= 0:  # Check to skip the noise label
            if label not in labels_info:
                labels_info[label] = {
                    'key_press_count': 0,
                    'duration_in_ms': 0,
                    'normalised_hz': 0,
                    'overlap': 0,
                    'trials': [0] * total_trials  # 36 trials
                }
            # Mark presence of the label in this specific trial
            labels_info[label]['trials'][i] = 1 if label in trial_counts else 0

# Ensure the base path exists
base_path = f'./data/{day}/{pt}/'
os.makedirs(base_path, exist_ok=True)  # Create base path

# Create directories for each label
for label in labels_info:
    label_dir = os.path.join(base_path, f'Label_{label:02d}')
    os.makedirs(label_dir, exist_ok=True)  # Create a path for each label

# Define all the fieldnames including 'TnW' columns for normalised weights
fieldnames = ['label', 'key_press_count', 'duration_in_ms', 'normalised_hz', 'overlap']
for i in range(total_trials):
    fieldnames += [f'T{i+1}', f'T{i+1}w']

# Write to CSV with additional weight columns
filename = f'{base_path}{pt}_label_data.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for label, info in labels_info.items():
        if label >= 0:  # Ensure noise label is not written
            row_data = {
                'label': label,
                'key_press_count': info['key_press_count'],
                'duration_in_ms': info['duration_in_ms'],  # Include duration_in_ms
                'normalised_hz': info['normalised_hz'],
                'overlap': info['overlap'],
            }
            # Include trial presence and weight data
            for i in range(total_trials):
                presence_key = f'T{i+1}'
                weight_key = f'T{i+1}w'
                row_data[presence_key] = info['trials'][i]
                # Retrieve the normalised weight for this label in the current trial and convert to percentage
                row_data[weight_key] = normalised_trial_counts_including_noise[i].get(label, 0) * 100  # Convert to percentage
            writer.writerow(row_data)
#%%

###
# STOP HERE. Consult the a_labels_data list for information about each synergy;
# determine your synergy of interest; enter start and end frames of your
# synergy; timeseries and kinematics calculations are based on this range
###

syn_frame_start = 4912
syn_frame_end = syn_frame_start+112-28

fig3 = b_utils.plot_curr_cluster(embed, entire_data_density, syn_frame_start, x_grid, y_grid)

fig4, cluster = b_utils.plot_cluster_wav_mags(proj, labels, 8, features, frequencies, wave_correct=True, response_correct=True, mean_response=True, colour='lightblue')

#%%

###
# Calculate the time series of each channel, the frame of each keypress, the
# onset and offset of each keypress; calculate integrals of each channel
###

channels = ['Little', 'Ring', 'Middle', 'Index']
colors = sns.color_palette(["#A3FFD6","#7BC9FF","#8576FF","#1C1678"])
data = tracking_filtered[syn_frame_start:syn_frame_end, :] # Set data range

# Function to calculate the Euclidean distance between two multivariate data points
def euclidean_distance_multivariate(a, b):
    return np.linalg.norm(a - b)

# Function to find onset and offset around a peak within a window
def find_onset_offset(derivative, peak_index, window=15):
    onset_index = max(0, peak_index - window)
    offset_index = min(len(derivative) - 1, peak_index + window)
    return onset_index, offset_index

# Function to convert total peaks to frequency in Hz
def normalise_peaks_to_hz(total_peaks, frames_in_window, frame_rate=120):
    duration_seconds = frames_in_window / frame_rate
    frequency_hz = total_peaks / duration_seconds
    duration_in_ms = duration_seconds * 1000  # Convert seconds to milliseconds
    return frequency_hz, duration_in_ms

# Function to slice, normalise, and invert y values
def process_y_values(data):
    y_values = np.vstack((data[:, 1], data[:, 3], data[:, 5], data[:, 7])).T
    y_values_positive = np.abs(y_values)
    return y_values, y_values_positive

# Function to plot time series and detect key events
def time_series_events(ax, x_range, y_values, channels, colors, y_threshold=params['y_threshold'], plot=True):
    total_peaks_across_channels = 0
    onset_offset_data = {channel: [] for channel in channels}
    area_under_curve = []
    peak_indices = []

    for i, channel_name in enumerate(channels):
        derivative = np.diff(y_values[:, i], prepend=y_values[0, i])
        peaks, _ = find_peaks(y_values[:, i], height=y_threshold, prominence = 10)
        total_peaks_across_channels += len(peaks)
        for peak in peaks:
            peak_indices.append((peak, channel_name))
        peak_indices.sort(key=lambda x: x[0])

    for peak_idx, (peak, channel_name) in enumerate(peak_indices):
        onset, offset = find_onset_offset(derivative, peak)
        # Directly store onset, peak, and offset data without calculating an index
        onset_offset_data[channel_name].append({'onset': onset, 'peak': peak, 'offset': offset})
            
    if plot:
        for i, channel_name in enumerate(channels):
            ax.plot(x_range, y_values[:, i], label=channel_name, color=colors[i])
            for event_data in onset_offset_data[channel_name]:
                peak = event_data['peak']
                ax.plot(x_range[peak], y_values[peak, i], 'r*')
                ax.plot(x_range[event_data['onset']], y_values[event_data['onset'], i], 'go')
                ax.plot(x_range[event_data['offset']], y_values[event_data['offset'], i], 'mo')

        # Calculate area under curve using simpson's rule for each channel and append to list
        area_under_curve = [simpson(y_values_positive[:, i]) for i in range(y_values_positive.shape[1])]

    if plot:
        ax.legend()

    return total_peaks_across_channels, onset_offset_data, area_under_curve, y_threshold

# Function to identify segments within a full dataset that are similar to a target time series using Dynamic Time Warping (DTW)
def find_similar_series(y_values, y_values_full, syn_frame_start, syn_frame_end, num_segments=4, dist_threshold=None):
    dtw_distances = []
    num_channels = y_values.shape[1]
    target_length = y_values.shape[0]
    selected_ranges = [(syn_frame_start, syn_frame_end)]

    for i in range(len(y_values_full) - target_length + 1):
        if any(start <= i <= end or start <= i + target_length - 1 <= end for start, end in selected_ranges):
            continue

        total_distance = 0
        comparison_segment = y_values_full[i:i + target_length]

        for channel in range(num_channels):
            target_channel_data = np.array(y_values[:, channel]).flatten()
            comparison_channel_data = np.array(comparison_segment[:, channel]).flatten()
            
            # Now, pass the 1-D arrays to fastdtw
            distance, _ = fastdtw(target_channel_data, comparison_channel_data, dist=euclidean_distance_multivariate)
            total_distance += distance

        avg_distance = total_distance / num_channels
        if dist_threshold is None or avg_distance < dist_threshold:
            dtw_distances.append((i, avg_distance, i + target_length - 1))
    
    dtw_distances = sorted(dtw_distances, key=lambda x: x[1])[1:]
    similar_segment_indices = []
    for idx, _, end_idx in dtw_distances:
        if not any(start <= idx <= end or start <= end_idx <= end for start, end in selected_ranges):
            similar_segment_indices.append(idx)
            selected_ranges.append((idx, end_idx))
            if len(similar_segment_indices) >= num_segments:
                break
    
    return similar_segment_indices

# Function to calculate the mean of identified occurrences with CI
def mean_with_confidence_intervals(ax, data, colors, channel_names):

    for i, channel in enumerate(channel_names):
        mean_series = np.mean(data[:, :, i], axis=0)
        ci = sem(data[:, :, i], axis=0) * t.ppf((1 + 0.95) / 2., data.shape[0] - 1)
        ax.plot(mean_series, label=channel, color=colors[i])
        ax.fill_between(range(len(mean_series)), mean_series - ci, mean_series + ci, color=colors[i], alpha=0.2)
    ax.legend()
    ax.set_title("Mean of Time Series with Confidence Intervals")

# Function to store all identified occurrnces and plot the mean of each channel
def plot_similar_series(y_values_full, y_values, similar_segment_indices, channels, colors):

    # Data container for calculating mean and confidence interval
    all_data = np.empty((len(similar_segment_indices) + 1, y_values.shape[0], len(channels)))

    # Populate the first entry with the target time series
    all_data[0, :, :] = y_values

    # Populate subsequent entries with similar time series data
    for idx, segment_idx in enumerate(similar_segment_indices):
        start = segment_idx
        end = start + y_values.shape[0]
        similar_segment = y_values_full[start:end, :]
        all_data[idx + 1, :, :] = similar_segment

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot the mean of the target and similar time series segments with CI
    mean_with_confidence_intervals(ax, all_data, colors, channels)

    plt.tight_layout()
    plt.show()

# Process and plot time series for synergy of interest
y_values_syn, y_values_positive = process_y_values(data)
x_range = np.arange(syn_frame_start, syn_frame_end)

fig, ax = plt.subplots(figsize=(14, 6))

total_peaks_across_channels, onset_offset_data, area_under_curve, y_threshold = time_series_events(ax, x_range, y_values_syn, channels, colors)
normalised_frequency_hz, duration_in_ms = normalise_peaks_to_hz(total_peaks_across_channels, syn_frame_end - syn_frame_start + 1)

ax.set_title('Time Series of Channels with Detected Keypresses')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Y Position')
ax.text(0.025, 0.05, f"Normalised Frequency: {normalised_frequency_hz:.3f} Hz", transform=ax.transAxes, color='red')
ax.legend()
plt.show()

# Plot normal and non-negative curves
fig2, ax2 = plt.subplots(figsize=(14, 6))
for i, channel_name in enumerate(channels):
    ax2.plot(x_range, y_values_positive[:, i], label=channel_name, color=colors[i])
ax2.set_title('Normalised and Non-negative Curves')
ax2.set_xlabel('Frame Index')
ax2.set_ylabel('Adjusted Y Position')
ax2.legend()
plt.show()

# Plot integrals for each channel
plt.figure(figsize=(6, 4))
bottom_val = 0 # Starting point for stacking
handles = []
for i, (channel, area) in enumerate(zip(channels, area_under_curve)):
    bar = plt.bar('Total Area', area, bottom=bottom_val, color=colors[i], label=channel)
    handles.append(bar[0])  # Store the handle of the bar
    bottom_val += area

plt.title('Stacked Area Contribution of Each Channel')
plt.ylabel('Area Under Curve')
plt.legend(handles[::-1], channels[::-1]) # Invert legend handles for clarity
plt.show()

# Process and plot time series for additional occurrences
y_values_full, *_ = process_y_values(tracking_filtered)

#%%

###
# Calculate and plot the mean and 95% CI over all occurrences of the synergy of
# interest (fast Dynamic Time Warping (fDTW))
###

similar_segment_indices = find_similar_series(y_values_syn, y_values_full,
            syn_frame_start, syn_frame_end, num_segments=10, dist_threshold=1000)

plot_similar_series(y_values_full, y_values_syn, similar_segment_indices, 
            channels, colors)

#%%

###
# Calculate and plot average velocity and acceleration for each channel;
# calculate and plot overlap of movements over all channels
###

# Initialise empty lists to store values
average_velocities = []
std_devs_velocity = []

average_accelerations = []
std_devs_acceleration = []

def calculate_overlaps(onset_offset_data):
    total_overlap_frames = 0

    # Convert onset_offset_data to a flat list of dictionaries with added channel information
    all_events = []
    for channel, events in onset_offset_data.items():
        for event in events:
            # Append event with channel information
            all_events.append({**event, 'channel': channel})

    # Sort all events by time of the event ('onset' and 'offset') to maintain the sequence of events as they occurred
    all_events_sorted = sorted(all_events, key=lambda x: x['onset'])

    # Iterate through sorted events to find overlaps between offsets and subsequent onsets across channels
    for i in range(len(all_events_sorted) - 1):
        current_event = all_events_sorted[i]
        next_event = all_events_sorted[i + 1]
        
        # Ensure we are comparing an offset to a subsequent onset and they are from different channels
        if 'offset' in current_event and 'onset' in next_event and current_event['channel'] != next_event['channel']:
            # Calculate potential overlap
            overlap = current_event['offset'] - next_event['onset']
            
            # If the calculated value is positive, it indicates an overlap
            if overlap > 0:
                total_overlap_frames += overlap

    return total_overlap_frames, all_events_sorted

total_peaks, onset_offset_data, area_under_curve, y_threshold = time_series_events(ax, x_range, y_values_syn, channels, colors, plot=False)  # plot=True if you want to visualise
total_overlap_frames, all_events_sorted = calculate_overlaps(onset_offset_data)
percent_overlap = total_overlap_frames / len(y_values_syn) * 100

# Calculate magnitude of velocities and accelerations
for i in range(y_values_syn.shape[1]):
    velocities = np.abs(np.diff(y_values_syn[:, i]))
    accelerations = np.abs(np.diff(velocities))
    
    average_velocities.append(np.mean(velocities))
    std_devs_velocity.append(np.std(velocities))
    
    average_accelerations.append(np.mean(accelerations))
    std_devs_acceleration.append(np.std(accelerations))

# Velocity plot data
df_velocity = pd.DataFrame({
    'Channel': channels,
    'Average Velocity': average_velocities,
    'STD Velocity': std_devs_velocity
})

# Acceleration plot data
df_acceleration = pd.DataFrame({
    'Channel': channels,
    'Average Acceleration': average_accelerations,
    'STD Acceleration': std_devs_acceleration
})

# Plot velocities
plt.figure(figsize=(12, 6))
sns.barplot(x='Channel', y='Average Velocity', data=df_velocity, palette='Blues')
plt.errorbar(x=np.arange(len(channels)), y=df_velocity['Average Velocity'], yerr=df_velocity['STD Velocity'], fmt='none', c='darkblue', capsize=5)
plt.title('Average Velocities for Each Digit')
plt.ylabel('Velocity (pixels/s)')
plt.xlabel('Channel')
plt.show()

# Plot accelerations
plt.figure(figsize=(12, 6))
sns.barplot(x='Channel', y='Average Acceleration', data=df_acceleration, palette='Greens')
plt.errorbar(x=np.arange(len(channels)), y=df_acceleration['Average Acceleration'], yerr=df_acceleration['STD Acceleration'], fmt='none', c='darkgreen', capsize=5)
plt.title('Average Accelerations for Each Digit')
plt.ylabel('Acceleration')
plt.xlabel('Channel')
plt.show()

# Plot percent of frames detected as overlap
plt.figure(figsize=(6, 4))
sns.barplot(x=['Total Overlap'], y=[percent_overlap], palette='Purples')
plt.title('Total Percent Overlap Among Channels')
plt.ylabel('Percent Overlap')
plt.ylim(0, 100)  # Assuming percent values, adjust if necessary
plt.show()
# Print the result
print(f"Total percent overlap among channels: {percent_overlap:.2f}%")

#%%

###
# Update the participant specific CSV file with metrics for the analysed label
###

def update_csv_data(pt, cluster, kp, dm, hz, ol):
    # Path to the CSV file
    filename = f'./data/{day}/{pt}/{pt}_label_data.csv'
    
    # Read the existing CSV into a DataFrame
    df = pd.read_csv(filename)
    
    # Check if the cluster exists in the DataFrame
    if cluster in df['label'].values:
        # Update the specific row
        df.loc[df['label'] == cluster, 'key_press_count'] = total_peaks_across_channels
        df.loc[df['label'] == cluster, 'duration_in_ms'] = duration_in_ms
        df.loc[df['label'] == cluster, 'normalised_hz'] = normalised_frequency_hz
        df.loc[df['label'] == cluster, 'overlap'] = percent_overlap
    else:
        # Optionally handle the case where the label does not exist
        print(f"Label {cluster} not found in the CSV.")
    
    # Write the DataFrame back to the CSV
    df.to_csv(filename, index=False)
    print(f"Data for label {cluster} updated successfully.")

# Sample function calls with hypothetical values
update_csv_data(pt=pt, cluster=cluster, kp=total_peaks_across_channels, dm = duration_in_ms, hz=normalised_frequency_hz, ol=percent_overlap)

#%%

###
# Extract metrics for each label from participant specific CSVs; calculate
# weighted averages of each metric across participants and trials
###

def process_and_plot_data(participant_ids, num_trials):
    metrics = ['key_press_count', 'duration_in_ms', 'normalised_hz', 'overlap']
    trial_data = {metric: [[] for _ in range(num_trials)] for metric in metrics}
    trial_weights = {metric: [[] for _ in range(num_trials)] for metric in metrics}
    noise_data = [[] for _ in range(num_trials)]  # For storing noise percentages

    for pt in participant_ids:
        csv_file = f'./data/{day}/{pt}/{pt}_label_data.csv'
        if os.path.exists(csv_file):
            try:
                data = pd.read_csv(csv_file)
                for i in range(num_trials):
                    tn_col = f'T{i+1}w'
                    if tn_col in data.columns:
                        weights = data[tn_col].apply(round).tolist()
                        total_weight = sum(weights)
                        noise = 100 - total_weight  # Calculate noise as the remainder to 100
                        noise_data[i].append(noise)
                        for metric in metrics:
                            values = data[metric].tolist()
                            if weights:
                                weighted_mean = np.average(values, weights=weights)
                                weighted_std = np.sqrt(np.average((values - weighted_mean) ** 2, weights=weights))
                                trial_data[metric][i].append(weighted_mean)
                                trial_weights[metric][i].append(weighted_std)
            except Exception as e:
                print(f"Error processing file {csv_file}: {e}")
                continue

    # Plotting metrics with 2x2 subplot layout
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12), dpi=300)
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        means = [np.mean(values) if values else None for values in trial_data[metric]]
        stds = [np.mean(stds) if stds else None for stds in trial_weights[metric]]
        x_values = list(range(1, num_trials + 1))
        means = [v for v in means if v is not None]
        stds = [s for s in stds if s is not None]

        if means:
            scatter = ax.scatter(x_values, means, c=means, cmap='viridis', label=f'Weighted average for {metric}', s=100)
            ax.plot(x_values, means, color='gray', alpha=0.5)
            z_score = norm.ppf(0.975)
            ci = z_score * np.array(stds) / np.sqrt(len(means))
            ax.fill_between(x_values, np.array(means) - ci, np.array(means) + ci, color='gray', alpha=0.1)
            plt.colorbar(scatter, ax=ax, label='Metric Value')

        ax.set_title(f'Weighted Averages for {metric}')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Weighted Average Value')
        ax.set_xticks(x_values)
        ax.set_xticklabels([f'{i}' for i in range(1, num_trials + 1)])
        ax.legend(loc=2)

    plt.tight_layout()
    plt.show()

    # Plotting noise data in a separate figure
    noise_means = [np.mean(n) / 100 for n in noise_data]  # Convert sums to percentages
    noise_stds = [np.std(n) / 100 for n in noise_data]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.errorbar(x_values, noise_means, yerr=noise_stds, fmt='-o', label='Average Noise with 95% CI')
    ax.set_title('Average Noise per Trial')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Noise (%)')
    ax.set_ylim(0, 1)
    ax.legend()
    plt.show()

    # New figure with 1x3 subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), dpi=300)
    x_data = [np.mean(values) if values else None for values in trial_data['key_press_count']]
    x_data = [v for v in x_data if v is not None]

    for i, y_metric in enumerate(['duration_in_ms', 'normalised_hz', 'overlap']):
        ax = axes[i]
        y_data = [np.mean(values) if values else None for values in trial_data[y_metric]]
        y_data = [v for v in y_data if v is not None]

        if x_data and y_data:
            ax.scatter(x_data, y_data, c='blue', label=f'{y_metric} vs key_press_count', s=100)
            slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
            ax.plot(x_data, np.array(x_data) * slope + intercept, color='red', label='Line of best fit')
        
        ax.set_title(f'{y_metric} vs key_press_count')
        ax.set_xlabel('Weighted Average of key_press_count')
        ax.set_ylabel(f'Weighted Average of {y_metric}')
        ax.legend()

    plt.tight_layout()
    plt.show()

process_and_plot_data(participant_ids=[''], num_trials=12)

#%%

###
# Extract label distributions from each participant .spydata file, perform JS
# Divergence calculations between and within trials, permute results to derive
# an empirical p value for each pair, calculate a combined p value with
# Stouffer's method
###

# Directory containing the .spydata files; please adapt to your system
directory = f'/Users/wi11iamk/Documents/GitHub/HUB_DT/data/{day}spydata'

# Function to load a .spydata file and return the variables as a dictionary.
def load_spydata(file_path):
    try:
        data, error = io.load_dictionary(file_path)
        if error:
            raise ValueError(f"Error loading {file_path}: {error}")
        return data
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

# Function to clear all variables from the global workspace except those in vars_to_keep.
def clear_workspace(vars_to_keep=[]):
    global_vars = list(globals().keys())
    for var in global_vars:
        if var not in vars_to_keep and not var.startswith('_'):
            del globals()[var]

# Function to compare cluster memberships between two time points, excluding noise label (-1).
def compare_clusters(cluster_labels1, cluster_labels2, all_labels):
    counts1 = np.array([np.sum(cluster_labels1 == label) for label in all_labels if label != -1])
    counts2 = np.array([np.sum(cluster_labels2 == label) for label in all_labels if label != -1])
    return counts1, counts2

# Function to process each .spydata file in the given directory.
def process_directory(directory):
    a_labels_vars = []

    # Function to process a .spydata file.
    def process_file(file_path, file_name):
        data = load_spydata(file_path)
        if data is not None and 'a_labels' in data:
            a_labels_key = f'a_labels_{file_name}'
            globals()[a_labels_key] = data['a_labels'].flatten()  # Ensure it's 1D
            a_labels_vars.append((a_labels_key, file_name))
            print(f"Loaded and renamed 'a_labels' to '{a_labels_key}'")
            vars_to_keep = ['os', 'np', 'pd', 'plt', 'sns', 'jensenshannon', 
                            'norm', 'ttest_rel', 'io', 'load_spydata', 
                            'clear_workspace', 'compare_clusters', 'process_directory', 
                            'a_labels_vars', 'process_file', 'convert_to_proportions', 
                            'permutation_test_jsd', 'combine_pvalues', 'day', 
                            'trial', 'micro'] + [v[0] for v in a_labels_vars]
            clear_workspace(vars_to_keep=vars_to_keep)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".spydata") and os.path.isfile(file_path):
            process_file(file_path, filename.split('.')[0])
    
    return a_labels_vars

# Function to determine significance level
def determine_significance(p_value):
    if p_value < 0.001:
        return "p < 0.001"
    elif p_value < 0.01:
        return "p < 0.01"
    elif p_value < 0.05:
        return "p < 0.05"
    else:
        return f"p = {p_value:.3e}"

# Process the directory and retrieve all saved 'a_labels' variables
a_labels_vars = process_directory(directory)

# Collect all permutation test p-values from all participants
all_p_values = []

# Function to convert counts to proportions (probability distributions)
def convert_to_proportions(counts, epsilon=1e-10):
    total_counts = counts.sum()
    proportions = (counts + epsilon) / (total_counts + epsilon * len(counts))
    proportions /= proportions.sum()  # Ensure proportions sum to 1
    return proportions

# Function to perform permutation test to determine the significance of the observed JSD
def permutation_test_jsd(proportions1, proportions2, observed_jsd, n_permutations=100000):
    combined_counts = np.concatenate((proportions1, proportions2))
    n = len(proportions1)
    perm_jsds = []

    for _ in range(n_permutations):
        np.random.shuffle(combined_counts)
        perm_proportions1 = combined_counts[:n]
        perm_proportions2 = combined_counts[n:]
        perm_jsd = jensenshannon(perm_proportions1, perm_proportions2)
        perm_jsds.append(perm_jsd)

    perm_jsds = np.array(perm_jsds)
    p_value = np.sum(perm_jsds >= observed_jsd) / n_permutations
    return observed_jsd, p_value

# Data to be stored for each participant
trial_results = []
micro_results = []

# Process each 'a_labels' variable for slicing and clustering analysis
for a_labels_key, participant_id in a_labels_vars:
    a_labels = globals()[a_labels_key]

    # Get all possible unique labels from the entire a_labels array, excluding noise label (-1)
    all_labels = np.unique(a_labels)
    all_labels = all_labels[all_labels != -1]  # Exclude noise label (-1)

    # Calculate the mean JSD values for online and offline periods
    if day == 'D1':
        online_jsd = []
        offline_jsd = []

        for i in range(1, 13):
            # Define micro-segments for online and offline JSD calculations
            m1_key = f"{i}_m1"
            m2_key = f"{i}_m2"
            if i < 12:
                next_m1_key = f"{i + 1}_m1"
            
            # Online JSD (m1 vs m2 for the same trial)
            if m1_key in micro and m2_key in micro:
                counts1, counts2 = compare_clusters(a_labels[micro[m1_key][0]:micro[m1_key][1]+1], a_labels[micro[m2_key][0]:micro[m2_key][1]+1], all_labels)
                proportions1 = convert_to_proportions(counts1)
                proportions2 = convert_to_proportions(counts2)
                jsd_online = jensenshannon(proportions1, proportions2)
                online_jsd.append(jsd_online)

            # Offline JSD (m2 vs next m1 between trials)
            if i < 12 and m2_key in micro and next_m1_key in micro:
                counts1, counts2 = compare_clusters(a_labels[micro[m2_key][0]:micro[m2_key][1]+1], a_labels[micro[next_m1_key][0]:micro[next_m1_key][1]+1], all_labels)
                proportions1 = convert_to_proportions(counts1)
                proportions2 = convert_to_proportions(counts2)
                jsd_offline = jensenshannon(proportions1, proportions2)
                offline_jsd.append(jsd_offline)

        mean_online_jsd = (np.mean(online_jsd))
        mean_offline_jsd = (np.mean(offline_jsd))

        # Store the micro_results
        micro_results.append({
            'Participant ID': participant_id,
            'Mean Online JSD': round(mean_online_jsd, 4),
            'Mean Offline JSD': round(mean_offline_jsd, 4)
        })

    # Define slices
    slices = [trial['36'], trial['39']]  # <-- Set trials for comparison here
    
    # Initialise container for slice data segments
    cluster_labels_segments = []

    # Use the existing HDBSCAN labels
    for sl in slices:
        if sl:
            slice_labels = a_labels[sl[0]:sl[1]]
            cluster_labels_segments.append(slice_labels)

    # Compare clusters between slices
    counts1, counts2 = compare_clusters(cluster_labels_segments[0], cluster_labels_segments[1], all_labels)

    # Convert counts to proportions
    proportions1 = convert_to_proportions(counts1)
    proportions2 = convert_to_proportions(counts2)

    # Calculate the observed Jensen-Shannon Divergence between each pair of distributions
    observed_jsd = jensenshannon(proportions1, proportions2)

    # Perform permutation test
    _, p_value = permutation_test_jsd(proportions1, proportions2, observed_jsd, n_permutations=100000)

    # Calculate z-score from the two-tailed p-value
    z_score = norm.ppf(1 - p_value / 2)

    # Store the trial_results
    trial_results.append({
        'Participant ID': participant_id,
        'Observed JSD': round(observed_jsd, 4),
        'Permutation P-value': round(p_value, 4),
        'Z-score': round(z_score, 4)
    })

    # Print the trial_results for each participant
    print(f"Participant ID: {participant_id}, Observed JSD: {observed_jsd}, Permutation Test P-value: {p_value}, Z-score: {z_score}")

    # Store the p-value for combining
    all_p_values.append(p_value)

# Combine p-values using Stouffer's method
def combine_pvalues(p_values):
    z_scores = np.array([norm.ppf(1 - p / 2) for p in p_values])  # Convert two-tailed p-values
    combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
    combined_p = 2 * (1 - norm.cdf(np.abs(combined_z)))  # Convert back to a two-tailed p-value
    return combined_p

combined_p_value = combine_pvalues(all_p_values)

# Determine significance level for combined p-value
significance_combined = determine_significance(combined_p_value)

# Print the final combined p-value and significance
print(f"Combined P-value across all participants: {combined_p_value:.3e}")
print(f"Significance: {significance_combined}")

# Create a DataFrame from the trial_results
trial_results_df = pd.DataFrame(trial_results)

# Plot the trial_results table as a figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# Add title to the table
title = f"Comparison of trial slices: {slices[0]} vs {slices[1]}"
fig.suptitle(title, fontsize=14)

# Add the table to the figure
table_data = [trial_results_df.columns.values.tolist()] + trial_results_df.values.tolist()
tbl = ax.table(cellText=table_data, cellLoc='center', loc='center')

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.2)

# Add combined p-value to the table
tbl.add_cell(len(table_data), len(trial_results_df.columns) - 1, width=0.25, height=0.066, text=f"Stouffer's P: {combined_p_value:.3e}", loc='center')

plt.show()

# Calculate and plot mean JSD values for online and offline periods
if day == 'D1' and micro_results:
    micro_results_df = pd.DataFrame(micro_results)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=micro_results_df.melt(id_vars='Participant ID', var_name='Period', value_name='JSD'),
                x='Period', y='JSD', errorbar='sd')
    sns.stripplot(data=micro_results_df.melt(id_vars='Participant ID', var_name='Period', value_name='JSD'),
                  x='Period', y='JSD', color='black', alpha=0.5)

    # Perform paired t-test
    ttest_pvalue = ttest_rel(micro_results_df['Mean Online JSD'], micro_results_df['Mean Offline JSD']).pvalue

    # Determine significance level for paired t-test p-value
    significance_ttest = determine_significance(ttest_pvalue)
        
    plt.title("Mean Online and Offline JSD Values")
    plt.xlabel(f"Paired t-test p-value: {significance_ttest}")
    plt.show()

    print(f"Paired t-test p-value for online vs. offline JSD: {ttest_pvalue:.3e}")
#%%

# TODO TBD
