#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:55:56 2024

@author: wi11iamk
"""
#%%

###
# Import necessary libraries
###

import numpy as np, pandas as pd, umap, umap.plot
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from hubdt import data_loading, behav_session_params, wavelets, t_sne, hdb_clustering, b_utils
from scipy.stats import gaussian_kde, ks_2samp, sem, t
from scipy.integrate import simpson
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import jensenshannon
from fastdtw import fastdtw

#%%

###
# Initialise the HUB-DT session; import all DLC features and tracking data;
# apply Savitzky-Golay filter to each y time series; initialise dictionary to
# store specific y_threshold values for peak detection for each participant;
# initialise dictionary to store frame lengths of each trial 
###

mysesh = behav_session_params.load_session_params ('Mine')

features = data_loading.dlc_get_feats (mysesh)

del features[4:7] # Delete features

tracking = data_loading.load_tracking (mysesh, dlc=True, feats=features)

# Create variables for and apply Savitzky-Golay filter
tracking_filtered = tracking.copy() # Create copy of tracking to filter
window_length = 9  # Choose an odd number for the window size
polyorder = 1  # Polynomial order
for col in range(1, 8, 2):  # Iterate over y-value columns (1, 3, 5, 7)
    tracking_filtered[:, col] = savgol_filter(tracking[:, col], window_length, polyorder)
    
# Dictoionary to store y_threshold values specific to each participant
y_threshold_values = {
    '012': 330,
    '027': 350,
    '049': 400,
    # Add more participants as needed
}

participant_number = '012' # Enter participant number from HUB-DT session here
y_threshold = y_threshold_values.get(participant_number)

# Initialise a dictionary to store frame lengths for each trial
trial = {}

# Total number of trials
total_trials = 36

# Populate the dictionary with slices
for i in range(total_trials):
    key = f"{i+1}"  # Key name (e.g., '1' for trial 1)
    value = (i*1200, (i+1)*1200-1)  # Range for each slice
    trial[key] = value

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
# Fit wavelet projection into two dimensional embedded space (UMAP); plot
###

mapper = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0.2).fit(proj)

embed = mapper.embedding_

plt.scatter(embed[:, 0], embed[:, 1], s=0.25, c='blue', alpha=0.25)

umap.plot.connectivity(mapper)

umap.plot.diagnostic(mapper, diagnostic_type='pca')

local_dims = umap.plot.diagnostic(mapper, diagnostic_type='local_dim')

            
#%%

###
# Calculate and plot a gaussian KDE over embedded data; calculate a list of
# slices from the embedding and plot each as an overlay atop the gaussian KDE;
# calculate the Jensen-Shannon divergence and Kolmogorov-Smirnov statistic for
# each slice as a probability vector or data sample, respectively
###

# Function to calculate density on the fixed grid
def calc_density_on_fixed_grid(data, grid_coords):
    kde = gaussian_kde(data.T)
    density = kde(grid_coords).reshape(x_grid.shape)
    return density

# Function to normalise and flatten a grid to form a probability vector
def normalise_grid(grid):
    flattened = grid.flatten()
    normalised = flattened / np.sum(flattened)
    return normalised

# Calculate a density grid for the entire dataset
entire_data_density, x_grid, y_grid = t_sne.calc_density(embed)
grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])

# Define slices (you may use an empty tuple for the entire dataset)
slices = [trial['1'],trial['5'],trial['12'],trial['35']]

# Initialise container for normalised probability vectors and slice data segments
probability_vectors = []
slice_data_segments = []

# Prepare the plotting area based on the number of slices with actual data
fig, axes = plt.subplots(1, max(len(slices), 1), figsize=((len(slices)*6), 4)) if len(slices) > 1 else plt.subplots(figsize=(6, 4))
axes = np.atleast_1d(axes)  # Ensure axes is always iterable

# Iterate through slices to calculate densities and append to lists
for i, sl in enumerate(slices):
    ax = axes[i] if len(slices) > 1 else axes
    # Check if there's a slice defined
    pcm = ax.pcolormesh(x_grid, y_grid, entire_data_density, shading='auto', cmap='plasma')
    ax.set_title('Entire Dataset Density' if not sl else f'Slice {i+1}: {sl[0]}-{sl[1]}')
    fig.colorbar(pcm, ax=ax, label='Density')
    if sl:
        slice_data = embed[sl[0]:sl[1], :]
        slice_data_segments.append(slice_data)  # Append the slice data for KS tests
        slice_density = calc_density_on_fixed_grid(slice_data, grid_coords)
        slice_vector = normalise_grid(slice_density)
        probability_vectors.append(slice_vector)  # Append the normalised probability vector
        
        # Scatter plot for the slice data
        ax.scatter(slice_data[:, 0], slice_data[:, 1], color='deepskyblue', s=0.75, alpha=0.33, label=f'Slice {i+1} Data Points')
        ax.set_title(f'Slice {i+1}: {sl[0]}-{sl[1]}')
    else:
        # For an empty slice definition, skip or handle differently
        continue
    ax.legend()

plt.tight_layout()
plt.show()

# Ensure probability vectors sum to 1 within tolerance
for i, vector in enumerate(probability_vectors):
    if np.isclose(np.sum(vector), 1.0, atol=1e-8):
        print(f"Vector {i+1} sums to 1.0 within tolerance.")
    else:
        print(f"Vector {i+1} does not sum to 1.0; sum is {np.sum(vector)}.")
        
# Calculate pairwise JS divergences between listed probability_vectors
for i in range(len(probability_vectors) - 1):
    js_divergence = jensenshannon(probability_vectors[i], probability_vectors[i + 1])
    print(f"Jansen-Shannon Divergence between Vector {i+1} and Vector {i + 2}: {js_divergence}")

# Calculate pairwise KS statistics between listed slice_data_segments
for i in range(len(slice_data_segments) - 1):
    # Perform KS test between consecutive slice_data segments
    ks_stat, p_value = ks_2samp(slice_data_segments[i].ravel(), slice_data_segments[i + 1].ravel())
    
    # Determine whether to reject the null hypothesis
    if p_value <= 0.05:
        decision = "reject the null hypothesis (the distributions are different)"
    else:
        decision = "fail to reject the null hypothesis (no significant difference between the distributions)"
    
    # Print the results, including the decision
    print(f"KS Statistic between Slice {i+1} and Slice {i+2}: {ks_stat}, P-value: {p_value}. We {decision}.")

#%%

###
# Perform HDBSCAN clustering to obtain labels and probabilities from embedded
# data; plot cluster distance tree; plot cluster labels atop embedded data
###

clusterobj = hdb_clustering.hdb_scan(embed, 200, 20, selection='leaf', cluster_selection_epsilon=0.2)

labels = clusterobj.labels_

probabilities = clusterobj.probabilities_

n_colors = 20  # Number of colorus for clusters
color_palette = sns.color_palette("Paired", n_colors)
noise_color = (0.75,0.75,0.75)  # A distinct colour for noise
#noise_color = (1,1,1)
color_palette.append(noise_color)  # Append noise color at the end

fig, cluster_colors = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, color_palette, noise=False)

#%%

### 
# Calcuate the mean length in frames of each label; store the first frame of
# the first five occurrences of each label in a list of tuples; calculate total
# uses and percent of time in use for each label
###

a_labels = np.reshape(labels, (-1, 1))

# Function to tally continuous label occurrences, allowing for a threshold
def calculate_label_data(arr, threshold=0):
    continuous_counts = {}
    current_val = None
    count = 0
    start_frame = 0
    minus_one_counter = 0  # Counter for -1 values
    
    arr = arr.flatten()  # Ensure the array is flattened for iteration
    
    for idx, val in enumerate(arr):
        if val != -1:
            if val == current_val or minus_one_counter <= threshold and current_val is not None:
                if minus_one_counter <= threshold and current_val is not None:
                    count += minus_one_counter  # Include -1s within threshold in the count
                count += 1
                minus_one_counter = 0  # Reset the -1 counter
            else:
                if current_val is not None:
                    finalise_count(continuous_counts, current_val, count, start_frame)
                current_val = val
                count = 1
                start_frame = idx
        else:
            if minus_one_counter < threshold:
                minus_one_counter += 1  # Tolerate -1 within the threshold
            else:
                if current_val is not None:
                    finalise_count(continuous_counts, current_val, count, start_frame)
                    current_val = None
                    count = 0
                minus_one_counter = 0  # Reset -1 counter
    
    # Finalise the last sequence
    if current_val is not None and count > 0:
        finalise_count(continuous_counts, current_val, count, start_frame)
    
    process_counts(continuous_counts)
    
    # Prepare sorted results
    sorted_results = sorted([(k,) + v['stats'] for k, v in continuous_counts.items()], key=lambda x: x[0])
    
    return sorted_results

# Function to update or initialise label data in 'continuous_counts'
def finalise_count(continuous_counts, val, count, start_frame):
    if val not in continuous_counts:
        continuous_counts[val] = {'counts': [], 'start_frames': []}
    continuous_counts[val]['counts'].append(count)
    continuous_counts[val]['start_frames'].append(start_frame)

# Function to compute statistics for each label based on its counts
def process_counts(continuous_counts):
    total_all_frames = sum(sum(c for c in v['counts']) for v in continuous_counts.values())
    for val, data in continuous_counts.items():
        counts = data['counts']
        start_frames = data['start_frames']
        mean_count = np.mean(counts)
        first_one = start_frames[:1]

        data['stats'] = (
            mean_count,  # Mean length in frames
            *first_one,  # The first frame of the first occurrence
            len(counts),  # Total number of occurrences
            (sum(counts) / total_all_frames) * 100  # % of total frames
        )

a_labels_data = calculate_label_data(a_labels, threshold=5)

#%%

###
# FOR TRIALS: Calculate the distribution of labels within each trial, one plot
# with, and one without, noise
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
fig, ax = plt.subplots(figsize=(14, 8))
trial_indices = np.arange(1, len(normalised_trial_counts) + 1)
for label in range(0, max_label_non_neg + 1):
    bottom = np.sum(plot_matrix_non_neg[:, :label], axis=1)
    color = color_palette[label] if label < len(color_palette) else (1, 1, 1)  # Fallback to white
    ax.bar(trial_indices, plot_matrix_non_neg[:, label], bottom=bottom, color=color, label=f'Label {label}')
ax.set_xticks(trial_indices)
ax.set_title('Normalised Distribution Excluding Noise')
ax.legend(title="Labels", loc="best", bbox_to_anchor=(1.0, 1.0))

# Plot including noise
fig, ax = plt.subplots(figsize=(14, 8))
trial_indices = np.arange(1, len(normalised_trial_counts_including_noise) + 1)
for label in range(-1, max_label + 1):
    if label == -1:
        # Directly use noise_color for label -1
        color = noise_color
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
# Consult the a_labels_data list for information about each synergy; determine
# your synergy of interest; enter start and end frames of the synergy; 
# timeseries calculations are based on this range
###

syn_frame_start = 1220
syn_frame_end = syn_frame_start+240

fig3 = b_utils.plot_curr_cluster(embed, entire_data_density, syn_frame_start, x_grid, y_grid)

#fig4 = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, compare_to=True, comp_label=8)

fig5 = b_utils.plot_cluster_wav_mags(proj, labels, 2, features, frequencies, wave_correct=True, response_correct=True, mean_response=True, colour='lightblue')

#%%

###
# FOR SYNERGIES: Calculate and plot the time series of each channel, the frame 
# of each keypress, the onset and offset of each keypress; calculate and plot 
# each channel to a normalised frame; calculate integrals of each channel;
# calculate and plot the mean and 95% CI of all channels over all occurrences
# of the synergy of interest
###

channels = ['Little', 'Ring', 'Middle', 'Index']
colors = sns.color_palette(["#005F99","#FF449F","#FFEA6B","#00EAD3"])
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
    return frequency_hz

# Function to slice, normalise, and invert y values
def process_y_values(data):
    y_values_combined = np.vstack((data[:, 1], data[:, 3], data[:, 5], data[:, 7])).T
    normalised_y_values = y_values_combined - y_values_combined[0, :]
    positive_y_values = np.abs(normalised_y_values)
    return y_values_combined, normalised_y_values, positive_y_values

# Function to access full data for DTW analysis
def process_y_values_full(data):
    y_values_combined_full = np.vstack((data[:, 1], data[:, 3], data[:, 5], data[:, 7])).T
    return y_values_combined_full

# Function to plot time series and detect key events
def time_series_events(ax, x_range, y_values_combined, channels, colors, y_threshold, plot=True):
    total_peaks_across_channels = 0
    onset_offset_data = {channel: [] for channel in channels}
    area_under_curve = []
    peak_indices = []

    for i, channel_name in enumerate(channels):
        derivative = np.diff(y_values_combined[:, i], prepend=y_values_combined[0, i])
        peaks, _ = find_peaks(y_values_combined[:, i], height=y_threshold, prominence = 10)
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
            ax.plot(x_range, y_values_combined[:, i], label=channel_name, color=colors[i])
            for event_data in onset_offset_data[channel_name]:
                peak = event_data['peak']
                ax.plot(x_range[peak], y_values_combined[peak, i], 'r*')
                ax.plot(x_range[event_data['onset']], y_values_combined[event_data['onset'], i], 'go')
                ax.plot(x_range[event_data['offset']], y_values_combined[event_data['offset'], i], 'mo')

        # Calculate area under curve using simpson's rule for each channel and append to list
        area_under_curve = [simpson(positive_y_values[:, i]) for i in range(positive_y_values.shape[1])]

    if plot:
        ax.legend()

    return total_peaks_across_channels, onset_offset_data, area_under_curve

# Function to identify segments within a full dataset that are similar to a target time series using Dynamic Time Warping (DTW)
def find_similar_series(y_values_combined, full_dataset_y, syn_frame_start, syn_frame_end, num_segments=4, dist_threshold=None):
    dtw_distances = []
    num_channels = y_values_combined.shape[1]
    target_length = y_values_combined.shape[0]
    selected_ranges = [(syn_frame_start, syn_frame_end)]

    for i in range(len(full_dataset_y) - target_length + 1):
        if any(start <= i <= end or start <= i + target_length - 1 <= end for start, end in selected_ranges):
            continue

        total_distance = 0
        comparison_segment = full_dataset_y[i:i + target_length]

        for channel in range(num_channels):
            target_channel_data = np.array(y_values_combined[:, channel]).flatten()
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
def plot_similar_series(full_dataset_y, y_values_combined, similar_segment_indices, channels, colors):

    # Data container for calculating mean and confidence interval
    all_data = np.empty((len(similar_segment_indices) + 1, y_values_combined.shape[0], len(channels)))

    # Populate the first entry with the target time series
    all_data[0, :, :] = y_values_combined

    # Populate subsequent entries with similar time series data
    for idx, segment_idx in enumerate(similar_segment_indices):
        start = segment_idx
        end = start + y_values_combined.shape[0]
        similar_segment = full_dataset_y[start:end, :]
        all_data[idx + 1, :, :] = similar_segment

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot the mean of the target and similar time series segments with CI
    mean_with_confidence_intervals(ax, all_data, colors, channels)

    plt.tight_layout()
    plt.show()

# Process and plot time series for synergy of interest
data = tracking_filtered[syn_frame_start:syn_frame_end, :]
y_values_combined, normalised_y_values, positive_y_values = process_y_values(data)
x_range = np.arange(syn_frame_start, syn_frame_end)

fig, ax = plt.subplots(figsize=(14, 6))
total_peaks_across_channels, onset_offset_data, area_under_curve = time_series_events(ax, x_range, y_values_combined, channels, colors, y_threshold)
normalised_frequency_hz = normalise_peaks_to_hz(total_peaks_across_channels, syn_frame_end - syn_frame_start + 1)

ax.set_title('Time Series of Channels with Detected Keypresses')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Y Position')
ax.text(0.025, 0.05, f"Normalised Frequency: {normalised_frequency_hz:.3f} Hz", transform=ax.transAxes, color='red')
ax.legend()

plt.show()

# Plot normal and non-negative curves
fig2, ax2 = plt.subplots(figsize=(14, 6))
for i, channel_name in enumerate(channels):
    ax2.plot(x_range, positive_y_values[:, i], label=channel_name, color=colors[i])
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

#%%

# Process and plot time series for additional occurrences
full_dataset_y = process_y_values_full(tracking_filtered)

similar_segment_indices = find_similar_series(y_values_combined, full_dataset_y,
            syn_frame_start, syn_frame_end, num_segments=10, dist_threshold=1000)

plot_similar_series(full_dataset_y, y_values_combined,similar_segment_indices, 
            channels, colors)

#%%

###
# FOR SYNERGIES: Calculate and plot average velocity and acceleration for each
# channel; calculate and plot overlap of movements over all channels
###

# Initialise empty lists to store values
average_velocities = []
std_devs_velocity = []

average_accelerations = []
std_devs_acceleration = []

data = tracking_filtered[syn_frame_start:syn_frame_end, :] # Set data range
y_values_combined, normalised_y_values, positive_y_values = process_y_values(data)

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

total_peaks, onset_offset_data, area_under_curve = time_series_events(ax, x_range, y_values_combined, channels, colors, y_threshold, plot=False)  # plot=True if you want to visualise
total_overlap_frames, all_events_sorted = calculate_overlaps(onset_offset_data)
percent_overlap = total_overlap_frames / len(y_values_combined) * 100

# Calculate magnitude of velocities and accelerations
for i in range(y_values_combined.shape[1]):
    velocities = np.abs(np.diff(y_values_combined[:, i]))
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
# Display the result
print(f"Total percent overlap among channels: {percent_overlap:.2f}%")

#%%

###
# FOR TRIALS: Calculate and plot integrals of each channel per trial; calculate
# and plot the noralised frequency of peaks in hz per trial 
###

fig, ax = plt.subplots(figsize=(18, 6))
normalised_freq_per_trial = []

trial_keys = list(trial.keys())  # Get a list of trial keys to use for x-ticks
x_positions = range(1, len(trial_keys) + 1)  # Generate x positions for each bar

# Iterate over each trial using enumerate for proper positioning
for idx, (trial_number, (trial_start, trial_end)) in enumerate(trial.items(), start=1):
    data = tracking_filtered[trial_start:trial_end, :]
    y_values_combined = np.vstack((data[:, 1], data[:, 3], data[:, 5], data[:, 7])).T

    normalised_y_values = y_values_combined - y_values_combined[0, :]
    positive_y_values = np.abs(normalised_y_values)

    area_under_curve = [simpson(positive_y_values[:, i]) for i in range(positive_y_values.shape[1])]
    total_area = sum(area_under_curve)
    bottom_val = 0  

    # For each channel, add a segment to the bar at the correct x position
    for i, area in enumerate(area_under_curve):
        ax.bar(x_positions[idx-1], area, bottom=bottom_val, color=colors[i])
        bottom_val += area

    # Calculate normalised frequency for the trial
    total_peaks_across_channels = sum([len(find_peaks(y_values_combined[:, i], prominence=17)[0]) for i in range(y_values_combined.shape[1])])
    normalised_frequency_hz = normalise_peaks_to_hz(total_peaks_across_channels, trial_end - trial_start + 1)
    normalised_freq_per_trial.append(normalised_frequency_hz)

# Correctly setting x-ticks
ax.set_xticks(x_positions)
ax.set_xticklabels(trial_keys)

# Plotting the normalised frequency as a line plot on a secondary y-axis
ax2 = ax.twinx()
ax2.plot(x_positions, normalised_freq_per_trial, color='slategrey', marker='o', label='Normalised Frequency (Hz)')
ax2.set_ylabel('Normalised Frequency (Hz)', color='slategrey')
ax2.tick_params(axis='y', labelcolor='slategrey')

# Make sure the x-axis aligns for both plots
ax.set_xticks(x_positions)
ax.set_xticklabels(trial_keys)

# Adding labels and title
ax.set_xlabel('Trial', fontsize=14)
ax.set_ylabel('Total Area Under Curve', fontsize=14)
plt.title('Total Activity per Trial with Normalised Frequency', fontsize=16)

# Adding legend for the channels, and possibly for the normalised frequency if desired
legend_elements = [Patch(facecolor=colors[i], edgecolor='white', label=channels[i]) for i in range(len(channels))] + [Line2D([0], [0], color='slategrey', marker='o', label='Normalised Hz')]
ax.legend(handles=legend_elements[::-1], loc='upper left')

plt.tight_layout()
plt.show()

#%%

# TODO Convert CKPS calculations from Matlab to Python and add
# TODO Convert micro-online and micro-offline code from Matlab to Python and add
