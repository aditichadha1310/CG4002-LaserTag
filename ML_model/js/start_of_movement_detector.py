import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, signal
import csv
import random



def preprocess_data(df):
    
    # Compute features for each column
    def compute_mean(data):
        return np.mean(data)

    def compute_variance(data):
        return np.var(data)

    def compute_median_absolute_deviation(data):
        return np.median(data)

    def compute_root_mean_square(data):
        return np.sqrt(np.mean(np.square(data)))

    def compute_interquartile_range(data):
        return stats.iqr(data)

    def compute_percentile_75(data):
        return np.percentile(data, 75)

    def compute_kurtosis(data):
        return stats.kurtosis(data)

    def compute_min_max(data):
        return np.max(data) - np.min(data)

    def compute_signal_magnitude_area(data):
        return np.sum(data) / len(data)

    def compute_zero_crossing_rate(data):
        return ((data[:-1] * data[1:]) < 0).sum()

    def compute_spectral_centroid(data):
        spectrum = np.abs(np.fft.rfft(data))
        normalized_spectrum = spectrum / np.sum(spectrum)
        normalized_frequencies = np.linspace(0, 1, len(spectrum))
        spectral_centroid = np.sum(normalized_frequencies * normalized_spectrum)
        return spectral_centroid

    def compute_spectral_entropy(data):
        freqs, power_density = signal.welch(data)
        return stats.entropy(power_density)

    def compute_spectral_energy(data):
        freqs, power_density = signal.welch(data)
        return np.sum(np.square(power_density))

    def compute_principle_frequency(data):
        freqs, power_density = signal.welch(data)
        return freqs[np.argmax(np.square(power_density))]
    
    processed_data = []

    # Loop through each column and compute features
    for column in df.columns:
        column_data = df[column]

        # Compute features for the column
        mean = compute_mean(column_data)
        variance = compute_variance(column_data)
        median_absolute_deviation = compute_median_absolute_deviation(column_data)
        root_mean_square = compute_root_mean_square(column_data)
        interquartile_range = compute_interquartile_range(column_data)
        percentile_75 = compute_percentile_75(column_data)
        kurtosis = compute_kurtosis(column_data)
        min_max = compute_min_max(column_data)
        signal_magnitude_area = compute_signal_magnitude_area(column_data)
        zero_crossing_rate = compute_zero_crossing_rate(column_data)
        spectral_centroid = compute_spectral_centroid(column_data)
        spectral_entropy = compute_spectral_entropy(column_data)
        spectral_energy = compute_spectral_energy(column_data)
        principle_frequency = compute_principle_frequency(column_data)

        # Store features in list
        processed_column_data = [mean, variance, median_absolute_deviation, root_mean_square, 
                                interquartile_range, percentile_75, kurtosis, min_max, 
                                signal_magnitude_area, zero_crossing_rate, spectral_centroid, 
                                spectral_entropy, spectral_energy, principle_frequency]
        print(processed_column_data)
        # Append processed column data to main processed data array
        processed_data.append(processed_column_data)

    processed_data_arr = np.concatenate(processed_data)

    return processed_data_arr


def MLP(data):
    return [random.random() for _ in range(4)]

def generate_simulated_wave():
    # simulate game movement with noise and action

    # base noise 10s long -> 20Hz*10 = 200 samples
    t = np.linspace(0, 5, 200) # Define the time range
    x1 = 0.2 * np.sin(t) + 0.2 * np.random.randn(200) 
    x1[(x1 > -1) & (x1 < 1)] = 0.0 # TODO - sensor noise within margin of error auto remove
    
    # movement motion
    period = 2  # seconds
    amplitude = 5
    t = np.linspace(0, 2, int(2 / 0.05)) # Define the time range
    x2 = amplitude * np.sin(2 * np.pi * t / period)[:40] # Compute the sine wave for only one cycle

    x = x1 
    # Add to the 40th-80th elements
    x[20:60] += x2

    x[80:120] += x2

    return x

def generate_simulated_data():
    yaw = random.uniform(-180, 180)
    pitch = random.uniform(-180, 180)
    roll = random.uniform(-180, 180)
    accX = random.uniform(-1000, 1000)
    accY = random.uniform(-1000, 1000)
    accZ = random.uniform(-1000, 1000)
    flex1 = random.uniform(-180, 180)
    flex2 = random.uniform(-180, 180)
    return [flex1, flex2, yaw, pitch, roll, accX, accY, accZ]

# Define the window size and threshold factor
window_size = 11
threshold_factor = 2

# Define N units for flagging movement, 20Hz -> 2s = 40 samples
N = 40

# Initialize empty arrays for data storage
t = []
x = []
filtered = []
threshold = []
movement_detected = []
last_movement_time = -N  # set last movement time to negative N seconds ago
wave = generate_simulated_wave()
i = 0
timenow = 0
    
if __name__ == "__main__":
    # Create empty dataframe
    df = pd.DataFrame(columns=['flex1', 'flex2', 'yaw', 'pitch', 'roll', 'accX', 'accY', 'accZ'])

    while True:

        # Create plot window
        plt.ion()
        plt.show()

        data = generate_simulated_data()

        # Append new data to dataframe
        df.loc[len(df)] = data

        # Compute absolute acceleration values
        # x.append(np.abs(data[5:8])) # abs of accX, accY, accZ
        x.append(wave[i]) # abs of accX, accY, accZ

        # time
        t.append(timenow)

        # Compute moving window median
        if len(x) < window_size:
            filtered.append(0)
        else:
            filtered.append(np.median(x[-window_size:], axis=0))

        # Compute threshold using past median data, threshold = mean + k * std
        if len(filtered) < window_size:
            threshold.append(0)
        else:
            past_filtered = filtered[-window_size:]
            threshold.append(np.mean(past_filtered, axis=0) + (threshold_factor * np.std(past_filtered, axis=0)))

        # Identify movement
        if len(filtered) > window_size:
            # checking if val is past threshold and if last movement was more than N samples ago
            if np.all(filtered[-1] > threshold[-1]) and len(t) - last_movement_time >= N:
                movement_detected.append(len(df) - 1)
                last_movement_time = len(t)  # update last movement time
                print(f"Movement detected at sample {len(df) - 1}")

        # if movement has been detected for more than N samples, preprocess and feed into neural network
        if len(movement_detected) > 0 and len(df) - movement_detected[-1] >= N:
            # extract movement data
            start = movement_detected[-1]
            end = len(df)
            movement_data = df.iloc[start:end, :]

            # print the start and end index of the movement
            print(f"Processing movement detected from sample {start} to {end}")

            # perform data preprocessing
            preprocessed_data = preprocess_data(movement_data)
            
            # feed preprocessed data into neural network
            output = MLP(preprocessed_data)
            print(f"output: \n {output} \n") # print output of MLP

            np_output = np.array(output)
            largest_index = np_output.argmax()
            print(f"largest index: {largest_index} \n") # print largest index of MLP output

            # reset movement_detected list
            movement_detected.clear()

        i +=1
        timenow += 1

        if i == 200:
            i = 0

        plt.clf()
        plt.plot(t, x, label='original signal')
        plt.plot(t, filtered, label='filtered signal')
        plt.plot(t, threshold, label='threshold function')
        plt.legend()
        plt.draw()
        plt.pause(0.01)

        time.sleep(0.05)

    # # Simulate incoming data every 0.05 seconds
    # for i in range(len(wave)):
    #     # new_t = time.time()
    #     new_t = i
    #     new_x = wave[i] # TODO change to data comms line

    #     # process data sub-function
    #     # Append new data to arrays
    #     t.append(new_t)
    #     x.append(new_x)
        
    #     # Compute moving window median
    #     if len(x) < window_size:
    #         filtered.append(0)
    #     else:
    #         filtered.append(np.median(x[-window_size:]))
        
    #     # Compute threshold using past median data, threshold = mean + k * std
    #     if len(filtered) < window_size:
    #         threshold.append(0)
    #     else:
    #         past_filtered = filtered[-window_size:]
    #         threshold.append(np.mean(past_filtered) + (threshold_factor * np.std(past_filtered)))
        
    #     # Identify movement
    #     if len(filtered) > window_size:
    #         # checking if val is past threshold and if last movement was more than N seconds ago
    #         if filtered[-1] > threshold[-1] and t[-1] - last_movement_time >= N:
    #             movement_detected.append(t[-1])
    #             last_movement_time = t[-1]  # update last movement time
    #             print(f"Movement detected at {t[-1]}")
        
    #     # # Output data to a CSV file
    #     # df = pd.DataFrame({'time': t, 'original': x, 'filtered': filtered, 'threshold': threshold})
    #     # df.to_csv('output.csv', index=False)

    #     # Plot data
    #     plt.clf()
    #     plt.plot(t, x, label='original signal')
    #     plt.plot(t, filtered, label='filtered signal')
    #     plt.plot(t, threshold, label='threshold function')
    #     plt.legend()
    #     plt.draw()
    #     plt.pause(0.01)

    #     time.sleep(0.01)

    # # Close plot window
    # plt.close()


# t = np.linspace(0, 10, 50)
# x = np.sin(t) + 0.2 * np.random.randn(50)
# x[20] = 10
# x[10:15] = 10
# x[30:35] = 10

# # Define the window size and threshold factor
# window_size = 7
# threshold_factor = 2

# # Apply median filtering with a dynamic threshold
# filtered = np.zeros_like(x)
# threshold = np.zeros_like(x)

# # Start processing data
# for i in range(window_size, len(t)):
#     # Get the latest datapoint and add it to the filtered array
#     x_latest = np.sin(t[i]) + 0.2 * np.random.randn(1)
#     x_latest_filtered = np.median(x_latest)  # filtered latest data point

#     # Update filtered array
#     filtered = np.concatenate((filtered, [x_latest_filtered]))
#     filtered = filtered[1:]

#     # Compute threshold using past median data, threshold = mean + k * std
#     past_filtered = filtered[-window_size:]
#     threshold_value = np.mean(past_filtered) + (threshold_factor * np.std(past_filtered))
#     threshold = np.concatenate((threshold, [threshold_value]))
#     threshold = threshold[1:]

#     # Check if movement is detected
#     if filtered[-1] > threshold[-1]:
#         movement_detected = t[i]
#         print(f"Movement detected at {i}, {t[i]}")

#     # Plot the filtered signal and threshold
#     plt.clf()
#     plt.plot(t[:i+1], filtered)
#     plt.plot(t[:i+1], threshold)
#     plt.pause(0.01)

# # Compute moving window median
# for i in range(window_size // 2, len(x) - window_size // 2):
#     filtered[i] = np.median(x[i - window_size // 2:i + window_size // 2 + 1])

#     # Compute threshold using past median data, threshold = mean + k * std
#     past_filtered = filtered[i - window_size // 2:i + window_size // 2]
#     threshold[i] = np.mean(past_filtered) + (threshold_factor * np.std(past_filtered))
#     print(f"{i}, {filtered[i]}, {threshold[i]}\n")

# # Identify movement
# movement_detected = []
# for i in range(window_size // 2, len(x) - window_size // 2):
#     if filtered[i] > threshold[i]:
#         movement_detected.append(t[i])

# Output individual datapoint values to a CSV file
# df = pd.DataFrame({'time': t, 'original': x, 'threshold': threshold, 'filtered': filtered})
# df.to_csv('output.csv', index=False)

# Plot the original and filtered signals
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.plot(t, x, ':', label='Original', linewidth=2.5)
# ax.plot(t, threshold, ':', label='Threshold', color='red')
# ax.plot(t, filtered, ':', label='Filtered', linewidth=2.5)
# ax.vlines(movement_detected, ymin=filtered.min(), ymax=filtered.max(), colors='green', label='Movement Detected')
# ax.set_ylabel('Signal')
# ax.set_xlabel('Time (s)')
# ax.legend()
# plt.show()

