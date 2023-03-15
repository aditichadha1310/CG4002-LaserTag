# Import necessary libraries
import numpy as np
from collections import deque
from scipy.signal import find_peaks

# Define function for detecting movement start
def detect_movement_start(accel_data, threshold=0.5, window_size=5, peak_height=0.5):
    """
    Detect the start of a movement in accelerometer data.

    Parameters:
        accel_data (numpy array): Accelerometer data, with shape (N,3) for N samples and 3 axes.
        threshold (float): Threshold for detecting significant changes in acceleration.
        window_size (int): Size of moving average window for smoothing data.
        peak_height (float): Minimum height of a peak in the smoothed data to be considered significant.

    Returns:
        start_index (int): Index of the first sample in the movement.
    """
    
    
    # arr = np.array(accel_data)
    # print(arr)

    accX = list()
    accY = list()
    accZ = list()
    for row in accel_data:
        accX.append(row[0])
        accY.append(row[1])
        accZ.append(row[2])
    
    # print(accX)
    # print(accY)

    diffX = float(max(accX[5:])) - float(max(accX[0:]))
    diffY = float(max(accY[5:])) - float(max(accY[0:]))
    diffZ = float(max(accZ[5:])) - float(max(accZ[0:]))

    value = diffX + diffY + diffZ

    print(value,end=" ")
    if value < -11:
        print("-----------------------------------------------------------------------Grenade thrown")
    else:
        print()
    return value

    # if( value > threshold):


    # ave1 = np.mean(arr,axis=0)
    # ave2 = np.mean(arr,axis=0)
    # diff = ave2 - ave1

    # print(ave2,end="  ")
    # print(ave1,end="  ")
    # print(diff)


    # action = accel_data[-1]
    # print(action,end=" ")
    # accel_data = accel_data[0:3]
    # print(accel_data, end=" ")
    # counter+=1
    # sum_x += accel_data[0]
    # if counter == 5:
    #     mean_x1 = sum_x/5
    # if counter == 10:
    #     mean_x2 = sum_x/5
    #     counter = 0
    #     diff = mean_x2 - mean_x1
    #     print(diff, end=" ")


    # Compute magnitude of acceleration vector
    # accel_mag = np.linalg.norm(accel_data)
    # print(accel_mag)

    # Smooth data with moving average window
    # smoothed_data = np.convolve(accel_mag, np.ones(window_size)/window_size, mode='valid')

    # Compute difference between consecutive samples
    # diff_data = np.diff(smoothed_data)

    # Find peaks in the difference data
    # peak_indices, _ = find_peaks(diff_data, height=peak_height)

    # Check if any peaks are above the threshold
    # for i in peak_indices:
    #     print(diff_data[i])
    #     if diff_data[i] > threshold:
    #         # start_index = i
    #         print("Start of move detected")
    #     else:
    #         print("No move")
    #         break

    # return start_index
