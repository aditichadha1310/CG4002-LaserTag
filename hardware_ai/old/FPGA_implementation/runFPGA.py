import csv
import time
import numpy as np
# import start_of_move_test2
# from collections import deque
# from scipy.signal import find_peaks

# Define function for detecting movement start
def detect_movement_start(accel_data):
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



min = 0
isChanged = False

# with open('final_data.csv') as csv_file:
def to_fpga():
    # csv_reader = csv.reader(csv_file)
    window = []
    # window = np.array(window)

    # Skip the header row if present
    # next(csv_reader)
    counter = 0
    change = [0,0]

    for row in csv_reader:
        # Process the row here
        # print(row)

        nparr = np.array(row)
        action = int(nparr[-1])
        change[0] = change[1]
        change[1] = action
        if change[1]-change[0]!=0:
            isChanged = True
        print(action,end=" ")
        counter += 1
        # print(counter,end=" ")
        if counter > 99 and action != 0:
            print()
            print("------------------------------Grenade-------------------------------")
            counter = -50
        nparr = nparr[0:3]
        window.append(nparr)
        if len(window) == 10:
            output = detect_movement_start(window)
            if output < min:
                min = output
            output = str(output)
            # with open('output.csv','a') as out:
            #     out.write(str(action))
            #     out.write(",")
            #     out.write(output)
            #     out.write('\n')
            if isChanged:
                with open('output.csv','a') as out:
                    out.write(str(change[0]))
                    out.write(",")
                    out.write(str(min))
                    out.write('\n')
                isChanged = False
                min = 0
            window = window[1:]

        # Delay for 1 second
        # time.sleep(0.02)
