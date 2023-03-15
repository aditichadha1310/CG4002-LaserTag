import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import csv
import time
# import numpy as np
# import start_of_move_test2
from statistics import mean

# df = pd.read_csv("hardware_ai/new/data_processing/15032023_raw_data.csv")
# df = pd.read_csv('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/15032023_50Hz.csv')
# print(df.head(10))
# print(df.shape)

# df = df.dropna()
# print(df.shape)

# # No. of samples per action
# sns.set_style('whitegrid')
# plt.figure(figsize = (10, 5))
# sns.countplot(x = 'label', data = df)
# plt.title("No of samples by activity")
# plt.show()





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

    accX = []
    accY = []
    accZ = []
    for row in accel_data:
        accX.append(float(row[0]))
        accY.append(float(row[1]))
        accZ.append(float(row[2]))
    
    # print(mean(accX))
    # print(accY)
    # print(accZ)

    diffX = mean(accX[20:]) - mean(accX[0:20])
    diffY = mean(accY[20:]) - mean(accY[0:20])
    diffZ = mean(accZ[20:]) - mean(accZ[0:20])

    value = diffX + diffY + diffZ

    print(value,end=" ")
    if value > 30:
        print("-----------------------------------------------------------------------Grenade thrown")
    else:
        print()
    return value






with open('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/15032023_50Hz.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    window = []
    # window = np.array(window)

    # Skip the header row if present
    next(csv_reader)
    counter = 0
    change = [0,0]

    maxx = 0
    isChanged = False

    for row in csv_reader:
        # Process the row here
        # print(row)

        nparr = np.array(row)
        action = int(nparr[-1])
        change[0] = change[1]
        change[1] = action
        if change[1]-change[0]!=0:
            isChanged = True
        counter += 1
        # print(counter,end=" ")
        if counter > 90 and action != 0:
            print()
            print("------------------------------Action-------------------------------")
            counter = -50
        print(action,end=" ")
        nparr = nparr[0:3]
        window.append(nparr)
        if len(window) == 40:
            # output = detect_movement_start(window)
            detect_movement_start(window)
            # if output > maxx:
            #     maxx = output

            # output = str(output)
            # with open('output.csv','a') as out:
            #     out.write(str(action))
            #     out.write(",")
            #     out.write(output)
            #     out.write('\n')
            if isChanged:
                with open('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/output.csv','a') as out:
                    out.write(str(change[0]))
                    out.write(",")
                    out.write(str(maxx))
                    out.write('\n')
                isChanged = False
                maxx = 0
            window = window[1:]

        # Delay for 1 second
        time.sleep(0.02)
