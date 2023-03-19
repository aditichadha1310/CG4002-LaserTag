import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import csv
import time
# import numpy as np
# import start_of_move_test2
from statistics import mean

def print_action(action):
    print()
    if action == 1:
        print("------------------------------Grenade-------------------------------")
    if action == 2:
        print("------------------------------Reload-------------------------------")
    if action == 3:
        print("------------------------------Shield-------------------------------")
    if action == 4:
        print("------------------------------End Game-------------------------------")

def detect_movement_start(accel_data):

    accX = []
    accY = []
    accZ = []
    for row in accel_data:
        accX.append(float(row[0]))
        accY.append(float(row[1]))
        accZ.append(float(row[2]))

    diffX = mean(accX[WINDOW_HALF:]) - mean(accX[0:WINDOW_HALF])
    diffY = mean(accY[WINDOW_HALF:]) - mean(accY[0:WINDOW_HALF])
    diffZ = mean(accZ[WINDOW_HALF:]) - mean(accZ[0:WINDOW_HALF])

    mean_acc_change = diffX + diffY + diffZ

    print(mean_acc_change,end=" ")
    if mean_acc_change > 30:
        print("--------------------------------------------------------------------------------------------------------Grenade thrown")
    else:
        print()
    return mean_acc_change


# Main function
window = []
counter = 0
actionBeforeCurrent = [0,0]
max_value = 0
isActionChanged = False
mean_acc_change = 0
BEFORE = 0
CURRENT = 1
WINDOW_SIZE = 12
WINDOW_HALF = 6

with open('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/17032023_20Hz.csv') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Skip the header row if present
    next(csv_reader)

    for row in csv_reader:
        np_arr = np.array(row)
        action = int(np_arr[-1])
        counter += 1
        actionBeforeCurrent[BEFORE] = actionBeforeCurrent[CURRENT]
        actionBeforeCurrent[CURRENT] = action
        if actionBeforeCurrent[CURRENT]-actionBeforeCurrent[BEFORE]!=0:
            isActionChanged = True
        if counter > 90 and action != 0:
            print_action(action)
            counter = -50
        print(action,end=" ")
        acc_values = np_arr[0:3]
        window.append(acc_values)
        # Check for formation of a complete window
        if len(window) == WINDOW_SIZE:
            mean_acc_change = detect_movement_start(window)
            window = window[1:]
            # Record action and change in mean acc
        with open('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/output.csv','a') as out:
            out.write(str(action))
            out.write(",")
            out.write(str(mean_acc_change))
            out.write('\n')
            # if isActionChanged:
            #     with open('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/output.csv','a') as out:
            #         out.write(str(actionBeforeCurrent[0]))
            #         out.write(",")
            #         out.write(str(max_value))
            #         out.write('\n')
            #     isActionChanged = False
            #     maxx = 0
        # else:
        #     print()

        # Delay for 0.05 s --> 20 Hz sampling rate
        # time.sleep(0.05)
