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


# Main function
window = []
counter = 0
actionBeforeCurrent = [0,0]
max_value = 0
isActionChanged = False
BEFORE = 0
CURRENT = 1
WINDOW_SIZE = 12

with open('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/15032023_50Hz.csv') as csv_file:
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
            isChanged = True
        if counter > 90 and action != 0:
            print_action(action)
            counter = -50
        print(action,end=" ")
        acc_values = np_arr[0:3]
        window.append(acc_values)
        if len(window) == WINDOW_SIZE:
            detect_movement_start(window)
            if isChanged:
                with open('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/output.csv','a') as out:
                    out.write(str(actionBeforeCurrent[0]))
                    out.write(",")
                    out.write(str(max_value))
                    out.write('\n')
                isChanged = False
                maxx = 0
            window = window[1:]

        # Delay for 1 second
        time.sleep(0.02)
