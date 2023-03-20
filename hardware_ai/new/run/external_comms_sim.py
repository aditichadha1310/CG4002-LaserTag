# Send data to start_of_move_checker, one row at a time
# start_of_move_checker gathers data row by row to form window
# Once window has 12 rows, check for possible start of move
#   - if mean_acc_change passes a predetermined threshold, return 1
#   - else, return 0

# If 1 is returned,
# Gather another 8 rows of data to form a window of 20 rows
# Once formed, pass 20-row window into data processor
#   - Feature engineering
#   - 6 features to 100 features
#   - Compress 20 rows into 1 row

# Pass the 1 row of 100 columns into the overlay function

import time
import csv
from start_of_move_checker import StartOfMoveClass
import feature_engineering
import overlay
import numpy as np

checker = StartOfMoveClass()

with open('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/17032023_20Hz.csv') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Skip the header row if present
    next(csv_reader)

    for row in csv_reader:
        np_arr = np.array(row)
        output = checker.input(np_arr)
        if output != 0:
            output.append(np_arr)
            if len(output) == 20:
                engineered_features = feature_engineering(output)
                action = overlay(engineered_features)

        # Delay for 0.05 s --> 20 Hz sampling rate
        time.sleep(0.05)