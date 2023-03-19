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