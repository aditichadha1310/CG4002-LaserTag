import csv
import time
import numpy as np
import start_of_move_test2

with open('test/lzh_total_data2.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    window = []
    # window = np.array(window)

    # Skip the header row if present
    next(csv_reader)
    counter = 0
    change = [0,0]

    min = 0
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
            print("------------------------------Grenade-------------------------------")
            counter = -50
        print(action,end=" ")
        nparr = nparr[0:3]
        window.append(nparr)
        if len(window) == 10:
            output = start_of_move_test2.detect_movement_start(window)
            if output < min:
                min = output
            # output = str(output)
            # with open('output.csv','a') as out:
            #     out.write(str(action))
            #     out.write(",")
            #     out.write(output)
            #     out.write('\n')
            if isChanged:
                with open('test/output.csv','a') as out:
                    out.write(str(change[0]))
                    out.write(",")
                    out.write(str(min))
                    out.write('\n')
                isChanged = False
                min = 0
            window = window[1:]

        # Delay for 1 second
        # time.sleep(0.02)
