## Hardware AI
All files currently used are in the 'New' folder; the 'Old' folder is an archive for experimental files in the intial development phase
### data_processing
This folder contains all files used to process data.
- `cpp_array_maker_v2.py`
    - This file is used to convert the weights and biases resulting from model training into the C++ 2D array format, by enclosing each row in curly brackets, separating each element with commas etc.
- `window_feature_engineering.py`
    - This file takes in raw data collected for training, validation and testing and generates new features required for time-series data
    - It converts every group of 30 rows and 6 columns into 1 row and 100 columns
### datasets
Contains every dataset used for training, validation and testing, labeled accordingly.

The latest dataset used were the 20230412_30_processed .csv files
### run
- `ai_interface.py`
    - Is the only file in the Ultra96
    - Has 4 functions:
        1) `load_overlay`
            - Called once in external comms to load overlay
        2) `confirmAction`
            - Called every time external comms receives IMU readings from internal comms
            - Data is in the form of 30 rows and 6 columnns
            - Data is sent to the 'feature_engineering' function first
        3) `feature_engineering`
            - Data coverted from 30 rows and 6 columns to 1 row and 100 columns
        4) `feed_overlay`
            - Engineered features, in the form of 100 floats, are then fed into the bitstream, which will then return an integer from 1 to 4, representing the predicted action
- `amlpfunc16.cpp`
    - C++ code to implement classifier in Vivado HLS
    - Consists of 4 main components:
        1) Read input from S_AXIS
            - Input in the form of an array of 100 floats
        2) Create 2D arrays to store weights and biases
        3) Layer logic for Hidden Layers 1 and 2, and Output Layer
            - Matrix multiplication with weights
            - Matrix addition with biases
        4) Write output to M_AXIS
- `test_amlpfunc16.cpp`
    - Testbench to simulate amlpfunc16.cpp with
### train
- `train_v2.py`
    - Takes in feature-engineered dataset and trains model
    - Returns weights and biases to be used in C++ code
