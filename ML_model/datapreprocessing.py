import pandas as pd
import numpy as np

# Read dataset for training and testing
data = pd.read_csv("final_data.csv")

print(data.head())
# data = data.drop(columns=['date', 'time', 'username', 'wrist', 'activity'])
# print(data.head())

acc_data = data[['acc_x', 'acc_y', 'acc_z']].values
gyro_data = data[['gyro_x', 'gyro_y', 'gyro_z']].values

data['acc_mean'] = np.mean(acc_data, axis=1)
data['acc_std'] = np.std(acc_data, axis=1)
data['acc_max'] = np.max(acc_data, axis=1)
data['acc_min'] = np.min(acc_data, axis=1)

data['gyro_mean'] = np.mean(gyro_data, axis=1)
data['gyro_std'] = np.std(gyro_data, axis=1)
data['gyro_max'] = np.max(gyro_data, axis=1)
data['gyro_min'] = np.min(gyro_data, axis=1)

data['lab'] = data['label']
data = data.drop(columns='label')

print(data.head())
data.to_csv('datapreprocessed.csv', index=False)
