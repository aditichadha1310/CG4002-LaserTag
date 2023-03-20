import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

WINDOW_SIZE = 20 # i.e. 1 sec of data to determine action
STEP_SIZE = 10
window = []
acc_x_list = []
acc_y_list = []
acc_z_list = []
gyro_x_list = []
gyro_y_list = []
gyro_z_list = []
train_labels = []

def feature_engineering(window):
    df = pd.DataFrame(window)
    xs = df[0].values
    ys = df[1].values
    zs = df[2].values
    xg = df[3].values
    yg = df[4].values
    zg = df[5].values

    acc_x_list.append(xs)   
    acc_y_list.append(ys)
    acc_z_list.append(zs)
    gyro_x_list.append(xg)   
    gyro_y_list.append(yg)
    gyro_z_list.append(zg)

# Statistical Features on raw x, y and z in time domain
X_train = pd.DataFrame()
y_train = np.array(train_labels)

# mean
X_train['acc_x_mean'] = pd.Series(acc_x_list).apply(lambda x: x.mean())
X_train['acc_y_mean'] = pd.Series(acc_y_list).apply(lambda x: x.mean())
X_train['acc_z_mean'] = pd.Series(acc_z_list).apply(lambda x: x.mean())
X_train['gyro_x_mean'] = pd.Series(gyro_x_list).apply(lambda x: x.mean())
X_train['gyro_y_mean'] = pd.Series(gyro_y_list).apply(lambda x: x.mean())
X_train['gyro_z_mean'] = pd.Series(gyro_z_list).apply(lambda x: x.mean())

# std dev
X_train['acc_x_std'] = pd.Series(acc_x_list).apply(lambda x: x.std())
X_train['acc_y_std'] = pd.Series(acc_y_list).apply(lambda x: x.std())
X_train['acc_z_std'] = pd.Series(acc_z_list).apply(lambda x: x.std())
X_train['gyro_x_std'] = pd.Series(gyro_x_list).apply(lambda x: x.std())
X_train['gyro_y_std'] = pd.Series(gyro_y_list).apply(lambda x: x.std())
X_train['gyro_z_std'] = pd.Series(gyro_z_list).apply(lambda x: x.std())

# avg absolute diff
X_train['acc_x_aad'] = pd.Series(acc_x_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
X_train['acc_y_aad'] = pd.Series(acc_y_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
X_train['acc_z_aad'] = pd.Series(acc_z_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
X_train['gyro_x_aad'] = pd.Series(gyro_x_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
X_train['gyro_y_aad'] = pd.Series(gyro_y_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
X_train['gyro_z_aad'] = pd.Series(gyro_z_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

# min
X_train['acc_x_min'] = pd.Series(acc_x_list).apply(lambda x: x.min())
X_train['acc_y_min'] = pd.Series(acc_y_list).apply(lambda x: x.min())
X_train['acc_z_min'] = pd.Series(acc_z_list).apply(lambda x: x.min())
X_train['gyro_x_min'] = pd.Series(gyro_x_list).apply(lambda x: x.min())
X_train['gyro_y_min'] = pd.Series(gyro_y_list).apply(lambda x: x.min())
X_train['gyro_z_min'] = pd.Series(gyro_z_list).apply(lambda x: x.min())

# max
X_train['acc_x_max'] = pd.Series(acc_x_list).apply(lambda x: x.max())
X_train['acc_y_max'] = pd.Series(acc_y_list).apply(lambda x: x.max())
X_train['acc_z_max'] = pd.Series(acc_z_list).apply(lambda x: x.max())
X_train['gyro_x_max'] = pd.Series(gyro_x_list).apply(lambda x: x.max())
X_train['gyro_y_max'] = pd.Series(gyro_y_list).apply(lambda x: x.max())
X_train['gyro_z_max'] = pd.Series(gyro_z_list).apply(lambda x: x.max())

# max-min diff
X_train['acc_x_maxmin_diff'] = X_train['acc_x_max'] - X_train['acc_x_min']
X_train['acc_y_maxmin_diff'] = X_train['acc_y_max'] - X_train['acc_y_min']
X_train['acc_z_maxmin_diff'] = X_train['acc_z_max'] - X_train['acc_z_min']
X_train['gyro_x_maxmin_diff'] = X_train['gyro_x_max'] - X_train['gyro_x_min']
X_train['gyro_y_maxmin_diff'] = X_train['gyro_y_max'] - X_train['gyro_y_min']
X_train['gyro_z_maxmin_diff'] = X_train['gyro_z_max'] - X_train['gyro_z_min']

# median
X_train['acc_x_median'] = pd.Series(acc_x_list).apply(lambda x: np.median(x))
X_train['acc_y_median'] = pd.Series(acc_y_list).apply(lambda x: np.median(x))
X_train['acc_z_median'] = pd.Series(acc_z_list).apply(lambda x: np.median(x))
X_train['gyro_x_median'] = pd.Series(gyro_x_list).apply(lambda x: np.median(x))
X_train['gyro_y_median'] = pd.Series(gyro_y_list).apply(lambda x: np.median(x))
X_train['gyro_z_median'] = pd.Series(gyro_z_list).apply(lambda x: np.median(x))

# median abs dev 
X_train['acc_x_mad'] = pd.Series(acc_x_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
X_train['acc_y_mad'] = pd.Series(acc_y_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
X_train['acc_z_mad'] = pd.Series(acc_z_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
X_train['gyro_x_mad'] = pd.Series(gyro_x_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
X_train['gyro_y_mad'] = pd.Series(gyro_y_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
X_train['gyro_z_mad'] = pd.Series(gyro_z_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))

# interquartile range
X_train['acc_x_IQR'] = pd.Series(acc_x_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
X_train['acc_y_IQR'] = pd.Series(acc_y_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
X_train['acc_z_IQR'] = pd.Series(acc_z_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
X_train['gyro_x_IQR'] = pd.Series(gyro_x_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
X_train['gyro_y_IQR'] = pd.Series(gyro_y_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
X_train['gyro_z_IQR'] = pd.Series(gyro_z_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

# negtive count
X_train['acc_x_neg_count'] = pd.Series(acc_x_list).apply(lambda x: np.sum(x < 0))
X_train['acc_y_neg_count'] = pd.Series(acc_y_list).apply(lambda x: np.sum(x < 0))
X_train['acc_z_neg_count'] = pd.Series(acc_z_list).apply(lambda x: np.sum(x < 0))
X_train['gyro_x_neg_count'] = pd.Series(gyro_x_list).apply(lambda x: np.sum(x < 0))
X_train['gyro_y_neg_count'] = pd.Series(gyro_y_list).apply(lambda x: np.sum(x < 0))
X_train['gyro_z_neg_count'] = pd.Series(gyro_z_list).apply(lambda x: np.sum(x < 0))

# positive count
X_train['acc_x_pos_count'] = pd.Series(acc_x_list).apply(lambda x: np.sum(x > 0))
X_train['acc_y_pos_count'] = pd.Series(acc_y_list).apply(lambda x: np.sum(x > 0))
X_train['acc_z_pos_count'] = pd.Series(acc_z_list).apply(lambda x: np.sum(x > 0))
X_train['gyro_x_pos_count'] = pd.Series(gyro_x_list).apply(lambda x: np.sum(x > 0))
X_train['gyro_y_pos_count'] = pd.Series(gyro_y_list).apply(lambda x: np.sum(x > 0))
X_train['gyro_z_pos_count'] = pd.Series(gyro_z_list).apply(lambda x: np.sum(x > 0))

# values above mean
X_train['acc_x_above_mean'] = pd.Series(acc_x_list).apply(lambda x: np.sum(x > x.mean()))
X_train['acc_y_above_mean'] = pd.Series(acc_y_list).apply(lambda x: np.sum(x > x.mean()))
X_train['acc_z_above_mean'] = pd.Series(acc_z_list).apply(lambda x: np.sum(x > x.mean()))
X_train['gyro_x_above_mean'] = pd.Series(gyro_x_list).apply(lambda x: np.sum(x > x.mean()))
X_train['gyro_y_above_mean'] = pd.Series(gyro_y_list).apply(lambda x: np.sum(x > x.mean()))
X_train['gyro_z_above_mean'] = pd.Series(gyro_z_list).apply(lambda x: np.sum(x > x.mean()))

# number of peaks
X_train['acc_x_peak_count'] = pd.Series(acc_x_list).apply(lambda x: len(find_peaks(x)[0]))
X_train['acc_y_peak_count'] = pd.Series(acc_y_list).apply(lambda x: len(find_peaks(x)[0]))
X_train['acc_z_peak_count'] = pd.Series(acc_z_list).apply(lambda x: len(find_peaks(x)[0]))
X_train['gyro_x_peak_count'] = pd.Series(gyro_x_list).apply(lambda x: len(find_peaks(x)[0]))
X_train['gyro_y_peak_count'] = pd.Series(gyro_y_list).apply(lambda x: len(find_peaks(x)[0]))
X_train['gyro_z_peak_count'] = pd.Series(gyro_z_list).apply(lambda x: len(find_peaks(x)[0]))

# skewness
X_train['acc_x_skewness'] = pd.Series(acc_x_list).apply(lambda x: stats.skew(x))
X_train['acc_y_skewness'] = pd.Series(acc_y_list).apply(lambda x: stats.skew(x))
X_train['acc_z_skewness'] = pd.Series(acc_z_list).apply(lambda x: stats.skew(x))
X_train['gyro_x_skewness'] = pd.Series(gyro_x_list).apply(lambda x: stats.skew(x))
X_train['gyro_y_skewness'] = pd.Series(gyro_y_list).apply(lambda x: stats.skew(x))
X_train['gyro_z_skewness'] = pd.Series(gyro_z_list).apply(lambda x: stats.skew(x))

# kurtosis
X_train['acc_x_kurtosis'] = pd.Series(acc_x_list).apply(lambda x: stats.kurtosis(x))
X_train['acc_y_kurtosis'] = pd.Series(acc_y_list).apply(lambda x: stats.kurtosis(x))
X_train['acc_z_kurtosis'] = pd.Series(acc_z_list).apply(lambda x: stats.kurtosis(x))
X_train['gyro_x_kurtosis'] = pd.Series(gyro_x_list).apply(lambda x: stats.kurtosis(x))
X_train['gyro_y_kurtosis'] = pd.Series(gyro_y_list).apply(lambda x: stats.kurtosis(x))
X_train['gyro_z_kurtosis'] = pd.Series(gyro_z_list).apply(lambda x: stats.kurtosis(x))

# energy
X_train['acc_x_energy'] = pd.Series(acc_x_list).apply(lambda x: np.sum(x**2)/100)
X_train['acc_y_energy'] = pd.Series(acc_y_list).apply(lambda x: np.sum(x**2)/100)
X_train['acc_z_energy'] = pd.Series(acc_z_list).apply(lambda x: np.sum(x**2)/100)
X_train['gyro_x_energy'] = pd.Series(gyro_x_list).apply(lambda x: np.sum(x**2)/100)
X_train['gyro_y_energy'] = pd.Series(gyro_y_list).apply(lambda x: np.sum(x**2)/100)
X_train['gyro_z_energy'] = pd.Series(gyro_z_list).apply(lambda x: np.sum(x**2)/100)

# avg resultant
X_train['acc_avg_result'] = [i.mean() for i in ((pd.Series(acc_x_list)**2 + pd.Series(acc_y_list)**2 + pd.Series(acc_z_list)**2)**0.5)]
X_train['gyro_avg_result'] = [i.mean() for i in ((pd.Series(gyro_x_list)**2 + pd.Series(gyro_y_list)**2 + pd.Series(gyro_z_list)**2)**0.5)]

# signal magnitude area
X_train['acc_sma'] =    pd.Series(acc_x_list).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(acc_y_list).apply(lambda x: np.sum(abs(x)/100)) \
                  + pd.Series(acc_z_list).apply(lambda x: np.sum(abs(x)/100))
X_train['gyro_sma'] =    pd.Series(gyro_x_list).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(gyro_y_list).apply(lambda x: np.sum(abs(x)/100)) \
                  + pd.Series(gyro_z_list).apply(lambda x: np.sum(abs(x)/100))

X_train['labels'] = pd.Series(train_labels)

print(X_train.head(60))
print(X_train['labels'].value_counts())
print(X_train.shape)
# print(y_train.shape)
# X_train.to_csv('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/19032023_train.csv', index=False)

# with open('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/19032023_20Hz_clean.csv') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     # Skip the header row if present
#     next(csv_reader)
#     for row in csv_reader:
#         np_arr = np.array(row)


#         window.append(np_arr)
#         if len(window) == WINDOW_SIZE:
#             window_feature_engineering(window)
#             window = window[10:]