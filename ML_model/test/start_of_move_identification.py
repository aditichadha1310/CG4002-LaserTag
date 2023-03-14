import matplotlib.pyplot as plt
import pandas as pd

datain = pd.read_csv('datapreprocessed.csv')
datain2 = datain[datain['lab'] == 1]
datain3 = datain[datain['lab'] == 0]
datain2 = datain2.iloc[:500]
datain3 = datain3.iloc[:1000]

# Create scatter plot
# plt.scatter(x = 'ind', y = 'acc_x', c= 'pink', data = datain2)
# plt.scatter(x = 'ind', y = 'acc_y', c= 'green', data = datain2)
# plt.scatter(x = 'ind', y = 'acc_z', c= 'yellow', data = datain2)
plt.scatter(x = 'ind', y = 'acc_mean', c= 'red', data = datain2)
plt.scatter(x = 'ind', y = 'acc_mean', c= 'blue', data = datain3)

# Create title, xlabel and ylabel
plt.title('Grenade')
plt.xlabel('Time')
plt.ylabel('Acc')
# Show plot
plt.show()