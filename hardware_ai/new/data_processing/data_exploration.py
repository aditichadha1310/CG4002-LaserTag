import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv("hardware_ai/new/data_processing/15032023_raw_data.csv")
df = pd.read_csv('/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/15032023_50Hz.csv')
print(df.head(10))
print(df.shape)

df = df.dropna()
print(df.shape)

# No. of samples per action
sns.set_style('whitegrid')
plt.figure(figsize = (10, 5))
sns.countplot(x = 'label', data = df)
plt.title("No of samples by activity")
plt.show()