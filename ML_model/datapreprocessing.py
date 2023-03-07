import pandas as pd
import numpy as np

# Read dataset for training and testing
data = pd.read_csv("ML_model\dataset.csv")

# Create an example dictionary
users = {'Name': ['Amit', 'Cody', 'Drew'],
    'Age': [20,21,25]}
 
# Create DataFrame
df = pd.DataFrame(users, columns=['Name','Age'])
 
print("Original DataFrame:")
print(df)
 
df.to_csv('datapreprocessed.csv')
new_df = pd.read_csv('datapreprocessed.csv')

print('Data from Users.csv:')
print(new_df)