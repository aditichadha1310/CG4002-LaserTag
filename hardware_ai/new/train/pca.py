import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv(
    '/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/20230403_processed_train.csv')
test_data = pd.read_csv(
    '/Users/edly/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/20230403_processed_test.csv')
X_train = train_data.iloc[:, :100].values
y_train = train_data.iloc[:, 100:101].values
X_test = test_data.iloc[:, :100].values
y_test = test_data.iloc[:, 100:101].values

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train[0])

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(pca.explained_variance_ratio_.cumsum())