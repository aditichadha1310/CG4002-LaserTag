# Defining neural networks with Keras sequential API and Keras functional API

# Sequential API
# Must have layers ordered in sequence
# - Input layer
# - Some number of hidden layers
# - Output layer

# Training and evaluation
# 1) Load and clean data
# 2) Define model
# 3) Train and evaluate model
# 4) Evaluate model

# Building a sequential model

# Import tensorflow
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# from tensorflow import keras
import pandas as pd
import numpy as np

# Define constants
LABELS = ['nothing',
          'grenade',
          'reload',
          'shield',
          'end']


# data = pd.read_csv("Supervised ML model\WISDM_ar_v1.1_raw.txt", sep=",", )
data = pd.read_csv("data\datapreprocessed.csv")
print(data.head())

# Change pandas dataframe to np array
# X = data.iloc[:, 5:11].values
# y = data.iloc[:, 11:12].values

X = data.iloc[:, 0:14].values
y = data.iloc[:, 14:15].values

print(X[:10])
print(y[:10])

# Normalisation of data
sc = StandardScaler()
X = sc.fit_transform(X)

print(X[:10])

# One hot encode labels
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

print(y[:10])

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Define a sequential model
model = tf.keras.Sequential()

# Define first hidden layer, 16-node dense layer
# - input_shape = dimensions of input reshaped into a vector
# model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)))
# model.add(tf.keras.layers.Flatten(input_shape=(20, 14)))
model.add(tf.keras.layers.Dense(16, input_dim=14, activation='relu'))
# model.add(tf.keras.layers.Dense(16, activation='relu'))

# Define second hidden layer, 8-node dense layer
model.add(tf.keras.layers.Dense(8, activation='relu'))

# Define output layer, 4 outputs
model.add(tf.keras.layers.Dense(5, activation='softmax'))

# At this point, the model has been defined, but not yet trained
# We must first complete the compilation step: define the optimiser and loss function

# Compile the model with the adam optimiser and categorical crossentropy loss function (for classification problems with more than 2 classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model
# fit() operation
#   - Required arguments
#       - features
#       - labels
#   - Optional arguments
#       - batch_size (default = 32 examples in each batch)
#       - epochs (number of times the whole set of batches is trained), using multiple epochs allows the model to revisit the same batches but with different weights and biases,
#         and possibly optimiser parameters, since they are updated after each batch
#       - validation_split (splits the data into a training and validation set)
history = model.fit(X_train, y_train, epochs=100,
                    batch_size=64, validation_data=(X_test, y_test))

# Evaluate the test set
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predict using test data
# prediction = model.predict(X_test)

# pred = np.argmax(prediction, axis=1)[:5]
# label = np.argmax(y_test, axis=1)[:5]

# print(pred)
# print(label)
# print(prediction.flatten())
# print(y_test)

# Visualise traning and validation losses and accuracies
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Summarise the model
print(model.summary())

# Save weights
# for layer in model.layers:
weights1 = model.layers[0].get_weights()[0]
biases1 = model.layers[0].get_weights()[1]
np.savetxt('Output_Weights1.txt', weights1, delimiter=', ')
np.savetxt('Output_Biases1.txt', biases1, delimiter=', ')
weights2 = model.layers[1].get_weights()[0]
biases2 = model.layers[1].get_weights()[1]
np.savetxt('Output_Weights2.txt', weights2, delimiter=', ')
np.savetxt('Output_Biases2.txt', biases2, delimiter=', ')
weights3 = model.layers[2].get_weights()[0]
biases3 = model.layers[2].get_weights()[1]
np.savetxt('Output_Weights3.txt', weights3, delimiter=', ')
np.savetxt('Output_Biases3.txt', biases3, delimiter=', ')
# print(weights)

if os.path.isfile('Supervised ML model\weights.h5') is False:
    model.save_weights('Supervised ML model\weights.h5')
