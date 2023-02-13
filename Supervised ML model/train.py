# Defining neural networks in TensorFlow using both linear algebra and higher level Keras operations

import tensorflow as tf
import pandas as pd
import numpy as np

# Define input data
read_data = pd.read_csv('Supervised ML model\dataset.csv')
data = np.array(read_data, np.float32)

# We only need 6 features:
# acceleration_x,
# acceleration_y,
# acceleration_z,
# gyro_x,
# gyro_y,
# gyro_z
data = data.drop(columns=["date", "time", "username"])

# Look at the right wrist first
data = data[data.wrist == 0]

# Label of each row should be the activity column since its a supervised ML model i.e. needs labelled data

# Plot the data

# Define weights for the 1st layer as a 6x truncated random normal variable
weights1 = tf.Variable(tf.random.truncated_normal([6, 32]))

# Define bias for the 1st layer
bias1 = tf.Variable(tf.ones([32]))

# Define weights for the 2nd layer as a 500x500 truncated random normal variable
weights2 = tf.Variable(tf.random.truncated_normal([32, 16]))

# Define bias for the 2nd layer
bias2 = tf.Variable(0.0)

# Define dense layer 1 with the glorot uniform initialiser
dense1 = tf.keras.layers.Dense(32, activation='relu')(data)

# Define dense layer 2 with the glorot uniform initialiser
dense2 = tf.keras.layers.Dense(16, activation='relu')(dense1)

# Avoid overfitting the neural network (instead of memorising examples, train the model to recognise general patterns)
# Use dropouts to solve this because it forces the neural network to develop more robust rules for classification
# since it cannot rely on any particular nodes being passed to an activation function
# to improve out of sample performance

# Apply dropout operation (this drops weights connected to 25% of nodes randomly)
dropout1 = tf.keras.layers.Dropout(0.25)(dense2)

# Define output layer
outputs = tf.keras.layers.Dense(2, activation='relu')(dropout1)

# Define the model

# model_m.compile


def model(weights1, bias1, weights2, bias2, data):
    # Apply relu activation function to layer1
    layer = tf.keras.activations.relu(tf.matmul(data, weights1) + bias1)

    # Apply dropout rate of 0.25
    dropout = tf.keras.layers.Dropout(0.25)(layer)

    # Apply relu activation function to layer2
    output = tf.keras.activations.relu(tf.matmul(dropout, weights2) + bias2)

    return output


def loss_function(weights1, bias1, weights2, bias2, data, target_data):
    predicted_data = model(weights1, bias1, weights2, bias2, data)

    # Pass target data and predicted data to the cross entropy loss
    loss = tf.keras.losses.binary_crossentropy(target_data, predicted_data)

    return loss


# Train the model
for j in range(100):
    # Complete the optimizer
    tf.opt.minimize(lambda: loss_function(weights1, bias1, weights2, bias2),
                    var_list=[weights1, bias1, weights2, bias2])

# Make predictions with model using test features
model_predictions = model(weights1, bias1, weights2, bias2, data)

# Construct the confusion matrix
# confusion_matrix(test_targets, model_predictions)
