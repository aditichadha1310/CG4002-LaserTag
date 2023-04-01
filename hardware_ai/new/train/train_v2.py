from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv(
    'C:/Users/edly1/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/20230401_train.csv')
val_data = pd.read_csv(
    'C:/Users/edly1/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/20230331_20Hz_validation_processed.csv')
test_data = pd.read_csv(
    'C:/Users/edly1/Documents/GitHub/CG4002-LaserTag/hardware_ai/new/datasets/20230401_test.csv')

# Change pandas dataframe to np array
X_train = train_data.iloc[:, :100].values
y_train = train_data.iloc[:, 100:101].values
X_val = val_data.iloc[:, :100].values
y_val = val_data.iloc[:, 100:101].values
X_test = test_data.iloc[:, :100].values
y_test = test_data.iloc[:, 100:101].values

# One-hot encode the target variable
y_train_encoded = to_categorical(y_train)
y_val_encoded = to_categorical(y_val)
y_test_encoded = to_categorical(y_test)

# Define the MLP model
model = Sequential()
model.add(Dense(12, input_dim=100, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_encoded, validation_data=(X_val, y_val_encoded), epochs=50, batch_size=32)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print('Test loss: ', loss)
print('Test accuracy: ', accuracy)

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
def numpy_to_cpp_array(np_array, destination_file):
    cpp_array = ""
    if np_array.ndim == 2:
        rows, cols = np_array.shape
        for i in range(rows):
            cpp_array += "{"
            for j in range(cols):
                cpp_array += str(np_array[i][j])
                if j < cols - 1:
                    cpp_array += ", "
            cpp_array += "}"
            if i < rows - 1:
                cpp_array += ",\n"

    else:
        for i in range(len(np_array)):
            cpp_array += str(np_array[i])
            if i < len(np_array) -1:
                cpp_array += ",\n"

    with open(destination_file, "w") as file:
        file.write(cpp_array)
    return cpp_array

# for layer in model.layers:
weights1 = model.layers[0].get_weights()[0]
biases1 = model.layers[0].get_weights()[1]
numpy_to_cpp_array(weights1,'Output_Weights1.txt')
numpy_to_cpp_array(biases1,'Output_Biases1.txt')
weights2 = model.layers[1].get_weights()[0]
biases2 = model.layers[1].get_weights()[1]
numpy_to_cpp_array(weights2,'Output_Weights2.txt')
numpy_to_cpp_array(biases2,'Output_Biases2.txt')
weights3 = model.layers[2].get_weights()[0]
biases3 = model.layers[2].get_weights()[1]
numpy_to_cpp_array(weights3,'Output_Weights3.txt')
numpy_to_cpp_array(biases3,'Output_Biases3.txt')