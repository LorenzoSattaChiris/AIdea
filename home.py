# Importing the required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Loading the variables
training_test_ratio = 0.3
layer_numbers = 2
layers = [128, 10]
activations = ["relu", "softmax"]
optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]
epochs = 1
batch_size = 32

# Loading the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Combine the loaded training and test sets
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

# Split the data into training and test sets with a different ratio
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=training_test_ratio, random_state=42)

# Defining the model
model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28, 28)))

for i in range(layer_numbers):
    model.add(keras.layers.Dense(layers[i], activation=activations[i]))

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Start the timer
start_time = time.time()

# Training the model
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# Evaluating the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', test_accuracy)

# Generating the Report
elapsed_time = time.time() - start_time

train_accuracy = history.history['accuracy']
train_loss = history.history['loss']

val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

# # Plotting accuracy
# plt.plot(train_accuracy, label='Training Accuracy')
# plt.plot(val_accuracy, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # Plotting loss
# plt.plot(train_loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

