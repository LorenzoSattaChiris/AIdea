# Our AI model creator and report generator

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from sklearn.model_selection import train_test_split
import psutil

# Importing the required libraries
import matplotlib.pyplot as plt

# Loading the variables
dataset = keras.datasets.mnist.load_data()
training_test_ratio = 0.7
layer_numbers = 2
layers = [128,10] 
activations = [ 'ReLU', 'Softmax'] 
optimizer = 'Adam'
loss = 'sparse_categorical_crossentropy'
epochs = 10
batch_size = 32

# Loading the dataset
(x_train, y_train), (x_test, y_test) = dataset

# Combine the loaded training and test sets
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

# Split the data into training and test sets with a different ratio
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=training_test_ratio, random_state=42)

# Defining the model
model = keras.Sequential()
metrics = ["accuracy"]

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

# Find the best epoch for accuracy
best_epoch_acc = np.argmax(val_accuracy) + 1
best_acc = val_accuracy[best_epoch_acc - 1]

# Find the best epoch for loss
best_epoch_loss = np.argmin(val_loss) + 1
best_loss = val_loss[best_epoch_loss - 1]

print('Best epoch for accuracy:', best_epoch_acc)
print('Best accuracy:', best_acc)
print('Best epoch for loss:', best_epoch_loss)
print('Best loss:', best_loss)
print("Elapsed time: ", elapsed_time)
print("Time per epoch (Estimate): ", elapsed_time/epochs)
print('Final Training Loss:', train_loss[-1])
print('Final Validation Loss:', val_loss[-1])
print(model.summary())
print('CPU usage:', psutil.cpu_percent())
print('Memory usage:', psutil.virtual_memory().percent)
for layer in model.layers:
    print('Weights:', layer.get_weights())
# print('Learning rate for each epoch:', learning_rate_schedule)
# print('Time per epoch:', time_per_epoch)


# Plotting accuracy
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting loss
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
