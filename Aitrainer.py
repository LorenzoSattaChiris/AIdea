import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

def create_model(layer_sizes, activation, optimizer):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    for size in layer_sizes:
        model.add(keras.layers.Dense(size, activation=activation))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Define the search space
layer_sizes_options = [[128], [256], [128, 64], [256, 128]]
activations_options = ["relu", "tanh"]
optimizer_options = ["adam", "sgd"]
epochs = 1
batch_size = 32

# Define the number of trials
num_trials = 20

best_accuracy = 0
best_hyperparameters = None

for _ in range(num_trials):
    # Randomly select hyperparameters
    layer_sizes = random.choice(layer_sizes_options)
    activation = random.choice(activations_options)
    optimizer = random.choice(optimizer_options)
    
    # Create and train the model
    model = create_model(layer_sizes, activation, optimizer)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=0)
    
    # Evaluate the model
    val_accuracy = max(history.history['val_accuracy'])
    
    # Update the best model if current model is better
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_hyperparameters = (layer_sizes, activation, optimizer)

print("Best Hyperparameters:", best_hyperparameters)
print("Best Validation Accuracy:", best_accuracy)
