# 3 Different Ways of AI Model Improvements

# Initialization 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random
from bayes_opt import BayesianOptimization

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (Numpy Arrays)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)


def create_model(layer_sizes, activation, optimizer):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    for size in layer_sizes:
        model.add(keras.layers.Dense(size, activation=activation))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Define the search space
layer_sizes_options = [[128], [256], [128, 64], [256, 128]]
activations_options = ["relu", "tanh"]
optimizer_options = ["adam", "sgd"]
epochs = 1
batch_size = 32

# Model 1 - Random Search
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
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=0,
    )

    # Evaluate the model
    val_accuracy = max(history.history["val_accuracy"])

    # Update the best model if current model is better
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_hyperparameters = (layer_sizes, activation, optimizer)

print("Best Hyperparameters:", best_hyperparameters)
print("Best Validation Accuracy:", best_accuracy)

# Model 2 - Grid Search
grid_search_results = []

for layer_sizes in layer_sizes_options:
    for activation in activations_options:
        for optimizer in optimizer_options:
            # Create and train the model
            model = create_model(layer_sizes, activation, optimizer)
            history = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                verbose=0,
            )

            # Evaluate the model
            val_accuracy = max(history.history["val_accuracy"])

            # Store the results
            grid_search_results.append(
                {
                    "layer_sizes": layer_sizes,
                    "activation": activation,
                    "optimizer": optimizer,
                    "val_accuracy": val_accuracy,
                }
            )

# Find the best hyperparameters and accuracy
best_grid_search_result = max(grid_search_results, key=lambda x: x["val_accuracy"])
best_hyperparameters = (
    best_grid_search_result["layer_sizes"],
    best_grid_search_result["activation"],
    best_grid_search_result["optimizer"],
)
best_accuracy = best_grid_search_result["val_accuracy"]

print("Best Hyperparameters:", best_hyperparameters)
print("Best Validation Accuracy:", best_accuracy)

# Model 3 - Bayesian Optimization
# Define the search space for Bayesian Optimization
pbounds = {
    "layer1_size": (64, 256),
    "layer2_size": (32, 128),
    "activation": ["relu", "tanh"],
    "optimizer": ["adam", "sgd"],
}

# Define the function to optimize
def optimize_model(layer1_size, layer2_size, activation, optimizer):
    model = create_model([layer1_size, layer2_size], activation, optimizer)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=0,
    )
    val_accuracy = max(history.history["val_accuracy"])
    return val_accuracy

# Perform Bayesian Optimization
optimizer = BayesianOptimization(f=optimize_model, pbounds=pbounds)
optimizer.maximize(init_points=5, n_iter=15)

# Get the best hyperparameters and accuracy
best_hyperparameters = optimizer.max["params"]
best_accuracy = optimizer.max["target"]

print("Best Hyperparameters:", best_hyperparameters)
print("Best Validation Accuracy:", best_accuracy)
