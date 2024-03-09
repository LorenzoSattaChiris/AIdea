// Select the button element
const button = document.querySelector('.submit');

// Add an event listener to the button
button.addEventListener('click', () => {
    // Code to be executed when the button is clicked
    // Code to be executed when the button is clicked
    code = `
    # Importing the required libraries
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    # Loading the dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Defining the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=32)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_accuracy)
    `;
});


