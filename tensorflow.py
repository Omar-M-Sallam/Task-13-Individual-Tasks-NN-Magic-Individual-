import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten the images
train_images = train_images.reshape((train_images.shape[0], 28*28)) / 255.0
test_images = test_images.reshape((test_images.shape[0], 28*28)) / 255.0

# Define the neural network model
model = models.Sequential()
model.add(layers.Dense(30, activation='sigmoid', input_shape=(28*28,)))  # Hidden layer
model.add(layers.Dense(10, activation='softmax'))  # Output layer for 10 classes

# Compile the model
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=30, batch_size=10, validation_data=(test_images, test_labels))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")
