import numpy as np
import random
import mnist

# Neural Network Class
class NeuralNetwork:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Randomly initializing weights and biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, a):
        """Return the output of the network if `a` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        """Return a tuple `(nabla_b, nabla_w)` representing the gradient for the cost function."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activations = [x]  # List to store all activations layer by layer
        zs = []  # List to store all z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                accuracy = self.evaluate(test_data)
                print(f"Epoch {epoch}: {accuracy} / {n_test}")
            else:
                print(f"Epoch {epoch} complete")

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives ∂C/∂a for the output activations."""
        return output_activations - y

# Helper Functions

# Convert labels into one-hot encoding
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Load MNIST Dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Preprocess the data: Normalize and flatten the images
train_images = train_images.reshape((train_images.shape[0], 28*28, 1)) / 255.0
test_images = test_images.reshape((test_images.shape[0], 28*28, 1)) / 255.0

# Convert labels to one-hot encoding
train_labels = [vectorized_result(y) for y in train_labels]
test_labels = [vectorized_result(y) for y in test_labels]

# Combine images and labels into tuples
training_data = list(zip(train_images, train_labels))
test_data = list(zip(test_images, test_labels))

# Initialize the network with 784 input neurons, 30 hidden neurons, and 10 output neurons.
net = NeuralNetwork([784, 30, 10])

# Train the network using MNIST data
net.train(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

# Evaluate performance after training
accuracy = net.evaluate(test_data)
print(f"Final Accuracy: {accuracy} / {len(test_data)}")