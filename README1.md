# Neural Networks and Deep Learning (Chapter 1 Summary)

## Introduction
This project explores building a neural network from scratch, based on Chapter 1 of *Neural Networks and Deep Learning* by Michael Nielsen. The goal is to classify handwritten digits from the MNIST dataset.

## Key Learnings from Chapter 1

- **Neurons and Perceptrons**: Neural networks are built from neurons that process inputs and produce outputs. Perceptrons are simple neurons using step functions.
- **Network Structure**: The network consists of an input layer, hidden layers, and an output layer. Non-linear activation functions like **sigmoid** are crucial for solving complex problems.
- **Forward Propagation**: Data flows through the network layer by layer, with each neuron applying weights and activation functions to generate predictions.
- **Cost Function**: Measures how well the network’s predictions match the true labels. The **quadratic cost function** is introduced.
- **Gradient Descent**: Used to optimize the network by adjusting weights and biases in small steps, guided by the gradient of the cost function.
- **Learning Rate**: Controls the size of these steps, impacting how quickly or effectively the network learns.

## Implementation
- Neural network from scratch using Python/NumPy
- A single hidden layer with 30 neurons
- Training on the MNIST dataset