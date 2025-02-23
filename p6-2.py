# Complete Neural Network with Multiple Layers and Activations
# This code combines everything we've learned into a working neural network

import numpy as np
import nnfs
from nnfs.datasets import spiral_data  # Gets us sample data to test with

nnfs.init()  # Makes sure we get consistent results

# Layer of neurons - processes inputs using weights and biases
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Create weights and biases with good starting values
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation - converts negative numbers to zero
class Activation_ReLU:
    def forward(self, inputs):
        # Return input if > 0, otherwise return 0
        self.output = np.maximum(0, inputs)

# Softmax activation - converts outputs to probabilities
class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Create dataset
X, y = spiral_data(samples=100, classes=3)  # Creates 100 points in 3 categories

# Create neural network
dense1 = Layer_Dense(2, 3)         # First layer: 2 inputs -> 3 neurons
activation1 = Activation_ReLU()     # ReLU activation for first layer
dense2 = Layer_Dense(3, 3)         # Second layer: 3 inputs -> 3 neurons
activation2 = Activation_Softmax()  # Softmax activation for output layer

# Process data through the network
dense1.forward(X)                   # Pass data through first layer
activation1.forward(dense1.output)  # First activation
dense2.forward(activation1.output)  # Pass data through second layer
activation2.forward(dense2.output)  # Final activation

# Show first 5 probabilities
print(activation2.output[:5])