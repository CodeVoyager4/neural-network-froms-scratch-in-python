# Neural Network with Activation Functions
# This code adds non-linear processing to our network using ReLU activation

import numpy as np
import nnfs  # Helper library for neural network examples
from nnfs.datasets import spiral_data  # Provides sample data for testing

# Initialize settings for consistent results
nnfs.init()

# Get sample spiral data: X contains points, y contains categories
X, y = spiral_data(100, 3)  # Creates 100 points in 3 categories

# Layer of neurons - same as before but organized in a class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Create weights and biases
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU Activation Function - adds non-linearity to the network
class Activation_ReLU:
    def forward(self, inputs):
        # Calculate output values from inputs
        # ReLU returns x if x > 0, or 0 if x <= 0
        self.output = np.maximum(0, inputs)

# Create network structure
layer1 = Layer_Dense(2, 5)        # First layer: 2 inputs -> 5 neurons
activation1 = Activation_ReLU()    # Activation function for first layer

# Process the data
layer1.forward(X)                  # Pass data through first layer
activation1.forward(layer1.output) # Pass result through activation function

# Show the final output
print(activation1.output)
