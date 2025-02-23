# Neural Network with Object-Oriented Design
# This code shows how to organize a neural network using classes

import numpy as np  # NumPy helps us do math operations efficiently

# Set a random seed so we get the same random numbers each time
np.random.seed(0)

# Our input data - three sets of 4 numbers each
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

# This class represents a layer of neurons in our network
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Create weights and biases for the layer
        # weights: slightly random numbers, scaled down by 0.10
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # biases: start at zero for each neuron
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        # Calculate outputs for all neurons in this layer
        self.output = np.dot(inputs, self.weights) + self.biases

# Create two layers:
# First layer: takes 4 inputs, produces 5 outputs
layer1 = Layer_Dense(4, 5)
# Second layer: takes 5 inputs (from layer1), produces 2 outputs
layer2 = Layer_Dense(5, 2)

# Process the data through both layers
layer1.forward(X)          # Feed input data through first layer
layer2.forward(layer1.output)  # Feed layer1's output through second layer

# Show the final output
print(layer2.output)















'''import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]

weights2 = [[0.1, -0.14, -0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases = [2, 3, 0.5]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)'''

