# Neural Network Layer Using NumPy
# This code shows a more efficient way to calculate multiple neuron outputs

import numpy as np  # NumPy helps us do math operations more efficiently

# Our input data - 4 numbers we want to process
inputs = [1, 2, 3, 2.5]

# Three sets of weights, one for each neuron
# Each neuron looks at all 4 inputs
weights1 = [0.2, 0.8, -0.5, 1.0]     # First neuron's weights
weights2 = [0.5, -0.91, 0.26, -0.5]  # Second neuron's weights
weights3 = [-0.26, -0.27, 0.17, 0.87] # Third neuron's weights

# Combine all weights into a single list
weights = [weights1, weights2, weights3]

# Each neuron has its own bias value
bias1 = 2    # First neuron's bias
bias2 = 3    # Second neuron's bias
bias3 = 0.5  # Third neuron's bias

# Combine all biases into a single list
biases = [bias1, bias2, bias3]

# Calculate all neuron outputs at once using NumPy
# This replaces the manual multiplication and addition from p2.py
output = np.dot(weights, inputs) + biases

# Show the final output from all three neurons
print(output)