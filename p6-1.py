# Neural Network with Softmax Activation
# This code shows how to convert neuron outputs into probabilities

import numpy as np
import nnfs  # Helper library for consistent results
nnfs.init()

# Sample outputs from a layer of 3 neurons, for 3 different inputs
layer_outputs = [[4.8, 1.21, 2.385],     # First input's outputs
                 [8.9, -1.81, 0.2],      # Second input's outputs
                 [1.41, 1.051, 0.026]]   # Third input's outputs

# Convert the outputs to exponential values (e^x)
exp_values = np.exp(layer_outputs)

# Normalize the values to get probabilities:
# Each value is divided by the sum of all values in its row
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)  # Show the final probabilities

'''
# Here's the old way of doing it (commented out for reference):
layer_outputs = [4.8, 1.21, 2.385]  # Single input example
E = math.e

# Calculate exponential values
exp_values = []
for output in layer_outputs:
    exp_values.append(E**output)

# Normalize values
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
'''