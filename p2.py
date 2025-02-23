# A Simple Neural Network Layer
# This code shows how multiple neurons work together in a layer

# Our starting data - 4 numbers that we want to process
inputs = [1, 2, 3, 2.5]

# Each neuron has its own set of weights
# These weights determine what patterns the neuron looks for
weights1 = [0.2, 0.8, -0.5, 1.0]    # Weights for first neuron
weights2 = [0.5, -0.91, 0.26, -0.5]  # Weights for second neuron
weights3 = [-0.26, -0.27, 0.17, 0.87] # Weights for third neuron

# Each neuron also has its own bias
# Bias helps adjust the neuron's sensitivity
bias1 = 2    # Bias for first neuron
bias2 = 3    # Bias for second neuron
bias3 = 0.5  # Bias for third neuron

# Calculate the output for all three neurons at once
# Each line below represents one neuron's calculation
output = [
    # First neuron: multiply each input by its weight, add them all up, then add bias
    inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
    # Second neuron: same process but with different weights and bias
    inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
    # Third neuron: same process but with different weights and bias
    inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3       
]

# Show the final output from all three neurons
print(f"Output: {output}")