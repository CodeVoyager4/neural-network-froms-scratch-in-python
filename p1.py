# This code demonstrates a single neuron in a neural network
# Each neuron processes multiple inputs to produce one output

# Input values - these could be features of our data
inputs = [1.2, 5.1, 2.1]

# Weights - each input has a corresponding weight that determines its importance
weights = [3.1, 2.1, 8.7]

# Bias - an additional value that helps the neuron learn patterns
bias = 3

# Calculate the neuron's output:
# 1. Multiply each input by its corresponding weight
# 2. Add up all these multiplied values
# 3. Add the bias to get the final result
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias

# Display the final output value
print(output)