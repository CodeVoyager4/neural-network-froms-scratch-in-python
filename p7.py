# Neural Network Loss Calculation
# This code shows how to measure how well our network is performing

import math  # For mathematical operations

# Example outputs from our neural network (probabilities)
softmax_output = [0.7, 0.1, 0.2]  # Network's prediction (70% first class, 10% second, 20% third)

# The actual correct answer (called "ground truth")
# [1, 0, 0] means the first class is correct
target_output = [1, 0, 0]

# Calculate how wrong our prediction was (the loss)
# We do this by:
# 1. Looking at the predicted probability for the correct answer
# 2. Taking the negative log of that number
# This gives us our loss value - lower is better
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)  # Show the loss value

# Simpler example: when we know which class is correct
loss = -math.log(softmax_output[0])  # If first class is correct
print(loss)

# Some example calculations to understand the math:
print(-math.log(0.7))  # Loss when we're 70% confident and correct
print(-math.log(0.5))  # Loss when we're 50% confident and correct