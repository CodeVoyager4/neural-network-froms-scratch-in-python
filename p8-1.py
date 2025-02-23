# Neural Network Loss Calculation - Batch Processing
# This code shows how to calculate loss for multiple predictions at once

import numpy as np

# Example outputs from our neural network for 3 different inputs
softmax_outputs = np.array([[0.7, 0.1, 0.2],     # First prediction: 70% class 0, 10% class 1, 20% class 2
                           [0.1, 0.5, 0.4],     # Second prediction: 10% class 0, 50% class 1, 40% class 2
                           [0.02, 0.9, 0.08]])   # Third prediction: 2% class 0, 90% class 1, 8% class 2

# The correct answers (0 = first class, 1 = second class, 2 = third class)
class_targets = [0, 1, 1]  # First should be class 0, second and third should be class 1

# Get the predicted probability for the correct class of each input
# This finds the probability our network gave for the actual correct answer
print(softmax_outputs[[0, 1, 2], class_targets])

# Calculate the loss for each prediction
# Loss is negative log of the predicted probability for the correct class
print(-np.log(softmax_outputs[[0, 1, 2], class_targets]))