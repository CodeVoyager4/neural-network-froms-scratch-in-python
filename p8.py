# Complete Neural Network with Loss Calculation
# This code combines all components into a working neural network that can measure its performance

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Dense Layer - Processes inputs using weights and biases
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # Random starting weights
        self.biases = np.zeros((1, n_neurons))                      # Start biases at zero
    
    def forward(self, inputs):
        # Calculate output values
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU Activation - Converts negative numbers to zero
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)  # Keep positive values, zero out negatives

# Softmax Activation - Converts outputs to probabilities
class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities (subtract max for numerical stability)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them into probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Loss Calculation - Measures how well the network is performing
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Categorical Cross-Entropy Loss - Specific type of loss for classification
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Handle both one-hot encoded and sparse labels
        if len(y_true.shape) == 1:  # Sparse labels
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  # One-hot encoded labels
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Calculate negative log likelihoods (the loss)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Create dataset
X, y = spiral_data(samples=100, classes=3)  # Get spiral data with 100 points, 3 classes

# Create network layers
dense1 = Layer_Dense(2, 3)          # First layer: 2 inputs -> 3 neurons
activation1 = Activation_ReLU()      # ReLU activation for first layer
dense2 = Layer_Dense(3, 3)          # Second layer: 3 inputs -> 3 neurons
activation2 = Activation_Softmax()   # Softmax activation for output layer

# Forward pass through the network
dense1.forward(X)                    # First layer
activation1.forward(dense1.output)   # First activation
dense2.forward(activation1.output)   # Second layer
activation2.forward(dense2.output)   # Final activation

# Calculate loss
loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print(activation2.output[:5])  # Show first 5 predictions
print(f'Loss: {loss}')        # Show how well we're doing