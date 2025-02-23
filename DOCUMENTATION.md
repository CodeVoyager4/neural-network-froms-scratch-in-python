# Neural Network From Scratch in Python
## Project Overview
This project implements a neural network from scratch using Python, NumPy, and Matplotlib. The implementation progresses from basic neural network concepts to more complex implementations.

## File Structure and Functionality

### Basic Neural Network Components
- **p1.py** (Lines 1-7): Demonstrates basic neuron computation with single output
  - Single neuron implementation
  - Basic weighted sum with bias

- **p2.py** (Lines 1-17): Expands to multiple neurons
  - Multiple weight sets and biases
  - Manual implementation of layer calculations

- **p3.py** (Lines 1-49): Introduces NumPy operations
  - Vectorized implementation using np.dot
  - More efficient layer calculations

### Advanced Implementation
- **p4.py** (Lines 1-58): Object-oriented implementation
  - Introduces Layer_Dense class
  - Implements forward propagation
  - Handles batch processing

- **p5.py** (Lines 1-26): Activation functions
  - Implements ReLU activation
  - Uses spiral dataset for testing
  - Structured layer organization

### Utility Files
- **versions.py**: Version checking utility
  - Displays Python version
  - Shows NumPy version
  - Shows Matplotlib version

- **cd.py** (Lines 1-11): Data generation utility
  - Creates synthetic data for training
  - Generates spiral dataset

## Key Components

### Layer_Dense Class
The dense layer implementation includes:
- Weight initialization using normalized random values
- Bias initialization
- Forward propagation method

### Activation_ReLU Class
Implements the Rectified Linear Unit activation function:
- Forward propagation method
- Non-linear activation

## Dependencies
- NumPy: For efficient matrix operations
- Matplotlib: For visualization
- NNFS: Neural Network From Scratch library for datasets

## Summary
This codebase demonstrates the progressive development of a neural network, starting from basic mathematical operations and evolving into a structured, object-oriented implementation. The code shows:

1. Basic neuron calculations
2. Layer implementations
3. Vectorized operations
4. Activation functions
5. Structured neural network architecture

The implementation focuses on educational purposes, showing each step in building a neural network from fundamental principles rather than using existing frameworks. 