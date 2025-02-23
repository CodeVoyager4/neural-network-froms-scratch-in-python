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

### 2. Multiple Neurons Layer
- Expands to 4 inputs and 3 neurons
- Each neuron has its own weights and bias
- Manual matrix multiplication implementation

### 3. Vectorized Implementation
Reference: 

### 4. Dense Layer Class
Reference:

Key features:
- Weight initialization using normalized random values (0.10 * randn)
- Zero initialization for biases
- Forward propagation using dot product

### 5. ReLU Activation
Reference:

- Implements Rectified Linear Unit
- Used after dense layer for non-linearity

### 6. Softmax Implementation
Reference:

- Exponential normalization
- Probability distribution output

## Data Handling
- Uses spiral dataset for testing
- Supports batch processing
- Implements data generation utilities

## Dependencies
- NumPy: Matrix operations
- NNFS: Dataset generation
- Matplotlib: Visualization

## Usage Example

## Project Structure
- Progressive implementation from basic to advanced concepts
- Each file builds upon previous concepts
- Clear separation of layer and activation functions
- Object-oriented design for reusability

This implementation serves as an educational tool for understanding neural networks at a fundamental level, avoiding the abstraction of existing frameworks.

- NumPy implementation
- Efficient dot product operations
- Vectorized layer calculations

### 4. Object-Oriented Implementation (p4.py)
Reference:

Key features:
- Dense layer class
- Weight initialization: `0.10 * np.random.randn()`
- Bias initialization with zeros
- Forward propagation using dot product

### 5. Activation Functions (p6-2.py)
Reference:

- ReLU Activation: `max(0, x)`
- Softmax Activation for classification
- Proper numerical stability handling

### 6. Complete Network Architecture
Reference:

- Two-layer neural network
- ReLU activation in hidden layer
- Softmax activation in output layer
- Forward propagation pipeline

## Core Components

### Layer_Dense Class
Properties:
- Weights: Randomly initialized with scaling
- Biases: Zero initialization
- Forward method: Implements `output = inputs @ weights + biases`

### Activation Functions
1. ReLU (Rectified Linear Unit):
   - Forward method: `max(0, x)`
   - Used in hidden layers

2. Softmax:
   - Exponential normalization
   - Numerical stability with max subtraction
   - Probability distribution output

## Data Handling
- Uses spiral dataset for testing
- Supports batch processing
- Implements data generation utilities

## Dependencies
- NumPy: Matrix operations and efficient computations
- NNFS: Dataset generation and utilities
- Matplotlib: Visualization tools

## Usage Example