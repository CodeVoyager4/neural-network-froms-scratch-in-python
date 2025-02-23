import numpy as np
from numpy import array
import matplotlib.pyplot as plt

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

print(output)

'''plt.plot(output, marker='o')
#plt.bar(range(len(output)), output)
plt.title("List viz")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()
'''
from scipy.optimize import curve_fit

# Your list (Y-values)
y_data = output

# Assume X-values (e.g., a linear range from -2 to 2, adjust as needed)
x_data = np.linspace(-2, 2, len(y_data))

# Define the sigmoid function
def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Initial parameter guesses: [max value, steepness, midpoint]
initial_guess = [max(y_data), 1, 0]

# Fit the sigmoid function to the data
popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=initial_guess, maxfev=5000)

# Extract fitted parameters
L_fit, k_fit, x0_fit = popt

# Generate smooth X-values for the fitted curve
x_smooth = np.linspace(min(x_data), max(x_data), 100)
y_fit = sigmoid(x_smooth, L_fit, k_fit, x0_fit)

# Plot
plt.scatter(x_data, y_data, color='blue', label='Data Points')  # Original data
plt.plot(x_smooth, y_fit, color='red', label=f'Sigmoid Fit (L={L_fit:.1f}, k={k_fit:.2f}, x0={x0_fit:.2f})')
plt.title("Fitting Sigmoid Activation Function")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Print fitted parameters
print(f"Fitted Parameters: L = {L_fit:.2f}, k = {k_fit:.2f}, x0 = {x0_fit:.2f}")
