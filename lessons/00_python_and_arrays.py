"""Run: python lessons/00_python_and_arrays.py. Read, predict, then execute."""

import numpy as np

# A variable names a value. A list stores several values. A function packages
# a calculation so we can call it with different inputs.
pixels = [0, 128, 255]


def normalize(values):
    """Convert a Python list into an array and scale its values to [0, 1]."""
    return np.array(values, dtype=float) / 255


print("Normalized pixels:", normalize(pixels))
# Python indexing starts at 0. Shape is a tuple describing the array's axes.
X = np.array([[1., 2.], [3., 4.]])  # two examples, two features each
W = np.array([[0.1], [0.2]])        # two incoming weights, one neuron
b = np.array([0.5])                # an offset shared by both examples
print("X shape:", X.shape, "W shape:", W.shape)
print("Weighted sums X @ W + b:\n", X @ W + b)
# First row: 1*0.1 + 2*0.2 + 0.5 = 1.0. The second row is your exercise.
# @ is matrix multiplication; * is elementwise multiplication in NumPy.
# Try changing b. Which examples change? Try W.T: why does X @ W.T fail?
