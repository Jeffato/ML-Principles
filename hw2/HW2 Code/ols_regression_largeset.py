from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

polyDegree = 10

# Train- 100 data points
dir = Path(__file__).resolve().parent.parent / "HW2_data/P3_data"

# Load test/train data sets
train_x_100=np.load(dir / "train_100.npz")["x"]
train_y_100=np.load(dir / "train_100.npz")["y"]

# Create Data matrix with columns 1, x, x^2 .. x^9
train_data_matrix = np.column_stack([train_x_100**i for i in range(polyDegree)])

# Linear Regression- Normal Equation
inverse = np.linalg.pinv(train_data_matrix.T @ train_data_matrix)
weights = inverse @ train_data_matrix.T @ train_y_100

# Plot Check
# Data points
plt.scatter(train_x_100, train_y_100, color='blue', label='Data Points')

# Line of Best Fit
x_vals = np.linspace(0, 1, 200)
y_vals = np.polyval(weights[::-1], x_vals)
plt.plot(x_vals, y_vals, color='red', label='Best Fit Line', linewidth=2)

# Graph info
plt.title('Data Points and Polynomial Regression Line')
plt.legend()
plt.show()