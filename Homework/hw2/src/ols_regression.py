# Q 3.1
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

polyDegree = 10
saveFlag = False

# Load test/train data sets
dir = Path(__file__).resolve().parent.parent / "data/P3_data"
save_dir = Path(__file__).resolve().parent / "regressionWeights/regression"

train_x=np.load(dir / "train.npz")["x"]
train_y=np.load(dir / "train.npz")["y"]

test_x=np.load(dir / "test.npz")["x"]
test_y=np.load(dir / "test.npz")["y"]

# If time, code out KFolds
kf = KFold(n_splits=5, shuffle=True, random_state=5)

mse_scores = []
count = 0

# Start Training
for train_index, test_index in kf.split(train_x):
    X_train, X_test = train_x[train_index], train_x[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]

    # Create Data matrix with columns 1, x, x^2 .. x^9
    train_data_matrix = np.column_stack([X_train**i for i in range(polyDegree)])
    test_data_matrix = np.column_stack([X_test**i for i in range(polyDegree)])

    # Linear Regression- Normal Equation
    inverse = np.linalg.pinv(train_data_matrix.T @ train_data_matrix)
    weights = inverse @ train_data_matrix.T @ y_train

    predictions = test_data_matrix @ weights

    fold_mse = np.mean((y_test - predictions) ** 2)
    mse_scores.append(fold_mse)
    
    # # Plot Check
    # # Data points
    # plt.scatter(train_x, train_y, color='blue', label='Data Points')

    # # Line of Best Fit
    # x_vals = np.linspace(0, 1, 200)
    # y_vals = np.polyval(weights[::-1], x_vals)
    # plt.plot(x_vals, y_vals, color='red', label='Best Fit Line', linewidth=2)

    # # Graph info
    # plt.title('Data Points and Polynomial Regression Line')
    # plt.legend()
    # plt.show()
    
    if saveFlag:
        np.savetxt(save_dir / f"Fold:{count}", weights)
        count = count + 1

# Average MSE across all folds
average_mse = np.mean(mse_scores)

print(f'5-Fold Cross-Validation MSE: {average_mse:.4f}')