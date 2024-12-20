# Q 3.2
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

polyDegree = 10
saveFlag = False

save_dir = Path(__file__).resolve().parent.parent / "regressionWeights/ridge_regression"

# Load test/train data sets
dir = Path(__file__).resolve().parent.parent / "HW2_data/P3_data"

train_x=np.load(dir / "train.npz")["x"]
train_y=np.load(dir / "train.npz")["y"]

test_x=np.load(dir / "test.npz")["x"]
test_y=np.load(dir / "test.npz")["y"]

# If time, code out KFolds
kf = KFold(n_splits=5, shuffle=True, random_state=5)
# regularization_params = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.0003, 0.001, 0.01, 0.5, 1]
# regularization_params = [i for i in range(1, 100, 1)]
regularization_params = [0.0001, 0.0003, 0.001, 0.01, 0.5, 1]
regularization_params = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.00011]

minScore = 999999
minReg = 0

avg_mse_scores = []

for reg in regularization_params:
    mse_scores = []
    # reg = reg / 10000

    for train_index, test_index in kf.split(train_x):
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]

        # Create Data matrix with columns 1, x, x^2 .. x^9
        train_data_matrix = np.column_stack([X_train**i for i in range(polyDegree)])
        test_data_matrix = np.column_stack([X_test**i for i in range(polyDegree)])

        # Linear Regression- Normal Equation w/ ridge regression
        inner_product = train_data_matrix.T @ train_data_matrix + reg * np.eye(10)
        inverse = np.linalg.pinv(inner_product)
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
            np.savetxt(save_dir / f"ridge_lambda{reg}", weights)

    average_mse = np.mean(mse_scores)
    print(f'5-Fold Cross-Validation MSE: {average_mse:.4f}, lambda = {reg}')

    avg_mse_scores.append(average_mse)

    if minScore > average_mse:
        minScore = average_mse
        minReg = reg

print(f'Optimal lambda is {minReg} with a MMSE of {minScore}')

# Graph info
plt.scatter(regularization_params, avg_mse_scores, color='blue')
plt.title('Regularization Parameters v. MSE Score')
plt.xlabel("Regularization Parameters")
plt.ylabel("MSE Scores")
plt.legend()
plt.show()