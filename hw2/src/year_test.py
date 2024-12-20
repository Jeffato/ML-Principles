# Q 5.1 + 5.2
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Q5 Dirs and Flags
graph_dir = Path(__file__).resolve().parent.parent / "data/P5_data"

def pre_process(matrix):
    # SVD
    centered_matrix = matrix - np.mean(matrix, axis=0)
    covariance = np.cov(centered_matrix, rowvar=False)
    eigenValues, eigenVectors = np.linalg.eigh(covariance)

    # Select top 2 PCs
    idx = eigenValues.argsort()[::-1]   
    eigenVectors = eigenVectors[:,idx]
    eigenValues = eigenValues[idx]

    top_pc = 2
    pca_eigenVectors = eigenVectors[:, :top_pc]
    pca_eigenValues = eigenValues[:top_pc]
    proj_pca = centered_matrix @ pca_eigenVectors

    inv_sqrt_eigenValues = np.diag(1.0 / np.sqrt(pca_eigenValues))
    whitened_matrix = proj_pca @ inv_sqrt_eigenValues

    return whitened_matrix

def construct_data_matrix(matrix, polyDegree):
    feature_1= np.column_stack([matrix[:, 0]**i for i in range(polyDegree)])
    feature_2= np.column_stack([matrix[:, 1]**i for i in range(1, polyDegree)])
    data_matrix = np.hstack((feature_1, feature_2))

    return data_matrix

# Load Data
train_logit=np.load(graph_dir / "vgg16_train.npz", allow_pickle=True)["logit"]
train_year=np.load(graph_dir / "vgg16_train.npz", allow_pickle=True)["year"]

test_logit=np.load(graph_dir / "vgg16_test.npz", allow_pickle=True)["logit"]
test_year=np.load(graph_dir / "vgg16_test.npz", allow_pickle=True)["year"]
test_filename=np.load(graph_dir / "vgg16_test.npz", allow_pickle=True)["filename"] 

# Pre-process data
train_logit_processed = pre_process(train_logit)
test_logit_processed = pre_process(test_logit)

degree = 2

# Construct data matrix
train_data_matrix = construct_data_matrix(train_logit_processed, degree)
test_data_matrix = construct_data_matrix(test_logit_processed, degree)

# Linear Regression- Normal Equation
inverse = np.linalg.pinv(train_data_matrix.T @ train_data_matrix)
weights = inverse @ train_data_matrix.T @ train_year

# Test MSE
test_pred = test_data_matrix @ weights
test_mse = np.mean((test_year - test_pred) ** 2)

# Find smallest and biggest error
error = np.abs(test_year - test_pred)

smallest_idx = np.argmin(error)
largest_idx = np.argmax(error) 

print(f"Model MSE: {test_mse:.3f}")
print(f"Largest Error- Predicted: {test_pred[largest_idx]:.0f} Actual: {test_year[largest_idx]:.0f} Name: {test_filename[largest_idx]}")
print(f"Smallest Error- Predicted: {test_pred[smallest_idx]:.0f} Actual: {test_year[smallest_idx]:.0f} Name: {test_filename[smallest_idx]}")