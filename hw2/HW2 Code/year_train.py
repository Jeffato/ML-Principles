# Q 5.1 + 5.2
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Q5 Dirs and Flags
graph_dir = Path(__file__).resolve().parent.parent / "HW2_data/P5_data"

def pre_process(matrix):
    centered_matrix = matrix - np.mean(matrix, axis=0)
    covariance = np.cov(centered_matrix, rowvar=False)
    eigenValues, eigenVectors = np.linalg.eigh(covariance)

    inv_sqrt_eigenValues = np.diag(1.0 / np.sqrt(eigenValues))
    whitened_data = inv_sqrt_eigenValues @ eigenVectors.T @ centered_matrix.T
    whitened_matrix = whitened_data[(0,1), :].T

    return whitened_matrix

# def construct_data_matrix(matrix, polyDegree):
#     feature_1= np.column_stack([matrix[:, 0]**i for i in range(polyDegree)])
#     feature_2= np.column_stack([matrix[:, 1]**i for i in range(1, polyDegree)])
#     data_matrix = np.hstack((feature_1, feature_2))

#     return data_matrix

def construct_data_matrix(X, degree):
    n_samples, n_features = X.shape
    poly_matrix = np.ones((n_samples, (degree + 1) * n_features))

    for i in range(n_features):
        for j in range(1, degree + 1):
            poly_matrix[:, i * (degree + 1) + j] = X[:, i] ** j

    return poly_matrix

# Load Data
train_logit=np.load(graph_dir / "vgg16_train.npz", allow_pickle=True)["logit"]
train_year=np.load(graph_dir / "vgg16_train.npz", allow_pickle=True)["year"]
train_filename=np.load(graph_dir / "vgg16_train.npz", allow_pickle=True)["filename"] 

test_logit=np.load(graph_dir / "vgg16_test.npz", allow_pickle=True)["logit"]
test_year=np.load(graph_dir / "vgg16_test.npz", allow_pickle=True)["year"]
test_filename=np.load(graph_dir / "vgg16_test.npz", allow_pickle=True)["filename"] 

# Pre-process data
train_logit_processed = pre_process(train_logit)

# Split into train and validation
whitened_train_logit = train_logit_processed[:5000]
whitened_valid_logit = train_logit_processed[5000:]

sub_train_year = train_year[:5000]
sub_valid_year = train_year[5000:]

polyDegree = [i for i in range(2, 10)]
valid_mse_list = []

for degree in polyDegree:
    # Construct data matrix
    train_data_matrix = construct_data_matrix(whitened_train_logit, degree)
    validation_data_matrix = construct_data_matrix(whitened_valid_logit, degree)

    # Linear Regression- Normal Equation
    inverse = np.linalg.pinv(train_data_matrix.T @ train_data_matrix)
    weights = inverse @ train_data_matrix.T @ sub_train_year

    # Valid MSE
    valid_pred = validation_data_matrix @ weights
    valid_mse = np.mean((sub_valid_year - valid_pred) ** 2)

    # train_mse_list.append(valid_pred)
    valid_mse_list.append(valid_mse)

polyDegree = [i-1 for i in polyDegree]

plt.scatter(polyDegree, valid_mse_list, color='red', label='Valid')

# Graph info
plt.title('')
plt.legend()
plt.show()