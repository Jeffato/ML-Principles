# Q 5.1 + 5.2
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Q5 Dirs and Flags
graph_dir = Path(__file__).resolve().parent.parent / "HW2_data/P5_data"

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
train_filename=np.load(graph_dir / "vgg16_train.npz", allow_pickle=True)["filename"] 

# Pre-process data
train_logit_processed = pre_process(train_logit)

# Split into train and validation
whitened_train_logit = train_logit_processed[:5000]
whitened_valid_logit = train_logit_processed[5000:]

sub_train_year = train_year[:5000]
sub_valid_year = train_year[5000:]

polyDegree = [i for i in range(2, 9)]
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
    valid_mse_list.append(valid_mse)

polyDegree = [i-1 for i in polyDegree]

plt.scatter(polyDegree, valid_mse_list, color='red', label='Validation MSE')

# Graph info
plt.title('Polynomial Basis vs MSE')
plt.legend()
plt.show()