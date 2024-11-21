from pathlib import Path
import numpy as np
import pandas as pd

# Q4 Dirs and Flags
dir = Path(__file__).resolve().parent.parent
saveFlag = True
ec = True
ec_file = "spam_ham_ec.csv"
file = "spam_ham.csv"

def pre_process(matrix):
    # SVD
    centered_matrix = matrix - np.mean(matrix, axis=0)
    covariance = np.cov(centered_matrix, rowvar=False)
    eigenValues, eigenVectors = np.linalg.eigh(covariance)

    # Select top 50 PCs
    idx = eigenValues.argsort()[::-1]   
    eigenVectors = eigenVectors[:,idx]
    eigenValues = eigenValues[idx]

    top_pc = 50
    pca_eigenVectors = eigenVectors[:, :top_pc]
    pca_eigenValues = eigenValues[:top_pc]
    proj_pca = centered_matrix @ pca_eigenVectors

    inv_sqrt_eigenValues = np.diag(1.0 / np.sqrt(pca_eigenValues))
    whitened_matrix = proj_pca @ inv_sqrt_eigenValues

    return whitened_matrix

# Load Data
tokenized_data = pd.read_csv(dir / ec_file)

# Remove unnamed, and extract labels -> dim 2000
tokenized_data = tokenized_data.drop(columns=['Unnamed: 0'])
labels = tokenized_data.pop('cls').to_numpy()

processed_data = pre_process(tokenized_data)

# Split into train and validation
train4_2 = processed_data[0:3500]
test4_2 = processed_data[3500:4000]

mail = processed_data[4000:]

label_train = labels[0:3500]
label_test= labels[3500:4000]

if saveFlag & (not ec):
    np.savez("train4_2.npz", x = train4_2, y = label_train)
    np.savez("test4_2.npz",  x = test4_2, y = label_test)

if saveFlag & ec:
    np.savez("train4_2_ec.npz", x = train4_2, y = label_train)
    np.savez("test4_2_ec.npz",  x = test4_2, y = label_test)
    np.savez("mail_ec.npz",  x = mail)