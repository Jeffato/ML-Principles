from PIL import Image
from pathlib import Path

import numpy as np
import os

# Q1- MMSE Regression
def Q1():
    # Q 1.1 
    data_matrix = np.array([
        [1, 4, 1, 1], 
        [1, 7, 0, 2], 
        [1, 10, 1, 3], 
        [1, 13, 0, 4]])

    y = np.array([16, 23, 36, 43]) 

    # Q 1.2

    # Since Augmented rank is < n -> overdetermined. Same hyperplane
    augmented_matrix = np.array([
        [1, 4, 1, 1, 16], 
        [1, 7, 0, 2, 23], 
        [1, 10, 1, 3, 36], 
        [1, 13, 0, 4, 43]])

    print("Augmented Rank:", np.linalg.matrix_rank(augmented_matrix))

    # Q 1.3

    # Since data matrix is singular, must use pseudo inverse
    dm_pseudo_inv = np.linalg.pinv(data_matrix.T @ data_matrix)
    weights = dm_pseudo_inv @ data_matrix.T @ y
    print(weights)

    # Q 1.5
    data_matrix_added = np.array([
        [1, 4, 1, 1], 
        [1, 7, 0, 2], 
        [1, 10, 1, 3], 
        [1, 13, 0, 4],
        [1, 16, 1, 5],
        [1, 19, 0, 6],
        [1, 22, 1, 7],
        [1, 25, 0, 8],])

    y_added = np.array([16, 23, 36, 43, 56.04, 62.77, 76.04, 82.96])

    dm_added_pseudo_inv = np.linalg.pinv(data_matrix_added.T @ data_matrix_added)
    weights_added = dm_added_pseudo_inv @ data_matrix_added.T @ y_added
    print(weights_added)

    # Q 1.6
    for i in range(data_matrix.shape[1]):
        X_reduced = np.delete(data_matrix, i, axis=1)
        reduced_rank = np.linalg.matrix_rank(X_reduced)
        print(f"Rank without column {i}:", reduced_rank)

train_dir = Path(__file__).resolve().parent.parent / "HW2_data/P4_data/train"
test_dir  = Path(__file__).resolve().parent.parent / "HW2_data/P4_data/test"

# Q4- EigenFace
def Q4():
    # 1. Load Images- Create an data matrix with each row an image from the training set
    print("Loading Training Images...")
    images = []

    # Iterate through Train folder
    for file in os.listdir(train_dir):
        image = np.asarray(Image.open(os.path.join(train_dir, file)))
        images.append(image.flatten())

    images = np.array(images)

    # 2. Compute E[X] - Average value of each column
    print("Starting calculations...")
    mean_image = np.mean(images, axis=0)

    # 3. Compute COV(X,X)
    covariance = np.cov(images, rowvar=False)

    # 4. Spectral Decomposition of COV(X,X)
    eigenValues, eigenVectors = np.linalg.eigh(covariance)

    # 5. Sort Eigenvectors by largest Eigenvalues (Will need M many for modeling)
    print("Sorting eigenVectors...")
    idx = eigenValues.argsort()[::-1]   
    eigenVectors = eigenVectors[:,idx]

    # 6. Model Test Image
    print("Generating Test Images...")

    # Load image- Iterate through Train folder
    for file in os.listdir(test_dir):
        image = np.asarray(Image.open(os.path.join(test_dir, file)))
        test_image = np.array(image.flatten())

        M = [2, 10, 100, 1000, 4000]

        for num_components in M:
            m_eigen_vec = eigenVectors[:, 0:num_components]
            center_image = test_image - mean_image
            est_image = mean_image + m_eigen_vec @ m_eigen_vec.T @ center_image

            # Convert the array to a PIL Image object and display
            constructed_img = Image.fromarray(est_image.reshape(60,80))

            print("Displaying image with M=", num_components)
            constructed_img.show()

Q4()