import numpy as np

def Q1():
    # Q 1.1 
    data_matrix = np.array([
        [1, 4, 1, 1], 
        [1, 7, 0, 2], 
        [1, 10, 1, 3], 
        [1, 13, 0, 4]])

    y = np.array([16, 23, 36, 43]) 

    # Q 1.2
    print("Data Matrix Rank:", np.linalg.matrix_rank(data_matrix))
    print("Dim Y", y.shape[0])

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

Q1()