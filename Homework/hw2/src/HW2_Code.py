from PIL import Image
from pathlib import Path
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# Q3 Dirs and Flags
reg_dir = Path(__file__).resolve().parent / "regressionWeights"
load_regression = "regression"
load_ridge_reg =  "ridge_regression/ridge_lambda0.0003.txt"
dir = Path(__file__).resolve().parent / "HW2_data/P3_data"

def Q3():
    weight_list = []

    # Average reg lambda
    for file in os.listdir(reg_dir / load_regression):
        file_path = os.path.join(reg_dir / load_regression, file)
        weights = np.loadtxt(file_path)
        weight_list.append(weights)

    weight_list = np.array(weight_list)
    merged_weights = np.mean(weight_list, axis=0)

    # Optimal ridge reg lambda
    weights_ridge = np.loadtxt(reg_dir / load_ridge_reg)

    # Plot Check
    # Line of Best Fit
    x_vals_reg = np.linspace(0, 1, 200)
    y_vals_reg = np.polyval(merged_weights[::-1], x_vals_reg)
    plt.plot(x_vals_reg, y_vals_reg, color='red', label='No Regularization', linewidth=2)

    x_vals_ridge_reg = np.linspace(0, 1, 200)
    y_vals_ridge_reg = np.polyval(weights_ridge[::-1], x_vals_ridge_reg)
    plt.plot(x_vals_ridge_reg, y_vals_ridge_reg, color='blue', label='Regularization', linewidth=2)

    # Graph info
    plt.title('Comparison between regularized and non-reg models')
    plt.legend()
    # plt.show()

    # Test_set
    test_x = np.load(dir / "test.npz")["x"]
    test_y = np.load(dir / "test.npz")["y"]   

    test_data_matrix = np.column_stack([test_x**i for i in range(10)])

    predictions_reg = test_data_matrix @ merged_weights
    predictions_ridge_reg = test_data_matrix @ weights_ridge

    mse_reg = np.mean((test_y - predictions_reg) ** 2)
    mse_ridge_reg = np.mean((test_y - predictions_ridge_reg) ** 2)

    print(f"MSE for no Reg: {mse_reg}")
    print(f"MSE for Reg: {mse_ridge_reg}")

# Q4 Dirs and Flags
train_dir = Path(__file__).resolve().parent.parent / "data/P4_data/train"
test_dir  = Path(__file__).resolve().parent.parent / "data/P4_data/test"
save_dir = Path(__file__).resolve().parent / "EigenFaces"

saveFlag = False
genTestImagesFlag = True
displayEigenFlag= False
display_faces = False

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

    # 6. Model Test Image (4.2)
    if genTestImagesFlag:
        print("Generating Test Images...")

        # Load image- Iterate through Train folder
        for file, fileNum in zip(os.listdir(test_dir), range(0,10)):
            curr_test_image = Image.open(os.path.join(test_dir, file))
            image = np.asarray(curr_test_image)
            test_image = np.array(image.flatten())
            curr_test_image.show()

            M = [2, 10, 100, 1000, 4000]

            for num_components in M:
                m_eigen_vec = eigenVectors[:, 0:num_components]
                center_image = test_image - mean_image
                est_image = mean_image + m_eigen_vec @ m_eigen_vec.T @ center_image

                if display_faces:
                    constructed_img = Image.fromarray(est_image.reshape(60,80)).convert("L")
                    print("Displaying image with M=", num_components)
                    constructed_img.show()

                # Save the reconstructed image
                if saveFlag:
                    output_filename = f"pic{fileNum}_M{num_components}.png"
                    constructed_img.save(save_dir / output_filename)
                    print(f"Image saved as {output_filename} with M={num_components}")
            
        print("Done!")

    # 7. Display eigenvectors (4.3)
    if displayEigenFlag:
        print("Displaying Greatest Eigenvectors...")
        for i in range(10):
            # Process eigenvector
            curr_eigen_vec = eigenVectors[:, i]
            color_eigen =(curr_eigen_vec - np.min(curr_eigen_vec)) / (np.max(curr_eigen_vec) - np.min(curr_eigen_vec)) * 256
            reshaped_eigen = color_eigen.reshape(60,80)
            reshaped_eigen = np.uint8(reshaped_eigen)

            # Display Image
            constructed_img = Image.fromarray(reshaped_eigen)
            print("Displaying Eigenvector", i)
            constructed_img.show()

            # Save the reconstructed image
            if saveFlag:
                output_filename = f"eigenvector{i}.png"
                constructed_img.save(save_dir / output_filename)
                print(f"Image saved as {output_filename}")
    
        print("Done!")

# Q5- Graphing
# Q 5.1 + 5.2
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Q5 Dirs 
graph_dir = Path(__file__).resolve().parent.parent / "data/P5_data"

def PCA(matrix):
    # SVD
    centered_matrix = matrix - np.mean(matrix, axis=0)
    covariance = np.cov(centered_matrix, rowvar=False)
    eigenValues, eigenVectors = np.linalg.eigh(covariance)

    # Sort eigen val/vec to find principle components
    idx = eigenValues.argsort()[::-1]   
    pca_eigenVectors = eigenVectors[:,idx]
    pca_eigenValues = eigenValues[idx]

    # PCA
    pca_inv_sqrt_eigenValues = np.diag(1.0 / np.sqrt(pca_eigenValues))
    pca_matrix = pca_inv_sqrt_eigenValues @ pca_eigenVectors.T @ matrix.T

    return pca_matrix

def Q5():
    # Load Data
    train_logit=np.load(graph_dir / "vgg16_train.npz", allow_pickle=True)["logit"]
    train_year=np.load(graph_dir / "vgg16_train.npz", allow_pickle=True)["year"]

    # Generate PCA Graphs
    pca_logit = PCA(train_logit)

    # 2-D Graph
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=1148, vmax=2012)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax = plt.gca()) 
    ax.scatter(pca_logit[0,:], pca_logit[1,:], train_year, c=train_year, s=2,picker=4)
    plt.show()

    # 1-D Graph
    fig, ax = plt.subplots()
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=1148, vmax=2012)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.scatter(pca_logit[0, :], train_year, c=train_year, cmap=cmap, s=2)
    plt.show()

# if __name__ == '__main__':
#     Q1()