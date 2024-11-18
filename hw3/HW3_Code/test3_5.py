from pathlib import Path
import math
import numpy as np
import pandas as pd

# Dirs, Train from 3_2
p3_data = Path(__file__).resolve().parent.parent / "HW3_data/P3_data/data_2" 

class GDA_Classifier:
    def __init__(self):
        self.mean_pos = np.array([0.013075402904720767, 0.06295251252263563])
        self.mean_neg = np.array([-0.023139418380441288, -0.021149522420830142])
        self.cov_pos = np.array([[0.98285498, 0.00612046], [0.00612046, 1.05782804]])
        self.cov_neg =  np.array([[ 1.00329037, -0.01142356], [-0.01142356, 4.97693356]])
        self.pos_prior = 0.5
        self.neg_prior = 0.5

    # Gaussian Distribution
    def mult_gaussian_prob(self, x, mean, cov):
        exponent = math.exp(-0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean)))
        normalization = math.sqrt(2 * math.pi * np.linalg.det(cov))

        return normalization * exponent
    
    def multivariate_gaussian_likelihood(x, mu, cov):
        d = len(mu)
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        diff = x - mu
        exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        normalization = 1 / (np.sqrt((2 * np.pi) ** d * cov_det))
        likelihood = normalization * np.exp(exponent)
        return likelihood

    # Prediction Function
    def predict(self, x):
        pos = self.mult_gaussian_prob(x, self.mean_pos, self.cov_pos) * self.pos_prior
        neg = self.mult_gaussian_prob(x, self.mean_neg, self.cov_neg) * self.neg_prior

        if pos > neg : return 1
        else: return -1
    
    # Evaluation Function
    def evaluate(self, x, y):
        predictions = x.apply(lambda val: self.predict(val), axis=1)
        accuracy = np.mean(predictions == y['Result'])

        return accuracy

def main():
    # Construct Model
    classifier = GDA_Classifier()

    # Load the dataset
    x_test = np.load(p3_data / "test.npz")["x"]
    y_test = np.load(p3_data / "test.npz")["y"]
    x_df = pd.DataFrame(x_test)
    y_df = pd.DataFrame(y_test, columns=['Result'])

    accuracy = classifier.evaluate(x_df, y_df)

    print(f"Model Accuracy on Test Set: {accuracy*100:.2f}%")

main()