from pathlib import Path
import math
import numpy as np
import pandas as pd

# Dirs, Train from 3_2
p3_data = Path(__file__).resolve().parent.parent / "data/P3_data/data_2" 

class GDA_Classifier:
    def __init__(self):
        self.mean_pos = np.array([0, 0])
        self.cov_pos = np.eye(2)

        self.mean_1_neg = np.array([0, 2])
        self.cov_1_neg  = self.cov_pos = np.eye(2)

        self.mean_2_neg = np.array([0, -2])
        self.cov_2_neg  = self.cov_pos = np.eye(2)

        self.pos_prior = self.neg_prior = neg_weight = 0.5

    # Gaussian Distribution
    def mult_gaussian_prob(self, x, mean, cov):
        exponent = math.exp(-0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean)))
        normalization = 1 / math.sqrt(2 * math.pi * np.linalg.det(cov))
        
        return normalization * exponent

    # Prediction Function
    def predict(self, x):
        pos = self.mult_gaussian_prob(x, self.mean_pos, self.cov_pos) * self.pos_prior
        neg = (0.5 * self.mult_gaussian_prob(x, self.mean_1_neg, self.cov_1_neg) + 0.5 * self.mult_gaussian_prob(x, self.mean_2_neg, self.cov_2_neg) )* self.neg_prior

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