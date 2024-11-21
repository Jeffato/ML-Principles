from pathlib import Path
import math
import numpy as np
import pandas as pd

# Dirs, Train from 3_1
p3_data = Path(__file__).resolve().parent.parent / "HW3_data/P3_data/data_1" 

class GDA_Classifier:
    def __init__(self, pos_prior, neg_prior):
        self.mean_pos = -0.0721922106722285
        self.mean_neg = 0.9401561132214228
        self.var_pos = 1.3096715040939155
        self.var_neg = 1.9437063405522659
        self.pos_prior = pos_prior
        self.neg_prior = neg_prior

    # Gaussian Distribution
    def gaussian_prob(self, x, mean, variance):
        return (math.exp(-((x - mean) ** 2) / (2 * variance)) / math.sqrt(2 * math.pi * variance))

    # Prediction Function
    def predict(self, x):
        pos = self.gaussian_prob(x, self.mean_pos, self.var_pos) * self.pos_prior
        neg = self.gaussian_prob(x, self.mean_neg, self.var_neg) * self.neg_prior

        if pos > neg : return 1
        else: return -1
    
    # Evaluation Function
    def evaluate(self, x, y):
        predictions = x.apply(lambda val: self.predict(val), axis=1)
        accuracy = np.mean(predictions == y['Result'])

        return accuracy

def main():
    # Load the dataset
    x_test = np.load(p3_data / "test.npz")["x"]
    y_test = np.load(p3_data / "test.npz")["y"]
    x_df = pd.DataFrame(x_test)
    y_df = pd.DataFrame(y_test, columns=['Result'])

    # Calculate priors
    prior_pos = x_df[y_df['Result'] == 1].shape[0] / len(x_df)
    prior_neg = x_df[y_df['Result'] == -1].shape[0] / len(x_df)
    
    # Construct Model
    classifier = GDA_Classifier(prior_pos, prior_neg)

    accuracy = classifier.evaluate(x_df, y_df)

    # 90.00%
    print(f"Model Accuracy on Test Set: {accuracy*100:.2f}%")

main()