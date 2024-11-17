from pathlib import Path
import math
import numpy as np
import pandas as pd

# Dirs, Train from 3_1
p3_data = Path(__file__).resolve().parent.parent / "HW3_data/P3_data/data_1" 

class GDA_Classifier:
    def __init__(self):
        self.mean_pos = -0.0721922106722285
        self.mean_neg = 0.9401561132214228
        self.var_pos = 1.3096715040939155
        self.var_neg = 1.9437063405522659

    # Gaussian Distribution
    def gaussian_prob(self, x, mean, variance):
        return (math.exp(-((x - mean) ** 2) / (2 * variance)) / math.sqrt(2 * math.pi * variance))

    # Prediction Function
    def predict(self, x):
        pos = self.gaussian_prob(x, self.mean_pos, self.var_pos)
        neg = self.gaussian_prob(x, self.mean_neg, self.var_neg)

        if pos > neg : return 1
        else: return -1
    
    # Evaluation Function
    def evaluate(self, x, y):
        predictions = x.apply(lambda val: self.predict(val), axis=1)

        print(predictions.shape)
        print(y.shape)

        predictions = predictions.reset_index(drop=True)
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

    # 61.50%
    print(f"Model Accuracy on Test Set: {accuracy*100:.2f}%")

main()