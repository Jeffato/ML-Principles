from pathlib import Path
import numpy as np
import pandas as pd
import math

# File path constants
dir = 'data\DiabetesData'
testDataFile = 'test.csv'
learnedWeights = 'weights.txt'

class NaiveBayes:
    def __init__(self, weights):
        self.mean_glu_pos = weights[0]
        self.var_glu_pos = weights[1]
        self.mean_glu_neg = weights[2]
        self.var_glu_neg = weights[3]
        self.mean_bp_pos = weights[4]
        self.var_bp_pos = weights[5]
        self.mean_bp_neg = weights[6]
        self.var_bp_neg = weights[7]
        self.prior_pos = weights[8]
        self.prior_neg = weights[9]
    
    # Helper function to find factor probability using Gaussian densities
    # Formula 1.18 in Murphy Textbook
    # Input x: factor data point, mean: factor mean, variance: factor variance (mean/variance found from training)
    def gaussian_prob(self, x, mean, variance):
        return (math.exp(-((x - mean) ** 2) / (2 * variance)) / math.sqrt(2 * math.pi * variance))

    # Prediction Function
    def predict(self, glu, bp):
        # Probability D+: Multiply glucose and bp probabilities
        p_glu_pos = self.gaussian_prob(glu, self.mean_glu_pos, self.var_glu_pos)
        p_bp_pos = self.gaussian_prob(bp, self.mean_bp_pos, self.var_bp_pos)
        p_diabetes_pos = p_glu_pos * p_bp_pos * self.prior_pos

        # Probability D-: Multiply glucose and bp probabilities
        p_glu_neg = self.gaussian_prob(glu, self.mean_glu_neg, self.var_glu_neg)
        p_bp_neg = self.gaussian_prob(bp, self.mean_bp_neg, self.var_bp_neg)
        p_diabetes_neg = p_glu_neg * p_bp_neg * self.prior_neg

        if p_diabetes_pos > p_diabetes_neg : return 1
        else: return 0

    # Evaluation Function
    def evaluate(self, x, y):
        predictions = x.apply(lambda row: self.predict(row['glucose'], row['bloodpressure']), axis=1)
        accuracy = np.mean(predictions == y)
        # for x_val, y_val in zip(predictions, y): print(x_val == y_val)
        return accuracy

# Load Weights from a given file.
# Input: file name
# Output: Array of weights mean_glu_pos, var_glu_pos, mean_glu_neg, var_glu_neg, mean_bp_pos, var_bp_pos, mean_bp_neg, var_bp_neg
def load_Weights(filepath):
    with open(Path(__file__).resolve().parent.parent / dir / filepath, 'r') as file: 
        data = [line.rstrip() for line in file]
        return data
    
def main():
    # Construct model
    weights = load_Weights(learnedWeights)
    classifier = NaiveBayes(np.array(weights).astype(np.float64))

    # Load test data
    with open(Path(__file__).resolve().parent.parent / dir / testDataFile, 'r') as file: 
        testData = pd.read_csv(file)

    # Evaluate the classifier using test data
    x_test = testData[['glucose', 'bloodpressure']]
    y_test = testData['diabetes']
    accuracy = classifier.evaluate(x_test, y_test)
    
    print(f"Model Accuracy on test set: {accuracy*100:.2f}%")

main()