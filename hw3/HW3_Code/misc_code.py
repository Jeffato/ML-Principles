import numpy
import pandas as pd
import math

# Entropy of the Y-outcome label, in this case it is Play -> Y/N
def entropy(df):
    true_prior = df["Play"].mean()
    false_prior = 1 - true_prior
    bits = true_prior * math.log2(true_prior) + false_prior * math.log2(false_prior) 

    return -bits

def cond_entropy(df, feature_label):
    bits = 0
    feature_values = df[feature_label].unique()

    for something in somethings:
        conditional = 0
        for values in feature_values:
            print()

        bits += temp
            
    return -bits

# Q1
headers = ["Day", "Weather", "Temperature", "Humidity", "Wind", "Play"]
data = [
    [1, "Sunny", "Hot", "High", "Weak", False],
    [2, "Cloudy", "Hot", "High", "Weak", True],
    [3, "Sunny", "Mild", "Normal", "Strong", True],
    [4, "Cloudy", "Mild", "High", "Strong", True],
    [5, "Rainy", "Mild", "High", "Strong", False],
    [6, "Rainy", "Cool", "Normal", "Strong", False],
    [7, "Rainy", "Mild", "High", "Weak", True],
    [8, "Sunny", "Hot", "High", "Strong", False],
    [9, "Cloudy", "Hot", "Normal", "Weak", True],
    [10, "Sunny", "Mild", "High", "Strong", False],
]
main_df = pd.DataFrame(data, columns=headers) 

print(entropy(main_df))