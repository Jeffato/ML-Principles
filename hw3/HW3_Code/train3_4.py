from pathlib import Path
import numpy as np
import pandas as pd

# Dirs
p3_data = Path(__file__).resolve().parent.parent / "HW3_data/P3_data/data_2" 

# Load the dataset
train_x=np.load(p3_data / "train.npz")["x"]
train_y=np.load(p3_data / "train.npz")["y"]

x_df = pd.DataFrame(train_x)
y_df = pd.DataFrame(train_y, columns=['Result'])

# Partition Data
pos = x_df[y_df['Result'] == 1]
neg = x_df[y_df['Result'] == -1]

# Mean and Cov for pos/neg
mean_pos = pos.mean()
mean_neg = neg.mean()

#COV
cov_pos  = np.cov(pos, rowvar = False)
cov_neg = np.cov(neg, rowvar = False)

#Prior values: Going to be saved alongside mean and variance for 3.3
prior_pos = np.mean(train_y==1)
prior_neg = np.mean(train_y==-1)

print(f'Positive Set: Mean: Class 1: {mean_pos[0]}, Mean: Class 2: {mean_pos[1]} Covariance: {cov_pos}, Prior: {prior_pos}')
print(f'Negative Set: Mean: Class 1: {mean_neg[0]}, Mean: Class 2: {mean_neg[1]} Covariance: {cov_neg}, Prior: {prior_neg}')