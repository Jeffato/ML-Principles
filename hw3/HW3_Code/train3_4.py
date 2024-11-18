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

print(pos.shape)
print(neg.shape)

# Mean and Variance for pos/neg
mean_pos = pos.mean().mean()
mean_neg = neg.mean().mean()

var_pos = pos.var().mean()
var_neg = neg.var().mean()

print(f'Positive Set: Mean {mean_pos}, Variance: {var_pos}')
print(f'Negative Set: Mean {mean_neg}, Variance: {var_neg}')