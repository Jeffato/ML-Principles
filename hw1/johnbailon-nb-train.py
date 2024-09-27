from pathlib import Path
import numpy as np
import pandas as pd

# File path constants
dir = 'DiabetesData'
readFileName = 'train.csv'
writeFileName = 'weights'

# Load the dataset
with open(Path(__file__).resolve().parent / dir / readFileName, 'r') as file: df = pd.read_csv(file)

# Partition data by variable and disease state (ie glucose/bp and diabetes/no diabetes)
glucose_pos = df[df['diabetes'] == 1]['glucose']
glucose_neg = df[df['diabetes'] == 0]['glucose']
bp_pos = df[df['diabetes'] == 1]['bloodpressure']
bp_neg = df[df['diabetes'] == 0]['bloodpressure']

# Find the mean for glucose and bp 
mean_glu_pos = glucose_pos.mean()
mean_glu_neg = glucose_neg.mean()
mean_bp_pos = bp_pos.mean()
mean_bp_neg = bp_neg.mean()

# Find the variance for glucose and bp 
var_glu_pos = glucose_pos.var()
var_glu_neg = glucose_neg.var()
var_bp_pos = bp_pos.var()
var_bp_neg = bp_neg.var()

# Priors
prior_pos = df[df['diabetes'] == 1].shape[0] / len(df)
prior_neg = df[df['diabetes'] == 0].shape[0] / len(df)

# Output
print(f'Mean D+:{mean_glu_pos} Var D+:{var_glu_pos}')
print(f'Mean D-:{mean_glu_neg} Var D-:{var_glu_neg}')
print(f'Mean BP+:{mean_bp_pos} Var BP+:{var_bp_pos}')
print(f'Mean BP-:{mean_bp_neg} Var BP-:{var_bp_neg}')
print(f'Diabetes+:{prior_pos} Diabetes-:{prior_neg}')

output = [
    mean_glu_pos, var_glu_pos, 
    mean_glu_neg, var_glu_neg,
    mean_bp_pos, var_bp_pos, 
    mean_bp_neg, var_bp_neg,
    prior_pos, prior_neg
] 

# Save to txt file
np.savetxt(writeFileName, output)