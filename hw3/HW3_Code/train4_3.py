from pathlib import Path
import numpy as np

# Dirs
dir = Path(__file__).resolve().parent.parent
saveFlag = False

def neg_log_likelihood():
    pass

def cost_function():
    pass

def gradient_descent():
    pass

# Load Data
train_x = np.load("train4_2.npz")["x"]
train_y = np.load("train4_2.npz")["y"]

test_x = np.load("test4_2.npz")["x"]
test_y = np.load("test4_2.npz")["y"]

