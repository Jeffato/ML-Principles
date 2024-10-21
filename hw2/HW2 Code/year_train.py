# Q 5.2
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Q3 Dirs and Flags
dir = Path(__file__).resolve().parent.parent / "HW2_data/P5_data"

# Load Data
train_logit=np.load(dir / "vgg16_train.npz", allow_pickle=True)["logit"]
train_year=np.load(dir / "vgg16_train.npz", allow_pickle=True)["year"]
train_filename=np.load(dir / "vgg16_train.npz", allow_pickle=True)["filename"] 

test_logit=np.load(dir / "vgg16_test.npz", allow_pickle=True)["logit"]
test_year=np.load(dir / "vgg16_test.npz", allow_pickle=True)["year"]
test_filename=np.load(dir / "vgg16_test.npz", allow_pickle=True)["filename"] 

# Standardize

# Graph info
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=1148, vmax=2012)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax = plt.gca()) 
# ax.scatter(pca_logit[0,:],pca_logit[1,:],year,c=year,s=2,picker=4)