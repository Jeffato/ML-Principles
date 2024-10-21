train_logit=np.load("vgg16_train.npz")["logit"]
train_year=np.load("vgg16_train.npz")["year"]
train_filename=np.load("vgg16_train.npz")["filename"] 

test_logit=np.load("vgg16_test.npz")["logit"]
test_year=np.load("vgg16_test.npz")["year"]
test_filename=np.load("vgg16_test.npz")["filename"] 

#maybe useful for problem 5.1

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=1148, vmax=2012)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
ax.scatter(pca_logit[0,:],pca_logit[1,:],year,c=year,s=2,picker=4)
