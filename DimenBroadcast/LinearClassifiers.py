from sklearn.datasets import make_blobs
import numpy as np 

from sklearn.preprocessing import OneHotEncoder

X_values, y_flat = make_blobs(n_features=2, n_samples=800, centers=3, random_state=500)
y=OneHotEncoder().fit_transform(y_flat.reshape(-1,1)).todense()
y=np.array()

from matplotlib import pyplot as plt 
 
plt.rcParams['figure.figsize']=(24,10)

plt.scatter(X_values[:,0],X_values[:,1],c=y_flat,alpha=0.4,s=150)