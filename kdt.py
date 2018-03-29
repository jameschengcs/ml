import numpy as np
from sklearn.neighbors import KDTree 

np.random.seed(0)
X = np.random.random((10, 3))  # 10 points in 3 dimensions
tree = KDTree(X, leaf_size=2)              
dist, ind = tree.query([X[0]], k=3)                
print(ind)      # indices of 3 closest neighbors
print(dist)
