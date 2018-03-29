import numpy as np
from sklearn.neighbors import KDTree 

np.random.seed(0)
X = np.random.random((10, 3))  # 10 points in 3 dimensions
tree = KDTree(X, leaf_size=2)            
print(tree.query_radius([X[0]], r=0.3, count_only=True))
ind, dist = tree.query_radius([X[0]], 
                              r = 0.3, 
                              count_only = False, 
                              return_distance = True)
print(ind)
print(dist)