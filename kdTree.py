# KD Tree example
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KDTree 

iris = load_iris()
print(iris.data.shape)
print(iris.data[0])
tree = KDTree(iris.data, leaf_size=1)    

q = np.array([[4.0, 2.1, 5.1, 1.5]])         
dist, ind = tree.query(q, k=5)      
print(ind)      # indices of 3 closest neighbors
print(dist)

lst = []
for i in ind[0]:
    print(iris.data[i], iris.target[i])
    lst.append(iris.target[i])
    
target = max(lst,key=lst.count)    
print("Result: ", target, iris.target_names[target])    

