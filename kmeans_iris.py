import numpy as np
from numpy import linalg as LA
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
for nK in range(1, 6):
    kmeans = KMeans(n_clusters=nK, random_state=0).fit(iris.data)

    sumd = np.zeros(shape=(nK, 1))
    print("!", sumd)
    for i in range(len(iris.data)):
        k = kmeans.labels_[i]
        ui = kmeans.cluster_centers_[k]
        d = LA.norm(iris.data[i] - ui)
        sumd[k] += d * d
    
    ak = np.sum(sumd) / nK     
    print("k = ", nK, ", ak = ", ak)
    


