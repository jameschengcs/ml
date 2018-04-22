from sklearn.cluster import KMeans
import numpy as np
#####
X = np.array([[0.5, 2], [1, 4.5], [1, 0.25],
             [4, 2], [4, 4], [4, 0]])
###
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)

targets = kmeans.predict([[0, 0], [4, 3]])
print(targets)
