import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import MeanShift, estimate_bandwidth

iris = load_iris()
#X = iris.data
X = iris.data[:, 0:2]  # we only take the first two features.
Y = iris.target
bw = estimate_bandwidth(X)
# bw    # try to assign a value  for bandwisth
print(bw)
clustering = MeanShift().fit(X)

print(len(clustering.labels_))
print(clustering.labels_)
print(Y)

# Plot the real groups
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()
# Plot the groups clustered by mean shift
plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()