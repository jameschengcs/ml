import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
import pydot

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# query from dataset
q = iris.data[:1, :]
result = clf.predict(q)
print(q)
print(result)
print(iris.target_names[result])

# customized  query 
q = np.array([[8.1, 0.5, 5.4, 8.2]])
result = clf.predict(q)
print(q)
print(iris.target_names[result])

# Visualization  (tree->dot file->image file)
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('mytree.png')
#

#(graph,) = pydot.graph_from_dot_file('mytree.dot')
#graph.write_png('mytree.png')

