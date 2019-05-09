from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

iris = load_iris()
test_id = [1,51,101]
test_data = iris.data[test_id]

t_data = np.delete(iris.data, test_id, axis=0)
t_target = np.delete(iris.target, test_id)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(t_data, t_target)
print (clf.predict(test_data))
