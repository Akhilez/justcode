"""
Machine learning algorithm
dataset:
-------------features-----------|----label-----
weight          |bumpiness      |type
----------------------------------------------- Train-Set
140             |1  (yes)       |0  (orange)
130             |1  (yes)       |0  (orange)
150             |0  (no)        |1  (apple)
170             |0  (no)        |1  (apple)
----------------------------------------------- Test-Set
160             |0  (no)        |1  (apple)
-----------------------------------------------
"""
from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[160, 0]]))