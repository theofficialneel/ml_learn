from sklearn import tree
ft = [[140,1],[130,1],[150,0],[170,0]]
lb = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(ft, lb)
print (clf.predict([[150,0]]))


