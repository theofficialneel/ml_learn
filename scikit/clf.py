from scipy.spatial import distance
def eucDist(a, b):
    return distance.euclidean(a, b)

class basicKNN():
    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
    
    def predict(self, X_test):
        preds = []
        for row in X_test:
            pred_label = self.closest(row)
            preds.append(pred_label)

        return preds

    def closest(self, row):
        close = eucDist(row, self.X[0])
        close_index = 0
        for i in range(1, len(self.X)):
            d = eucDist(row, self.X[i])
            if d < close:
                close = d
                close_index = i

        return self.y[close_index]

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = basicKNN()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print (accuracy_score(y_test, preds))
