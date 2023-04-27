import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Makes decision based upon x_test[col] and split
        if x_test[self.col] < self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)



class LeafNode:
    def __init__(self, y, prediction):
        "Creates leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        self.y = y

    def predict(self, x_test):
        return self.y


def gini(x):
    """
    Returns the gini impurity score for values in y
    Assumes y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    uniques = list(set(x))
    d_c = {}
    for unique in uniques:
        d_c[unique] = 0 #initialization
    for val in x:
        d_c[val] += 1
    ps = list(d_c.values())
    ps_sum = sum(ps)
    ps = [(x/ps_sum) ** 2 for x in ps]
    return 1 - sum(ps)

    
def find_best_split(X, y, loss, min_samples_leaf, max_features):
    cols = range(0, len(X[0]))
    if max_features:
        cols = np.random.choice(len(X[0]), int(max_features * len(X[0])), replace=False)
    best = (-1, -1, loss(y))
    for col in cols:
        candidates = np.random.choice(X[:, col], 11)
        for split in candidates:
            yl = y[X[:, col] < split]
            yr = y[X[:, col] >= split]
            if len(yl) < min_samples_leaf or len(yr) < min_samples_leaf:
                continue
            l = (len(yl) * loss(yl) + len(yr) * loss(yr)) / (len(yl) + len(yr))
            if l == 0:
                return (col, split)
            if l < best[-1]:
                best = (col, split, l)
    return (best[0], best[1])
    
    
class DecisionTree:
    def __init__(self, min_samples_leaf=1, loss=None, max_features=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var for regression or gini for classification
        self.max_features = max_features
        
    def fit(self, X, y):
        """
        Creates a decision tree fit to (X,y) and saves as self.root, the root of
        the decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function acts as a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        Recursively creates and returns a decision tree fit to (X,y) for
        either a classification or regression.  This function calls self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree.create_leaf() or ClassifierTree.create_leaf() depending
        on the type of self.
        """
        if len(X) <= self.min_samples_leaf or len(np.unique(X)) == 1:
            return self.create_leaf(y)
        col, split = find_best_split(X, y, self.loss, self.min_samples_leaf, self.max_features)
        if col == -1:
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:, col] < split], y[X[:, col] < split])
        rchild = self.fit_(X[X[:, col] >= split], y[X[:, col] >= split])
        return DecisionNode(col, split, lchild, rchild)


    def predict(self, X_test):
        """
        Makes a prediction for each record in X_test and returns as array.
        This method is inherited by RegressionTree and ClassifierTree and
        works for both without modification.
        """
        y_pred = []
        for x_test in X_test:
            y_pred.append(self.root.predict(x_test))
        return y_pred


class RegressionTree(DecisionTree):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.var, max_features=0.3)

    def score(self, X_test, y_test):
        "Returns the R^2 of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)

    def create_leaf(self, y):
        """
        Returns a new LeafNode for regression, passes y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree(DecisionTree):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Returns the accuracy_score() of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def create_leaf(self, y):
        """
        Returns a new LeafNode for classification, passes y and mode(y) to
        the LeafNode constructor. 
        """
        return LeafNode(y, stats.mode(y)[0])