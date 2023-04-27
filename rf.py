import numpy as np
from sklearn.utils import resample

from dtree import *

def set_diff2d(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)], 'formats':ncols * [A.dtype]}
    C = np.setdiff1d(A.copy().view(dtype), B.copy().view(dtype))
    return C

class RandomForest:
    def __init__(self, n_estimators=10, oob_score=False, loss=None, min_samples_leaf=3, max_features=0.3, reg=True):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.loss = loss
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.reg = reg

    def fit(self, X, y):
        """
        Reads (X, y) training set, fits all n_estimators trees to different,
        bootstrapped versions of the training data.  Keeps track of the indexes of
        the OOB records for each tree.  
        Computes the OOB validation score estimate and stores as self.oob_score_, to
        mimic sklearn.
        """
        scores_ = []
        trees = []
        if self.reg:
            Y_pred_sum = [0] * len(y)
            Y_pred_count = [0] * len(y)
        else:
            Y_pred_, Y_val = [], []
            uniq = list(sorted(set(y)))
        for _ in range(self.n_estimators):
            X_train, y_train, train_indx = resample(X, y, np.array(range(0, len(y))),n_samples=len(y), replace=True)
            val_indx = sorted([i for i in range(0, len(X)) if i not in train_indx])
            X_val, y_val = X[val_indx], y[val_indx]
            if self.reg:
                dtree = RegressionTree621(self.min_samples_leaf)
                dtree.fit(X_train, y_train)
                y_pred = dtree.predict(X_val)
                for i in range(len(y_pred)):
                    Y_pred_sum[val_indx[i]] += sum(y_pred[i])
                    Y_pred_count[val_indx[i]] += len(y_pred[i])
            else:
                dtree = ClassifierTree621(self.min_samples_leaf)
                dtree.fit(X_train, y_train)
                y_pred = dtree.predict(X_val)
                y_pred = [stats.mode(p)[0][0] for p in y_pred]
                Y_pred_.extend(y_pred)
                Y_val.extend(y_val)
                self.trees = [dtree]
                scores_.append(self.score(X_val, y_val)) 
            trees.append(dtree)
        if self.reg:
            new_y, new_pred = [], []
            for i in range(len(Y_pred_sum)):
                if Y_pred_count[i] == 0:
                    continue
                new_pred.append(Y_pred_sum[i]/Y_pred_count[i])
                new_y.append(y[i])
            self.oob_score_ = r2_score(new_y, new_pred)
        else:
            self.oob_score_ = (sum(scores_) / len(scores_))
        self.trees = trees
            
            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score, 
                        loss=np.var, min_samples_leaf=min_samples_leaf, max_features=max_features, reg=True)
        #self.min_samples_leaf = min_samples_leaf
        #self.trees = ...

    def predict(self, X_test) -> np.ndarray:
        """
        Reads a 2D nxp array with one or more records, computes the weighted average
        prediction from all trees in this forest. Weights each trees prediction by
        the number of observations in the leaf making that prediction.  Returns a 1D vector
        with the predictions for each input record of X_test.
        """
        Y_pred = []
        for tree in self.trees:
            Y_pred.append(tree.predict(X_test))
        Y_pred = np.array(Y_pred)
        fin_Y_pred = []
        for col in Y_pred.T:
            c, s = 0, 0
            for rs in col:
                s+= sum(rs)
                c+=len(rs)
            fin_Y_pred.append(s / c)
        return fin_Y_pred

        
    def score(self, X_test, y_test) -> float:
        """
        Reads a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collects the prediction for each record and then computes R^2 on that and y_test.
        """
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)
        
class RandomForestClassifier621(RandomForest):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score, 
                    loss=gini, min_samples_leaf=min_samples_leaf, max_features=max_features, reg=False)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        #self.trees = ...

    def predict(self, X_test) -> np.ndarray:
        Y_pred = []
        for tree in self.trees:
            Y_pred.append(tree.predict(X_test))
        Y_pred = np.array(Y_pred)
        fin_Y_pred = []
        for col in Y_pred.T:
            c = []
            for rs in col:
                c.extend(rs)
            fin_Y_pred.append( stats.mode(c)[0])
        return fin_Y_pred
        
    def score(self, X_test, y_test) -> float:
        """
        Reads a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collects the predicted class for each record and then computes accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)  