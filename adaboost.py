import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_data(filename):
    """ 
    Reads and parses a text file as input

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)

    Returns X and Y numpy arrays
    """
    data = np.genfromtxt(filename, dtype=float,
                     encoding=None, delimiter=",")
    X = np.delete(data, -1, axis=1).astype(float)
    Y = data[:,-1]
    convert = lambda t: -1 if t == 0 else 1
    f = np.vectorize(convert)
    Y = f(Y).astype(float)
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """
    Reads numpy matrix X, a array y and num_iter 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}

    Returns trees and weights 
    """
    trees = []
    trees_weights = [] 
    N, _ = X.shape
    d = np.ones(N) / N
    for m in range(num_iter):
        h = DecisionTreeClassifier(max_depth=1, random_state=0)
        h.fit(X, y, sample_weight=d)
        equ = h.predict(X) != y
        err = np.sum(np.multiply(d, equ)) / np.sum(d)
        if err == 0:
            alpha = np.inf
        else: 
            alpha = np.log((1-err)/err)
        trees.append(h)
        trees_weights.append(alpha)
        d = np.where(equ == 0, d, d * np.exp(alpha))
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """
    Reads X, trees and weights and predicts Y
    """
    y = np.zeros(X.shape[0])
    for i in range(len(trees)):
        y += trees_weights[i] * trees[i].predict(X)
    convert = lambda t: -1 if t == 0 else t
    f = np.vectorize(convert)
    y = f(np.sign(y)).astype(float)
    return y