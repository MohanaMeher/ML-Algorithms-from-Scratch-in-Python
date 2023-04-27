import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def normalize(X): # creating standard variables here (u-x)/sigma
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        s = np.std(X[:,j])
        X[:,j] = (X[:,j] - u) / s

def loss_gradient(X, y, B, lmbda):
    return -np.dot(np.transpose(X), y - np.dot(X, B))

def loss_ridge(X, y, B, lmbda):
    return np.dot(np.transpose(y - (X*B)), y - (X*B)) + (lmbda * np.transpose(B) * B)

def loss_gradient_ridge(X, y, B, lmbda):
    return -np.dot(np.transpose(X), y - np.dot(X, B)) + (lmbda * B)

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def log_likelihood(X, y, B,lmbda):
    n = len(X)
    arr = []
    for i in range(n):
        arr.append(-np.dot(y[i]*X[i], B) - np.log(1 + np.exp(np.dot(X[i], B))))
    return np.sum(arr)


def log_likelihood_gradient(X, y, B, lmbda):
    return -np.dot(np.transpose(X), y - sigmoid(np.dot(X, B)))

def minimize(X, y, loss_gradient,
              eta=0.00001, lmbda=0.0,
              max_iter=1000, addB0=True,
              precision=1e-9):
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    # For linear or logistic regression, estimates B0 by adding a column of 1s and increase p by 1
    # for Ridge regression we will set addB0 to False and estimate B0 as part of the RidgeRegression fit method
    if addB0:
        X0 = np.ones((n,1))
        X = np.hstack((X0, X))
        p += 1

    # initiates a random vector of Bs of size p
    B = np.random.random_sample(size=(p, 1)) * 2 - 1  # make between [-1,1)

    # minimization procedure 
    prev_B = B
    eps = 1e-5 # prevent division by 0
    
    # retains the history of the gradients to use as part of our iteration procedure
    # stopping condition L2-norm of the gradient <= precision
    
    h, i = 0, 0
    gradient = loss_gradient(X, y, B, lmbda)
    while np.linalg.norm(gradient) > precision and i <= max_iter:
        h += (gradient ** 2)
        B = prev_B - (eta / (np.sqrt(h + eps))) * gradient
        gradient = loss_gradient(X, y, B, lmbda)
        prev_B = B
        i+=1
    return B

    

class LinearRegression: 
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class LogisticRegression: 
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict_proba(self, X):
        """
        Computes the probability that the target is 1. 
        """
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        pre = np.dot(X, self.B)
        return sigmoid(pre)

    def predict(self, X):
        """
        Calls self.predict_proba() to get probabilities then, for each x in X,
        return a 1 if P(y==1,x) > 0.5 else 0.
        """
        def proba(x): 
            if x > 0.5:
                return 1
            return 0
        return np.array([proba(x) for x in list(self.predict_proba(X))])

    def fit(self, X, y):
        self.B = minimize(X, y,
                          log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class RidgeRegression:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        B = minimize(X, y,
                          loss_gradient_ridge,
                          self.eta,
                          self.lmbda,
                          self.max_iter, addB0=False)
        y_mean = np.mean(y.reshape(-1, 1))
        B = np.vstack(([y_mean], B))
        self.B = B