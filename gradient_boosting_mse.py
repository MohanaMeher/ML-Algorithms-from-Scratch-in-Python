import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def load_dataset(path="data/rent-ideal.csv"):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    y = dataset[:, -1]
    X = dataset[:, 0:- 1]
    return X, y

def gradient_boosting_mse(X, y, num_iter, max_depth=1, nu=0.1):
    """Reads predictors X, an array y, and num_iter (big M in the sum)

    Returns the y_mean and trees 
   
    Input: X, y, num_iter
           max_depth
           nu (shrinkage parameter)

    Outputs: y_mean, array of trees from DecisionTreeRegressor 
    """
    trees = []
    N, _ = X.shape
    y_mean = np.mean(y)
    y_pred = np.full((X.shape[0],), y_mean)
    fm = y_mean
    for m in range(num_iter):
        r = np.subtract(y, y_pred)
        Tm = DecisionTreeRegressor(max_depth=max_depth)
        Tm.fit(X, r)
        y_pred = y_pred + (nu * Tm.predict(X))
        trees.append(Tm)
    return y_mean, trees  

def gradient_boosting_predict(X, trees, y_mean,  nu=0.1):
    """
    Reads X, trees, y_mean predict y_hat
    """
    y_pred = np.full((X.shape[0],), y_mean)
    for tree in trees:
        y_pred = np.add(y_pred, nu * tree.predict(X))
    return y_pred

