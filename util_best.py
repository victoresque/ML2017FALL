import numpy as np
import pandas as pd
from param_best import *

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def rmse(x, y):
    return rms(y - x)

def gradient(w, X, y, reg=0):
    return np.sum(2 * np.dot(y - np.dot(w, X.T), -X), axis=0) + 2 * reg * w

def load_w(w_path):
    return pd.read_csv(w_path, header=None).values.T

def load_test_data(test_path):
    test_data = pd.read_csv(test_path, header=None).replace('NR', '0').as_matrix()
    n_rows = test_data.shape[0] // n_categories
    test_X = np.zeros((0, x_len))
    for i in range(n_rows):
        td = np.zeros((1, 0))
        for cat in categories:
            td = np.append(td, test_data[i * n_categories + cat, -feature_len:])
        test_X = np.append(test_X, [td.flatten().astype(np.float)], axis=0)
    return test_X.astype(np.float)

# Feature scaling, bugged
def feature_scaling(X):
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i])

# converting [x] to [1 x]
def to_homogeneous(X):
    return np.insert(X, 0, [1], axis=1)

def add_dimension(X):
    X_2nd = np.zeros((X.shape[0], 0))
    X_all2nd = np.zeros((X.shape[0], 0))
    X_3rd = np.zeros((X.shape[0], 0))

    n_cat = n_categories
    f_len = feature_len
    for i in cat_2nd:
        X_2nd = np.append(X_2nd, np.square(X[:,(i+1)*f_len-lookback_2nd:(i+1)*f_len]), axis=1)
    for i in cat_all2nd:
        X_all2nd = np.append(X_all2nd, np.array(X[:,(i+1)*f_len-lookback_all_2nd:(i+1)*f_len]), axis=1)
    for i in cat_3rd:
        X_3rd = np.append(X_3rd, np.power(X[:,i*f_len:(i+1)*f_len], 3), axis=1)

    return np.append(np.append(np.append(X, X_2nd, axis=1), X_3rd, axis=1),
                     [np.dot(np.transpose([x]), [x]).flatten() for x in X_all2nd],
                     axis=1)

def transform(X):
    #feature_scaling(X)
    return to_homogeneous(add_dimension(X))