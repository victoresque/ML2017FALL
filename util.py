import pandas as pd
import numpy as np
from param import *

def load_train_data():
    train_X = pd.read_csv('preproc/pre_X.csv', header=None).values
    train_y = pd.read_csv('preproc/pre_y.csv', header=None).values
    train_data = np.append(train_X, train_y, axis=1)
    if local_valid:
        np.random.shuffle(train_data)
    train_X = train_data[:, :-1]
    train_y = train_data[:, -1:]
    valid_X = None
    valid_y = None

    n_rows = train_data.shape[0]

    if local_valid:
        v = n_rows // valid_fold // vn
        valid_X = [train_X[v * i:v * (i + 1), :] for i in range(vn)]
        valid_y = [train_y[v * i:v * (i + 1), :] for i in range(vn)]
        train_X = train_X[v * vn:n_rows, :]
        train_y = train_y[v * vn:n_rows, :]

    return train_X, train_y, valid_X, valid_y

def load_test_data():
    test_data = pd.read_csv('data/test.csv', header=None).replace('NR', '0').values
    n_rows = test_data.shape[0] // n_categories
    test_X = np.zeros((0, feature_len))
    for i in range(n_rows):
        test_X = np.append(test_X, test_data[i * n_categories:(i + 1) * n_categories, -feature_len:], axis=0)
    return test_X.astype(np.float)

def load_w():
    return pd.read_csv('result/w.csv', header=None).values.T

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def rmse(x, y):
    return rms(y - x)

def gradient(w, X, y, reg=0):
    return np.sum(2 * np.dot(y.T - np.dot(w, X.T), -X), axis=0) + 2 * reg * w

def extract(X):
    X_ext = np.zeros((0, feature_len))
    for i in range(X.shape[0] // n_categories):
        for ci in range(len(categories)):
            x = X[i*n_categories + categories[ci]]
            X_ext = np.append(X_ext, [x], axis=0)
            if cat_order[ci] >= 2:
                X_ext = np.append(X_ext, np.power([x], 2), axis=0)
            if cat_order[ci] >= 3:
                X_ext = np.append(X_ext, np.power([x], 3), axis=0)
    return X_ext

def flatten(X):
    feat_rows = np.sum(cat_order)
    X_flat = np.zeros((0, feat_rows * feature_len))
    for i in range(X.shape[0] // feat_rows):
        X_flat = np.append(X_flat, [X[i * feat_rows:(i + 1) * feat_rows, :].flatten()], axis=0)
    return X_flat

# converting [x] to [1 x]
def to_homogeneous(X):
    return np.insert(X, 0, [1], axis=1)

def clamp(y, y0):
    for i in range(y.shape[0]):
        if y[i] > y0[i] + clamp_thres:
            y[i] = y0[i] + clamp_thres
        if y[i] < y0[i] - clamp_thres:
            y[i] = y0[i] - clamp_thres
    return y
